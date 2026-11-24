using System.Collections.Concurrent;
using System.Threading.Channels;
using H3NetRuntime;

namespace H3NetRuntime.WebUI;

/// <summary>
/// Serialized generation queue. The native engine runs one job at a time (a
/// single DiT residency), so jobs are queued rather than fanned out; the queue
/// exists to make the WebUI feel responsive while a 60-second render runs.
/// </summary>
public sealed class JobManager : IDisposable
{
    private const int MaxPixels = 768 * 1344;
    private const int MinAlignedFrames = 22;
    /* The engine's Ref2VA path refuses anything larger. */
    private const int MaxReferenceImages = 9;
    private const int MaxReferenceVideos = 3;
    private const int MaxReferenceAudios = 3;
    private const int MaxReferences = 12;

    private readonly WebUiOptions _options;
    private readonly RuntimeHost _host;
    private readonly HistoryStore _history;
    private readonly ILogger<JobManager> _logger;
    private readonly Channel<Job> _queue;
    private readonly ConcurrentDictionary<string, Job> _jobs = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim _engineGate = new(1, 1);
    private readonly CancellationTokenSource _shutdown = new();
    private Task? _worker;
    private int _disposed;

    public JobManager(WebUiOptions options, RuntimeHost host, HistoryStore history,
                      ILogger<JobManager> logger)
    {
        _options = options;
        _host = host;
        _history = history;
        _logger = logger;
        _queue = Channel.CreateUnbounded<Job>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false
        });
    }

    public void Start()
    {
        _worker ??= Task.Factory.StartNew(WorkerLoopAsync, CancellationToken.None,
            TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
            TaskScheduler.Default).Unwrap();
    }

    // ---------------------------------------------------------------- submit

    /// <summary>Validates, applies the edition rules and enqueues a job.</summary>
    public JobView Submit(GenerateApiRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (string.IsNullOrWhiteSpace(request.Prompt))
            throw new ArgumentException("提示词不能为空。");

        PresetDefinition preset = Presets.Find(request.Preset);
        // The preset supplies canvas/duration defaults in both editions; the
        // general edition additionally locks every sampling knob to it.
        Presets.ApplyDefaults(request, preset);
        if (!_options.Advanced) Presets.Force(request, preset);

        ModeInfo mode = GenerationModes.Normalize(request);
        Validate(request, mode);

        string id = NewId();
        string directory = Path.Combine(_options.OutputDir, id);
        Directory.CreateDirectory(directory);

        var job = new Job
        {
            Id = id,
            Kind = "generate",
            Request = request,
            Prompt = request.Prompt!.Trim(),
            Preset = preset.Id,
            Mode = mode.Id,
            CreatedAt = DateTimeOffset.Now,
            Status = "queued",
            LivePreview = request.LivePreview ?? true,
            OutputPath = Path.Combine(_options.OutputDir, id + ".mp4"),
            Directory = directory
        };
        _jobs[id] = job;
        _logger.LogInformation("queued {Job} {Width}x{Height} {Frames}f preset={Preset} mode={Mode}",
            id, request.Width, request.Height, request.Frames, preset.Id, mode.Id);

        if (!_queue.Writer.TryWrite(job))
            throw new InvalidOperationException("任务队列已关闭。");
        return Snapshot(job);
    }

    private void Validate(GenerateApiRequest request, ModeInfo mode)
    {
        int width = request.Width ?? 0;
        int height = request.Height ?? 0;
        if (width < 32 || height < 32 || width % 32 != 0 || height % 32 != 0)
            throw new ArgumentException("宽高必须是 32 的倍数且不小于 32。");
        if ((long)width * height > MaxPixels)
            throw new ArgumentException("画布超过 768×1344 像素上限。");

        int frames = request.Frames ?? 0;
        if (frames < 5)
            throw new ArgumentException("帧数不能小于 5。");
        if (Presets.AlignFrames(frames) < MinAlignedFrames)
            throw new ArgumentException("引擎至少需要一个 22 帧的解码块。");
        if (Presets.AlignFrames(frames) > 362)
            throw new ArgumentException("帧数超过 362 上限。");

        int steps = request.Steps ?? 0;
        if (steps < 2 || steps > 1000)
            throw new ArgumentException("采样步数必须在 2..1000 之间。");
        if (request.Pdd8 == true && steps != 8)
        {
            // HyperFlow checkpoints sample their own 4/6/8-step grid and ignore
            // the legacy PDD adaptor. Dropping the flag here keeps the advanced
            // UI usable when it is left checked from an older configuration;
            // a real PDD pack still fails in the native validator if steps != 8.
            request.Pdd8 = false;
        }

        if (mode.Id == GenerationModes.FirstLastFrame)
        {
            bool hasFirst = !string.IsNullOrWhiteSpace(request.FirstFrame);
            bool hasLast = !string.IsNullOrWhiteSpace(request.LastFrame);
            if (!hasFirst && !hasLast)
                throw new ArgumentException("首尾帧模式至少需要一张首帧或尾帧。");
            if (hasFirst && ResolveUpload(request.FirstFrame) is null)
                throw new ArgumentException("首帧文件不存在，请重新上传。");
            if (hasLast && ResolveUpload(request.LastFrame) is null)
                throw new ArgumentException("尾帧文件不存在，请重新上传。");
        }

        if (mode.RequiresRef2Va)
        {
            (int images, int videos, int audios) = CountReferences(request);
            if (images + videos + audios == 0)
                throw new ArgumentException("多图参考模式至少需要一张参考图。");
            if (images > MaxReferenceImages)
                throw new ArgumentException($"参考图最多 {MaxReferenceImages} 张。");
            if (videos > MaxReferenceVideos)
                throw new ArgumentException($"参考视频最多 {MaxReferenceVideos} 段。");
            if (audios > MaxReferenceAudios)
                throw new ArgumentException($"参考音频最多 {MaxReferenceAudios} 段。");
            if (images + videos + audios > MaxReferences)
                throw new ArgumentException($"参考素材总数最多 {MaxReferences} 个。");
            if (!_host.Ref2VaAvailable)
                throw new ArgumentException(
                    "当前模型包不含 Ref2VA 权重，无法做多图参考。" +
                    "请先按“模型下载指引”补上 Ref2VA/transformer，或改用文生视频 / 首尾帧模式。");
        }
    }

    private static (int Images, int Videos, int Audios) CountReferences(GenerateApiRequest request)
    {
        int images = 0, videos = 0, audios = 0;
        if (request.References is { Count: > 0 })
        {
            foreach (ReferenceInput reference in request.References)
            {
                switch ((reference.Kind ?? "image").ToLowerInvariant())
                {
                    case "image": images++; break;
                    case "video": videos++; break;
                    case "audio": audios++; break;
                    case "video_audio":
                    case "video+audio": videos++; audios++; break;
                    default: images++; break;
                }
            }
        }
        if (request.ReferenceImages is { Count: > 0 }) images += request.ReferenceImages.Count;
        if (!string.IsNullOrWhiteSpace(request.ReferenceImage)) images++;
        if (!string.IsNullOrWhiteSpace(request.ReferenceVideo)) videos++;
        if (!string.IsNullOrWhiteSpace(request.ReferenceAudio)) audios++;
        return (images, videos, audios);
    }

    // ---------------------------------------------------------------- warm up

    /// <summary>
    /// Reads and prepares the DiT for the given layout so the first real
    /// generation does not pay for it. Refuses to interleave with a running job.
    /// </summary>
    public PreloadOutcome Preload(PreloadApiRequest request)
    {
        if (!_engineGate.Wait(0))
            throw new InvalidOperationException("引擎正在生成，请等当前任务结束后再预热。");
        try
        {
            H3Runtime runtime = _host.Ensure();
            var native = new H3GenerateRequest
            {
                Width = request.Width,
                Height = request.Height,
                Frames = request.Frames is null ? null : Presets.AlignFrames(request.Frames.Value),
                Steps = request.Steps,
                Pdd8 = request.Pdd8,
                DitLayers = request.DitLayers,
                DenoiseReuse = request.DenoiseReuse,
                CoreReuse = request.CoreReuse
            };
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            runtime.Preload(native);
            stopwatch.Stop();
            double elapsed = stopwatch.Elapsed.TotalMilliseconds;
            _logger.LogInformation("preload finished in {Ms:F0} ms", elapsed);
            return new PreloadOutcome(elapsed, _host.ModelDir);
        }
        finally
        {
            _engineGate.Release();
        }
    }

    // ----------------------------------------------------------------- query

    public JobView? Get(string id) =>
        _jobs.TryGetValue(id, out Job? job) ? Snapshot(job) : null;

    /// <summary>Queued and running jobs, in queue order.</summary>
    public List<JobView> Active()
    {
        Job[] jobs = _jobs.Values
            .Where(j => j.Status is "queued" or "running" or "cancelling")
            .OrderBy(j => j.CreatedAt)
            .ToArray();
        var views = new List<JobView>(jobs.Length);
        int position = 0;
        foreach (Job job in jobs)
        {
            JobView view = Snapshot(job);
            view.QueuePosition = job.Status == "queued" ? ++position : 0;
            views.Add(view);
        }
        return views;
    }

    public IReadOnlyList<JobSummary> History() => _history.Snapshot();

    /// <summary>
    /// Cancels a job by full id or by a unique id prefix. A prefix that matches
    /// more than one active job is rejected so a short string can never cancel
    /// the wrong generation.
    /// </summary>
    public bool Cancel(string idOrPrefix)
    {
        if (string.IsNullOrWhiteSpace(idOrPrefix)) return false;
        string id = idOrPrefix.Trim();
        if (!_jobs.TryGetValue(id, out Job? job))
        {
            Job[] matches = _jobs.Values
                .Where(j => j.Status is "queued" or "running" or "cancelling")
                .Where(j => j.Id.StartsWith(id, StringComparison.OrdinalIgnoreCase))
                .ToArray();
            if (matches.Length != 1) return false;
            job = matches[0];
            id = job.Id;
        }
        lock (job.Sync)
        {
            if (job.Status is "completed" or "failed" or "cancelled") return false;
            if (job.Status == "queued")
            {
                job.Status = "cancelled";
                job.Phase = "已取消";
                job.FinishedAt = DateTimeOffset.Now;
            }
            else
            {
                job.Status = "cancelling";
            }
        }
        try { job.Cts.Cancel(); }
        catch (ObjectDisposedException) { }
        _logger.LogInformation("cancel requested for {Job}", id);
        return true;
    }

    // --------------------------------------------------------------- worker

    private async Task WorkerLoopAsync()
    {
        try
        {
            await foreach (Job job in _queue.Reader.ReadAllAsync(_shutdown.Token))
            {
                if (job.Cts.IsCancellationRequested)
                {
                    Finish(job, "cancelled", null);
                    continue;
                }
                await _engineGate.WaitAsync(_shutdown.Token).ConfigureAwait(false);
                try
                {
                    await Task.Factory.StartNew(() => Execute(job), CancellationToken.None,
                        TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
                        TaskScheduler.Default).ConfigureAwait(false);
                }
                finally
                {
                    _engineGate.Release();
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _logger.LogCritical(ex, "job worker stopped unexpectedly");
        }
    }

    private void Execute(Job job)
    {
        if (job.Cts.IsCancellationRequested)
        {
            Finish(job, "cancelled", null);
            return;
        }

        if (!_host.TryEnsure(out H3Runtime? runtime, out string? error) || runtime is null)
        {
            Finish(job, "failed", (int)H3Status.Model, error ?? "模型加载失败。");
            return;
        }

        if (_options.Advanced && !string.IsNullOrWhiteSpace(job.Request.ModelDir) &&
            !PathsEqual(job.Request.ModelDir!, _host.ModelDir))
        {
            try
            {
                _host.Recreate(job.Request.ModelDir, null);
                runtime = _host.Ensure();
            }
            catch (Exception ex)
            {
                Finish(job, "failed", (int)H3Status.Model, "切换模型失败：" + ex.Message);
                return;
            }
        }

        PreviewPipeline? preview = null;
        if (job.LivePreview)
        {
            preview = new PreviewPipeline(job.Id, job.Directory, _logger);
            lock (job.Sync) job.Preview = preview;
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        lock (job.Sync)
        {
            job.Status = "running";
            job.StartedAt = DateTimeOffset.Now;
            job.StartedTicks = Environment.TickCount64;
            job.Phase = "准备中";
            job.Progress.Configure(job.Mode, job.Request.Frames ?? 56,
                job.Request.Steps ?? 8, job.Request.DitLayers ?? 50);
            job.Progress.Update("prepare", 0, 1, job.StartedTicks);
        }
        _logger.LogInformation("[{Job}] {Line}", job.Id, "准备       -         0.0%  0.0s");

        H3GenerateRequest native = BuildNativeRequest(job);
        var progress = new InlineProgress<H3Progress>(value => OnProgress(job, value));
        Action<H3Frame>? onFrame = preview is null ? null : frame => OnFrame(preview, job, frame);
        int every = Math.Max(1, job.Request.PreviewEvery ?? 4);
        job.PreviewStride = every;

        try
        {
            H3GenerateResult result = runtime.Generate(native, progress, onFrame,
                captureFrames: preview is not null, cancellationToken: job.Cts.Token);
            stopwatch.Stop();
            preview?.CompleteAsync().GetAwaiter().GetResult();
            Complete(job, result, stopwatch.Elapsed.TotalMilliseconds);
        }
        catch (OperationCanceledException)
        {
            stopwatch.Stop();
            preview?.CompleteAsync().GetAwaiter().GetResult();
            // Generation has returned, so no native thread still owns the
            // buffers. Drop cached decoder/embeddings now instead of leaving
            // the cancelled job's VRAM held until the next cache clear.
            try { runtime.ClearCache(); }
            catch (Exception ex) { _logger.LogDebug(ex, "cache clear after cancel failed"); }
            Finish(job, "cancelled", null);
            TryDelete(job.OutputPath);
        }
        catch (H3RuntimeException ex)
        {
            stopwatch.Stop();
            preview?.CompleteAsync().GetAwaiter().GetResult();
            _logger.LogError(ex, "generation {Job} failed", job.Id);
            Finish(job, "failed", (int)ex.Status, ex.Message);
            TryDelete(job.OutputPath);
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            preview?.CompleteAsync().GetAwaiter().GetResult();
            _logger.LogError(ex, "generation {Job} crashed", job.Id);
            Finish(job, "failed", (int)H3Status.Generate, ex.Message);
            TryDelete(job.OutputPath);
        }
    }

    private H3GenerateRequest BuildNativeRequest(Job job)
    {
        GenerateApiRequest r = job.Request;
        long? seed = r.Seed;
        if (seed is null or < 0) seed = Random.Shared.NextInt64(0, int.MaxValue);

        var native = new H3GenerateRequest
        {
            Prompt = job.Prompt,
            Width = r.Width,
            Height = r.Height,
            Frames = Presets.AlignFrames(r.Frames ?? 22),
            Steps = r.Steps,
            Seed = (ulong)seed.Value,
            OutputPath = job.OutputPath,
            FirstFramePath = ResolveUpload(r.FirstFrame),
            LastFramePath = ResolveUpload(r.LastFrame),
            Pdd8 = r.Pdd8,
            DitLayers = r.DitLayers,
            DenoiseReuse = r.DenoiseReuse,
            CoreReuse = r.CoreReuse,
            RenderWidth = r.RenderWidth,
            RenderHeight = r.RenderHeight,
            ReferenceImageSize = r.ReferenceImageSize,
            TokenReduction = r.TokenReduction,
            SolAttn = r.SolAttn,
            Fbc = r.Fbc,
            PreviewDenoise = r.PreviewDenoise,
            SsdStreaming = r.SsdStreaming,
            UseInt8RowFc2 = r.UseInt8RowFc2,
            UseReferenceRope = r.UseReferenceRope,
            KeepDit = r.KeepDit ?? _options.KeepDit,
            ReferenceImage = ResolveUpload(r.ReferenceImage),
            ReferenceVideo = ResolveUpload(r.ReferenceVideo),
            ReferenceAudio = ResolveUpload(r.ReferenceAudio),
            NegativePrompt = r.NegativePrompt
        };

        if (r.ReferenceImages is { Count: > 0 })
            native.ReferenceImages = r.ReferenceImages
                .Select(ResolveUpload)
                .Where(p => !string.IsNullOrEmpty(p))
                .Select(p => p!)
                .ToList();

        if (r.References is { Count: > 0 })
        {
            var ordered = new List<H3Reference>();
            foreach (ReferenceInput input in r.References)
            {
                string? path = ResolveUpload(input.Path);
                if (path is null) continue;
                ordered.Add(new H3Reference
                {
                    Kind = string.IsNullOrWhiteSpace(input.Kind) ? "image" : input.Kind!,
                    Path = path,
                    AudioPath = ResolveUpload(input.AudioPath),
                    IncludeEmbeddedAudio = input.IncludeEmbeddedAudio ?? false
                });
            }
            if (ordered.Count > 0) native.References = ordered;
        }
        return native;
    }

    /// <summary>Accepts either an absolute path or a /files/uploads/... URL.</summary>
    private string? ResolveUpload(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return null;
        string trimmed = value.Trim();
        const string prefix = "/files/uploads/";
        if (trimmed.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            string name = Uri.UnescapeDataString(trimmed[prefix.Length..]);
            string candidate = Path.Combine(_options.UploadDir, name);
            return File.Exists(candidate) ? candidate : null;
        }
        return File.Exists(trimmed) ? trimmed : null;
    }

    private void OnProgress(Job job, H3Progress value)
    {
        string? line;
        lock (job.Sync)
        {
            job.Phase = value.Phase;
            job.Completed = value.Completed;
            job.Total = value.Total;
            line = job.Progress.Update(value.Phase, value.Completed, value.Total,
                job.StartedTicks);
        }
        // Console keeps the timeline: the page has the bar, the log has "which
        // stage was slow" without opening devtools.
        if (line is not null)
            _logger.LogInformation("[{Job}] {Line}", job.Id, line);
    }

    private static void OnFrame(PreviewPipeline preview, Job job, H3Frame frame)
    {
        if (frame.Pixels.Length == 0) return;
        int count = Interlocked.Increment(ref job.FrameCount);
        bool last = frame.FrameCount > 0 && frame.FrameIndex >= frame.FrameCount - 1;
        if (count % job.PreviewStride != 0 && !last) return;
        preview.Push(frame.Pixels, frame.Width, frame.Height, frame.Stride,
            frame.FrameIndex, frame.FrameCount, frame.DenoiseStep, frame.DenoiseSteps);
    }

    private void Complete(Job job, H3GenerateResult result, double elapsedMs)
    {
        string? videoUrl = ToUrl(result.OutputPath);
        PreviewPipeline? preview;
        lock (job.Sync)
        {
            job.Status = "completed";
            job.FinishedAt = DateTimeOffset.Now;
            job.FinishedTicks = Environment.TickCount64;
            job.Result = result;
            job.VideoUrl = videoUrl;
            job.Phase = "完成";
            job.Completed = job.Total;
            job.Progress.Finish();
            preview = job.Preview;
            if (preview is not null) job.PosterUrl = preview.PosterUrl;
        }

        _history.Add(new JobSummary
        {
            Id = job.Id,
            Prompt = job.Prompt.Length > 160 ? job.Prompt[..160] : job.Prompt,
            Preset = job.Preset,
            CreatedAt = job.CreatedAt,
            ElapsedMs = Math.Round(elapsedMs),
            Width = result.Width,
            Height = result.Height,
            Frames = result.Frames,
            Seed = result.Seed,
            VideoUrl = videoUrl,
            PosterUrl = preview?.PosterUrl,
            Status = "completed"
        });
        _logger.LogInformation("job {Job} completed in {Seconds:F1}s ({Fps:F2} fps)",
            job.Id, elapsedMs / 1000.0, result.FramesPerSecond);
    }

    private void Finish(Job job, string status, int? statusCode, string? error = null)
    {
        lock (job.Sync)
        {
            job.Status = status;
            job.FinishedAt ??= DateTimeOffset.Now;
            job.FinishedTicks = Environment.TickCount64;
            job.ErrorStatus = statusCode;
            if (error is not null) job.Error = error;
            if (status == "cancelled") job.Phase = "已取消";
        }
        if (status == "failed")
            _logger.LogWarning("job {Job} failed: {Error}", job.Id, error);
    }

    private string? ToUrl(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return null;
        string relative = Path.GetRelativePath(_options.OutputDir, path).Replace('\\', '/');
        return relative.StartsWith("..", StringComparison.Ordinal)
            ? null
            : "/files/outputs/" + relative;
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch { /* best effort */ }
    }

    private static bool PathsEqual(string left, string right)
    {
        try
        {
            return string.Equals(Path.GetFullPath(left).TrimEnd('\\', '/'),
                                 Path.GetFullPath(right).TrimEnd('\\', '/'),
                                 StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }

    private static string NewId() => Guid.NewGuid().ToString("N")[..10];

    /// <summary>
    /// Remaining time. <see cref="JobProgress"/> already smooths the rate and
    /// stops the estimate from climbing once the run is under way, so the page
    /// shows exactly the number the console prints. A plain elapsed/overall
    /// ratio is only the fallback for the first few seconds.
    /// </summary>
    private static double? EstimateEta(Job job)
    {
        if (job.StartedTicks == 0 || job.FinishedTicks != 0) return null;
        double overall = job.Progress.Overall;
        if (overall < 0.05) return null;
        double elapsed = Environment.TickCount64 - job.StartedTicks;
        if (elapsed < 3000) return null;
        double eta = job.Progress.EtaSeconds;
        double remaining = eta > 0 ? eta * 1000.0
            : elapsed * (1 - overall) / overall;
        return Math.Round(Math.Min(remaining, TimeSpan.FromHours(6).TotalMilliseconds));
    }

    // -------------------------------------------------------------- snapshot

    private JobView Snapshot(Job job)
    {
        lock (job.Sync)
        {
            var view = new JobView
            {
                Id = job.Id,
                Kind = job.Kind,
                Status = job.Status,
                Prompt = job.Prompt,
                Preset = job.Preset,
                Mode = job.Mode,
                Request = job.Request,
                CreatedAt = job.CreatedAt,
                StartedAt = job.StartedAt,
                FinishedAt = job.FinishedAt,
                Result = job.Result,
                Error = job.Error,
                ErrorStatus = job.ErrorStatus,
                VideoUrl = job.VideoUrl,
                PosterUrl = job.PosterUrl ?? job.Preview?.PosterUrl,
                LivePreview = job.LivePreview,
                VaePath = _options.KeepResidentVae ? "resident-slow" : "fast",
                Progress = job.Phase.Length == 0 && job.Status == "queued"
                    ? null
                    : new JobProgressView
                    {
                        Phase = job.Phase,
                        Stage = job.Progress.Label.Length > 0
                            ? job.Progress.Label
                            : job.Phase,
                        Completed = job.Completed,
                        Total = job.Total,
                        Fraction = job.Total > 0
                            ? Math.Clamp((double)job.Completed / job.Total, 0, 1)
                            : null,
                        Overall = Math.Round(job.Progress.Overall, 4),
                        ElapsedMs = job.StartedTicks == 0
                            ? 0
                            : Math.Round((double)((job.FinishedTicks != 0
                                ? job.FinishedTicks
                                : Environment.TickCount64) - job.StartedTicks)),
                        EtaMs = EstimateEta(job)
                    }
            };
            view.PreviewUrl = job.Preview?.PreviewUrl;
            return view;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _queue.Writer.TryComplete();
        _shutdown.Cancel();
        try { _worker?.Wait(TimeSpan.FromSeconds(5)); } catch { /* shutdown */ }
        foreach (Job job in _jobs.Values)
        {
            try { job.Cts.Cancel(); } catch { /* shutdown */ }
            job.Cts.Dispose();
        }
        _engineGate.Dispose();
        _shutdown.Dispose();
    }

    // ------------------------------------------------------------------ types

    /// <summary>Result of a warm-up call.</summary>
    public sealed record PreloadOutcome(double ElapsedMs, string ModelDir);

    private sealed class Job
    {
        public readonly object Sync = new();
        public string Id = string.Empty;
        public string Kind = "generate";
        public GenerateApiRequest Request = new();
        public string Prompt = string.Empty;
        public string? Preset;
        public string Mode = GenerationModes.Text;
        public DateTimeOffset CreatedAt;
        public DateTimeOffset? StartedAt;
        public DateTimeOffset? FinishedAt;
        public string Status = "queued";
        public string Phase = string.Empty;
        public int Completed;
        public int Total;
        public readonly JobProgress Progress = new();
        public long StartedTicks;
        public long FinishedTicks;
        public H3GenerateResult? Result;
        public string? Error;
        public int? ErrorStatus;
        public string? VideoUrl;
        public string? PosterUrl;
        public bool LivePreview;
        public string OutputPath = string.Empty;
        public string Directory = string.Empty;
        public PreviewPipeline? Preview;
        public int PreviewStride = 4;
        public int FrameCount;
        public readonly CancellationTokenSource Cts = new();
    }

    /// <summary>IProgress that runs inline; the engine callback must not queue.</summary>
    private sealed class InlineProgress<T> : IProgress<T>
    {
        private readonly Action<T> _sink;
        public InlineProgress(Action<T> sink) => _sink = sink;
        public void Report(T value) => _sink(value);
    }
}
