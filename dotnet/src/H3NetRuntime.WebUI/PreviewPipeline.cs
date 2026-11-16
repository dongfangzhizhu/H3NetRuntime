using System.Threading.Channels;

namespace H3NetRuntime.WebUI;

/// <summary>
/// Turns engine frames into PNG files for the browser. The engine callback
/// only hands a copy of the pixels to a single-slot channel, so a slow disk
/// or a slow PNG encode can never stall the CUDA pipeline: the newest frame
/// wins and intermediate ones are dropped.
/// </summary>
internal sealed class PreviewPipeline
{
    private const int KeepPreviews = 8;

    private readonly string _jobId;
    private readonly string _directory;
    private readonly ILogger _logger;
    private readonly Channel<Payload> _channel;
    private readonly Task _writer;
    private int _counter;
    private string? _posterUrl;
    private volatile string? _previewUrl;

    public PreviewPipeline(string jobId, string directory, ILogger logger)
    {
        _jobId = jobId;
        _directory = directory;
        _logger = logger;
        _channel = Channel.CreateBounded<Payload>(new BoundedChannelOptions(1)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.DropOldest
        });
        _writer = Task.Factory.StartNew(WriteLoopAsync, CancellationToken.None,
            TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
            TaskScheduler.Default).Unwrap();
    }

    /// <summary>Latest preview image URL, or null before the first frame lands.</summary>
    public string? PreviewUrl => _previewUrl;

    /// <summary>Poster image URL (first fully decoded frame).</summary>
    public string? PosterUrl => Volatile.Read(ref _posterUrl);

    /// <summary>Called from the native frame thread; must stay allocation-light.</summary>
    public void Push(byte[] pixels, int width, int height, int stride,
                     int frameIndex, int frameCount, int denoiseStep, int denoiseSteps)
    {
        _channel.Writer.TryWrite(new Payload(pixels, width, height, stride,
            frameIndex, frameCount, denoiseStep, denoiseSteps));
    }

    /// <summary>Flushes pending frames and waits for the writer to drain.</summary>
    public async Task CompleteAsync()
    {
        _channel.Writer.TryComplete();
        try
        {
            await _writer.WaitAsync(TimeSpan.FromSeconds(20)).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "preview writer for {Job} did not drain in time", _jobId);
        }
    }

    private async Task WriteLoopAsync()
    {
        ChannelReader<Payload> reader = _channel.Reader;
        while (await reader.WaitToReadAsync().ConfigureAwait(false))
        {
            while (reader.TryRead(out Payload payload))
            {
                try { Write(payload); }
                catch (Exception ex) { _logger.LogWarning(ex, "cannot write preview for {Job}", _jobId); }
            }
        }
    }

    private void Write(Payload payload)
    {
        if (payload.Pixels.Length == 0 || payload.Width <= 0 || payload.Height <= 0) return;

        int index = ++_counter;
        byte[] preview = PngEncoder.Encode(payload.Pixels, payload.Width, payload.Height,
            payload.Stride, 384);
        if (preview.Length == 0) return;
        Save($"preview-{index}.png", preview);
        _previewUrl = $"/files/outputs/{_jobId}/preview-{index}.png";

        // A decoded frame makes a far better poster than a denoise preview.
        if (_posterUrl is null && payload.FrameCount > 0)
        {
            byte[] poster = PngEncoder.Encode(payload.Pixels, payload.Width, payload.Height,
                payload.Stride, 720);
            if (poster.Length > 0)
            {
                Save("poster.png", poster);
                Volatile.Write(ref _posterUrl, $"/files/outputs/{_jobId}/poster.png");
            }
        }

        if (index > KeepPreviews)
        {
            string stale = Path.Combine(_directory, $"preview-{index - KeepPreviews}.png");
            try { File.Delete(stale); } catch { /* best effort */ }
        }
    }

    private void Save(string name, byte[] bytes)
    {
        string path = Path.Combine(_directory, name);
        string temp = path + ".tmp";
        File.WriteAllBytes(temp, bytes);
        File.Move(temp, path, overwrite: true);
    }

    private readonly record struct Payload(
        byte[] Pixels, int Width, int Height, int Stride,
        int FrameIndex, int FrameCount, int DenoiseStep, int DenoiseSteps);
}
