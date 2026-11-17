using System.Runtime.InteropServices;
using System.Text.Json;

namespace H3NetRuntime;

/// <summary>
/// Managed entry point for the Python-free MiniMax-H3 / HyperFlow H3 Flash
/// runtime. One instance owns one resident engine context; calls on the same
/// instance are serialized by the native layer.
/// </summary>
public sealed class H3Runtime : IDisposable
{
    private readonly H3RuntimeHandle _handle;
    private int _disposed;

    private H3Runtime(H3RuntimeHandle handle) => _handle = handle;

    /// <summary>Native runtime version (e.g. "1.0.0").</summary>
    public static string Version => NativeMethods.ReadString(NativeMethods.Version());

    /// <summary>C ABI version; the managed wrapper checks it at load time.</summary>
    public static int AbiVersion => NativeMethods.AbiVersion();

    /// <summary>JSON build/runtime information from the native library.</summary>
    public static string BuildInfoJson =>
        NativeMethods.TakeString(NativeMethods.BuildInfo()) ?? "{}";

    /// <summary>
    /// Loads the model metadata and initialises the CUDA device. Weights stay
    /// unmapped until the first generation.
    /// </summary>
    public static H3Runtime Create(H3RuntimeOptions? options = null)
    {
        if (AbiVersion != 1)
            throw new H3RuntimeException(
                $"native h3rt ABI {AbiVersion} is incompatible with this wrapper (ABI 1)");

        nint errorBuffer = Marshal.AllocHGlobal(1024);
        try
        {
            Marshal.WriteByte(errorBuffer, 0, 0);
            int status = NativeMethods.Create(options?.ToJson() ?? "{}",
                out nint handle, errorBuffer, 1024);
            if (status != 0)
            {
                string message = Marshal.PtrToStringUTF8(errorBuffer) ?? "unknown error";
                throw new H3RuntimeException((H3Status)status, message);
            }
            return new H3Runtime(new H3RuntimeHandle(handle));
        }
        finally
        {
            Marshal.FreeHGlobal(errorBuffer);
        }
    }

    /// <summary>JSON device details (name, VRAM, compute capability, arch check).</summary>
    public string DeviceInfoJson =>
        NativeMethods.TakeString(NativeMethods.DeviceInfo(_handle.DangerousGetHandle())) ?? "{}";

    /// <summary>JSON model pack inventory (bytes, files, tensors per component).</summary>
    public string ModelInfoJson =>
        NativeMethods.TakeString(NativeMethods.ModelInfo(_handle.DangerousGetHandle())) ?? "{}";

    /// <summary>The pack directory the engine actually resolved and loaded.</summary>
    public string ModelDir
    {
        get
        {
            H3PackLocation location = Location;
            return location.ModelDir.Length > 0 ? location.ModelDir : location.PackDir;
        }
    }

    /// <summary>
    /// The directory the engine reads from. When <see cref="H3ModelPaths"/> was
    /// used this is the assembled link pack, not the pack the caller named.
    /// </summary>
    public string PackDir => Location.PackDir;

    /// <summary>True when the components were linked together into a pack.</summary>
    public bool IsComposedPack => Location.Composed;

    private readonly record struct H3PackLocation(string ModelDir, string PackDir, bool Composed);

    private H3PackLocation _location;
    private bool _locationResolved;

    private H3PackLocation Location
    {
        get
        {
            if (_locationResolved) return _location;
            try
            {
                using JsonDocument document = JsonDocument.Parse(ModelInfoJson);
                JsonElement root = document.RootElement;
                string Read(string name) =>
                    root.TryGetProperty(name, out JsonElement value)
                        ? value.GetString() ?? string.Empty
                        : string.Empty;
                bool composed = root.TryGetProperty("composed_pack", out JsonElement flag) &&
                                flag.ValueKind == JsonValueKind.True;
                _location = new H3PackLocation(Read("model_dir"), Read("pack_dir"), composed);
            }
            catch (JsonException)
            {
                _location = new H3PackLocation(string.Empty, string.Empty, false);
            }
            _locationResolved = true;
            return _location;
        }
    }

    /// <summary>JSON resident-cache state.</summary>
    public string CacheInfoJson =>
        NativeMethods.TakeString(NativeMethods.CacheInfo(_handle.DangerousGetHandle())) ?? "{}";

    /// <summary>Releases cached embeddings / prepared weights.</summary>
    public void ClearCache() => Check(NativeMethods.CacheClear(Handle));

    /// <summary>Enables the on-disk prompt embedding cache.</summary>
    public void SetTextCacheDir(string dir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dir);
        Check(NativeMethods.SetTextCacheDir(Handle, dir));
    }

    /// <summary>Cooperatively cancels the running generation.</summary>
    public void Cancel() => Check(NativeMethods.Cancel(Handle));

    /// <summary>Whether cancellation has been requested.</summary>
    public bool IsCancelled => NativeMethods.IsCancelled(Handle) != 0;

    /// <summary>
    /// Warms the denoiser so the first user-visible generation does not pay
    /// the DiT weight read. Layout fields in <paramref name="request"/> must
    /// match the first real request (width/height/frames/steps/pdd8).
    /// </summary>
    public void Preload(H3GenerateRequest? request = null)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        var probe = request ?? new H3GenerateRequest();
        nint errorBuffer = Marshal.AllocHGlobal(1024);
        try
        {
            Marshal.WriteByte(errorBuffer, 0, 0);
            int status = NativeMethods.Preload(Handle, probe.ToJson(), errorBuffer, 1024);
            if (status != 0)
            {
                string message = Marshal.PtrToStringUTF8(errorBuffer) ?? "preload failed";
                throw new H3RuntimeException((H3Status)status, message);
            }
        }
        finally
        {
            Marshal.FreeHGlobal(errorBuffer);
        }
    }

    /// <summary>Runs <see cref="Preload"/> on a dedicated thread.</summary>
    public Task PreloadAsync(H3GenerateRequest? request = null,
                             CancellationToken cancellationToken = default) =>
        Task.Factory.StartNew(() => Preload(request), cancellationToken,
            TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
            TaskScheduler.Default);

    /// <summary>
    /// Generates an MP4 (video + audio) and reports progress / decoded frames.
    /// This call blocks; use <see cref="GenerateAsync"/> from UI code.
    /// </summary>
    /// <param name="request">ComfyUI-workflow-equivalent parameters.</param>
    /// <param name="progress">Optional phase progress sink.</param>
    /// <param name="onFrame">Optional per-frame callback (called on a native thread).</param>
    /// <param name="captureFrames">Copy RGB pixels into <see cref="H3Frame.Pixels"/>.</param>
    /// <param name="cancellationToken">Cancels at the next engine callback.</param>
    public H3GenerateResult Generate(
        H3GenerateRequest request,
        IProgress<H3Progress>? progress = null,
        Action<H3Frame>? onFrame = null,
        bool captureFrames = false,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(request);
        if (string.IsNullOrWhiteSpace(request.Prompt))
            throw new ArgumentException("Prompt is required.", nameof(request));

        NativeMethods.ProgressCallback progressCallback = (phasePtr, completed, total, _) =>
        {
            try
            {
                if (cancellationToken.IsCancellationRequested) return 1;
                if (progress is not null)
                    progress.Report(new H3Progress(NativeMethods.ReadString(phasePtr),
                                                   completed, total));
                return 0;
            }
            catch { return 0; } // never let an exception cross the native boundary
        };

        NativeMethods.FrameCallback frameCallback = (framePtr, _) =>
        {
            try
            {
                if (cancellationToken.IsCancellationRequested) return 1;
                if (onFrame is null) return 0;
                var native = Marshal.PtrToStructure<NativeMethods.NativeFrame>(framePtr);
                byte[] pixels = Array.Empty<byte>();
                if (captureFrames && native.Rgb != nint.Zero &&
                    native.Stride > 0 && native.Height > 0)
                {
                    pixels = new byte[checked(native.Stride * native.Height)];
                    Marshal.Copy(native.Rgb, pixels, 0, pixels.Length);
                }
                onFrame(new H3Frame
                {
                    Width = native.Width,
                    Height = native.Height,
                    Stride = native.Stride,
                    FrameIndex = native.FrameIndex,
                    FrameCount = native.FrameCount,
                    DenoiseStep = native.DenoiseStep,
                    DenoiseSteps = native.DenoiseSteps,
                    Pixels = pixels
                });
                return 0;
            }
            catch { return 0; }
        };

        var callbacks = new NativeMethods.NativeCallbacks
        {
            OnProgress = Marshal.GetFunctionPointerForDelegate(progressCallback),
            OnFrame = Marshal.GetFunctionPointerForDelegate(frameCallback),
            Opaque = nint.Zero
        };
        string requestJson = request.ToJson();

        using CancellationTokenRegistration registration = cancellationToken.Register(
            static state =>
            {
                try { ((H3Runtime)state!).Cancel(); }
                catch (ObjectDisposedException) { /* shutting down */ }
            }, this);
        cancellationToken.ThrowIfCancellationRequested();

        bool addRef = false;
        _handle.DangerousAddRef(ref addRef);
        try
        {
            int status = NativeMethods.Generate(_handle.DangerousGetHandle(),
                requestJson, ref callbacks, out nint resultPtr);
            // Keep the delegate thunks alive until the native call returns.
            GC.KeepAlive(progressCallback);
            GC.KeepAlive(frameCallback);

            if (status != 0)
            {
                string message = NativeMethods.ReadString(
                    NativeMethods.LastError(_handle.DangerousGetHandle()));
                if (string.IsNullOrEmpty(message))
                    message = NativeMethods.ReadString(NativeMethods.StatusText(status));
                if (status == (int)H3Status.Cancelled)
                    throw new OperationCanceledException(message, cancellationToken);
                throw new H3RuntimeException((H3Status)status, message);
            }

            string json = NativeMethods.TakeString(resultPtr)
                ?? throw new H3RuntimeException("the native library returned no result");
            return H3GenerateResult.Parse(json);
        }
        finally
        {
            if (addRef) _handle.DangerousRelease();
        }
    }

    /// <summary>Long-running wrapper around <see cref="Generate"/> on a dedicated thread.</summary>
    public Task<H3GenerateResult> GenerateAsync(
        H3GenerateRequest request,
        IProgress<H3Progress>? progress = null,
        Action<H3Frame>? onFrame = null,
        bool captureFrames = false,
        CancellationToken cancellationToken = default) =>
        Task.Factory.StartNew(
            () => Generate(request, progress, onFrame, captureFrames, cancellationToken),
            cancellationToken,
            TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
            TaskScheduler.Default);

    private nint Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed != 0, this);
            return _handle.DangerousGetHandle();
        }
    }

    private void Check(int status)
    {
        if (status == 0) return;
        string message = NativeMethods.ReadString(NativeMethods.LastError(Handle));
        if (string.IsNullOrEmpty(message))
            message = NativeMethods.ReadString(NativeMethods.StatusText(status));
        throw new H3RuntimeException((H3Status)status, message);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}
