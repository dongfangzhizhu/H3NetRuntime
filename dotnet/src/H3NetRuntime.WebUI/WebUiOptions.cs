using H3NetRuntime;

namespace H3NetRuntime.WebUI;

public sealed class WebUiOptions
{
    public string Edition { get; set; } = "general";
    public string OutputDir { get; set; } = "work/outputs";
    public string UploadDir { get; set; } = "work/uploads";
    public string HistoryFile { get; set; } = "work/history.json";
    public string? ModelDir { get; set; }
    public string? ConfigPath { get; set; }
    public string? ModelRoot { get; set; }

    /// <summary>
    /// Per-component weight paths. Anything left empty falls back to the same
    /// slot under the discovered pack, so the usual install only needs
    /// <see cref="ModelDir"/> (or nothing at all, thanks to auto-discovery).
    /// </summary>
    public H3ModelPaths Models { get; set; } = new();

    /// <summary>Where the composed link packs live; empty means the temp dir.</summary>
    public string? PackCacheDir { get; set; }

    public bool PreloadOnStart { get; set; }

    /// <summary>
    /// Keep the ~9.5 GiB resident video VAE decoder mapped between jobs. Off
    /// by default: on a 24 GiB card, combining this with <see cref="KeepDit"/>
    /// fills VRAM and the next job can degrade to a multi-hour-per-step path.
    /// Legacy name <c>CacheEnabled</c> is honoured as an alias.
    /// </summary>
    public bool KeepResidentVae { get; set; }

    /// <summary>
    /// Legacy alias for <see cref="KeepResidentVae"/> from older config files.
    /// When set it is OR-ed in during <see cref="Normalize"/>.
    /// </summary>
    public bool? CacheEnabled { get; set; }

    /// <summary>
    /// Keep the DiT mapped after each generation (faster repeat calls). Safe to
    /// leave on when <see cref="KeepResidentVae"/> is off — the DiT shares pages
    /// with the one-shot VAE through the stream allocator.
    /// </summary>
    public bool KeepDit { get; set; } = true;

    public int MaxHistory { get; set; } = 60;
    public int Port { get; set; } = 7867;

    /// <summary>Bind address; loopback by default, use 0.0.0.0 to expose on LAN.</summary>
    public string Host { get; set; } = "127.0.0.1";

    public bool Advanced => string.Equals(Edition, "advanced", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Makes every path absolute and creates the working directories. Working
    /// directories resolve against <paramref name="baseDirectory"/> (the launch
    /// directory, so outputs land where the user started the app); paths that
    /// come from appsettings — the model pack, the config table, the component
    /// overrides — resolve against <paramref name="assetsDirectory"/> (the
    /// directory the appsettings file lives in), which is the same no matter
    /// where the process was started from.
    /// </summary>
    public void Normalize(string baseDirectory, string? assetsDirectory = null)
    {
        string assets = assetsDirectory ?? baseDirectory;
        Edition = Advanced ? "advanced" : "general";
        if (string.IsNullOrWhiteSpace(Host)) Host = "127.0.0.1";
        // Backwards compatibility: older configs called this switch CacheEnabled.
        if (CacheEnabled == true) KeepResidentVae = true;
        OutputDir = Resolve(baseDirectory, OutputDir);
        UploadDir = Resolve(baseDirectory, UploadDir);
        HistoryFile = Resolve(baseDirectory, HistoryFile);
        if (!string.IsNullOrWhiteSpace(ModelDir)) ModelDir = Resolve(assets, ModelDir);
        if (!string.IsNullOrWhiteSpace(ConfigPath)) ConfigPath = Resolve(assets, ConfigPath);
        if (!string.IsNullOrWhiteSpace(ModelRoot)) ModelRoot = Resolve(assets, ModelRoot);
        if (!string.IsNullOrWhiteSpace(PackCacheDir)) PackCacheDir = Resolve(baseDirectory, PackCacheDir);

        Models.TextEncoder = ResolveOptional(assets, Models.TextEncoder);
        Models.Transformer = ResolveOptional(assets, Models.Transformer);
        Models.VideoVae = ResolveOptional(assets, Models.VideoVae);
        Models.AudioVae = ResolveOptional(assets, Models.AudioVae);
        Models.Tokenizer = ResolveOptional(assets, Models.Tokenizer);
        Models.PddLora = ResolveOptional(assets, Models.PddLora);
        Models.Ref2vaTransformer = ResolveOptional(assets, Models.Ref2vaTransformer);
        Models.Ref2vaTextEncoder = ResolveOptional(assets, Models.Ref2vaTextEncoder);
        Models.Ref2vaVideoVae = ResolveOptional(assets, Models.Ref2vaVideoVae);
        Models.Ref2vaAudioVae = ResolveOptional(assets, Models.Ref2vaAudioVae);
        Models.Ref2vaTokenizer = ResolveOptional(assets, Models.Ref2vaTokenizer);

        Directory.CreateDirectory(OutputDir);
        Directory.CreateDirectory(UploadDir);
        Directory.CreateDirectory(Path.GetDirectoryName(HistoryFile) ?? OutputDir);
    }

    private static string Resolve(string baseDirectory, string path) =>
        Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(baseDirectory, path));

    private static string? ResolveOptional(string baseDirectory, string? path) =>
        string.IsNullOrWhiteSpace(path) ? null : Resolve(baseDirectory, path);
}
