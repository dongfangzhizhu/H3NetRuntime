using H3NetRuntime;

namespace H3NetRuntime.WebUI;

/// <summary>
/// Owns the single resident H3Runtime. Creation is lazy so the UI can start
/// (and explain what is missing) even when the model pack is not installed.
/// </summary>
public sealed class RuntimeHost : IDisposable
{
    private readonly object _sync = new();
    private readonly WebUiOptions _options;
    private readonly ILogger<RuntimeHost> _logger;
    private H3Runtime? _runtime;
    private string _modelDir = string.Empty;
    private string? _lastError;
    private bool _disposed;

    public RuntimeHost(WebUiOptions options, ILogger<RuntimeHost> logger)
    {
        _options = options;
        _logger = logger;
        _modelDir = options.ModelDir ?? string.Empty;
    }

    public string ModelDir
    {
        get { lock (_sync) return _modelDir; }
    }

    public string? LastError
    {
        get { lock (_sync) return _lastError; }
    }

    public bool IsLoaded
    {
        get { lock (_sync) return _runtime is not null; }
    }

    /// <summary>Creates the runtime if needed and returns it; throws on failure.</summary>
    public H3Runtime Ensure()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_runtime is not null) return _runtime;
            try
            {
                _runtime = H3Runtime.Create(new H3RuntimeOptions
                {
                    ModelDir = string.IsNullOrWhiteSpace(_modelDir) ? null : _modelDir,
                    ConfigPath = _options.ConfigPath,
                    KeepDit = true,
                    CacheEnabled = true,
                    TextCacheDir = Environment.GetEnvironmentVariable("H3_TEXT_CACHE_DIR")
                });
                _modelDir = _runtime.ModelDir;
                _lastError = null;
                _logger.LogInformation("model pack loaded: {ModelDir}", _modelDir);
                return _runtime;
            }
            catch (Exception ex)
            {
                _lastError = ex.Message;
                _logger.LogError(ex, "cannot create h3rt runtime");
                throw;
            }
        }
    }

    public bool TryEnsure(out H3Runtime? runtime, out string? error)
    {
        try
        {
            runtime = Ensure();
            error = null;
            return true;
        }
        catch (Exception ex)
        {
            runtime = null;
            error = ex.Message;
            return false;
        }
    }

    /// <summary>Switches to another model pack. Caller must ensure no job runs.</summary>
    public void Recreate(string? modelDir, string? configPath)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            H3Runtime? previous = _runtime;
            _runtime = null;
            previous?.Dispose();

            string? resolved = string.IsNullOrWhiteSpace(modelDir) ? _modelDir : Path.GetFullPath(modelDir);
            try
            {
                _runtime = H3Runtime.Create(new H3RuntimeOptions
                {
                    ModelDir = resolved,
                    ConfigPath = string.IsNullOrWhiteSpace(configPath) ? _options.ConfigPath : Path.GetFullPath(configPath),
                    KeepDit = true,
                    CacheEnabled = true
                });
                _modelDir = _runtime.ModelDir;
                _lastError = null;
                _logger.LogInformation("model pack switched to {ModelDir}", _modelDir);
            }
            catch (Exception ex)
            {
                _lastError = ex.Message;
                _logger.LogError(ex, "cannot switch model pack");
                throw;
            }
        }
    }

    public string BuildInfoJson => H3Runtime.BuildInfoJson;

    public string DeviceInfoJson
    {
        get
        {
            try { return Ensure().DeviceInfoJson; }
            catch { return "{}"; }
        }
    }

    public string ModelInfoJson
    {
        get
        {
            try { return Ensure().ModelInfoJson; }
            catch { return "{}"; }
        }
    }

    public string CacheInfoJson
    {
        get
        {
            try { return Ensure().CacheInfoJson; }
            catch { return "{}"; }
        }
    }

    public List<ModelPackInfo> ScanPacks()
    {
        var packs = new List<ModelPackInfo>();
        string? root = _options.ModelRoot;
        if (string.IsNullOrWhiteSpace(root))
        {
            string current = ModelDir;
            root = string.IsNullOrWhiteSpace(current)
                ? Environment.GetEnvironmentVariable("H3_MODEL_ROOT")
                : Path.GetDirectoryName(current.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        }
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return packs;

        void Add(string path)
        {
            string marker = Path.Combine(path, "FL2VA", "transformer", "config.json");
            if (!File.Exists(marker)) return;
            packs.Add(new ModelPackInfo
            {
                Name = new DirectoryInfo(path).Name,
                Path = path.Replace('\\', '/'),
                HasRef2Va = File.Exists(Path.Combine(path, "Ref2VA", "transformer", "model.safetensors.index.json")),
                Active = string.Equals(Path.GetFullPath(path), Path.GetFullPath(ModelDir),
                                       StringComparison.OrdinalIgnoreCase)
            });
        }

        Add(root);
        foreach (string child in Directory.EnumerateDirectories(root))
            Add(child);
        return packs.OrderBy(p => p.Name, StringComparer.OrdinalIgnoreCase).ToList();
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            _runtime?.Dispose();
            _runtime = null;
        }
    }
}
