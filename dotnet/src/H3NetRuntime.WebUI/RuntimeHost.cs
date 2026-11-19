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
    private ModelDiscovery? _discovery;
    private string _modelDir = string.Empty;
    private string? _lastError;
    private bool _disposed;

    public RuntimeHost(WebUiOptions options, ILogger<RuntimeHost> logger)
    {
        _options = options;
        _logger = logger;
        _modelDir = options.ModelDir ?? string.Empty;
    }

    /// <summary>
    /// The weight layout as it was found on disk. Cheap (file probes only), so
    /// the config endpoint can report what is missing before the engine is
    /// loaded.
    /// </summary>
    public ModelDiscovery Discovery
    {
        get { lock (_sync) return _discovery ??= ModelLocator.Resolve(_options); }
    }

    /// <summary>Ref2VA weights exist, so reference-conditioned video can run.</summary>
    public bool Ref2VaAvailable => Discovery.Ref2VaPresent;

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

            ModelDiscovery discovery = _discovery ??= ModelLocator.Resolve(_options);
            if (!discovery.IsUsable)
            {
                _lastError = discovery.Error;
                throw new H3RuntimeException(H3Status.Model, discovery.Error ?? "model weights not found");
            }

            try
            {
                _runtime = H3Runtime.Create(new H3RuntimeOptions
                {
                    // The pack that was found is passed as an absolute path, so
                    // the engine never falls back to its own guess.
                    ModelDir = discovery.PackDir,
                    Models = discovery.Models.IsEmpty ? null : discovery.Models,
                    PackCacheDir = _options.PackCacheDir,
                    // A missing explicit config file is a hard error in the
                    // engine, so only forward it when it exists.
                    ConfigPath = UsableConfigPath(),
                    KeepDit = _options.KeepDit,
                    CacheEnabled = _options.KeepResidentVae,
                    TextCacheDir = Environment.GetEnvironmentVariable("H3_TEXT_CACHE_DIR")
                });
                _modelDir = _runtime.ModelDir;
                _lastError = null;
                foreach (string line in discovery.Describe()) _logger.LogInformation("{Line}", line);
                _logger.LogInformation(
                    "model ready: engine reads {PackDir} (composed={Composed}), model_dir={ModelDir}",
                    _runtime.PackDir, _runtime.IsComposedPack, _modelDir);
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

            if (!string.IsNullOrWhiteSpace(modelDir))
                _options.ModelDir = Path.GetFullPath(modelDir);
            if (!string.IsNullOrWhiteSpace(configPath))
                _options.ConfigPath = Path.GetFullPath(configPath);
            _discovery = ModelLocator.Resolve(_options);

            if (!_discovery.IsUsable)
            {
                _lastError = _discovery.Error;
                _logger.LogError("cannot switch model pack: {Error}", _lastError);
                throw new H3RuntimeException(H3Status.Model, _lastError ?? "model weights not found");
            }

            try
            {
                _runtime = H3Runtime.Create(new H3RuntimeOptions
                {
                    ModelDir = _discovery.PackDir,
                    Models = _discovery.Models.IsEmpty ? null : _discovery.Models,
                    PackCacheDir = _options.PackCacheDir,
                    ConfigPath = UsableConfigPath(),
                    KeepDit = _options.KeepDit,
                    CacheEnabled = _options.KeepResidentVae,
                    TextCacheDir = Environment.GetEnvironmentVariable("H3_TEXT_CACHE_DIR")
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

    /// <summary>Re-runs the file probes (after the user drops weights in place).</summary>
    public void Refresh()
    {
        lock (_sync)
        {
            _discovery = ModelLocator.Resolve(_options);
            _lastError = null;
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
            string current = Discovery.PackDir ?? ModelDir;
            root = string.IsNullOrWhiteSpace(current)
                ? Environment.GetEnvironmentVariable("H3_MODEL_ROOT")
                : Path.GetDirectoryName(current.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        }
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return packs;

        void Add(string path)
        {
            if (!ModelLocator.IsPack(path)) return;
            packs.Add(new ModelPackInfo
            {
                Name = new DirectoryInfo(path).Name,
                Path = path.Replace('\\', '/'),
                HasRef2Va = HasRef2Va(path),
                Active = string.Equals(Path.GetFullPath(path), Path.GetFullPath(Discovery.PackDir ?? ModelDir),
                                       StringComparison.OrdinalIgnoreCase)
            });
        }

        Add(root);
        foreach (string child in Directory.EnumerateDirectories(root))
            Add(child);
        return packs.OrderBy(p => p.Name, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static bool HasRef2Va(string packDir) =>
        Directory.Exists(Path.Combine(packDir, "Ref2VA", "transformer")) &&
        Directory.EnumerateFiles(Path.Combine(packDir, "Ref2VA", "transformer"), "*.safetensors").Any();

    private string? UsableConfigPath() =>
        !string.IsNullOrWhiteSpace(_options.ConfigPath) && File.Exists(_options.ConfigPath)
            ? _options.ConfigPath
            : null;

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
