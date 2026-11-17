using System.Text.Json;
using H3NetRuntime;

namespace H3NetRuntime.WebUI;

/// <summary>One weight component as the UI reports it.</summary>
public sealed record ModelComponent(
    string Key, string Label, string Slot, string? Path, bool Required, bool Present);

/// <summary>
/// The result of looking for model weights: the pack root (if one exists), the
/// per-component overrides the user configured, and enough diagnostics for the
/// UI to explain what is missing instead of failing with "model error".
/// </summary>
public sealed class ModelDiscovery
{
    public string? PackDir { get; init; }
    public H3ModelPaths Models { get; init; } = new();
    public bool Ref2VaPresent { get; init; }
    /// <summary>
    /// True when the weights were found in a ComfyUI <c>models/</c> directory
    /// (diffusion_models/, text_encoders/, vae/) rather than an assembled
    /// FL2VA/ pack. The five released files live there, so this is the normal
    /// case for a fresh ComfyUI_MiniMax-H3-V8 install.
    /// </summary>
    public bool ComfyLayout { get; init; }
    public IReadOnlyList<ModelComponent> Components { get; init; } = Array.Empty<ModelComponent>();
    public IReadOnlyList<string> Searched { get; init; } = Array.Empty<string>();
    public string? Error { get; init; }
    public bool SourceFromConfigFile { get; init; }

    /// <summary>Every required component resolves to something on disk.</summary>
    public bool IsUsable => Components.Where(c => c.Required).All(c => c.Present);

    public IEnumerable<ModelComponent> Missing => Components.Where(c => c.Required && !c.Present);

    /// <summary>
    /// The resolved paths, one line per component, for the console. "Which file
    /// did it actually load" is the first question behind every bad output, and
    /// the answer is only knowable here.
    /// </summary>
    public IReadOnlyList<string> Describe()
    {
        var lines = new List<string>
        {
            $"模型来源  : {PackDir ?? "(未找到)"}" +
            (ComfyLayout ? "  (ComfyUI models/ 目录)" : string.Empty) +
            (SourceFromConfigFile ? "  (来自 h3.config.json)" : string.Empty)
        };
        foreach (ModelComponent component in Components)
        {
            string state = component.Present ? "OK  " : component.Required ? "缺失" : "可选";
            lines.Add($"  [{state}] {component.Label,-10} {component.Key,-22} {component.Path ?? "-"}");
        }
        if (Error is { Length: > 0 }) lines.Add($"  错误: {Error}");
        return lines;
    }
}

/// <summary>
/// Finds the MiniMax-H3 pack without being told, so a fresh install works by
/// dropping the weights in one of the usual places. The engine's own fallback
/// (<c>~/h3-models/pdd8</c>) is kept as the last candidate, but the pack that
/// is actually found is passed down as an absolute path — the engine never has
/// to guess.
/// </summary>
public static class ModelLocator
{
    private static readonly (string Key, string Label, string Slot, bool Required)[] Slots =
    {
        ("text_encoder", "文本编码器", "FL2VA/text_encoder", true),
        ("transformer", "FL2VA DiT", "FL2VA/transformer", true),
        ("video_vae", "视频 VAE", "FL2VA/video_vae/source", true),
        ("audio_vae", "音频 VAE", "FL2VA/audio_vae", true),
        ("tokenizer", "分词器", "FL2VA/tokenizer", true),
        ("pdd_lora", "PDD 8 步 LoRA", "FL2VA/pdd_lora.safetensors", false),
        ("ref2va_transformer", "Ref2VA DiT", "Ref2VA/transformer", false),
        ("ref2va_text_encoder", "Ref2VA 文本编码器", "Ref2VA/text_encoder", false),
        ("ref2va_video_vae", "Ref2VA 视频 VAE", "Ref2VA/video_vae/source", false),
        ("ref2va_audio_vae", "Ref2VA 音频 VAE", "Ref2VA/audio_vae", false),
        ("ref2va_tokenizer", "Ref2VA 分词器", "Ref2VA/tokenizer", false)
    };

    public static ModelDiscovery Resolve(WebUiOptions options)
    {
        var searched = new List<string>();
        bool fromConfig = false;
        string? pack = null;
        string? comfy = null;

        foreach ((string candidate, bool isConfig) in Candidates(options))
        {
            if (string.IsNullOrWhiteSpace(candidate)) continue;
            string full = SafeFullPath(candidate);
            searched.Add(full);
            if (IsPack(full))
            {
                pack = full;
                fromConfig = isConfig;
                break;
            }
            /* A ComfyUI models/ directory is a source of the five released
             * files, not an assembled pack: keep it as a fallback so a real
             * pack still wins when both exist. */
            if (comfy is null && IsComfyLayout(full))
            {
                comfy = full;
                fromConfig = isConfig;
            }
        }

        bool comfyLayout = pack is null && comfy is not null;
        H3ModelPaths models = comfyLayout
            ? DeriveComfy(comfy!, options.Models)
            : options.Models;

        var components = new List<ModelComponent>(Slots.Length);
        foreach ((string key, string label, string slot, bool required) in Slots)
        {
            string? path = Override(models, key);
            if (path is null && pack is not null)
                path = Path.Combine(pack, slot.Replace('/', Path.DirectorySeparatorChar));
            bool present = path is not null && Exists(path);
            components.Add(new ModelComponent(key, label, slot, path, required, present));
        }

        bool ref2va = components.First(c => c.Key == "ref2va_transformer").Present;
        var discovery = new ModelDiscovery
        {
            PackDir = pack ?? comfy,
            Models = models,
            Ref2VaPresent = ref2va,
            ComfyLayout = comfyLayout,
            Components = components,
            Searched = searched,
            SourceFromConfigFile = fromConfig
        };

        if (!discovery.IsUsable)
        {
            string missing = string.Join("、", discovery.Missing.Select(c => c.Label));
            string hint = pack is null && comfy is null
                ? "没有找到模型包（需要包含 FL2VA/transformer/config.json 的目录，或 ComfyUI 的 models/ 目录）。"
                : $"{discovery.PackDir} 缺少：{missing}。";
            return new ModelDiscovery
            {
                PackDir = discovery.PackDir,
                Models = models,
                Ref2VaPresent = ref2va,
                ComfyLayout = comfyLayout,
                Components = components,
                Searched = searched,
                SourceFromConfigFile = fromConfig,
                Error = hint + " 请按“模型下载指引”准备权重，或在 appsettings.json 里设置 H3:ModelDir / H3:Models。"
            };
        }
        return discovery;
    }

    /// <summary>
    /// A ComfyUI <c>models/</c> directory: that is where the five released files
    /// of ComfyUI_MiniMax-H3-V8 live.
    /// </summary>
    public static bool IsComfyLayout(string directory) =>
        Exists(Path.Combine(directory, "diffusion_models")) &&
        Exists(Path.Combine(directory, "text_encoders")) &&
        Exists(Path.Combine(directory, "vae"));

    /// <summary>
    /// Picks the five components out of a ComfyUI models/ tree. Explicit
    /// <c>H3:Models</c> entries always win; the tokenizer is not part of the
    /// ComfyUI download, so it is searched in the places h3rt itself looks.
    /// </summary>
    private static H3ModelPaths DeriveComfy(string directory, H3ModelPaths overrides)
    {
        string diffusion = Path.Combine(directory, "diffusion_models");
        string encoders = Path.Combine(directory, "text_encoders");
        string vae = Path.Combine(directory, "vae");

        return new H3ModelPaths
        {
            Transformer = Keep(overrides.Transformer) ?? Pick(diffusion, ".safetensors", "fl2va"),
            Ref2vaTransformer = Keep(overrides.Ref2vaTransformer) ?? Pick(diffusion, ".safetensors", "ref2va"),
            TextEncoder = Keep(overrides.TextEncoder) ?? Pick(encoders, ".gguf", "qwen") ?? Pick(encoders, ".gguf"),
            VideoVae = Keep(overrides.VideoVae) ?? Pick(vae, ".safetensors", "sol_vae") ?? Pick(vae, ".safetensors", "video_vae"),
            AudioVae = Keep(overrides.AudioVae) ?? Pick(vae, ".safetensors", "audio_vae"),
            Tokenizer = Keep(overrides.Tokenizer) ?? FindTokenizer(directory),
            PddLora = Keep(overrides.PddLora),
            Ref2vaTextEncoder = Keep(overrides.Ref2vaTextEncoder),
            Ref2vaVideoVae = Keep(overrides.Ref2vaVideoVae),
            Ref2vaAudioVae = Keep(overrides.Ref2vaAudioVae),
            Ref2vaTokenizer = Keep(overrides.Ref2vaTokenizer)
        };
    }

    private static string? Keep(string? path) =>
        string.IsNullOrWhiteSpace(path) ? null : path;

    /// <summary>First file in <paramref name="directory"/> matching every needle and the extension.</summary>
    private static string? Pick(string directory, string extension, params string[] needles)
    {
        if (!Directory.Exists(directory)) return null;
        foreach (string file in Directory.EnumerateFiles(directory).OrderBy(f => f, StringComparer.Ordinal))
        {
            string name = Path.GetFileName(file);
            if (!name.EndsWith(extension, StringComparison.OrdinalIgnoreCase)) continue;
            if (needles.All(n => name.Contains(n, StringComparison.OrdinalIgnoreCase)))
                return file;
        }
        return null;
    }

    /// <summary>
    /// tokenizer.json lives outside the ComfyUI model download (CLIPLoaderGGUF
    /// fetches it from HuggingFace). Look where h3rt looks, so the UI reports
    /// the same file the engine will actually use.
    /// </summary>
    private static string? FindTokenizer(string comfyModels)
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string? env = Environment.GetEnvironmentVariable("H3RT_TOKENIZER_DIR");
        var roots = new List<string>
        {
            Path.Combine(comfyModels, "tokenizer"),
            Path.Combine(comfyModels, "text_encoders", "tokenizer"),
            Path.Combine(comfyModels, "..", "tokenizer"),
            Path.Combine(AppContext.BaseDirectory, "tokenizer")
        };
        if (!string.IsNullOrWhiteSpace(env)) roots.Insert(0, env);
        /* In a source tree the native runtime (and the tokenizer shipped next
         * to it) lives in dist/<rid>/, which is several levels up. */
        string rid = OperatingSystem.IsWindows() ? "win-x64" : "linux-x64";
        DirectoryInfo? parent = new(AppContext.BaseDirectory);
        for (int depth = 0; parent is not null && depth < 8; depth++, parent = parent.Parent)
            roots.Add(Path.Combine(parent.FullName, "dist", rid, "tokenizer"));
        if (!string.IsNullOrEmpty(home))
        {
            roots.Add(Path.Combine(home, "h3-models", "pdd8", "FL2VA", "tokenizer"));
            roots.Add(Path.Combine(home, ".cache", "huggingface", "hub"));
        }
        foreach (string root in roots)
        {
            if (!Directory.Exists(root)) continue;
            string candidate = Path.Combine(root, "tokenizer.json");
            if (File.Exists(candidate)) return SafeFullPath(root);
        }
        return null;
    }

    private static IEnumerable<(string Path, bool FromConfig)> Candidates(WebUiOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.ModelDir))
            yield return (options.ModelDir!, false);

        string? env = Environment.GetEnvironmentVariable("H3_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(env)) yield return (env, false);

        string? configured = ConfigFilePackDir(options.ConfigPath);
        if (configured is not null) yield return (configured, true);

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrEmpty(home)) yield return (Path.Combine(home, "h3-models", "pdd8"), false);

        string baseDir = AppContext.BaseDirectory;
        yield return (Path.Combine(baseDir, "models", "pdd8"), false);
        yield return (Path.Combine(baseDir, "models"), false);
        yield return (Path.GetFullPath(Path.Combine(baseDir, "..", "h3-models", "pdd8")), false);
        yield return (Path.Combine(Directory.GetCurrentDirectory(), "models", "pdd8"), false);
        yield return (Path.Combine(Directory.GetCurrentDirectory(), "h3-models", "pdd8"), false);
        /* ComfyUI_MiniMax-H3-V8 drops the five released files in a plain
         * models/ tree; look next to the app and one level up. */
        yield return (Path.Combine(Directory.GetCurrentDirectory(), "ComfyUI", "models"), false);
        yield return (Path.Combine(Directory.GetCurrentDirectory(), "ComfyUI_MiniMax-H3-V8", "ComfyUI", "models"), false);
        yield return (Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "ComfyUI_MiniMax-H3-V8", "ComfyUI", "models")), false);
        // Last resort: let the native layer apply its own default.
        yield return (Path.Combine(home, "h3-models", "pdd8"), false);
    }

    /// <summary>Reads h3_dir out of an h3.config.json (platform section first).</summary>
    private static string? ConfigFilePackDir(string? configPath)
    {
        if (string.IsNullOrWhiteSpace(configPath) || !File.Exists(configPath)) return null;
        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(configPath));
            JsonElement root = document.RootElement;
            string section = OperatingSystem.IsWindows() ? "windows" : "linux";
            foreach (string name in new[] { section, "" })
            {
                JsonElement scope = name.Length == 0
                    ? root
                    : root.TryGetProperty(name, out JsonElement nested) ? nested : default;
                if (scope.ValueKind != JsonValueKind.Object) continue;
                if (scope.TryGetProperty("h3_dir", out JsonElement value) &&
                    value.GetString() is { Length: > 0 } text)
                    return text;
            }
        }
        catch (JsonException) { }
        catch (IOException) { }
        return null;
    }

    private static string? Override(H3ModelPaths models, string key) => key switch
    {
        "text_encoder" => models.TextEncoder,
        "transformer" => models.Transformer,
        "video_vae" => models.VideoVae,
        "audio_vae" => models.AudioVae,
        "tokenizer" => models.Tokenizer,
        "pdd_lora" => models.PddLora,
        "ref2va_transformer" => models.Ref2vaTransformer,
        "ref2va_text_encoder" => models.Ref2vaTextEncoder,
        "ref2va_video_vae" => models.Ref2vaVideoVae,
        "ref2va_audio_vae" => models.Ref2vaAudioVae,
        "ref2va_tokenizer" => models.Ref2vaTokenizer,
        _ => null
    };

    public static bool IsPack(string directory) =>
        Exists(Path.Combine(directory, "FL2VA", "transformer", "config.json"));

    private static bool Exists(string path) =>
        File.Exists(path) || Directory.Exists(path);

    private static string SafeFullPath(string path)
    {
        try { return Path.GetFullPath(path); }
        catch (Exception) { return path; }
    }
}
