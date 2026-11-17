using System.Text.Json;
using System.Text.Json.Serialization;

namespace H3NetRuntime;

/// <summary>Options for <see cref="H3Runtime.Create"/>.</summary>
public sealed class H3RuntimeOptions
{
    /// <summary>MiniMax-H3 pack containing FL2VA/ (and optionally Ref2VA/).</summary>
    public string? ModelDir { get; set; }

    /// <summary>
    /// Per-component weight paths. Use this when the checkpoints live in
    /// different directories (a DiT from one release, a text encoder from
    /// another); anything left unset falls back to <see cref="ModelDir"/>.
    /// </summary>
    public H3ModelPaths? Models { get; set; }

    /// <summary>
    /// Directory for the link packs built from <see cref="Models"/>. Defaults
    /// to <c>%TEMP%/h3rt-packs</c> (Linux: <c>$TMPDIR/h3rt-packs</c>). Point it
    /// at the same volume as the weights so component links stay free.
    /// </summary>
    public string? PackCacheDir { get; set; }

    /// <summary>Optional h3.config.json with platform-specific pack paths.</summary>
    public string? ConfigPath { get; set; }

    /// <summary>On-disk prompt embedding cache directory.</summary>
    public string? TextCacheDir { get; set; }

    /// <summary>Reuse resident weights / text embeddings between calls.</summary>
    public bool CacheEnabled { get; set; } = true;

    /// <summary>Keep the DiT mapped after each generation (faster repeat calls).</summary>
    public bool KeepDit { get; set; } = true;

    /// <summary>Extra options forwarded verbatim to the native layer.</summary>
    public Dictionary<string, object?> Extra { get; } = new();

    internal string ToJson()
    {
        var values = new Dictionary<string, object?>
        {
            ["model_dir"] = ModelDir,
            ["models"] = Models?.ToDictionary(),
            ["pack_cache_dir"] = PackCacheDir,
            ["config"] = ConfigPath,
            ["text_cache_dir"] = TextCacheDir,
            ["cache_enabled"] = CacheEnabled,
            ["keep_dit"] = KeepDit
        };
        foreach (var pair in Extra) values[pair.Key] = pair.Value;
        return JsonSerializer.Serialize(values, JsonOptions);
    }

    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
}

/// <summary>
/// One path per weight component. Each value is a directory that already holds
/// the files the engine expects for that component; <see cref="PddLora"/> is a
/// single <c>.safetensors</c> file.
/// </summary>
/// <remarks>
/// The engine reads a single pack root, so the runtime assembles the standard
/// layout as links (NTFS junctions on Windows, symlinks on Linux) in a
/// content-addressed directory and loads that. Nothing is copied unless a file
/// cannot be linked at all.
/// </remarks>
public sealed class H3ModelPaths
{
    /// <summary>Qwen3-VL text encoder (<c>FL2VA/text_encoder</c>).</summary>
    public string? TextEncoder { get; set; }

    /// <summary>FL2VA INT8 DiT directory: config.json + model.safetensors (<c>FL2VA/transformer</c>).</summary>
    public string? Transformer { get; set; }

    /// <summary>
    /// Video VAE weights (<c>FL2VA/video_vae/source</c>). Pointing at the
    /// parent <c>video_vae</c> directory that contains <c>source/</c> works too.
    /// </summary>
    public string? VideoVae { get; set; }

    /// <summary>Audio VAE weights (<c>FL2VA/audio_vae</c>).</summary>
    public string? AudioVae { get; set; }

    /// <summary>Tokenizer files: tokenizer.json, tokenizer_config.json, vocab.json (<c>FL2VA/tokenizer</c>).</summary>
    public string? Tokenizer { get; set; }

    /// <summary>PDD 8-step adaptor (<c>FL2VA/pdd_lora.safetensors</c>).</summary>
    public string? PddLora { get; set; }

    /// <summary>Ref2VA DiT directory, needed for reference-conditioned video.</summary>
    public string? Ref2vaTransformer { get; set; }

    public string? Ref2vaTextEncoder { get; set; }
    public string? Ref2vaVideoVae { get; set; }
    public string? Ref2vaAudioVae { get; set; }
    public string? Ref2vaTokenizer { get; set; }

    /// <summary>True when at least one component path is set.</summary>
    [JsonIgnore]
    public bool IsEmpty =>
        TextEncoder is null && Transformer is null && VideoVae is null &&
        AudioVae is null && Tokenizer is null && PddLora is null &&
        Ref2vaTransformer is null && Ref2vaTextEncoder is null &&
        Ref2vaVideoVae is null && Ref2vaAudioVae is null &&
        Ref2vaTokenizer is null;

    internal Dictionary<string, object?>? ToDictionary()
    {
        var values = new Dictionary<string, object?>
        {
            ["text_encoder"] = TextEncoder,
            ["transformer"] = Transformer,
            ["video_vae"] = VideoVae,
            ["audio_vae"] = AudioVae,
            ["tokenizer"] = Tokenizer,
            ["pdd_lora"] = PddLora,
            ["ref2va_transformer"] = Ref2vaTransformer,
            ["ref2va_text_encoder"] = Ref2vaTextEncoder,
            ["ref2va_video_vae"] = Ref2vaVideoVae,
            ["ref2va_audio_vae"] = Ref2vaAudioVae,
            ["ref2va_tokenizer"] = Ref2vaTokenizer
        };
        return IsEmpty ? null : values;
    }
}

/// <summary>One ordered reference (Ref2VA / multi-modal conditioning).</summary>
public sealed class H3Reference
{
    /// <summary>"image", "video", "audio" or "video_audio".</summary>
    public string Kind { get; set; } = "image";
    public string Path { get; set; } = string.Empty;
    public string? AudioPath { get; set; }
    public bool IncludeEmbeddedAudio { get; set; }

    internal Dictionary<string, object?> ToDictionary() => new()
    {
        ["kind"] = Kind,
        ["path"] = Path,
        ["audio_path"] = AudioPath,
        ["include_embedded_audio"] = IncludeEmbeddedAudio
    };
}

/// <summary>
/// A ComfyUI-workflow-equivalent generation request. Unknown fields are
/// ignored by the native layer, so new engine knobs do not break the ABI.
/// </summary>
public sealed class H3GenerateRequest
{
    /// <summary>Positive prompt (required).</summary>
    public string Prompt { get; set; } = string.Empty;

    /// <summary>Only the MiniMax-H3 / HyperFlow pipeline is served.</summary>
    public string Model { get; set; } = "h3";

    public int? Width { get; set; }
    public int? Height { get; set; }
    public int? Frames { get; set; }
    public int? Steps { get; set; }
    public ulong? Seed { get; set; }

    /// <summary>MP4 output path. A temp file is used when omitted.</summary>
    public string? OutputPath { get; set; }
    public string? FirstFramePath { get; set; }
    public string? LastFramePath { get; set; }

    /// <summary>8-step distilled PDD release (on by default).</summary>
    public bool? Pdd8 { get; set; }
    public int? DitLayers { get; set; }
    public int? DenoiseReuse { get; set; }
    public int? CoreReuse { get; set; }
    public int? RenderWidth { get; set; }
    public int? RenderHeight { get; set; }
    public string? ReferenceImageSize { get; set; }

    public bool? TokenReduction { get; set; }
    public bool? SolAttn { get; set; }
    public bool? Fbc { get; set; }
    public bool? PreviewDenoise { get; set; }
    public bool? SsdStreaming { get; set; }
    public bool? UseInt8RowFc2 { get; set; }
    public bool? UseReferenceRope { get; set; }
    public bool? KeepDit { get; set; }

    public string? ReferenceImage { get; set; }
    public string? ReferenceVideo { get; set; }
    public string? ReferenceAudio { get; set; }
    public string? ReferenceVideoAudio { get; set; }
    public IList<string>? ReferenceImages { get; set; }
    public IList<string>? ReferenceVideos { get; set; }
    public IList<string>? ReferenceAudios { get; set; }
    public IList<H3Reference>? References { get; set; }

    /// <summary>
    /// Accepted for ComfyUI compatibility. MiniMax-H3's distilled sampler uses
    /// cfg=1 and has no negative branch, so this field has no effect.
    /// </summary>
    public string? NegativePrompt { get; set; }

    /// <summary>Extra fields forwarded verbatim (forward compatibility).</summary>
    public Dictionary<string, object?> Extra { get; } = new();

    internal string ToJson()
    {
        var values = new Dictionary<string, object?>
        {
            ["model"] = Model,
            ["prompt"] = Prompt,
            ["width"] = Width,
            ["height"] = Height,
            ["frames"] = Frames,
            ["steps"] = Steps,
            ["seed"] = Seed,
            ["output"] = OutputPath,
            ["first_frame"] = FirstFramePath,
            ["last_frame"] = LastFramePath,
            ["pdd8"] = Pdd8,
            ["dit_layers"] = DitLayers,
            ["denoise_reuse"] = DenoiseReuse,
            ["core_reuse"] = CoreReuse,
            ["render_width"] = RenderWidth,
            ["render_height"] = RenderHeight,
            ["reference_image_size"] = ReferenceImageSize,
            ["token_reduction"] = TokenReduction,
            ["sol_attn"] = SolAttn,
            ["fbc"] = Fbc,
            ["preview_denoise"] = PreviewDenoise,
            ["ssd_streaming"] = SsdStreaming,
            ["use_int8_row_fc2"] = UseInt8RowFc2,
            ["use_reference_rope"] = UseReferenceRope,
            ["keep_dit"] = KeepDit,
            ["reference_image"] = ReferenceImage,
            ["reference_video"] = ReferenceVideo,
            ["reference_audio"] = ReferenceAudio,
            ["reference_video_audio"] = ReferenceVideoAudio,
            ["reference_images"] = ReferenceImages,
            ["reference_videos"] = ReferenceVideos,
            ["reference_audios"] = ReferenceAudios,
            ["negative"] = NegativePrompt
        };
        if (References is { Count: > 0 })
            values["references"] = References.Select(r => r.ToDictionary()).ToList();
        foreach (var pair in Extra) values[pair.Key] = pair.Value;
        return JsonSerializer.Serialize(values, H3RuntimeOptions.JsonOptions);
    }
}

/// <summary>Result of one generation.</summary>
public sealed class H3GenerateResult
{
    [JsonPropertyName("status")] public string Status { get; set; } = string.Empty;
    [JsonPropertyName("model")] public string Model { get; set; } = string.Empty;
    [JsonPropertyName("engine")] public string Engine { get; set; } = string.Empty;
    [JsonPropertyName("output")] public string OutputPath { get; set; } = string.Empty;
    [JsonPropertyName("content_type")] public string ContentType { get; set; } = string.Empty;
    [JsonPropertyName("width")] public int Width { get; set; }
    [JsonPropertyName("height")] public int Height { get; set; }
    [JsonPropertyName("frames")] public int Frames { get; set; }
    [JsonPropertyName("fps")] public int Fps { get; set; }
    [JsonPropertyName("sample_rate")] public int SampleRate { get; set; }
    [JsonPropertyName("seed")] public long Seed { get; set; }
    [JsonPropertyName("seed_string")] public string SeedString { get; set; } = string.Empty;
    [JsonPropertyName("elapsed_ms")] public double ElapsedMilliseconds { get; set; }
    [JsonPropertyName("frames_per_second")] public double FramesPerSecond { get; set; }
    [JsonPropertyName("bytes")] public long Bytes { get; set; }
    [JsonPropertyName("pdd8")] public bool Pdd8 { get; set; }

    /// <summary>Engine phase transitions with their offset from the start.</summary>
    [JsonPropertyName("phases")] public List<H3PhaseTiming> Phases { get; set; } = new();

    /// <summary>The full JSON document returned by the native library.</summary>
    [JsonIgnore] public string RawJson { get; private set; } = string.Empty;

    public byte[] ReadOutputBytes() => File.ReadAllBytes(OutputPath);

    internal static H3GenerateResult Parse(string json)
    {
        H3GenerateResult result = JsonSerializer.Deserialize<H3GenerateResult>(json)
            ?? throw new H3RuntimeException("the native library returned an empty result");
        result.RawJson = json;
        return result;
    }
}

/// <summary>One engine phase and when it started, relative to generation start.</summary>
public sealed class H3PhaseTiming
{
    [JsonPropertyName("phase")] public string Phase { get; set; } = string.Empty;
    [JsonPropertyName("at_ms")] public double AtMilliseconds { get; set; }
}

/// <summary>Progress reported by the engine.</summary>
public readonly record struct H3Progress(string Phase, int Completed, int Total)
{
    /// <summary>0..1 when Total is known, otherwise null.</summary>
    public double? Fraction => Total > 0 ? Math.Clamp((double)Completed / Total, 0.0, 1.0) : null;
}

/// <summary>A decoded RGB24 frame delivered during generation.</summary>
public sealed class H3Frame
{
    public int Width { get; init; }
    public int Height { get; init; }
    public int Stride { get; init; }
    public int FrameIndex { get; init; }
    public int FrameCount { get; init; }
    public int DenoiseStep { get; init; }
    public int DenoiseSteps { get; init; }

    /// <summary>
    /// A copy of the pixels, or empty when <c>captureFrames</c> was false.
    /// The native pointer is only valid inside the callback.
    /// </summary>
    public byte[] Pixels { get; init; } = Array.Empty<byte>();
}
