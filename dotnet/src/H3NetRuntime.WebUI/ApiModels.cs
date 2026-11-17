using System.Text.Json.Serialization;

namespace H3NetRuntime.WebUI;

/// <summary>Body of POST /api/generate. All fields optional except Prompt.</summary>
public sealed class GenerateApiRequest
{
    public string? Prompt { get; set; }
    public string? NegativePrompt { get; set; }
    public string? Preset { get; set; }          // turbo | balanced | quality | cinematic
    public string? ModelDir { get; set; }        // advanced: switch pack before running

    /// <summary>t2v (text only), fl2v (first/last frame) or ref2v (references).</summary>
    public string? Mode { get; set; }

    public int? Width { get; set; }
    public int? Height { get; set; }
    public int? Frames { get; set; }
    public int? Steps { get; set; }
    public long? Seed { get; set; }

    public bool? Pdd8 { get; set; }
    public int? DitLayers { get; set; }
    public int? DenoiseReuse { get; set; }
    public int? CoreReuse { get; set; }
    public int? RenderWidth { get; set; }
    public int? RenderHeight { get; set; }
    public bool? TokenReduction { get; set; }
    public bool? SolAttn { get; set; }
    public bool? Fbc { get; set; }
    public bool? PreviewDenoise { get; set; }
    public bool? SsdStreaming { get; set; }
    public bool? UseInt8RowFc2 { get; set; }
    public bool? UseReferenceRope { get; set; }
    public bool? KeepDit { get; set; }
    public string? ReferenceImageSize { get; set; }

    public string? FirstFrame { get; set; }
    public string? LastFrame { get; set; }
    public string? ReferenceImage { get; set; }
    public string? ReferenceVideo { get; set; }
    public string? ReferenceAudio { get; set; }
    public List<string>? ReferenceImages { get; set; }

    /// <summary>Ordered Ref2VA conditioning list (images, videos, audio).</summary>
    public List<ReferenceInput>? References { get; set; }

    public string? OutputPath { get; set; }
    public bool? LivePreview { get; set; }
    public int? PreviewEvery { get; set; }       // deliver every Nth frame, default 4
}

/// <summary>One ordered Ref2VA reference.</summary>
public sealed class ReferenceInput
{
    /// <summary>"image", "video", "audio" or "video_audio".</summary>
    public string? Kind { get; set; } = "image";
    public string? Path { get; set; }
    public string? AudioPath { get; set; }
    public bool? IncludeEmbeddedAudio { get; set; }
}

public sealed class PreloadApiRequest{
    public int? Width { get; set; }
    public int? Height { get; set; }
    public int? Frames { get; set; }
    public int? Steps { get; set; }
    public bool? Pdd8 { get; set; }
    public int? DitLayers { get; set; }
    public int? DenoiseReuse { get; set; }
    public int? CoreReuse { get; set; }
}

public sealed class ModelSelectRequest
{
    public string? ModelDir { get; set; }
    public string? ConfigPath { get; set; }
}

public sealed class UploadResponse
{
    public string Path { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public long Size { get; set; }
    public string Url { get; set; } = string.Empty;
}

public sealed class PresetDefinition
{
    public string Id { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public string Description { get; init; } = string.Empty;
    public int Width { get; init; }
    public int Height { get; init; }
    public int Frames { get; init; }
    public bool Pdd8 { get; init; }
    public int Steps { get; init; }
    public int DitLayers { get; init; }
    public int DenoiseReuse { get; init; }
}

public sealed class ResolutionPreset
{
    public string Label { get; init; } = string.Empty;
    public int Width { get; init; }
    public int Height { get; init; }
}

public sealed class ModelPackInfo
{
    public string Name { get; init; } = string.Empty;
    public string Path { get; init; } = string.Empty;
    public bool HasRef2Va { get; init; }
    public bool Active { get; init; }
}

public sealed class ConfigResponse
{
    public string Edition { get; set; } = "general";
    public string Version { get; set; } = string.Empty;
    public int AbiVersion { get; set; }
    public bool ModelLoaded { get; set; }
    public string? ModelError { get; set; }
    public string ModelDir { get; set; } = string.Empty;
    public string OutputDir { get; set; } = string.Empty;
    public string BuildInfoJson { get; set; } = "{}";
    public string DeviceInfoJson { get; set; } = "{}";
    public string ModelInfoJson { get; set; } = "{}";
    public string CacheInfoJson { get; set; } = "{}";
    public List<PresetDefinition> Presets { get; set; } = new();
    public List<ResolutionPreset> Resolutions { get; set; } = new();

    /// <summary>Only the engine-valid frame counts (17n+5, at least 22).</summary>
    public List<int> FrameChoices { get; set; } = new();
    public List<ModelPackInfo> Packs { get; set; } = new();
    public bool AdvancedEnabled { get; set; }
    public bool UploadEnabled { get; set; } = true;
    public int MaxFrames { get; set; } = 362;
    public int DefaultPort { get; set; }

    /// <summary>Generation modes the UI offers (text, first/last frame, references).</summary>
    public List<ModeInfo> Modes { get; set; } = new();

    /// <summary>Ref2VA weights are installed, so reference conditioning can run.</summary>
    public bool Ref2vaAvailable { get; set; }

    /// <summary>The pack the engine will actually read (empty when none found).</summary>
    public string ModelPackDir { get; set; } = string.Empty;

    /// <summary>Per-component resolution, so the UI can show what is missing.</summary>
    public List<ModelComponentView> ModelComponents { get; set; } = new();

    /// <summary>Directories that were probed for a pack (shown when nothing is found).</summary>
    public List<string> ModelSearchPaths { get; set; } = new();

    /// <summary>Model download / install guide shipped with the WebUI.</summary>
    public string GuideUrl { get; set; } = "/guide.html";

    /// <summary>Upload limits enforced by the engine's Ref2VA path.</summary>
    public int MaxReferenceImages { get; set; } = 9;
    public int MaxReferenceVideos { get; set; } = 3;
    public int MaxReferenceAudios { get; set; } = 3;
    public int MaxReferences { get; set; } = 12;
}

/// <summary>One generation mode.</summary>
public sealed class ModeInfo
{
    public string Id { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public string Description { get; init; } = string.Empty;
    public bool RequiresRef2Va { get; init; }
}

/// <summary>One weight component as resolved on disk.</summary>
public sealed class ModelComponentView
{
    public string Key { get; init; } = string.Empty;
    public string Label { get; init; } = string.Empty;
    public string Slot { get; init; } = string.Empty;
    public string Path { get; init; } = string.Empty;
    public bool Required { get; init; }
    public bool Present { get; init; }
}

/// <summary>
/// Engine state. Kept separate from <see cref="ConfigResponse"/> because
/// loading the model pack takes seconds and the shell must paint first.
/// </summary>
public sealed class StatusResponse
{
    public bool ModelLoaded { get; set; }
    public string? ModelError { get; set; }
    public string ModelDir { get; set; } = string.Empty;
    public string DeviceInfoJson { get; set; } = "{}";
    public string ModelInfoJson { get; set; } = "{}";
    public string CacheInfoJson { get; set; } = "{}";
    public bool Busy { get; set; }
}

public sealed class PreloadResponse
{
    public double ElapsedMs { get; set; }
    public string ModelDir { get; set; } = string.Empty;
}

public sealed class JobProgressView{
    [JsonPropertyName("phase")] public string Phase { get; set; } = string.Empty;
    /// <summary>Chinese stage name for the status line ("文本编码").</summary>
    [JsonPropertyName("stage")] public string Stage { get; set; } = string.Empty;
    [JsonPropertyName("completed")] public int Completed { get; set; }
    [JsonPropertyName("total")] public int Total { get; set; }
    /// <summary>Progress inside the current stage (0-1).</summary>
    [JsonPropertyName("fraction")] public double? Fraction { get; set; }
    /// <summary>Progress across the whole run (0-1), monotonic.</summary>
    [JsonPropertyName("overall")] public double Overall { get; set; }
    [JsonPropertyName("elapsedMs")] public double ElapsedMs { get; set; }
    /// <summary>Projected remaining time, null until the estimate is meaningful.</summary>
    [JsonPropertyName("etaMs")] public double? EtaMs { get; set; }
}

public sealed class JobView
{
    [JsonPropertyName("id")] public string Id { get; set; } = string.Empty;
    [JsonPropertyName("kind")] public string Kind { get; set; } = "generate";
    [JsonPropertyName("status")] public string Status { get; set; } = "queued";
    [JsonPropertyName("prompt")] public string Prompt { get; set; } = string.Empty;
    [JsonPropertyName("preset")] public string? Preset { get; set; }
    [JsonPropertyName("mode")] public string Mode { get; set; } = "t2v";

    /// <summary>The request as the server will actually run it (edition rules applied).</summary>
    [JsonPropertyName("request")] public GenerateApiRequest? Request { get; set; }
    [JsonPropertyName("createdAt")] public DateTimeOffset CreatedAt { get; set; }
    [JsonPropertyName("startedAt")] public DateTimeOffset? StartedAt { get; set; }
    [JsonPropertyName("finishedAt")] public DateTimeOffset? FinishedAt { get; set; }
    [JsonPropertyName("progress")] public JobProgressView? Progress { get; set; }
    [JsonPropertyName("result")] public H3GenerateResult? Result { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
    [JsonPropertyName("errorStatus")] public int? ErrorStatus { get; set; }
    [JsonPropertyName("videoUrl")] public string? VideoUrl { get; set; }
    [JsonPropertyName("posterUrl")] public string? PosterUrl { get; set; }
    [JsonPropertyName("previewUrl")] public string? PreviewUrl { get; set; }
    [JsonPropertyName("livePreview")] public bool LivePreview { get; set; }
    [JsonPropertyName("queuePosition")] public int QueuePosition { get; set; }
}

public sealed class JobSummary
{
    public string Id { get; set; } = string.Empty;
    public string Prompt { get; set; } = string.Empty;
    public string? Preset { get; set; }
    public DateTimeOffset CreatedAt { get; set; }
    public double ElapsedMs { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public int Frames { get; set; }
    public long Seed { get; set; }
    public string? VideoUrl { get; set; }
    public string? PosterUrl { get; set; }
    public string Status { get; set; } = "completed";
}
