namespace H3NetRuntime.WebUI;

/// <summary>
/// The "通用版" presets. They are the only knobs a general-edition user sees;
/// the advanced edition can override every one of them.
/// </summary>
public static class Presets
{
    public static readonly PresetDefinition[] All =
    {
        new()
        {
            Id = "turbo", Name = "极速", Description = "512×512 · 22 帧 · 8 步 PDD",
            Width = 512, Height = 512, Frames = 22, Pdd8 = true, Steps = 8,
            DitLayers = 45, DenoiseReuse = 1
        },
        new()
        {
            Id = "balanced", Name = "均衡", Description = "864×480 · 56 帧 · 8 步 PDD",
            Width = 864, Height = 480, Frames = 56, Pdd8 = true, Steps = 8,
            DitLayers = 50, DenoiseReuse = 1
        },
        new()
        {
            Id = "quality", Name = "画质", Description = "864×480 · 56 帧 · 20 步完整采样",
            Width = 864, Height = 480, Frames = 56, Pdd8 = false, Steps = 20,
            DitLayers = 50, DenoiseReuse = 1
        },
        new()
        {
            Id = "cinematic", Name = "长镜头", Description = "864×480 · 124 帧（约 5 秒）",
            Width = 864, Height = 480, Frames = 124, Pdd8 = true, Steps = 8,
            DitLayers = 50, DenoiseReuse = 1
        }
    };

    /// <summary>
    /// The engine aligns the requested frame count up to 17n+5 and refuses
    /// anything below 22, so only these values are offered in the UI.
    /// </summary>
    public static readonly int[] FrameChoices =
        Enumerable.Range(0, 22).Select(n => 5 + 17 * n).Where(n => n >= 22).ToArray();

    /// <summary>Mirrors h3_align_frame_count: round up to the next 17n+5.</summary>
    public static int AlignFrames(int requested)
    {
        int value = Math.Max(requested, 5);
        int remainder = (value - 5) % 17;
        if (remainder != 0) value += 17 - remainder;
        return value;
    }

    public static readonly ResolutionPreset[] Resolutions =
    {
        new() { Label = "512 × 512 (1:1)", Width = 512, Height = 512 },
        new() { Label = "768 × 448 (12:7)", Width = 768, Height = 448 },
        new() { Label = "864 × 480 (16:9)", Width = 864, Height = 480 },
        new() { Label = "640 × 480 (4:3)", Width = 640, Height = 480 },
        new() { Label = "480 × 640 (3:4 竖屏)", Width = 480, Height = 640 },
        new() { Label = "576 × 1024 (9:16 竖屏)", Width = 576, Height = 1024 },
        new() { Label = "1024 × 576 (16:9 宽屏)", Width = 1024, Height = 576 }
    };

    public static PresetDefinition Find(string? id) =>
        All.FirstOrDefault(p => string.Equals(p.Id, id, StringComparison.OrdinalIgnoreCase))
        ?? All[1];

    /// <summary>Applies a preset as defaults, keeping explicitly supplied values.</summary>
    public static void ApplyDefaults(GenerateApiRequest request, PresetDefinition preset)
    {
        request.Width ??= preset.Width;
        request.Height ??= preset.Height;
        request.Frames ??= preset.Frames;
        request.Pdd8 ??= preset.Pdd8;
        request.Steps ??= preset.Steps;
        request.DitLayers ??= preset.DitLayers;
        request.DenoiseReuse ??= preset.DenoiseReuse;
    }

    /// <summary>
    /// General edition: the preset owns every sampling knob and every engine
    /// toggle. Canvas and duration stay user-controlled (they are validated
    /// server-side) because picking a portrait clip is a legitimate need.
    /// </summary>
    public static void Force(GenerateApiRequest request, PresetDefinition preset)
    {
        request.Pdd8 = preset.Pdd8;
        request.Steps = preset.Steps;
        request.DitLayers = preset.DitLayers;
        request.DenoiseReuse = preset.DenoiseReuse;
        request.CoreReuse = 1;
        request.TokenReduction = false;
        request.SolAttn = false;
        request.Fbc = false;
        request.SsdStreaming = false;
        request.UseInt8RowFc2 = false;
        request.UseReferenceRope = false;
        request.RenderWidth = 0;
        request.RenderHeight = 0;
        request.PreviewDenoise = false;
        request.KeepDit = true;
    }
}
