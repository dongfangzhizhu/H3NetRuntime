using System.Text.Json;
using H3NetRuntime;

// Zero-dependency managed smoke test. It does not need a GPU or model pack;
// if H3_MODEL_DIR is set, it additionally exercises real model introspection.

int failures = 0;

void Check(bool ok, string what)
{
    Console.WriteLine($"[{(ok ? "PASS" : "FAIL")}] {what}");
    if (!ok) failures++;
}

void ExpectStatus(Action action, H3Status expected, string what)
{
    try
    {
        action();
        Check(false, $"{what} (no exception thrown)");
    }
    catch (H3RuntimeException ex)
    {
        Check(ex.Status == expected, $"{what} (got {ex.Status})");
        Console.WriteLine($"       {ex.Message}");
    }
}

Console.WriteLine($"h3rt {H3Runtime.Version}, ABI {H3Runtime.AbiVersion}");
Check(H3Runtime.AbiVersion == 1, "ABI version is 1");
Check(!string.IsNullOrWhiteSpace(H3Runtime.Version), "version string is non-empty");

string buildInfo = H3Runtime.BuildInfoJson;
Console.WriteLine($"  build: {buildInfo}");
using (JsonDocument doc = JsonDocument.Parse(buildInfo))
{
    Check(doc.RootElement.GetProperty("python_required").GetBoolean() == false,
          "native library declares no Python dependency");
    Check(doc.RootElement.GetProperty("comfyui_required").GetBoolean() == false,
          "native library declares no ComfyUI dependency");
    Check(doc.RootElement.GetProperty("engine_version").GetString() == "0.2.3",
          "engine snapshot version");
}

// The managed wrapper always emits valid JSON, so exercise the native error
// path through a missing model pack instead. The C smoke test covers the raw
// parser (h3rt_create with malformed JSON returns H3RT_ERR_INVALID_JSON).
ExpectStatus(() => H3Runtime.Create(new H3RuntimeOptions
{
    ModelDir = Path.Combine(Path.GetTempPath(), "h3rt-no-such-pack-" + Guid.NewGuid().ToString("N"))
}), H3Status.Model, "missing model pack rejected");

string? modelDir = Environment.GetEnvironmentVariable("H3_MODEL_DIR");
if (!string.IsNullOrEmpty(modelDir) && Directory.Exists(modelDir))
{
    Console.WriteLine($"model pack: {modelDir}");
    using var runtime = H3Runtime.Create(new H3RuntimeOptions { ModelDir = modelDir });
    using (JsonDocument device = JsonDocument.Parse(runtime.DeviceInfoJson))
    {
        string name = device.RootElement.GetProperty("name").GetString() ?? "";
        Check(name.Length > 0, $"device detected: {name}");
        Check(device.RootElement.GetProperty("build_arch").GetInt32() > 0,
              "CUDA build architecture recorded");
    }
    using (JsonDocument model = JsonDocument.Parse(runtime.ModelInfoJson))
        Check(model.RootElement.GetProperty("fl2va_transformer")
                  .GetProperty("files").GetInt32() > 0,
              "FL2VA transformer inventory loaded");

    using (JsonDocument cache = JsonDocument.Parse(runtime.CacheInfoJson))
        Check(cache.RootElement.TryGetProperty("dit_loads", out _),
              "cache info readable");

    runtime.ClearCache();
    runtime.Cancel();
    Check(runtime.IsCancelled, "cancel flag observable from managed code");

    try
    {
        runtime.Generate(new H3GenerateRequest { Prompt = " " });
        Check(false, "empty prompt rejected");
    }
    catch (ArgumentException)
    {
        Check(true, "empty prompt rejected before entering native code");
    }

    // Per-component paths: instead of one pack root, name every weight source
    // and let h3rt materialise the standard FL2VA/... layout as links. Opt-in
    // because it costs a second full weight load.
    if (Environment.GetEnvironmentVariable("H3_SMOKE_COMPONENTS") == "1")
    {
        string fl2va = Path.Combine(modelDir, "FL2VA");
        var components = new H3ModelPaths
        {
            TextEncoder = Path.Combine(fl2va, "text_encoder"),
            // A bare .safetensors: config.json and pdd_heads.safetensors must be
            // linked in next to it from the same slot.
            Transformer = Path.Combine(fl2va, "transformer", "model.safetensors"),
            // The parent of source/ is accepted for the video VAE.
            VideoVae = Path.Combine(fl2va, "video_vae"),
            AudioVae = Path.Combine(fl2va, "audio_vae"),
            Tokenizer = Path.Combine(fl2va, "tokenizer"),
            PddLora = Path.Combine(fl2va, "pdd_lora.safetensors")
        };

        Console.WriteLine("per-component paths:");
        using var composed = H3Runtime.Create(new H3RuntimeOptions
        {
            Models = components,
            PackCacheDir = Path.Combine(Path.GetTempPath(), "h3rt-smoke-packs")
        });

        Check(composed.IsComposedPack, "engine reads a composed link pack");
        Check(composed.PackDir is { Length: > 0 } &&
              !string.Equals(composed.PackDir, modelDir, StringComparison.OrdinalIgnoreCase),
              $"composed pack differs from the source pack: {composed.PackDir}");
        Check(Directory.Exists(Path.Combine(composed.PackDir!, "FL2VA", "transformer")),
              "FL2VA/transformer slot exists in the composed pack");
        Check(File.Exists(Path.Combine(composed.PackDir!, "FL2VA", "transformer",
                                       "pdd_heads.safetensors")),
              "sidecar pdd_heads.safetensors linked next to the bare weight file");

        using (JsonDocument model = JsonDocument.Parse(composed.ModelInfoJson))
        {
            Check(model.RootElement.GetProperty("text_encoder")
                      .GetProperty("tensors").GetInt32() > 0,
                  "text encoder tensors visible through the composed pack");
            Check(model.RootElement.GetProperty("fl2va_transformer")
                      .GetProperty("files").GetInt32() > 0,
                  "DiT inventory visible through the composed pack");
        }
    }
    else
    {
        Console.WriteLine("(set H3_SMOKE_COMPONENTS=1 to also test per-component paths)");
    }
}
else
{
    Console.WriteLine("(set H3_MODEL_DIR to a MiniMax-H3 pack to test model loading)");
}

Console.WriteLine($"\n{(failures == 0 ? "SMOKE OK" : "SMOKE FAILED")} ({failures} failure(s))");
return failures == 0 ? 0 : 1;
