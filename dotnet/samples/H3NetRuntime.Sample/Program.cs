using System.Globalization;
using H3NetRuntime;

// A minimal .NET client: no Python, no ComfyUI, no HTTP server. One P/Invoke
// call drives the whole text -> video+audio pipeline.

var argsMap = ParseArgs(args);
if (argsMap.ContainsKey("help") || argsMap.ContainsKey("h"))
{
    PrintUsage();
    return 0;
}

string? modelDir = Get(argsMap, "model-dir")
    ?? Environment.GetEnvironmentVariable("H3_MODEL_DIR");
string? config = Get(argsMap, "config");
if (string.IsNullOrEmpty(modelDir) && string.IsNullOrEmpty(config))
{
    Console.Error.WriteLine("ERROR: pass --model-dir <pack>, set H3_MODEL_DIR, " +
                            "or pass --config <h3.config.json>. See --help.");
    return 2;
}

string prompt = Get(argsMap, "prompt")
    ?? "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light.";
bool infoOnly = argsMap.ContainsKey("info-only");

Console.WriteLine($"h3rt runtime {H3Runtime.Version} (ABI {H3Runtime.AbiVersion})");
Console.WriteLine($"build: {H3Runtime.BuildInfoJson}");

var options = new H3RuntimeOptions
{
    ModelDir = modelDir,
    ConfigPath = config,
    KeepDit = true,
    CacheEnabled = true,
    TextCacheDir = Environment.GetEnvironmentVariable("H3_TEXT_CACHE_DIR")
};
using var runtime = H3Runtime.Create(options);
Console.WriteLine($"device: {runtime.DeviceInfoJson}");
Console.WriteLine($"model:  {runtime.ModelInfoJson}");

if (infoOnly)
{
    Console.WriteLine("info-only: no generation requested.");
    return 0;
}

var request = new H3GenerateRequest
{
    Prompt = prompt,
    Width = ParseInt(argsMap, "width"),
    Height = ParseInt(argsMap, "height"),
    Frames = ParseInt(argsMap, "frames"),
    Steps = ParseInt(argsMap, "steps"),
    Seed = ParseULong(argsMap, "seed"),
    OutputPath = Get(argsMap, "out"),
    FirstFramePath = Get(argsMap, "first-frame"),
    LastFramePath = Get(argsMap, "last-frame"),
    Pdd8 = !argsMap.ContainsKey("no-pdd8")
};

var progress = new Progress<H3Progress>(p =>
{
    string suffix = p.Total > 0 ? $" {p.Completed}/{p.Total} ({p.Fraction:P0})" : "";
    Console.WriteLine($"  [{p.Phase}]{suffix}");
});

int framesSeen = 0;
void OnFrame(H3Frame frame)
{
    framesSeen++;
    if (framesSeen <= 3 || frame.FrameIndex == frame.FrameCount - 1)
        Console.WriteLine($"  frame {frame.FrameIndex + 1}/{frame.FrameCount} " +
                          $"{frame.Width}x{frame.Height}, stride {frame.Stride}");
}

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    Console.WriteLine("Ctrl+C: cancelling at the next engine callback...");
    cts.Cancel();
};

Console.WriteLine("generating (this can take several minutes on first run)...");
var started = DateTime.UtcNow;
H3GenerateResult result;
try
{
    result = await runtime.GenerateAsync(request, progress, OnFrame,
        captureFrames: false, cancellationToken: cts.Token);
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("generation cancelled.");
    return 3;
}
catch (H3RuntimeException ex)
{
    Console.Error.WriteLine($"h3rt error [{ex.Status}]: {ex.Message}");
    return 4;
}

Console.WriteLine($"done in {(DateTime.UtcNow - started).TotalSeconds:F1}s " +
                  $"(engine {result.ElapsedMilliseconds / 1000.0:F1}s)");
Console.WriteLine(result.RawJson);
Console.WriteLine($"output : {result.OutputPath}");
Console.WriteLine($"size   : {result.Bytes:N0} bytes, {result.Width}x{result.Height}, " +
                  $"{result.Frames} frames, fps {result.Fps}, audio {result.SampleRate} Hz");
Console.WriteLine($"frames delivered to callback: {framesSeen}");
return 0;

static void PrintUsage()
{
    Console.WriteLine("""
        H3NetRuntime.Sample - generate a MiniMax-H3 / HyperFlow H3 Flash clip from .NET.

        Usage:
          H3NetRuntime.Sample --model-dir <pack> [options]
          H3NetRuntime.Sample --config <h3.config.json> [options]
          H3NetRuntime.Sample --help

        Options:
          --model-dir <dir>     MiniMax-H3 pack (contains FL2VA/). Env: H3_MODEL_DIR
          --config <file>       h3.config.json with platform pack paths
          --prompt <text>       positive prompt (default: fox in snow)
          --out <path>          MP4 output (default: temp file)
          --width / --height    multiples of 32 (default 864x480)
          --frames <n>          5..362 (default 56)
          --steps <n>           forced to 8 with --pdd8 (default on)
          --seed <n>            non-negative integer
          --no-pdd8             run the full 20-step schedule instead
          --first-frame <img>   FL2VA first-frame anchor
          --last-frame <img>    FL2VA last-frame anchor
          --info-only           load the pack and print device/model info, no generation

        The native h3rt.dll/libh3rt.so is found via H3RT_NATIVE_DIR, the app
        directory, or dist/<rid> above it. FFmpeg must be on PATH.
        """);
}

static Dictionary<string, string?> ParseArgs(string[] argv)
{
    var map = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
    for (int i = 0; i < argv.Length; i++)
    {
        string token = argv[i];
        if (!token.StartsWith("--", StringComparison.Ordinal)) continue;
        string key = token[2..];
        string? value = null;
        if (i + 1 < argv.Length && !argv[i + 1].StartsWith("--", StringComparison.Ordinal))
            value = argv[++i];
        map[key] = value;
    }
    return map;
}

static string? Get(Dictionary<string, string?> map, string key) =>
    map.TryGetValue(key, out string? value) ? value : null;

static int? ParseInt(Dictionary<string, string?> map, string key) =>
    int.TryParse(Get(map, key), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : null;

static ulong? ParseULong(Dictionary<string, string?> map, string key) =>
    ulong.TryParse(Get(map, key), NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong v) ? v : null;
