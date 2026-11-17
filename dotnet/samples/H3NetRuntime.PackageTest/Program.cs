using System.Runtime.InteropServices;
using System.Text.Json;
using H3NetRuntime;

namespace H3NetRuntime.PackageTest;

/// <summary>
/// End-to-end test of the published H3NetRuntime NuGet package against the
/// five ComfyUI_MiniMax-H3-V8 weight files on this machine.
///
///   dotnet run -c Release -- --mode all
///   dotnet run -c Release -- --mode t2v --frames 22 --width 512 --height 512
/// </summary>
internal static class Program
{
    private const string DefaultModelDir = @"D:\AI\ComfyUI_MiniMax-H3-V8\ComfyUI\models";

    private static readonly string[] AllModes = { "t2v", "fl2v", "ref2v" };

    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        Dictionary<string, string> options = ParseArgs(args);
        string modeArg = options.GetValueOrDefault("mode", "all").ToLowerInvariant();
        IEnumerable<string> modes = modeArg == "all" ? AllModes : new[] { modeArg };
        foreach (string mode in modes)
            if (!AllModes.Contains(mode))
                return Fail($"unknown --mode '{mode}'. Use t2v, fl2v, ref2v or all.");

        string repoRoot = FindRepoRoot();
        string modelDir = options.GetValueOrDefault("model-dir", DefaultModelDir);
        string outputDir = options.GetValueOrDefault("out-dir", Path.Combine(repoRoot, "work", "package-test"));
        string packCacheDir = options.GetValueOrDefault("pack-cache", Path.Combine(repoRoot, ".cache", "h3rt-packs"));
        int width = ParseInt(options, "width", 512);
        int height = ParseInt(options, "height", 512);
        int frames = ParseInt(options, "frames", 22);
        int steps = ParseInt(options, "steps", 8);
        ulong seed = options.TryGetValue("seed", out string? seedText) && ulong.TryParse(seedText, out ulong parsed)
            ? parsed : 42UL;

        if (options.ContainsKey("list"))
        {
            foreach (string mode in modes)
            {
                H3GenerateRequest plan = BuildRequest(
                    mode, "(not generated)", repoRoot, width, height, frames, steps, seed);
                Console.WriteLine($"[{mode}] prompt: {plan.Prompt}");
                if (plan.FirstFramePath is not null) Console.WriteLine($"       first: {plan.FirstFramePath}");
                if (plan.LastFramePath is not null) Console.WriteLine($"       last : {plan.LastFramePath}");
                foreach (H3Reference reference in plan.References ?? new List<H3Reference>())
                    Console.WriteLine($"       ref  : {reference.Kind} {reference.Path}");
            }
            return 0;
        }

        Directory.CreateDirectory(outputDir);
        string rid = OperatingSystem.IsWindows() ? "win-x64" : "linux-x64";
        string packageTokenizer = Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", "tokenizer", "tokenizer.json");
        string tokenizer = File.Exists(packageTokenizer)
            ? packageTokenizer
            : Path.Combine(repoRoot, "assets", "tokenizer", "tokenizer.json");

        Console.WriteLine($"h3rt version: {H3Runtime.Version}  ABI: {H3Runtime.AbiVersion}");
        Console.WriteLine($"build info  : {H3Runtime.BuildInfoJson}");
        Console.WriteLine($"model dir   : {modelDir}");
        Console.WriteLine($"output dir  : {outputDir}");
        Console.WriteLine($"tokenizer   : {tokenizer}");

        string diffusion = Path.Combine(modelDir, "diffusion_models");
        string encoders = Path.Combine(modelDir, "text_encoders");
        string vae = Path.Combine(modelDir, "vae");
        string fl2va = FindOne(diffusion, "fl2va", "hyperflow_flash_int8_convrot.safetensors");
        string? ref2va = FindOne(diffusion, new[] { "ref2va", "hyperflow_flash_int8_convrot.safetensors" }, required: false);
        string textEncoder = FindOne(encoders, "qwen3vl-32B-MiniMax-H3", ".gguf");
        string videoVae = FindOne(vae, "sol_vae", ".safetensors");
        string audioVae = FindOne(vae, "audio_vae", ".safetensors");
        if (!File.Exists(tokenizer)) return Fail($"tokenizer.json not found: {tokenizer}");

        var models = new H3ModelPaths
        {
            Transformer = fl2va,
            Ref2vaTransformer = ref2va,
            TextEncoder = textEncoder,
            VideoVae = videoVae,
            AudioVae = audioVae,
            Tokenizer = tokenizer
        };

        Console.WriteLine("\nResolved model components (from the NuGet package test):");
        Console.WriteLine($"  FL2VA DiT      {fl2va}");
        Console.WriteLine($"  Ref2VA DiT     {ref2va ?? "(not present)"}");
        Console.WriteLine($"  text encoder   {textEncoder}");
        Console.WriteLine($"  video VAE      {videoVae}");
        Console.WriteLine($"  audio VAE      {audioVae}");
        Console.WriteLine($"  tokenizer      {tokenizer}");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        int failures = 0;
        try
        {
            using H3Runtime runtime = H3Runtime.Create(new H3RuntimeOptions
            {
                ModelDir = modelDir,
                Models = models,
                PackCacheDir = packCacheDir,
                KeepDit = true,
                CacheEnabled = true
            });
            Console.WriteLine($"\nruntime pack: {runtime.ModelInfoJson}");

            foreach (string mode in modes)
            {
                if (mode == "ref2v" && ref2va is null)
                {
                    Console.WriteLine("\n[ref2v] SKIP: Ref2VA checkpoint is not installed.");
                    continue;
                }

                string output = Path.Combine(outputDir, $"{mode}.mp4");
                H3GenerateRequest request = BuildRequest(mode, output, repoRoot, width, height, frames, steps, seed);
                Console.WriteLine($"\n=== {mode} -> {output} ===");
                Console.WriteLine($"prompt: {request.Prompt}");

                var progress = new Progress<H3Progress>(p =>
                    Console.WriteLine($"  [{mode}] {p.Phase,-24} {p.Completed}/{p.Total}"));

                try
                {
                    H3GenerateResult result = await runtime.GenerateAsync(
                        request, progress, cancellationToken: cts.Token);

                    if (!File.Exists(output) || new FileInfo(output).Length == 0)
                        throw new InvalidOperationException("the engine returned success but wrote no MP4.");

                    Console.WriteLine($"  OK {output}  {new FileInfo(output).Length} bytes  " +
                                      $"{result.Width}x{result.Height} {result.Frames}f  " +
                                      $"elapsed {result.ElapsedMilliseconds / 1000.0:F1}s");
                    foreach (H3PhaseTiming phase in result.Phases.TakeLast(6))
                        Console.WriteLine($"    phase {phase.Phase,-24} at {phase.AtMilliseconds / 1000.0,7:F1}s");
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine($"  CANCELLED {mode}");
                    failures++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  FAILED {mode}: {ex.Message}");
                    failures++;
                }

                if (cts.IsCancellationRequested) break;
            }
        }
        catch (H3RuntimeException ex)
        {
            return Fail($"runtime error ({ex.Status}): {ex.Message}");
        }
        catch (Exception ex)
        {
            return Fail(ex.Message);
        }

        Console.WriteLine(failures == 0 ? "\nALL MODES PASSED" : $"\n{failures} MODE(S) FAILED");
        return failures == 0 ? 0 : 1;
    }

    private static H3GenerateRequest BuildRequest(
        string mode, string output, string repoRoot, int width, int height, int frames, int steps, ulong seed)
    {
        string demoRoot = Path.Combine(repoRoot, "dotnet", "src", "H3NetRuntime.WebUI", "wwwroot", "demo");
        string fox = Path.Combine(demoRoot, "fox-snow.jpg");
        string forest = Path.Combine(demoRoot, "snow-forest.jpg");

        var request = new H3GenerateRequest
        {
            Prompt = mode switch
            {
                "fl2v" => "以这一帧为起点：赤狐保持原有的姿态与毛色，向前迈出两步，耳朵轻微转动，" +
                          "鼻息带出白雾。镜头缓慢推近，背景略微虚化，雪地反光自然。",
                "ref2v" => "参考图 1 提供主体（赤狐的毛色、体态与面部特征），参考图 2 提供场景" +
                           "（积雪针叶林与暮色天空）。把这只赤狐放进参考图 2 的林线前，" +
                           "它站在齐踝的深雪里回头望向镜头。镜头从低角度缓慢推近。",
                _ => "A red fox trotting through fresh snow at golden hour, slow dolly-in, " +
                     "shallow depth of field, film grain, natural ambient sound."
            },
            Width = width,
            Height = height,
            Frames = frames,
            Steps = steps,
            Seed = seed,
            Pdd8 = true,
            OutputPath = output,
            KeepDit = true
        };

        switch (mode)
        {
            case "fl2v":
                if (!File.Exists(fox)) throw new FileNotFoundException("built-in first-frame demo asset is missing", fox);
                request.FirstFramePath = fox;
                break;
            case "ref2v":
                if (!File.Exists(fox) || !File.Exists(forest))
                    throw new FileNotFoundException("built-in reference demo assets are missing");
                request.References = new List<H3Reference>
                {
                    new() { Kind = "image", Path = fox },
                    new() { Kind = "image", Path = forest }
                };
                break;
        }
        return request;
    }

    private static string FindOne(string directory, params string[] needles)
    {
        string? path = FindOne(directory, needles, required: true);
        return path!;
    }

    private static string? FindOne(string directory, string[] needles, bool required)
    {
        if (Directory.Exists(directory))
        {
            string[] files = Directory.GetFiles(directory)
                .Where(f => needles.All(n => Path.GetFileName(f).Contains(n, StringComparison.OrdinalIgnoreCase)))
                .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (files.Length > 0) return files[0];
        }
        if (required)
            throw new FileNotFoundException(
                $"no file matching [{string.Join(", ", needles)}] under {directory}");
        return null;
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            if (!arg.StartsWith("--", StringComparison.Ordinal)) continue;
            string key = arg[2..];
            string? value = null;
            int equals = key.IndexOf('=');
            if (equals >= 0)
            {
                value = key[(equals + 1)..];
                key = key[..equals];
            }
            else if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                value = args[++i];
            }
            result[key] = value ?? "true";
        }
        return result;
    }

    private static int ParseInt(Dictionary<string, string> options, string key, int fallback) =>
        options.TryGetValue(key, out string? text) && int.TryParse(text, out int value) ? value : fallback;

    private static string FindRepoRoot()
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "README.md")) &&
                File.Exists(Path.Combine(dir.FullName, "dotnet", "H3NetRuntime.sln")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("could not locate the repository root (README.md + dotnet/H3NetRuntime.sln)");
    }

    private static int Fail(string message)
    {
        Console.Error.WriteLine($"ERROR: {message}");
        return 1;
    }
}
