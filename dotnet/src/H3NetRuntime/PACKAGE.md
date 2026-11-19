# H3NetRuntime

Runs **MiniMax-H3 / HyperFlow H3 Flash** video+audio generation inside a .NET
process. No Python, no ComfyUI, no HTTP round trip: a CUDA C engine (`h3rt`) is
P/Invoked through a small JSON ABI, and the native library for your platform
ships in this package.

```csharp
using H3NetRuntime;

using var runtime = H3Runtime.Create(new H3RuntimeOptions
{
    ModelDir = @"D:\models\ComfyUI_MiniMax-H3-V8\ComfyUI\models",
    KeepDit = true,        // default: keep the DiT mapped (safe on its own)
    CacheEnabled = false   // default: release the ~9.5 GiB VAE decoder between calls
});

var result = await runtime.GenerateAsync(new H3GenerateRequest
{
    Prompt = "A red fox walks through fresh snow, camera slowly pushes in.",
    Width = 864, Height = 480, Frames = 56, Pdd8 = true,
    OutputPath = @"D:\out\fox.mp4"
});

Console.WriteLine($"{result.OutputPath} {result.Width}x{result.Height} {result.Frames}f");
```

## Install

```bash
dotnet add package H3NetRuntime
```

Native assets are laid out as `runtimes/win-x64/native/h3rt.dll` and
`runtimes/linux-x64/native/libh3rt.so`; the loader probes those paths, the app
directory, `dist/<rid>`, the working directory and finally the OS loader. Set
`H3RT_NATIVE_DIR` to point at a build you made yourself.

## Requirements

| | |
|---|---|
| OS / arch | Windows x64, Linux x64 |
| GPU | NVIDIA, compute capability **matching the build** (`sm_89` by default) |
| Runtime | CUDA 12/13 runtime + cuBLAS on `PATH` / `LD_LIBRARY_PATH` |
| External | `ffmpeg` and `ffprobe` on `PATH` (MP4 muxing) |
| Weights | **not included** — five files from ComfyUI_MiniMax-H3-V8; see `docs/MODELS.md` |

## What it exposes

| | |
|---|---|
| `H3Runtime.Create(H3RuntimeOptions)` | load a pack, initialise the device |
| `Generate` / `GenerateAsync` | MP4 out, `IProgress<H3Progress>`, per-frame RGB callback, `CancellationToken` |
| `Preload` | read the DiT before the first request |
| `DeviceInfoJson` / `ModelInfoJson` / `CacheInfoJson` | introspection |
| `Cancel` | cooperative cancel at the next engine callback |

Requests mirror the ComfyUI-V8 workflow: `Prompt`, `Width`/`Height`/`Frames`,
`Steps`, `Seed`, `Pdd8`, `FirstFramePath`/`LastFramePath` (FL2VA),
`ReferenceImages` / `References` (Ref2VA, ordered), plus the engine's speed
knobs (`DitLayers`, `DenoiseReuse`, `CoreReuse`, `TokenReduction`, `SolAttn`,
`Fbc`, `SsdStreaming`, `RenderWidth`/`RenderHeight`).

### Per-component model paths

Checkpoints usually come from different downloads. Name each one instead of
building a directory tree; the runtime links them into the layout the engine
expects (junctions on Windows, symlinks on Linux — no copies):

```csharp
var runtime = H3Runtime.Create(new H3RuntimeOptions
{
    ModelDir = @"D:\models\ComfyUI_MiniMax-H3-V8\ComfyUI\models", // optional base
    Models = new H3ModelPaths
    {
        Transformer = @"E:\hf\minimax_h3_fl2va_hyperflow_flash_int8_convrot.safetensors",
        Ref2vaTransformer = @"E:\hf\minimax_h3_ref2va_hyperflow_flash_int8_convrot.safetensors",
        TextEncoder = @"E:\hf\qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
        VideoVae    = @"E:\hf\minimax_h3_sol_vae_int8_convrot.safetensors",
        AudioVae    = @"E:\hf\minimax_h3_audio_vae_fp32.safetensors"
    }
});
```

Unset components fall back to the same slot under `ModelDir`. `PackDir` reports
the directory the engine actually reads, and `IsComposedPack` says whether it
was assembled.

## Residency and consecutive generations

`CacheEnabled` (the resident video-VAE decoder) is **off by default**;
`KeepDit` stays on. On a 24 GiB card, enabling both keeps the ~16 GiB DiT and
the ~9.5 GiB decoder mapped at once with no headroom, so a second generation in
the same process can degrade to minutes-or-hours per step instead of failing.
Leave the defaults as-is for batch/back-to-back work. `H3RuntimeOptions.VaePath`
reports `"fast"` (safe) or `"resident-slow"` (risky) for health checks.

## Notes

- One handle runs one generation at a time (`H3RT_ERR_BUSY` otherwise). The
  engine keeps ~30 GB of weights resident, so a long-lived handle is the
  intended shape; a cold start reads the pack for minutes.
- Frame counts must be `17n+5` and at least 22 (`22, 39, 56 … 362`); the runtime
  rounds up.
- Cancellation lands at the next engine checkpoint, not immediately.
- The native library is compiled for one compute capability. Rebuild with
  `scripts/build-native.cmd <cc>` (Windows) or `scripts/build.sh <cc>` (Linux)
  for a different GPU.
