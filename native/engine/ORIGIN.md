# Vendored engine snapshot

`native/engine/` started as a source snapshot of the C/CUDA inference engine
from `h3-spark.c` (upstream: `https://github.com/alexhegit/h3-spark.c`,
originally `antirez/h3.c`). It is vendored so this repository builds
`h3rt.dll` / `libh3rt.so` without Python, ComfyUI or a second checkout — only
a C/CUDA toolchain and FFmpeg.

| | |
|---|---|
| Upstream revision | `a474eca` (snapshot taken 2026-11-16) |
| Engine version | `H3_VERSION 0.2.3` |
| CUDA default arch | `sm_89` (RTX 4090 / Ada); override with `-DH3_CUDA_ARCH=121` |
| Files taken | core `*.c` / `*.h`, `h3_gpu.cu`, `h3_gpu_stubs.c`, `platform/win32/**`, `third_party/cJSON.c/.h`, `LICENSE`, `THIRD_PARTY_NOTICES.md` |
| Omitted | `main.c`, `h3_cli.c`, `h3_serve.c`, `linenoise.c` and their headers, tests, Metal sources, binaries, model fixtures |

## Local changes on top of the snapshot

The snapshot is **not byte-identical to upstream**. It carries the fixes
needed to run the five files published with `ComfyUI_MiniMax-H3-V8`:

- GGUF text encoder support (`h3_gguf.c/.h`, `h3_text_encoder.c`,
  `h3_weights.c/.h`);
- HyperFlow Flash INT8 `convrot` checkpoints: dynamic DiT depth and
  rank-16 time embedder (`h3_dit.c`, `h3_dit_schedule.c/.h`), per-project
  low-rank/INT8 MLP mix, head-interleaved QKV fusion (`h3_gpu.cu/.h`);
- ComfyUI V8 audio/video VAE layout: `decoder.blocks.N`, folded group norms,
  normalization tensors read from the checkpoint instead of `config.json`
  (`h3_audio_vae.c`, `h3_video_vae.c/.h`, `h3_video_encoder.c`);
- per-component model links, path logging and progress callbacks
  (`h3.c`, `h3_host.c/.h`, `h3_safetensors.c/.h`).

The external C ABI lives in `../../include/h3rt.h` and `../../src/`; it is
additive and does not change the engine's own API. To build against a newer
upstream tree instead, point CMake at it:

```
cmake -S . -B build -DH3_ENGINE_DIR=/path/to/h3-spark.c
```

That tree must have the same layout (root sources, `platform/win32/`,
`third_party/cJSON.c`) and may need the fixes above re-applied.

Licensing: see `LICENSE` (upstream MIT) and `THIRD_PARTY_NOTICES.md`. Model
weights are **not** included and are subject to the MiniMax-H3 / HyperFlow
license on Hugging Face.
