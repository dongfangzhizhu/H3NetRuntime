# Vendored engine snapshot

`native/engine/` is a **source snapshot** of the C/CUDA inference engine from
`../h3-spark.c` (upstream: `https://github.com/alexhegit/h3-spark.c`,
originally `antirez/h3.c`). It is vendored so that this repository can build
`h3rt.dll` / `libh3rt.so` on a machine that has **no Python, no ComfyUI and no
h3-spark.c checkout** — only a C/CUDA toolchain and FFmpeg.

| | |
|---|---|
| Upstream revision | `a474eca` (local checkout, 2026-11-16) |
| Engine version | `H3_VERSION 0.2.3` |
| CUDA default arch | `sm_89` (RTX 4090 / Ada); override with `-DH3_CUDA_ARCH=121` |
| Files taken | Root `*.h`, the 38 core `*.c`, `h3_gpu.cu`, `h3_gpu_stubs.c`, `platform/win32/**`, `third_party/cJSON.c/.h`, `LICENSE`, `THIRD_PARTY_NOTICES.md` |
| Files intentionally omitted | `main.c`, `h3_cli.c`, `h3_serve.c`, `linenoise.c`, tests, Metal sources, binaries, `.o`/`.d`, model fixtures |

Only our own `h3rt` translation unit (`../../src/h3rt.c`) is added; **no
upstream source was modified**. If you want to pick up a newer engine, point
CMake at a checkout instead of the snapshot:

```
cmake -S . -B build -DH3_ENGINE_DIR=/path/to/h3-spark.c
```

The `H3_ENGINE_DIR` tree must have the same layout (root sources,
`platform/win32/`, `third_party/cJSON.c`).

Licensing: see `LICENSE` (upstream) and `THIRD_PARTY_NOTICES.md` (cJSON, ICU,
CUDA, FFmpeg). Model weights are **not** included and are subject to the
MiniMax-H3 / HyperFlow license on Hugging Face.
