#!/usr/bin/env bash
# Linux / WSL build for h3rt. Requires gcc, nvcc (CUDA 12/13), ICU, OpenMP,
# FFmpeg on PATH and a .NET 8 SDK. No Python and no ComfyUI are needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${1:-89}"

echo "=== [1/4] Building native h3rt (sm_${ARCH}) ==="
cmake -S "$ROOT" -B "$ROOT/build-linux" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DH3_CUDA_ARCH="$ARCH" \
    -DH3_ENGINE_DIR="$ROOT/native/engine"
cmake --build "$ROOT/build-linux" --target h3rt h3rt_smoke --parallel "$(nproc)"

echo "=== [2/4] Staging native binaries into dist/linux-x64 ==="
DIST="$ROOT/dist/linux-x64"
mkdir -p "$DIST"
cp "$ROOT/build-linux/bin/libh3rt.so" "$DIST/"
cp "$ROOT/build-linux/bin/h3rt_smoke" "$DIST/"

echo "=== [3/4] Building .NET solution ==="
export H3RT_NATIVE_DIR="$DIST"
dotnet build "$ROOT/dotnet/H3NetRuntime.sln" -c Release --nologo
cp "$DIST/libh3rt.so" "$ROOT/dotnet/samples/H3NetRuntime.Sample/bin/Release/net8.0/" 2>/dev/null || true
cp "$DIST/libh3rt.so" "$ROOT/dotnet/tests/H3NetRuntime.Smoke/bin/Release/net8.0/" 2>/dev/null || true

echo "=== [4/4] C ABI smoke test ==="
LD_LIBRARY_PATH="$DIST:${LD_LIBRARY_PATH:-}" "$DIST/h3rt_smoke"

echo "OK. Native library: $DIST/libh3rt.so"
