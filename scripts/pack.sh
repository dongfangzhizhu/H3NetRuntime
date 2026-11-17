#!/usr/bin/env bash
# Builds the H3NetRuntime NuGet package with the native runtime inside.
# Usage: bash scripts/pack.sh [CUDA_ARCH] [--skip-native]
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ARCH="${1:-89}"
SKIP_NATIVE=""
for arg in "$@"; do
    case "$arg" in
        --skip-native) SKIP_NATIVE=1 ;;
        [0-9]*) ARCH="$arg" ;;
    esac
done

WIN_DLL="$ROOT/dist/win-x64/h3rt.dll"
LINUX_SO="$ROOT/dist/linux-x64/libh3rt.so"

echo "=== [1/3] Native runtime ==="
if [ -n "$SKIP_NATIVE" ]; then
    echo "    skipped (--skip-native)"
elif [ -f "$LINUX_SO" ]; then
    echo "    using existing $LINUX_SO"
else
    bash "$ROOT/scripts/build.sh" "$ARCH"
fi

if [ ! -f "$WIN_DLL" ] && [ ! -f "$LINUX_SO" ]; then
    echo "ERROR: no native runtime under dist/win-x64 or dist/linux-x64." >&2
    echo "       Build one first: scripts/build.sh $ARCH (Linux) or scripts/build-native.cmd $ARCH (Windows)." >&2
    exit 1
fi
if [ ! -f "$WIN_DLL" ]; then
    echo "    WARNING: dist/win-x64/h3rt.dll is missing; the package will only"
    echo "             contain the Linux runtime."
fi

echo "=== [2/3] dotnet pack ==="
rm -rf "$ROOT/dist/nuget"
dotnet pack "$ROOT/dotnet/src/H3NetRuntime/H3NetRuntime.csproj" -c Release --nologo \
    -o "$ROOT/dist/nuget" ${H3RT_PACKAGE_VERSION:+-p:Version=$H3RT_PACKAGE_VERSION}

echo "=== [3/3] Contents ==="
PKG=$(ls -1 "$ROOT"/dist/nuget/*.nupkg | head -1)
echo "    $PKG"
unzip -l "$PKG" | grep -E "native/|PACKAGE.md|\.nuspec" || true

echo
echo "OK. Publish with:"
echo "    dotnet nuget push \"$PKG\" --source <your-feed> --api-key <key>"
