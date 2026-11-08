#!/usr/bin/env bash
# Print the path of a Mercan CLI built from this checkout, building it first if needed.
#
#   BIN="$(scripts/mercan_cli.sh)"
#
# Environment:
#   MERCAN_BIN    use this CLI instead of building one
#   MERCAN_JOBS   parallel build jobs (default: nproc)
#
# The build goes to build/cli/ with its own llama.cpp checkout in
# build/cli-llama-src/, so other build directories are left untouched.
# Build logs go to stderr; only the binary path is printed on stdout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build/cli"
LLAMA_DIR="$ROOT/build/cli-llama-src"
BIN="${MERCAN_BIN:-$BUILD/mercan}"
JOBS="${MERCAN_JOBS:-$(nproc)}"

if [[ ! -x "$BIN" ]]; then
  {
    echo "mercan CLI not found at $BIN; building it (one-time, a few minutes)..."
    for tool in git cmake cargo c++; do
      command -v "$tool" >/dev/null || { echo "missing build tool: $tool"; exit 1; }
    done
    # Shallow-fetch only the pinned llama.cpp commit instead of cloning its history.
    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
      mkdir -p "$LLAMA_DIR"
      git -C "$LLAMA_DIR" init -q
      git -C "$LLAMA_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git
    fi
    MERCAN_LLAMA_DIR="$LLAMA_DIR" "$ROOT/scripts/prepare_llama.sh"
    GENERATOR=()
    command -v ninja >/dev/null && GENERATOR=(-G Ninja)
    cmake -S "$ROOT" -B "$BUILD" "${GENERATOR[@]}" -DCMAKE_BUILD_TYPE=Release \
      -DMERCAN_LLAMA_DIR="$LLAMA_DIR" -DMERCAN_BUILD_SHARED=OFF
    cmake --build "$BUILD" --target mercan -j "$JOBS"
  } >&2
fi

echo "$BIN"
