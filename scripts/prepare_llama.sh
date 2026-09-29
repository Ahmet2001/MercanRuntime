#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_COMMIT="${MERCAN_LLAMA_COMMIT:-e71b80510c848c00175924ecf3c40333ccae8eb5}"
LLAMA_DIR="${MERCAN_LLAMA_DIR:-$ROOT/build/_deps/llama-src}"

mkdir -p "$(dirname "$LLAMA_DIR")"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
fi

git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_COMMIT"
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_COMMIT"
git -C "$LLAMA_DIR" reset --hard "$LLAMA_COMMIT"
git -C "$LLAMA_DIR" clean -fdx

mkdir -p "$LLAMA_DIR/vendor"
cp -a "$ROOT/runtime/nedo004-ffi" "$LLAMA_DIR/vendor/nedo004-ffi"
cp -a "$ROOT/runtime/NedoTokenizer" "$LLAMA_DIR/vendor/NedoTokenizer"

cargo build --release --manifest-path "$LLAMA_DIR/vendor/nedo004-ffi/Cargo.toml"

git -C "$LLAMA_DIR" apply "$ROOT/runtime/nedolm/nedolm-llama.patch"
cp "$ROOT/runtime/nedolm/nedolm.cpp" "$LLAMA_DIR/src/models/nedolm.cpp"
cp "$ROOT/runtime/libmercan/include/mercan_tensor.h" "$LLAMA_DIR/src/models/mercan_tensor.h"
cp "$ROOT/runtime/libmercan/include/mercan_graph.h" "$LLAMA_DIR/src/models/mercan_graph.h"
cp "$ROOT/runtime/libmercan/include/mercan_kv.h" "$LLAMA_DIR/src/models/mercan_kv.h"
cp "$ROOT/runtime/nedolm/mercan_graph_ggml.hpp" "$LLAMA_DIR/src/models/mercan_graph_ggml.hpp"

echo "Prepared patched llama.cpp at $LLAMA_DIR"
