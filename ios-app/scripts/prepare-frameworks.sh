#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="$ROOT/ios-app"
WORK="$ROOT/build/ios"
CARRIER="$WORK/silo-carrier"
LLAMA="$WORK/llama.cpp"
MERCAN_XC="$APP_DIR/MercanRuntime.xcframework"
SILO_COMMIT="1be4fa9aad0ec72dbb6f40ac07d537ea16648944"
LLAMA_COMMIT="19cce6919992fdffd92e0ac152caaf1e2b8f254e"

for f in \
  runtime/nedolm/nedolm-llama.patch \
  runtime/nedo004-ffi/Cargo.toml \
  runtime/nedo004-ffi/Cargo.lock \
  runtime/nedolm/nedolm.cpp \
  runtime/nedolm/mercan_graph_ggml.hpp \
  runtime/libmercan/include/mercan.h \
  runtime/libmercan/include/mercan_arch.h \
  runtime/libmercan/include/mercan_tokenizer.h \
  runtime/libmercan/include/mercan_graph.h \
  runtime/libmercan/include/mercan_tensor.h \
  runtime/libmercan/include/mercan_kv.h \
  core/model_loader.cpp \
  core/builtin_plugins.cpp \
  architectures/nedolm/architecture.cpp \
  architectures/nedolm/tokenizer.cpp \
  architectures/lfm3/architecture.cpp \
  architectures/gguf_vocab/tokenizer.cpp; do
  test -s "$ROOT/$f" || { echo "Missing required file: $f" >&2; exit 2; }
done

mkdir -p "$WORK"
rm -rf "$CARRIER" "$LLAMA" "$MERCAN_XC" "$APP_DIR/llama.xcframework" "$APP_DIR/whisper.xcframework"

rustup target add aarch64-apple-ios
cargo build \
  --manifest-path "$ROOT/runtime/nedo004-ffi/Cargo.toml" \
  --locked --release --target aarch64-apple-ios
FFI="$ROOT/runtime/nedo004-ffi/target/aarch64-apple-ios/release/libnedo004_ffi.a"
test -s "$FFI"

# Silo is now used only as the pinned carrier for whisper.xcframework.
git clone --quiet https://github.com/stevederico/silo.git "$CARRIER"
git -C "$CARRIER" checkout --quiet "$SILO_COMMIT"

# libmercan currently uses llama.cpp as its compiled backend for built-in NedoLM.
git clone --quiet https://github.com/ggml-org/llama.cpp.git "$LLAMA"
git -C "$LLAMA" checkout --quiet "$LLAMA_COMMIT"
git -C "$LLAMA" apply --check "$ROOT/runtime/nedolm/nedolm-llama.patch"
git -C "$LLAMA" apply "$ROOT/runtime/nedolm/nedolm-llama.patch"

cp "$ROOT/runtime/nedolm/nedolm.cpp" "$LLAMA/src/models/nedolm.cpp"
cp "$ROOT/runtime/nedolm/mercan_graph_ggml.hpp" "$LLAMA/src/models/mercan_graph_ggml.hpp"
cp "$ROOT"/runtime/libmercan/include/mercan_*.h "$LLAMA/src/models/"

mkdir -p "$LLAMA/vendor/nedo004-ffi/target/release"
cp "$FFI" "$LLAMA/vendor/nedo004-ffi/target/release/libnedo004_ffi.a"

# Linux-only system libraries from the generic patch are not valid on iOS.
python3 - "$LLAMA/src/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = 'target_link_libraries(llama PRIVATE nedo004_ffi dl pthread m)'
new = 'target_link_libraries(llama PRIVATE nedo004_ffi)'
if old not in s:
    raise SystemExit('NedoLM FFI CMake linkage anchor not found')
p.write_text(s.replace(old, new, 1))
PY

chmod +x "$ROOT/scripts/build-libmercan-ios.sh"
"$ROOT/scripts/build-libmercan-ios.sh" "$LLAMA" "$FFI" "$MERCAN_XC"

cp -R "$CARRIER/whisper.xcframework" "$APP_DIR/whisper.xcframework"

# Some pinned whisper XCFramework metadata references dSYM directories that are
# absent from the Git checkout. Xcode validates the paths even for unsigned builds.
python3 - "$APP_DIR/whisper.xcframework" <<'PY'
import plistlib
from pathlib import Path
import sys
xc = Path(sys.argv[1])
info_path = xc / 'Info.plist'
info = plistlib.loads(info_path.read_bytes())
for lib in info.get('AvailableLibraries', []):
    debug_path = lib.get('DebugSymbolsPath')
    identifier = lib.get('LibraryIdentifier')
    if debug_path and identifier:
        (xc / identifier / debug_path).mkdir(parents=True, exist_ok=True)
PY

test -s "$MERCAN_XC/ios-arm64/MercanRuntime.framework/MercanRuntime"
file "$MERCAN_XC/ios-arm64/MercanRuntime.framework/MercanRuntime"
lipo -info "$MERCAN_XC/ios-arm64/MercanRuntime.framework/MercanRuntime"
