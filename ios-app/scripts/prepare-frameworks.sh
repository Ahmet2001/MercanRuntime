#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="$ROOT/ios-app"
WORK="$ROOT/build/ios"
CARRIER="$WORK/silo-carrier"
LLAMA="$WORK/llama.cpp"
NEDO_XC="$WORK/nedolm-llama.xcframework"
SILO_COMMIT="1be4fa9aad0ec72dbb6f40ac07d537ea16648944"
LLAMA_COMMIT="e71b80510c848c00175924ecf3c40333ccae8eb5"

for f in \
  runtime/nedolm/nedolm-llama.patch \
  runtime/nedo004-ffi/Cargo.toml \
  runtime/nedo004-ffi/Cargo.lock \
  runtime/nedolm/nedolm.cpp \
  runtime/nedolm/mercan_graph_ggml.hpp \
  runtime/libmercan/include/mercan_graph.h \
  runtime/libmercan/include/mercan_tensor.h \
  runtime/libmercan/include/mercan_kv.h; do
  test -s "$ROOT/$f" || { echo "Missing required file: $f" >&2; exit 2; }
done

mkdir -p "$WORK"
rm -rf "$CARRIER" "$LLAMA" "$NEDO_XC" "$APP_DIR/llama.xcframework" "$APP_DIR/whisper.xcframework"

rustup target add aarch64-apple-ios
cargo build \
  --manifest-path "$ROOT/runtime/nedo004-ffi/Cargo.toml" \
  --locked --release --target aarch64-apple-ios
FFI="$ROOT/runtime/nedo004-ffi/target/aarch64-apple-ios/release/libnedo004_ffi.a"
test -s "$FFI"

git clone --quiet https://github.com/stevederico/silo.git "$CARRIER"
git -C "$CARRIER" checkout --quiet "$SILO_COMMIT"

git clone --quiet https://github.com/ggml-org/llama.cpp.git "$LLAMA"
git -C "$LLAMA" checkout --quiet "$LLAMA_COMMIT"
git -C "$LLAMA" apply --check "$ROOT/runtime/nedolm/nedolm-llama.patch"
git -C "$LLAMA" apply "$ROOT/runtime/nedolm/nedolm-llama.patch"

cp "$ROOT/runtime/nedolm/nedolm.cpp" "$LLAMA/src/models/nedolm.cpp"
cp "$ROOT/runtime/nedolm/mercan_graph_ggml.hpp" "$LLAMA/src/models/mercan_graph_ggml.hpp"
cp "$ROOT"/runtime/libmercan/include/mercan_*.h "$LLAMA/src/models/"

cat > "$LLAMA/src/nedolm-silo-abi.cpp" <<'CPP'
#include "llama.h"

extern "C" struct llama_model * nedolm_silo_model_load_from_file(
        const char * path_model,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    auto params = llama_model_default_params();
    params.progress_callback = progress_callback;
    params.progress_callback_user_data = progress_callback_user_data;
    return llama_model_load_from_file(path_model, params);
}

extern "C" struct llama_context * nedolm_silo_context_init(
        struct llama_model * model,
        uint32_t n_ctx,
        int32_t n_threads,
        int32_t n_threads_batch) {
    auto params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads_batch;
    return llama_init_from_model(model, params);
}

extern "C" struct llama_sampler * nedolm_silo_create_sampler_chain(
        const struct llama_model * model) {
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (chain == nullptr || model == nullptr) return chain;
    if (llama_model_get_vocab(model) == nullptr) {
        llama_sampler_free(chain);
        return nullptr;
    }
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return chain;
}
CPP

mkdir -p "$LLAMA/vendor/nedo004-ffi/target/release"
cp "$FFI" "$LLAMA/vendor/nedo004-ffi/target/release/libnedo004_ffi.a"

python3 - "$LLAMA/src/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = 'target_link_libraries(llama PRIVATE nedo004_ffi dl pthread m)'
new = 'target_link_libraries(llama PRIVATE nedo004_ffi)\ntarget_sources(llama PRIVATE nedolm-silo-abi.cpp)'
if old not in s:
    raise SystemExit('NedoLM FFI CMake linkage anchor not found')
p.write_text(s.replace(old, new, 1))
PY

chmod +x "$ROOT/scripts/build-nedolm-llama-ios.sh"
"$ROOT/scripts/build-nedolm-llama-ios.sh" "$LLAMA" "$FFI" "$NEDO_XC"

cp -R "$CARRIER/llama.xcframework" "$APP_DIR/llama.xcframework"
cp -R "$CARRIER/whisper.xcframework" "$APP_DIR/whisper.xcframework"

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

SRC_FW="$NEDO_XC/ios-arm64/llama.framework"
DST_FW="$APP_DIR/llama.xcframework/ios-arm64/llama.framework"
test -s "$SRC_FW/llama"
test -s "$DST_FW/Headers/llama.h"
cp "$SRC_FW/llama" "$DST_FW/llama"
chmod +x "$DST_FW/llama"

python3 - "$DST_FW/Headers/llama.h" <<'PY'
from pathlib import Path
import sys
header = Path(sys.argv[1])
s = header.read_text()
marker = '#ifdef __cplusplus\n}\n#endif\n\n#endif // LLAMA_H'
if marker not in s:
    raise SystemExit('Could not find llama.h closing marker')
bridge = r'''
    LLAMA_API struct llama_model * nedolm_silo_model_load_from_file(
            const char * path_model,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    LLAMA_API struct llama_context * nedolm_silo_context_init(
            struct llama_model * model,
            uint32_t n_ctx,
            int32_t n_threads,
            int32_t n_threads_batch);

    LLAMA_API struct llama_sampler * nedolm_silo_create_sampler_chain(
            const struct llama_model * model);
'''.strip('\n')
header.write_text(s.replace(marker, bridge + '\n\n' + marker, 1))
PY

grep -q 'nedolm_silo_model_load_from_file' "$DST_FW/Headers/llama.h"
grep -q 'nedolm_silo_context_init' "$DST_FW/Headers/llama.h"
grep -q 'nedolm_silo_create_sampler_chain' "$DST_FW/Headers/llama.h"
file "$DST_FW/llama"
lipo -info "$DST_FW/llama"
