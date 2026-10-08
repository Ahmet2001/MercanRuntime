# Mercan Architecture SDK v1

Mercan Architecture SDK separates model-family discovery and validation from the Mercan core runtime. A `.mercan` model declares its family with `general.architecture`. Mercan reads metadata first, resolves a registered `mercan_architecture_v1`, validates the model, resolves a tokenizer provider, and only then hands tensor execution to the inference backend.

## Stable ABI surfaces

```c
#define MERCAN_ARCH_ABI_VERSION 1
#define MERCAN_TOKENIZER_ABI_VERSION 1
#define MERCAN_GRAPH_ABI_VERSION 1
```

Public headers installed by Mercan:

- `mercan.h`
- `mercan_arch.h`
- `mercan_tokenizer.h`
- `mercan_graph.h`

The public ABI intentionally exposes no `llama_model`, `llama_context`, `ggml_tensor *`, or other llama.cpp/ggml internal C++ types.

## Descriptor compatibility

Architecture and tokenizer descriptors use an append-only v1 layout. The runtime accepts the stable minimum prefixes `MERCAN_ARCHITECTURE_V1_BASE_SIZE` and `MERCAN_TOKENIZER_V1_BASE_SIZE`, copies only the bytes advertised by `struct_size`, and zero-fills the remaining tail. New callbacks may therefore be appended without making an older v1 provider invalid.

Use `MERCAN_ARCH_HAS_V1(desc, member)` and `MERCAN_TOKENIZER_HAS_V1(desc, member)` when code needs to distinguish whether a tail field was actually supplied by the provider. Callback fields are optional unless a specific capability requires them.

The regression target can be built with:

```bash
cmake -S . -B build/sdk-test -DMERCAN_BUILD_SDK_TESTS=ON -DMERCAN_LLAMA_DIR=...
cmake --build build/sdk-test --target mercan-sdk-prefix-test
ctest --test-dir build/sdk-test -R mercan-sdk-prefix-compat --output-on-failure
```

## Stable tensor resolver

Architecture code that needs model weights should use Mercan Tensor ABI v1 rather than backend structs. The runtime exposes a `mercan_tensor_resolver_v1` from the graph builder when available. Providers declare names and resolve opaque handles with `tensor_by_name` / `require_tensor`. See [`TENSOR_ABI_V1.md`](TENSOR_ABI_V1.md).

The built-in NedoLM provider now declares the canonical `.mercan` weight names and resolves its embedding, norm, attention-output, FFN/MorphFFN and output weights through this resolver. Q/K/V are resolved and validated by name while the projection operation remains in the LoRA-aware backend helper.

## Architecture registry

An architecture provider exports a `mercan_architecture_v1` descriptor and registers it with `mercan_arch_register_v1()`.

```cpp
static const mercan_architecture_v1 anka_arch = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "anka",
    "AnkaLM",
    "ankatok",
    MERCAN_ARCH_BACKEND_MANAGED_GRAPH,
    anka_probe,
    anka_validate,
};

mercan_arch_register_v1(&anka_arch);
```

A `.mercan` file can then declare:

```text
general.architecture = anka
mercan.tokenizer.type = ankatok
```

Mercan core does not need an `if (arch == "anka")` branch.

## Metadata view

Providers receive `mercan_metadata_v1`, a stable read-only view with `has_key()`, `get_string()`, `get_i64()`, and `get_f64()`. Architecture-specific metadata should use an architecture-owned namespace such as:

```text
anka.block_count
anka.attention_type
anka.expert_count
```

Core Mercan metadata stays under `mercan.*`.

## Tokenizer registry

A tokenizer provider uses `mercan_tokenizer_v1`. The descriptor may either set `MERCAN_TOKENIZER_BACKEND_MANAGED` and let the compiled backend tokenize, or provide `create`, `destroy`, `encode`, and `decode_piece` callbacks.

NedoLM currently registers `ndsurf004` as a built-in backend-managed tokenizer because the exact NDSRF004 bridge is compiled into the current backend.

## Mercan Graph ABI v1

Graph ABI v1 is the first backend-independent tensor-operation boundary. Architecture code receives/uses opaque `mercan_tensor_handle_v1` values rather than raw backend tensor pointers.

The first primitive set contains:

- tensor dimension and byte-stride inspection
- `get_rows`
- `cast_f32`
- `swiglu_split`
- `view_2d`
- `mul`
- `add`
- `concat`
- plain `matmul`
- `rms_norm`
- `rope_ext` for NORMAL/NEOX RoPE

The operation table is carried by `mercan_graph_builder_v1`:

```c
const mercan_graph_api_v1 * api = builder->api;
mercan_tensor_handle_v1 z = api->swiglu_split(builder, gate, up);
```

A provider never receives a `ggml_tensor *`. The current ggml adapter converts opaque handles to backend tensors internally. If llama.cpp/ggml changes, the adapter can change without changing the public Graph ABI.

### Current experimental scope

Graph ABI v1 is intentionally incremental. NedoLM's MorphFFN-specific primitive sequence runs through this ABI, and the real NedoLM path now also uses Graph ABI for RMSNorm+weight application, Q/K RoPE, final-token row selection and residual adds. Cached self-attention now runs through Graph ABI v1, and KV cache batch views are available through KV Cache ABI v1. Cache mutation/lifetime, LoRA-aware matrix multiplication, and backend model ownership remain runtime-managed.

The same primitives are now also used by a fully external transformer plugin through the generic backend adapter. Runtime-owned attention and KV state therefore work on both the built-in NedoLM path and an architecture that has no compiled llama.cpp model class.

## NedoLM today

```text
general.architecture = nedolm
                 ↓
Mercan architecture registry
                 ↓
nedolm provider
                 ↓
NDSRF004 tokenizer provider
                 ↓
backend graph
        ├── attention / KV cache: backend-managed
        └── MorphFFN + RMSNorm + RoPE + residual primitives: Mercan Graph ABI v1
                 ↓
current ggml adapter
                 ↓
CPU / CUDA / Metal-capable backend
```

NedoLM advertises both `MERCAN_ARCH_BACKEND_MANAGED_GRAPH` and `MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES` while this migration is partial.

## Why opaque handles

Third-party architecture code must not depend on the representation behind a tensor. A `mercan_tensor_handle_v1` may currently map to a ggml tensor, but the ABI does not promise that representation. This leaves room for future CUDA-native, Metal-native, or other backend adapters without changing architecture source code.

## Developer flow today

1. Pick a unique architecture name such as `anka`.
2. Implement `mercan_architecture_v1`.
3. Implement `mercan_tokenizer_v1` if needed.
4. Use Mercan Graph ABI primitives where the current primitive set is sufficient.
5. Use runtime-owned `self_attention` + `mercan_kv_resolver_v1` for causal transformer cache state.
6. Write a converter that emits `.mercan` metadata and tensors.
7. Verify registration with `mercan arch list` and `mercan tokenizer list`.
8. Run the model with `mercan run ./anka-1b.mercan`; architectures using the external graph callback are dispatched through the generic backend without a llama.cpp model factory entry.

See `examples/custom_arch/` for a minimal provider template.

## Next compatibility milestone

External plugins can now load tensors, tokenize, build a causal transformer graph, use runtime-owned attention/KV state, and execute prefill plus cached decode without a llama.cpp model-factory entry. The next compatibility work is breadth rather than the basic independence boundary: expose reshape/permute/contiguous helpers needed for arbitrary multi-head layouts, optimize cache mutation to avoid the current retained-K/V copy step, and add production-scale external-model regressions.


## Runtime-owned cached self-attention

Graph ABI v1 exposes `self_attention` as an append-only capability. Architecture plugins pass opaque Q/K/V tensor handles plus the output projection weight; the runtime owns attention masks, sliding-window selection, and the concrete KV-cache implementation. This keeps llama.cpp cache/input classes out of the public SDK.

Plugins must probe the capability with `MERCAN_GRAPH_API_HAS_V1(api, self_attention)` before use. The generic external backend implements this callback with causal masking and persistent per-layer K/V state. Q/K/V may use a single-head 2D layout or a 3D head layout; the runtime keeps concrete cache storage private.


## KV Cache ABI v1

`mercan_kv.h` exposes opaque cache handles for the runtime-owned base and sliding-window cache views. Plugins can query the current batch's K/V placement indices and attention mask as opaque Tensor ABI handles without importing llama.cpp memory/cache classes. A window size of `0` means the runtime/full-context policy; the sliding-window handle reports its concrete window length.

Cache allocation, lifetime and mutation are deliberately runtime-owned in v1. Architectures that need custom write semantics can be supported by append-only KV ABI extensions without changing existing plugins.


## Graph output and finalization

Graph ABI v1 now provides append-only `set_output` and `finalize` capabilities. Architecture code publishes embedding/logit/hidden outputs through stable output kinds and finalizes the graph root without touching llama.cpp `llm_graph_result` or `ggml_cgraph` internals. Plugins must capability-probe both callbacks before use.


## Dynamic plugin loading

Mercan Plugin ABI v1 (`mercan_plugin.h`) loads external native libraries with `dlopen`/`dlsym` on Unix-like systems and `LoadLibrary`/`GetProcAddress` on Windows. A plugin exports `mercan_plugin_entry_v1()` and receives host callbacks for architecture/tokenizer registration; it does not need to link against Mercan internals. Loaded libraries remain resident for process lifetime so descriptor strings and callbacks stay valid.

CLI usage:

```bash
mercan plugin load ./libmercan_arch_anka.so
MERCAN_PLUGINS=./libmercan_arch_anka.so mercan arch list
mercan run model.mercan --plugin ./libmercan_arch_anka.so
```

`examples/plugins/anka` is a real separately-built `.so` transformer plugin. It proves discovery, ABI-safe registration, external tokenization, graph construction, runtime-owned attention/KV reuse, and CPU/CUDA execution without an Anka class inside llama.cpp.

### External graph callback ABI v1

An architecture descriptor may append `build_graph` and advertise
`MERCAN_ARCH_GRAPH_CALLBACK_V1`. The runtime/backend adapter supplies a
`mercan_arch_graph_invocation_v1` containing opaque token/position/output handles and a
`mercan_graph_builder_v1`. Plugins resolve weights through `builder->tensors` and build
ops through `builder->api`; no `ggml_tensor *`, llama model class, or cache implementation
is exposed. `mercan_arch_build_graph_v1()` validates the ABI boundary before dispatch.

The external Anka example now builds a complete tiny causal transformer block and is exercised by the generic-backend regression. The plugin resolves every weight through Tensor ABI v1, obtains causal placement/mask views through KV Cache ABI v1, and delegates cached attention to `self_attention`. No Anka-specific model class exists in llama.cpp.


## Generic external architecture backend

Architectures advertising `MERCAN_ARCH_GRAPH_CALLBACK_V1` without `MERCAN_ARCH_BACKEND_MANAGED_GRAPH`
are loaded by Mercan's generic GGUF/ggml backend adapter. The adapter loads named tensors directly from the
`.mercan` container, creates opaque Tensor/Graph ABI handles, invokes the plugin's `build_graph` callback, and
executes the resulting graph without requiring a compiled llama.cpp model class for that architecture.

The `anka` example also registers an external `anka-byte` tokenizer. The SDK regression builds a tiny Anka `.mercan` transformer and verifies `plugin load -> model load -> tokenize -> prefill -> cached decode -> logits`. The test compares a cached context against a fresh context decoding the same token, so a passing result proves that retained K/V history changes the logits. CPU is the baseline; when a GPU device is available and `n_gpu_layers != 0`, the same generic adapter runs the graph and persistent cache path on that ggml GPU backend.

## Standalone NedoTokenizer (NDSRF004) SDK

The Nedo tokenizer can now run **without loading a model** through the
original Rust FFI library. The Mercan C SDK installs `nedo004.h`; the Rust
`nedo004-ffi` Cargo crate produces a static library **and** a shared library
(`libnedo004_ffi.so` / `libnedo004_ffi.dylib` / `nedo004_ffi.dll`).

```bash
cargo build --release --manifest-path runtime/nedo004-ffi/Cargo.toml
cargo test --release --manifest-path runtime/nedo004-ffi/Cargo.toml
```

Functions: `nedo004_encode_copy`, `nedo004_decode_copy`,
`nedo004_vocab_size` and `nedo004_vocab_sha256`. The buffer-size query
uses `-4` for insufficient capacity; `out_len` reports required length.
Tokenization uses the exact built-in 32,000-entry NDSRF004 surface vocabulary
and the original morphological tokenizer implementation. BOS/EOS are excluded
from encoded user text, as in the existing native bridge. Decoding reconstructs
UTF-8 bytes, including Turkish text and byte fallbacks. This is a standalone
API next to `mercan_tokenizer_v1`, which remains unchanged for plugin ABI v1.

The shared Rust tokenizer is independent of libmercan and llama.cpp. It can
be consumed by Python (EthosoftLib's optional `ethosoftlib.nedo` adapter)
or other SDK clients. Existing model inference/tokenizer behavior is unchanged.
