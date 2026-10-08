# Mercan

Mercan is a native local inference stack for Mercan/NedoLM models. The user-facing workflow is intentionally simple:

```bash
mercan run model.mercan
```

or directly from Hugging Face:

```bash
mercan run MercanAI/Mercan-0.8B-SFT
```

The Hugging Face form downloads `model.mercan` once into `~/.cache/mercan` and reuses the cached copy on later launches. On Linux, the installer automatically selects the CUDA build when a working NVIDIA GPU/driver is detected with `nvidia-smi`; otherwise it installs the CPU build.

## Install on Linux x86_64

Once the GitHub release is available:

```bash
curl -fsSL https://raw.githubusercontent.com/Ahmet2001/mercanApp-test1/main/install.sh | sh
```

The installer automatically chooses CUDA on NVIDIA systems and CPU elsewhere. Force CPU with `MERCAN_FORCE_CPU=1`.

Then:

```bash
mercan --version
mercan run MercanAI/Mercan-0.8B-SFT
```

Without root access:

```bash
MERCAN_PREFIX="$HOME/.local" ./install.sh
export PATH="$HOME/.local/bin:$PATH"
```

## CLI

```text
mercan run <model.mercan|owner/repo[:file.mercan]>
mercan pull <owner/repo[:file.mercan]>
mercan arch list
mercan tokenizer list
mercan graph abi
mercan --version
```

Examples:

```bash
mercan run ./model.mercan
mercan run MercanAI/Mercan-0.8B-SFT
mercan run MercanAI/Mercan-0.8B-SFT:model.mercan -p "Merhaba"
```

Useful run options:

```text
-p, --prompt TEXT
-n, --max-tokens N
--temperature F
--top-k N
--top-p F
-t, --threads N
--gpu-layers N   # -1 = all; CUDA package defaults to all layers
```

## Architecture

```text
PyTorch / Hugging Face checkpoint
              |
              v
       mercan_convert.py
              |
              v
         model.mercan
              |
              v
           mercan CLI
              |
              v
          libmercan C ABI
              |
              v
 patched llama.cpp / ggml
              |
              v
 NedoLM + exact NDSRF004 tokenizer
```

`.mercan` v1 is a self-contained deployment artifact. It uses GGUF v3 as its physical tensor container but adds a Mercan compatibility contract and embeds tokenizer assets/identity metadata. No executable model-bundled code is stored in the model file.

Mercan also ships Architecture SDK v1. `libmercan` reads `general.architecture` before backend loading, resolves a registered `mercan_architecture_v1`, validates the model, then resolves a `mercan_tokenizer_v1`. NedoLM/NDSRF004 are the first built-in providers rather than special cases in Mercan model dispatch.

Mercan Graph ABI v1 adds a backend-independent tensor-operation boundary based on opaque `mercan_tensor_handle_v1` values. NedoLM exercises this ABI in real inference for MorphFFN-specific primitives, RMSNorm, Q/K RoPE, row selection and residual adds. Attention/KV-cache, LoRA-aware linear helpers and model/tensor ownership remain backend-managed during the staged migration.

See `spec/MERCAN_FORMAT_V1.md`, `docs/ARCHITECTURE_SDK.md` and `docs/GRAPH_ABI_V1.md`.

## Build from source

Required tools:

- CMake >= 3.16
- C/C++ compiler with C++17 support
- Rust/Cargo
- Git
- curl

Build:

```bash
git clone https://github.com/Ahmet2001/mercanApp-test1.git
cd mercanApp-test1
chmod +x scripts/*.sh install.sh
./scripts/build_linux.sh
./dist/mercan-linux-x86_64/bin/mercan --version

# CUDA build (requires CUDA toolkit)
MERCAN_CUDA=1 MERCAN_DIST_NAME=mercan-linux-x86_64-cuda ./scripts/build_linux.sh
```

The build script checks out the pinned llama.cpp commit, builds the NDSRF004 Rust static bridge, applies `runtime/nedolm/nedolm-llama.patch`, copies the NedoLM model implementation and builds `libmercan` plus the `mercan` CLI. The release CLI links the llama/ggml backend statically so users do not need separate backend shared libraries.

## Repository layout

```text
cli/main.cpp                         user-facing CLI
runtime/libmercan/                   stable native C ABI + SDK headers
core/                                architecture/tokenizer registry + metadata dispatch
architectures/nedolm/                built-in NedoLM/NDSRF004 SDK providers
examples/custom_arch/                third-party architecture provider template
runtime/nedolm/nedolm.cpp            NedoLM graph implementation
runtime/nedolm/nedolm-llama.patch    llama.cpp integration
runtime/nedo004-ffi/                 exact tokenizer bridge
runtime/NedoTokenizer/               tokenizer implementation/assets
scripts/mercan_convert.py            PyTorch/HF -> .mercan converter
scripts/prepare_llama.sh             pinned backend preparation
scripts/build_linux.sh               Linux build/package
scripts/export_model.sh              F16 export + Q4_K_M deployment artifact
scripts/upload_hf.py                 Hugging Face publisher
spec/MERCAN_FORMAT_V1.md             model format contract
```

## Export the final Mercan model

On a machine with the final checkpoint and PyTorch:

```bash
./scripts/export_model.sh /path/to/step_00023334.pt /path/to/output
```

This produces an F16 `.mercan` intermediate and a Q4_K_M deployment artifact named `model.mercan`.

## Publish to Hugging Face

With a valid Hugging Face login/token available to `huggingface_hub`:

```bash
python -m pip install -U huggingface_hub
python scripts/upload_hf.py /path/to/model.mercan --repo MercanAI/Mercan-0.8B-SFT
```

The CLI expects `model.mercan` as the default artifact when a Hugging Face repository name is passed to `mercan run`.

## Architecture SDK v1

Custom families register through the versioned public descriptors in `mercan_arch.h` and `mercan_tokenizer.h`. Model dispatch no longer needs per-family branches inside `mercan.cpp`.

Architecture/tokenizer descriptors are append-only within ABI v1: Mercan accepts a stable minimum prefix, copies only `struct_size` bytes and zero-fills newer tail fields. Optional regression coverage is available with `-DMERCAN_BUILD_SDK_TESTS=ON`.

Mercan Tensor ABI v1 adds stable named weight declarations and opaque `tensor_by_name` / `require_tensor` lookup. The built-in NedoLM graph uses the resolver for its `.mercan` weight names; third-party architecture code does not need `ggml_tensor*` for these lookups.

Architecture SDK v1 is paired with the experimental Mercan Graph ABI v1. The primitive table now covers tensor shape/stride inspection, row gathering, F32 cast, SwiGLU split, 2D views, multiplication, addition, concatenation, plain matrix multiplication, RMSNorm and NORMAL/NEOX RoPE. NedoLM uses these primitives in its real inference graph.

A completely new architecture can use these backend-independent primitives today, but operations not yet represented by Graph ABI v1 still require compiled backend support. External `.so`/`.dylib` graph plugins are therefore not declared stable yet.

See `docs/ARCHITECTURE_SDK.md`, `docs/GRAPH_ABI_V1.md` and `examples/custom_arch/`.

## Current v1 scope

Mercan v1 includes the portable model format, converter, exact tokenizer bridge, native runtime ABI, Architecture/Tokenizer SDK registries, experimental Graph ABI v1, local/Hugging Face CLI workflow and Linux packaging. An HTTP server/API layer and stable external shared-object plugin loader are intentionally left for later stages.

## Standalone NedoTokenizer SDK

The original NDSRF004 Rust tokenizer now exports a model-free, copy-based C
API in `runtime/nedo004-ffi/include/nedo004.h`.
Build a standalone shared library with
`cargo build --release --manifest-path runtime/nedo004-ffi/Cargo.toml`.
It supports encode, byte-exact decode, vocabulary size, and vocabulary SHA-256,
without loading Mercan or an LLM. The tokenizer remains the exact Rust
NedoTokenizer: no alternate segmentation algorithm has been introduced.
