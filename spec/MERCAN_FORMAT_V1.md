# Mercan Model Format v1

Mercan v1 is the portable deployment format used by the Mercan runtime and CLI.

## Container

A `.mercan` file is a single self-contained model artifact. Mercan v1 uses GGUF v3 as its physical tensor container so the runtime can reuse ggml/llama.cpp tensor loading while exposing a Mercan-specific compatibility contract.

The file MUST start with the GGUF magic and MUST use GGUF version 3.

A `.mercan` file MUST NOT contain executable model-bundled code. Runtime behavior is selected by metadata and implemented by the Mercan runtime plus registered Architecture/Tokenizer SDK providers.

## Core Mercan metadata

Every Mercan v1 model MUST provide:

- `mercan.format = "mercan"`
- `mercan.format_version = 1`
- `mercan.runtime_abi = 1`
- `mercan.model_family`
- `general.architecture`

New exporters SHOULD also write `mercan.tokenizer.type` with the registered Mercan tokenizer provider name. For compatibility with early Mercan v1 artifacts, a runtime MAY use the selected architecture provider's default tokenizer when this key is absent.

Architecture-specific metadata MUST use an architecture-owned namespace. For example, NedoLM uses `nedolm.*`, while a third-party `anka` architecture should use `anka.*`.

## Architecture dispatch

`general.architecture` is the model-family dispatch key used by Architecture SDK v1.

Examples:

```text
general.architecture = "nedolm"
general.architecture = "anka"
```

Mercan core resolves this value through the `mercan_architecture_v1` registry before backend model loading. An unsupported architecture MUST be rejected rather than guessed.

The architecture provider owns validation of its architecture-specific metadata. Tensor graph execution remains backend-managed in Architecture SDK v1; see `docs/ARCHITECTURE_SDK.md`.

## Tokenizer metadata

The tokenizer is data and runtime behavior, not executable model-bundled code. `mercan.tokenizer.type` identifies a registered `mercan_tokenizer_v1` provider when present.

Tokenizer-specific assets MUST live under `mercan.tokenizer.*` or an explicitly documented architecture compatibility key. A tokenizer provider is responsible for validating the metadata/assets it requires.

A tokenizer provider can either implement tokenization through the Mercan Tokenizer SDK callbacks or declare itself backend-managed.

## NedoLM v1 profile

NedoLM/Mercan 0.8B models use:

- `general.architecture = "nedolm"`
- Architecture SDK provider: `nedolm`
- Default tokenizer provider: `ndsurf004`
- tokenizer spec: `NDSRF004`

Existing NedoLM v1 artifacts contain:

- `mercan.tokenizer.spec`
- `mercan.tokenizer.surface_vocab_sha256`
- `mercan.tokenizer.surface_vocab` as `ARRAY<UINT8>`
- compatibility key `nedolm.vocab_sha256`

The runtime MUST verify the NDSRF004 vocabulary identity before tokenization.

NDSRF004 special tokens are:

- PAD = 0
- BOS = 1
- EOS = 2

## LFM3 v1 profile

LiquidAI LFM3 models (hybrid gated short-conv + sliding-window/global attention) use:

- `general.architecture = "lfm3"`
- Architecture SDK provider: `lfm3` (backend-managed graph: llama.cpp's own `lfm3` model class)
- Tokenizer provider: `gguf-vocab`

The `gguf-vocab` tokenizer is the standard `tokenizer.ggml.*` vocabulary data in the container, encoded/decoded by the backend's tokenizer. It contains no executable code. Architecture metadata keeps the upstream `lfm3.*` namespace unchanged.

Because the physical container is GGUF v3, an upstream LFM3 GGUF becomes a `.mercan` file by adding the core Mercan keys without touching tensors (`scripts/gguf_to_mercan.py`). The wrapper also records `mercan.source_kind = "gguf"`, `mercan.source_checkpoint` and `mercan.source_sha256`.

Chat format: `<|startoftext|>` BOS, ChatML with English roles (`system` / `user` / `assistant`), generation prompt `<|im_start|>assistant\n<think>`. End of generation is the vocabulary's EOG set (`<|im_end|>`, `<|endoftext|>`); NedoLM's fixed IDs 32000/32001 are ordinary tokens in this vocabulary and must not be used as stops.

## NedoLM architecture metadata

- `nedolm.vocab_size`
- `nedolm.context_length`
- `nedolm.embedding_length`
- `nedolm.block_count`
- `nedolm.feed_forward_length`
- `nedolm.attention.head_count`
- `nedolm.attention.head_count_kv`
- `nedolm.attention.layer_norm_rms_epsilon`
- `nedolm.attention.sliding_window`
- `nedolm.rope.dimension_count`
- `nedolm.rope.freq_base`
- `nedolm.morph.layer_count`
- `nedolm.morph.shared_width`
- `nedolm.morph.root_width`
- `nedolm.morph.suffix_width`
- `nedolm.vocab_sha256`

The MorphFFN widths MUST sum to `nedolm.feed_forward_length` and the MorphFFN layer count MUST NOT exceed the transformer block count.

## Chat template

Mercan 0.8B SFT models use `chatml_tr`:

```text
<|im_start|>{role}\n{content}<|im_end|>\n
```

The training corpus adds EOS once at the end of a conversation. The runtime/CLI may construct prompts from this template but the model file remains the source of tokenizer and architecture truth.

## NedoLM tensor names

Mercan v1 NedoLM tensor names follow the llama.cpp-compatible NedoLM mapping, including:

- `token_embd.weight`
- `blk.N.attn_norm.weight`
- `blk.N.attn_q.weight`
- `blk.N.attn_k.weight`
- `blk.N.attn_v.weight`
- `blk.N.attn_output.weight`
- `blk.N.ffn_norm.weight`
- MorphFFN or standard FFN weights for each block
- `output_norm.weight`
- `blk.0.ffn_gate_tid2eid.weight` for the token-role table

Other architectures define their own tensor namespace/mapping through their architecture profile and backend implementation.

## Compatibility

`mercan.runtime_abi` is the native runtime ABI generation. A runtime MUST reject an unsupported ABI or unsupported format version instead of guessing.

Architecture SDK and Tokenizer SDK each have their own versioned ABI constants. Model format compatibility and plugin ABI compatibility are separate concerns.

GGUF is an implementation detail of Mercan v1. Future `.mercan` versions may change the physical container while preserving the higher-level Mercan runtime contract.

## Distribution

The intended user flow is:

```bash
mercan run model.mercan
mercan run owner/model-repo
```

For Hugging Face repositories, the default artifact name is `model.mercan`. A repository can also be addressed as `owner/repo:filename.mercan`.

Server functionality and a stable external shared-object plugin loader are out of scope for format v1. The portable model contract, `libmercan`, and the in-process Architecture/Tokenizer SDK registries are in scope.
