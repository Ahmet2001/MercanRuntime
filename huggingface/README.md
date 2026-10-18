---
language:
- tr
library_name: mercan
pipeline_tag: text-generation
tags:
- mercan
- nedolm
- gguf
- turkish
- text-generation
license: apache-2.0
---

# Mercan 0.8B SFT

Mercan 0.8B SFT is a Turkish assistant model distributed in the self-contained Mercan v1 deployment format.

## Run

Install the Mercan CLI, then run this repository directly:

```bash
mercan run MercanAI/Mercan-0.8B-SFT
```

The CLI downloads `model.mercan` once into the local Mercan cache and then runs it locally.

You can also download the artifact yourself:

```bash
mercan run model.mercan
```

## Artifact

- `model.mercan`: Q4_K_M deployment model
- Mercan format v1
- physical tensor container: GGUF v3
- exact NDSRF004 tokenizer asset embedded in the model
- context length: 4096
- 24 transformer blocks
- hidden size: 1536
- 12 attention heads / 4 KV heads
- MorphFFN in the first 18 blocks

The `.mercan` file contains model weights, architecture metadata, tokenizer metadata/data and the token-role routing table. It does not contain executable model-bundled code.

## Runtime

The reference runtime and CLI are published in `Ahmet2001/MercanRuntime`. `libmercan` uses a stable C ABI over the native inference backend.

## Chat template

```text
<|im_start|>{role}\n{content}<|im_end|>\n
```

EOS is appended once at the end of a training conversation.
