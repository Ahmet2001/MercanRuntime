# Third-party notices

The `mercan` binary and `libmercan` statically include the components below.
Full license texts are installed next to this file in `share/licenses/mercan/`.

## llama.cpp / ggml

- Upstream: https://github.com/ggml-org/llama.cpp (pinned commit in `scripts/prepare_llama.sh`)
- License: MIT, see `llama.cpp-LICENSE` (copied from the pinned source tree at build time)
- Copyright (c) 2023-2026 The ggml authors

ggml includes code under its own MIT notices:

- `ggml/src/ggml-cpu/llamafile/sgemm.cpp`: Copyright 2024 Mozilla Foundation
- YaRN RoPE (`ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/rope.cu`):
  Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng

## NedoTokenizer (NDSRF004)

- Source: `runtime/NedoTokenizer`, `runtime/nedo004-ffi`
- License: Apache-2.0, see `APACHE-2.0.txt` and `NedoTokenizer-LICENSE`
- Copyright 2026 NedoTokenizer contributors
- Morphology data is compiled from Zemberek NLP (Apache-2.0), see `ZEMBEREK_NOTICE.md`

## Rust crates linked into the NedoTokenizer bridge

Where a crate offers a choice of licenses, Mercan uses it under Apache-2.0
(`APACHE-2.0.txt`) unless only MIT is listed below.

| Crate | Version | License used |
|---|---|---|
| block-buffer | 0.10.4 | Apache-2.0 |
| cfg-if | 1.0.4 | Apache-2.0 |
| cpufeatures | 0.2.17 | Apache-2.0 |
| crypto-common | 0.1.7 | Apache-2.0 |
| digest | 0.10.7 | Apache-2.0 |
| generic-array | 0.14.7 | MIT, Copyright (c) 2015 Bartłomiej Kamiński |
| itoa | 1.0.18 | Apache-2.0 |
| memchr | 2.8.3 | MIT, Copyright (c) 2015 Andrew Gallant |
| ryu | 1.0.23 | Apache-2.0 |
| serde, serde_core | 1.0.228 | Apache-2.0 |
| serde_json | 1.0.145 | Apache-2.0 |
| sha2 | 0.10.9 | Apache-2.0 |
| typenum | 1.20.1 | Apache-2.0 |

Versions follow `runtime/nedo004-ffi/Cargo.lock`.

## NVIDIA CUDA (CUDA packages only)

The `-cuda` packages statically link NVIDIA CUDA runtime and cuBLAS libraries,
redistributed under the NVIDIA CUDA Toolkit End User License Agreement:
https://docs.nvidia.com/cuda/eula/

## MIT License

The MIT-licensed components above are provided under the following terms,
together with their copyright notices listed above:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
