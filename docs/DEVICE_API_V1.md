# Device API v1 (MercanRuntime 0.3.0)

Initialize `mercan_backend_init()` before querying devices or loading models.
The base C ABI remains version 1, with unchanged model/context parameter layouts.
`mercan_device_api_version()` independently returns 1.

`mercan_runtime_devices_json_v1()` returns a thread-local UTF-8 JSON snapshot:
`api_version`, `cuda_compiled`, `cuda_available`, `backends`, and `devices`.
Each device includes its name, description, backend, type, and free/total
memory in bytes. Build support and usable devices are separate facts.
The snapshot pointer is borrowed until the next snapshot query on that thread.
`mercan_cuda_available_v1()` is a convenient usable-device predicate.
Discovery does not download drivers, compile code, or guarantee model allocation.

`mercan_model_load_device_v1(path, params, device)` accepts:

| device | behavior |
| --- | --- |
| `cpu` | Requires `n_gpu_layers=0`, restricts loading to CPU. |
| `cuda` | Requires nonzero GPU layers, CUDA build and usable CUDA devices. |
| `auto` | With zero layers, chooses all layers on usable CUDA or CPU otherwise. Nonzero layers require usable CUDA. |

Use resolved `cpu`/`cuda` with corresponding layer counts if your binding
distinguishes an explicit zero override from an automatic default (as the
Python SDK does). Values below -1 and conflicting settings fail.
CUDA loading uses an explicit CUDA-only device list. Allocation/load errors
propagate; a failed GPU request never retries CPU. Legacy `mercan_model_load`
still supports other registered GPU backends with nonzero layers, but now
rejects unavailable GPU offload, including generic plugin architectures.
Generic graph plugins currently offload their entire model for any nonzero
layer request; per-layer placement is supported by the llama-managed backend.

`mercan_model_device_v1(model)` returns `cpu`, `cuda`, or `gpu` for another GPU
backend. `mercan_model_device_json_v1(model)` returns actual weight placement:
`device`, `placement` (`cpu`, `gpu`, `mixed`), `gpu_layers_requested`, and
actual device names/backends. These strings belong to the model until free.
The implementation inspects tensor buffers, treating pinned host buffers as
CPU memory. Weight placement does not imply every operation runs on GPU.
The pinned llama.cpp internal model header is used only inside the runtime;
it is not exposed by the public C ABI.

CLI equivalents:

```sh
mercan devices
mercan run model.mercan --device auto -p Merhaba
mercan run model.mercan --device cuda --gpu-layers 12 -p Merhaba
mercan run model.mercan --device cpu -p Merhaba
```

Build with `-DMERCAN_BUILD_SHARED=ON -DGGML_CUDA=ON` for CUDA. Install the
CUDA Toolkit for compilation and compatible runtime/cuBLAS libraries plus
an NVIDIA driver for execution. The default CPU build remains supported.
