#include "mercan_arch.h"
#include "mercan_internal.hpp"

#include <cstdio>
#include <cstring>

// LiquidAI LFM3 (hybrid gated short-conv + sliding-window/global attention).
//
// The tensor graph, hybrid conv/KV memory and RoPE variants are implemented by
// llama.cpp's own `lfm3` model class (added upstream in #31252), so this
// provider only claims and validates the metadata; execution is backend-managed.

namespace {

int lfm3_probe(const mercan_metadata_v1 * metadata) {
    if (!metadata || !metadata->get_string) return -1;
    const char * arch = metadata->get_string(metadata, "general.architecture");
    if (!arch) return 0;
    return std::strcmp(arch, "lfm3") == 0 ? 100 : 0;
}

int lfm3_validate(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity) {
    const char * arch = metadata && metadata->get_string
        ? metadata->get_string(metadata, "general.architecture") : nullptr;
    if (!arch || std::strcmp(arch, "lfm3") != 0) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "expected general.architecture=lfm3");
        return -1;
    }
    static const char * const required[] = {
        "lfm3.block_count",
        "lfm3.embedding_length",
        "lfm3.shortconv.l_cache",
    };
    for (const char * key : required) {
        if (!metadata->has_key || !metadata->has_key(metadata, key)) {
            if (error && error_capacity) std::snprintf(error, error_capacity, "LFM3 model is missing %s", key);
            return -2;
        }
    }
    return 0;
}

const mercan_architecture_v1 LFM3_ARCH = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "lfm3",
    "LiquidAI LFM3",
    "gguf-vocab",
    MERCAN_ARCH_BACKEND_MANAGED_GRAPH | MERCAN_ARCH_BUILTIN,
    lfm3_probe,
    lfm3_validate,
    nullptr, // backend-managed graph (llama.cpp lfm3 model class)
};

}

void mercan_register_lfm3_architecture(void) {
    (void) mercan_arch_register_v1(&LFM3_ARCH);
}
