#include "mercan_tokenizer.h"
#include "mercan_internal.hpp"

#include <cstdio>
#include <cstring>

// Generic backend-managed tokenizer: the vocabulary, merges and special tokens
// are the standard `tokenizer.ggml.*` data stored in the .mercan (GGUF v3)
// container, and llama.cpp's own tokenizer implementation encodes/decodes them.
// Used by architectures whose upstream llama.cpp model class is reused as-is
// (e.g. lfm3). It holds no executable code, only data the backend interprets.

namespace {

int gguf_vocab_probe(const mercan_metadata_v1 * metadata) {
    if (!metadata || !metadata->get_string) return -1;
    const char * explicit_type = metadata->get_string(metadata, "mercan.tokenizer.type");
    return explicit_type && std::strcmp(explicit_type, "gguf-vocab") == 0 ? 100 : 0;
}

int gguf_vocab_validate(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity) {
    if (!metadata || !metadata->get_string || !metadata->has_key) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "metadata view unavailable");
        return -1;
    }
    const char * model = metadata->get_string(metadata, "tokenizer.ggml.model");
    if (!model || !*model) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "tokenizer.ggml.model is missing");
        return -2;
    }
    if (std::strcmp(model, "nedolm") == 0) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "NedoLM vocabularies must use the ndsurf004 tokenizer");
        return -3;
    }
    if (!metadata->has_key(metadata, "tokenizer.ggml.tokens")) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "tokenizer.ggml.tokens is missing");
        return -4;
    }
    return 0;
}

const mercan_tokenizer_v1 GGUF_VOCAB = {
    MERCAN_TOKENIZER_ABI_VERSION,
    sizeof(mercan_tokenizer_v1),
    "gguf-vocab",
    "GGUF vocabulary (llama.cpp tokenizer)",
    MERCAN_TOKENIZER_BACKEND_MANAGED | MERCAN_TOKENIZER_BUILTIN,
    gguf_vocab_probe,
    gguf_vocab_validate,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}

void mercan_register_gguf_vocab_tokenizer(void) {
    (void) mercan_tokenizer_register_v1(&GGUF_VOCAB);
}
