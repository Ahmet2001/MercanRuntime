#pragma once

#include "mercan_arch.h"
#include "mercan_tokenizer.h"

#include <string>

struct gguf_context;

struct mercan_metadata_probe {
    gguf_context * ctx = nullptr;
    mercan_metadata_v1 view{};

    mercan_metadata_probe();
    ~mercan_metadata_probe();
    mercan_metadata_probe(const mercan_metadata_probe &) = delete;
    mercan_metadata_probe & operator=(const mercan_metadata_probe &) = delete;
};

bool mercan_metadata_open(const char * path, mercan_metadata_probe & out, std::string & error);
std::string mercan_metadata_string(const mercan_metadata_v1 * metadata, const char * key);
void mercan_ensure_builtin_plugins(void);
void mercan_register_nedolm_architecture(void);
void mercan_register_ndsurf004_tokenizer(void);
void mercan_register_lfm3_architecture(void);
void mercan_register_gguf_vocab_tokenizer(void);
