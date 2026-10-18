#ifndef MERCAN_H
#define MERCAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef MERCAN_BUILD_SHARED
#    define MERCAN_API __declspec(dllexport)
#  elif defined(MERCAN_USE_SHARED)
#    define MERCAN_API __declspec(dllimport)
#  else
#    define MERCAN_API
#  endif
#else
#  define MERCAN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t mercan_token;

// ABI 1 parameter layouts are frozen. Add new versioned entry points instead
// of extending structs passed by value. Bindings can check these before use.
#define MERCAN_ABI_VERSION 1

typedef struct mercan_model mercan_model;
typedef struct mercan_context mercan_context;

typedef struct mercan_model_params {
    int32_t n_gpu_layers;
    bool use_mmap;
    bool check_tensors;
} mercan_model_params;

typedef struct mercan_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    int32_t n_threads;
    int32_t n_threads_batch;
} mercan_context_params;

MERCAN_API const char * mercan_version(void);
MERCAN_API uint32_t mercan_abi_version(void);
MERCAN_API size_t mercan_model_params_size(void);
MERCAN_API size_t mercan_context_params_size(void);
MERCAN_API const char * mercan_last_error(void);

MERCAN_API void mercan_backend_init(void);
MERCAN_API void mercan_backend_free(void);

MERCAN_API mercan_model_params mercan_model_default_params(void);
MERCAN_API mercan_context_params mercan_context_default_params(void);

MERCAN_API mercan_model * mercan_model_load(const char * path, mercan_model_params params);
MERCAN_API void mercan_model_free(mercan_model * model);
MERCAN_API const char * mercan_model_architecture(const mercan_model * model);
MERCAN_API const char * mercan_model_tokenizer(const mercan_model * model);

MERCAN_API mercan_context * mercan_context_create(mercan_model * model, mercan_context_params params);
MERCAN_API void mercan_context_free(mercan_context * ctx);

// Returns token count on success. If capacity is too small, returns -required_count.
MERCAN_API int32_t mercan_tokenize(
    mercan_model * model,
    const char * text,
    size_t text_len,
    bool add_special,
    bool parse_special,
    mercan_token * out_tokens,
    int32_t capacity);

// Decodes a contiguous token batch. Position tracking is maintained by the context.
MERCAN_API int32_t mercan_decode(mercan_context * ctx, const mercan_token * tokens, int32_t n_tokens);

MERCAN_API const float * mercan_logits(mercan_context * ctx);
MERCAN_API int32_t mercan_vocab_size(mercan_model * model);
MERCAN_API uint32_t mercan_context_size(mercan_context * ctx);
MERCAN_API mercan_token mercan_bos_token(mercan_model * model);
MERCAN_API mercan_token mercan_eos_token(mercan_model * model);
MERCAN_API mercan_token mercan_pad_token(mercan_model * model);

// Returns bytes written, or negative required capacity.
MERCAN_API int32_t mercan_token_to_piece(
    mercan_model * model,
    mercan_token token,
    char * out,
    int32_t capacity,
    bool special);

#ifdef __cplusplus
}
#endif

#endif
