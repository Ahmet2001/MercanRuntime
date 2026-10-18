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

// Device API v1. Call after backend_init(). JSON strings are UTF-8 snapshots;
// the runtime snapshot is thread-local and valid until its next snapshot call.
// Model placement strings are borrowed until model_free(). ABI 1 structs stay frozen.
#define MERCAN_DEVICE_API_VERSION 1
MERCAN_API uint32_t mercan_device_api_version(void);
MERCAN_API bool mercan_cuda_available_v1(void);
MERCAN_API const char * mercan_runtime_devices_json_v1(void);
// cpu requires zero layers, cuda nonzero. auto + zero chooses automatically;
// auto + nonzero is strict CUDA offload. See docs/DEVICE_API_V1.md.
MERCAN_API mercan_model * mercan_model_load_device_v1(
    const char * path, mercan_model_params params, const char * device);
MERCAN_API const char * mercan_model_device_v1(const mercan_model * model);
MERCAN_API const char * mercan_model_device_json_v1(const mercan_model * model);

MERCAN_API mercan_context * mercan_context_create(mercan_model * model, mercan_context_params params);
MERCAN_API void mercan_context_free(mercan_context * ctx);
// Discard a decoded suffix. Call only while idle; recurrent backends may reject
// partial truncation, in which case sessions safely recreate their context.
MERCAN_API int mercan_context_truncate_v1(mercan_context * ctx, uint32_t keep_tokens);

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

// Generation API v1: one owned request/context per handle, borrowed model.
// Keep the model alive until free(). One thread may run a handle once; cancel()
// is safe from another thread. Free only after run() has returned. No exceptions
// cross this boundary. Text is length-delimited UTF-8; output pieces can split a
// UTF-8 character and must be decoded incrementally by text consumers.
#define MERCAN_GENERATION_API_VERSION 1
typedef struct mercan_generation mercan_generation;
typedef struct mercan_generation_params_v1 {
    uint32_t struct_size;
    int32_t max_tokens;
    float temperature;
    int32_t top_k;
    float top_p;
    float repeat_penalty;
    int32_t repeat_last_n;
    uint32_t seed;
    uint32_t use_seed;
} mercan_generation_params_v1;

typedef struct mercan_chat_message_v1 {
    const char * role; // canonical: system, user, assistant
    const char * content;
    size_t content_len;
} mercan_chat_message_v1;

enum mercan_generation_finish_v1 {
    MERCAN_GENERATION_PENDING = 0,
    MERCAN_GENERATION_STOP = 1,
    MERCAN_GENERATION_LIMIT = 2,
    MERCAN_GENERATION_CANCELLED = 3,
    MERCAN_GENERATION_ERROR = 4
};

// Callback returns nonzero to continue, zero to cancel. Bytes are borrowed for
// the callback duration only. It must not throw or free the generation/model.
typedef int (*mercan_generation_callback_v1)(const char *, size_t, void *);
MERCAN_API uint32_t mercan_generation_api_version(void);
MERCAN_API size_t mercan_generation_params_size_v1(void);
MERCAN_API int mercan_generation_default_params_v1(mercan_generation_params_v1 *, size_t);
// begin() copies the input, tokenizes and creates a fresh context. Prefill and
// the complete sampling/decode loop execute inside run(). Null params use defaults.
MERCAN_API mercan_generation * mercan_generate_begin_v1(
    mercan_model *, const char *, size_t, mercan_context_params,
    const mercan_generation_params_v1 *);
// Chat formatting and content delimiter escaping are shared by all consumers.
// Null role headers select sistem/kullanici/asistan. No system prompt/history
// is inserted; the caller supplies the complete message list.
MERCAN_API mercan_generation * mercan_chat_begin_v1(
    mercan_model *, const mercan_chat_message_v1 *, size_t,
    const char * role_system, const char * role_user, const char * role_assistant,
    mercan_context_params, const mercan_generation_params_v1 *);
// 0 = completed (STOP or LIMIT), 1 = cancelled, -1 = error (mercan_last_error).
// A null callback discards output. Cancellation is cooperative between decode
// batches/tokens; it does not interrupt an in-flight device operation.
MERCAN_API int mercan_generation_run_v1(mercan_generation *, mercan_generation_callback_v1, void *);
MERCAN_API void mercan_generation_cancel_v1(mercan_generation *);
MERCAN_API int mercan_generation_finish_reason_v1(const mercan_generation *);
MERCAN_API void mercan_generation_free_v1(mercan_generation *);

// Session API v1: borrowed model, owned persistent context/KV cache. Supply the
// complete prompt/transcript each turn; an exact token prefix is reused.
// One outstanding request per session, including completed but unfreed requests.
// Serialize calls per handle except generation_cancel(). Keep the model alive.
#define MERCAN_SESSION_API_VERSION 1
typedef struct mercan_session mercan_session;
MERCAN_API uint32_t mercan_session_api_version(void);
MERCAN_API mercan_session * mercan_session_create_v1(mercan_model *, mercan_context_params);
MERCAN_API mercan_generation * mercan_session_generate_begin_v1(
    mercan_session *, const char *, size_t, const mercan_generation_params_v1 *);
MERCAN_API mercan_generation * mercan_session_chat_begin_v1(
    mercan_session *, const mercan_chat_message_v1 *, size_t,
    const char *, const char *, const char *, const mercan_generation_params_v1 *);
// Explicitly retain a successfully completed turn before freeing the request.
// Uncommitted, failed or cancelled work invalidates the cache if it was touched.
MERCAN_API int mercan_generation_commit_session_v1(mercan_generation *);
MERCAN_API int mercan_session_reset_v1(mercan_session *);
// Borrowed thread-local JSON until next stats call. Requires an idle session.
MERCAN_API const char * mercan_session_stats_json_v1(const mercan_session *);
// Returns -1 while a request is outstanding; the session remains alive.
MERCAN_API int mercan_session_free_v1(mercan_session *);

// Embedding API v1: one pooled vector of the final normalized hidden states per
// text, for retrieval. Borrowed model, owned context; serialize calls per handle.
// Vectors are not L2-normalized. Texts longer than n_ctx tokens are truncated.
// Only backend-managed architectures are supported; others fail at create().
#define MERCAN_EMBEDDING_API_VERSION 1
enum mercan_pooling_v1 {
    MERCAN_POOLING_MEAN = 0,
    MERCAN_POOLING_LAST = 1
};
typedef struct mercan_embedder mercan_embedder;
MERCAN_API uint32_t mercan_embedding_api_version(void);
MERCAN_API mercan_embedder * mercan_embedder_create_v1(mercan_model *, mercan_context_params, int pooling);
MERCAN_API int32_t mercan_embedder_size_v1(const mercan_embedder *);
// Returns the number of floats written (the embedding size), or -1 on error
// (mercan_last_error). capacity must be at least mercan_embedder_size_v1().
MERCAN_API int32_t mercan_embed_v1(mercan_embedder *, const char * text, size_t text_len,
                                   float * out, int32_t capacity);
MERCAN_API void mercan_embedder_free_v1(mercan_embedder *);

#ifdef __cplusplus
}
#endif

#endif
