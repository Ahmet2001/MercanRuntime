#ifndef NEDO004_H
#define NEDO004_H

/* Standalone model-free NDSRF004 tokenizer C ABI, implemented by the original
 * Rust NedoTokenizer. Does not require mercan_model or llama.cpp. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEDO004_VOCAB_SIZE 32000u
#define NEDO004_PAD_ID 0u
#define NEDO004_BOS_ID 1u
#define NEDO004_EOS_ID 2u

/* Returns the lowercase SHA-256 of the embedded 32000-entry surface vocabulary.
 * The returned string has process lifetime. NULL means initialization failed. */
const char * nedo004_vocab_sha256(void);
uint32_t nedo004_vocab_size(void);

/* Copy-based thread-safe APIs. On -4, out_len reports required capacity.
 * Return 0 on success, -1 invalid args, -2 unavailable tokenizer,
 * -3 encode/decode error, -4 insufficient capacity.
 * The output may omit BOS/EOS, matching the Mercan NDSRF004 bridge. */
int32_t nedo004_encode_copy(
    const uint8_t * data, size_t len,
    uint16_t * output, size_t capacity, size_t * out_len);
int32_t nedo004_decode_copy(
    const uint16_t * ids, size_t len,
    uint8_t * output, size_t capacity, size_t * out_len);

#ifdef __cplusplus
}
#endif
#endif
