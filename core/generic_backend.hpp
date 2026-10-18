#pragma once

#include "mercan.h"
#include "mercan_arch.h"
#include "ggml-backend.h"

#include <string>

struct mercan_generic_model;
struct mercan_generic_context;

mercan_generic_model * mercan_generic_model_load(const char * path,
                                                  const mercan_architecture_v1 * architecture,
                                                  mercan_model_params params,
                                                  std::string & error,
                                                  ggml_backend_dev_t preferred = nullptr);
ggml_backend_dev_t mercan_generic_model_device(const mercan_generic_model * model);
void mercan_generic_model_free(mercan_generic_model * model);

mercan_generic_context * mercan_generic_context_create(mercan_generic_model * model,
                                                        mercan_context_params params,
                                                        std::string & error);
void mercan_generic_context_free(mercan_generic_context * ctx);
int32_t mercan_generic_decode(mercan_generic_context * ctx,
                              const mercan_token * tokens,
                              int32_t n_tokens,
                              std::string & error);
const float * mercan_generic_logits(mercan_generic_context * ctx);
int32_t mercan_generic_vocab_size(const mercan_generic_model * model);
uint32_t mercan_generic_context_size(const mercan_generic_context * ctx);
mercan_token mercan_generic_bos_token(const mercan_generic_model * model);
mercan_token mercan_generic_eos_token(const mercan_generic_model * model);
mercan_token mercan_generic_pad_token(const mercan_generic_model * model);
