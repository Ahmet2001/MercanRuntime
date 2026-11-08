#include "mercan.h"
#include "mercan_arch.h"
#include "mercan_tokenizer.h"
#include "mercan_internal.hpp"
#include "generic_backend.hpp"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

struct mercan_model {
    llama_model * impl = nullptr;
    mercan_generic_model * generic = nullptr;
    const mercan_architecture_v1 * architecture = nullptr;
    const mercan_tokenizer_v1 * tokenizer = nullptr;
    void * tokenizer_state = nullptr;
    std::string architecture_name;
    std::string tokenizer_name;
};

struct mercan_context {
    llama_context * impl = nullptr;
    mercan_generic_context * generic = nullptr;
    mercan_model * model = nullptr;
    std::vector<mercan_token> token_history;
    std::mt19937 rng;
    bool rng_seeded = false;
    uint32_t rng_seed_value = MERCAN_DEFAULT_SEED;
};

static thread_local std::string g_last_error;

static void set_error(const std::string & s) {
    g_last_error = s;
}

static void mercan_llama_log(enum ggml_log_level level, const char * text, void *) {
    const char * verbose = std::getenv("MERCAN_VERBOSE");
    const bool show_all = verbose && *verbose && std::strcmp(verbose, "0") != 0;
    if (show_all || level == GGML_LOG_LEVEL_ERROR) {
        if (text) std::fputs(text, stderr);
    }
}

extern "C" {

const char * mercan_version(void) {
    return "0.1.2";
}

const char * mercan_last_error(void) {
    return g_last_error.c_str();
}

void mercan_backend_init(void) {
    g_last_error.clear();
    mercan_ensure_builtin_plugins();
    llama_log_set(mercan_llama_log, nullptr);
    llama_backend_init();
}

void mercan_backend_free(void) {
    llama_backend_free();
}

mercan_model_params mercan_model_default_params(void) {
    mercan_model_params p{};
    p.n_gpu_layers = 0;
    p.use_mmap = true;
    p.check_tensors = false;
    return p;
}

mercan_context_params mercan_context_default_params(void) {
    mercan_context_params p{};
    p.n_ctx = 4096;
    p.n_batch = 512;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    p.n_threads = static_cast<int32_t>(hw);
    p.n_threads_batch = static_cast<int32_t>(hw);
    return p;
}

mercan_sampler_params mercan_sampler_default_params(void) {
    mercan_sampler_params p{};
    p.temperature = 0.7f;
    p.top_k = 40;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    p.repeat_penalty = 1.10f;
    p.repeat_last_n = 64;
    p.seed = MERCAN_DEFAULT_SEED;
    return p;
}

mercan_model * mercan_model_load(const char * path, mercan_model_params params) {
    g_last_error.clear();
    if (!path || !*path) {
        set_error("model path is empty");
        return nullptr;
    }
    try {
        mercan_ensure_builtin_plugins();

        mercan_metadata_probe metadata;
        std::string metadata_error;
        if (!mercan_metadata_open(path, metadata, metadata_error)) {
            set_error(metadata_error);
            return nullptr;
        }

        const std::string mercan_format = mercan_metadata_string(&metadata.view, "mercan.format");
        if (mercan_format != "mercan") {
            set_error(mercan_format.empty()
                ? "model is missing mercan.format=mercan"
                : "unsupported mercan.format='" + mercan_format + "'");
            return nullptr;
        }
        const int64_t format_version = metadata.view.get_i64(&metadata.view, "mercan.format_version", -1);
        if (format_version != 1) {
            set_error("unsupported Mercan model format version " + std::to_string(format_version) + " (runtime supports v1)");
            return nullptr;
        }
        const int64_t runtime_abi = metadata.view.get_i64(&metadata.view, "mercan.runtime_abi", -1);
        if (runtime_abi != 1) {
            set_error("unsupported Mercan runtime ABI " + std::to_string(runtime_abi) + " (runtime supports ABI 1)");
            return nullptr;
        }

        std::string arch_name = mercan_metadata_string(&metadata.view, "general.architecture");
        const mercan_architecture_v1 * arch = nullptr;
        if (!arch_name.empty()) {
            arch = mercan_arch_find_v1(arch_name.c_str());
        } else {
            int best_score = 0;
            for (size_t i = 0; i < mercan_arch_count_v1(); ++i) {
                const mercan_architecture_v1 * candidate = mercan_arch_at_v1(i);
                if (!candidate || !candidate->probe) continue;
                const int score = candidate->probe(&metadata.view);
                if (score > best_score) {
                    best_score = score;
                    arch = candidate;
                }
            }
            if (arch) arch_name = arch->name;
        }

        if (!arch) {
            set_error(arch_name.empty()
                ? "model architecture is missing and no Mercan Architecture SDK plugin matched"
                : "unsupported Mercan architecture '" + arch_name + "' (register a mercan_architecture_v1 provider)");
            return nullptr;
        }

        if (arch->validate) {
            char error[512] = {};
            if (arch->validate(&metadata.view, error, sizeof(error)) != 0) {
                set_error(std::string("architecture '") + arch->name + "' rejected model: " + (error[0] ? error : "validation failed"));
                return nullptr;
            }
        }

        std::string tokenizer_name = mercan_metadata_string(&metadata.view, "mercan.tokenizer.type");
        if (tokenizer_name.empty() && arch->default_tokenizer) tokenizer_name = arch->default_tokenizer;

        const mercan_tokenizer_v1 * tokenizer = nullptr;
        if (!tokenizer_name.empty()) tokenizer = mercan_tokenizer_find_v1(tokenizer_name.c_str());
        if (!tokenizer && !tokenizer_name.empty()) {
            set_error("unsupported Mercan tokenizer '" + tokenizer_name + "' (register a mercan_tokenizer_v1 provider)");
            return nullptr;
        }
        if (tokenizer && tokenizer->validate) {
            char error[512] = {};
            if (tokenizer->validate(&metadata.view, error, sizeof(error)) != 0) {
                set_error(std::string("tokenizer '") + tokenizer->name + "' rejected model: " + (error[0] ? error : "validation failed"));
                return nullptr;
            }
        }

        const bool generic_external = (arch->flags & MERCAN_ARCH_GRAPH_CALLBACK_V1) &&
                                      !(arch->flags & MERCAN_ARCH_BACKEND_MANAGED_GRAPH);
        llama_model * lm = nullptr;
        mercan_generic_model * gm = nullptr;
        if (generic_external) {
            std::string generic_error;
            gm = mercan_generic_model_load(path, arch, params, generic_error);
            if (!gm) { set_error(generic_error); return nullptr; }
        } else {
            llama_model_params lp = llama_model_default_params();
            lp.n_gpu_layers = params.n_gpu_layers;
            lp.load_mode = params.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
            lp.check_tensors = params.check_tensors;
            lm = llama_model_load_from_file(path, lp);
            if (!lm) {
                set_error(std::string("backend failed to load ") + arch->name + " model: " + path);
                return nullptr;
            }
        }

        void * tokenizer_state = nullptr;
        if (tokenizer && tokenizer->create) {
            char error[512] = {};
            tokenizer_state = tokenizer->create(&metadata.view, error, sizeof(error));
            if (!tokenizer_state) {
                if (lm) llama_model_free(lm);
                if (gm) mercan_generic_model_free(gm);
                set_error(std::string("tokenizer '") + tokenizer->name + " initialization failed: " + (error[0] ? error : "unknown error"));
                return nullptr;
            }
        }

        mercan_model * out = new mercan_model();
        out->impl = lm;
        out->generic = gm;
        out->architecture = arch;
        out->tokenizer = tokenizer;
        out->tokenizer_state = tokenizer_state;
        out->architecture_name = arch_name;
        out->tokenizer_name = tokenizer_name;
        return out;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown error while loading model");
        return nullptr;
    }
}

void mercan_model_free(mercan_model * model) {
    if (!model) return;
    if (model->tokenizer && model->tokenizer->destroy && model->tokenizer_state) {
        model->tokenizer->destroy(model->tokenizer_state);
    }
    if (model->impl) llama_model_free(model->impl);
    if (model->generic) mercan_generic_model_free(model->generic);
    delete model;
}

const char * mercan_model_architecture(const mercan_model * model) {
    return model ? model->architecture_name.c_str() : "";
}

const char * mercan_model_tokenizer(const mercan_model * model) {
    return model ? model->tokenizer_name.c_str() : "";
}

mercan_context * mercan_context_create(mercan_model * model, mercan_context_params params) {
    g_last_error.clear();
    if (!model || (!model->impl && !model->generic)) {
        set_error("invalid model");
        return nullptr;
    }
    try {
        llama_context * lc = nullptr;
        mercan_generic_context * gc = nullptr;
        if (model->generic) {
            std::string e; gc = mercan_generic_context_create(model->generic, params, e);
            if (!gc) { set_error(e); return nullptr; }
        } else {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = params.n_ctx; cp.n_batch = params.n_batch; cp.n_threads = params.n_threads; cp.n_threads_batch = params.n_threads_batch;
            lc = llama_init_from_model(model->impl, cp);
            if (!lc) { set_error("failed to create llama context"); return nullptr; }
        }
        mercan_context * out = new mercan_context();
        out->impl = lc; out->generic = gc;
        out->model = model;
        return out;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown error while creating context");
        return nullptr;
    }
}

void mercan_context_free(mercan_context * ctx) {
    if (!ctx) return;
    if (ctx->impl) llama_free(ctx->impl);
    if (ctx->generic) mercan_generic_context_free(ctx->generic);
    delete ctx;
}

int32_t mercan_context_reset(mercan_context * ctx, bool clear_data) {
    g_last_error.clear();
    if (!ctx || (!ctx->impl && !ctx->generic)) {
        set_error("invalid context");
        return -1;
    }
    if (ctx->generic) {
        std::string e;
        const int rc = mercan_generic_context_reset(ctx->generic, clear_data, e);
        if (rc != 0) set_error(e);
        if (rc != 0) return rc;
    } else {
        llama_memory_t mem = llama_get_memory(ctx->impl);
        if (!mem) {
            set_error("backend context does not expose sequence memory");
            return -1;
        }
        llama_memory_clear(mem, clear_data);
    }
    ctx->token_history.clear();
    ctx->rng_seeded = false;
    return 0;
}

int32_t mercan_context_rewind(mercan_context * ctx, uint32_t token_count) {
    g_last_error.clear();
    if (!ctx || (!ctx->impl && !ctx->generic)) {
        set_error("invalid context");
        return -1;
    }
    if (token_count > ctx->token_history.size()) {
        set_error("rewind position exceeds decoded token history");
        return -1;
    }
    if (ctx->generic) {
        std::string e;
        const int rc = mercan_generic_context_rewind(ctx->generic, token_count, e);
        if (rc != 0) set_error(e);
        if (rc != 0) return rc;
    } else {
        llama_memory_t mem = llama_get_memory(ctx->impl);
        if (!mem) {
            set_error("backend context does not expose sequence memory");
            return -1;
        }
        if (!llama_memory_seq_rm(mem, -1, static_cast<llama_pos>(token_count), -1)) {
            set_error("backend memory cannot partially rewind this sequence");
            return -1;
        }
    }
    ctx->token_history.resize(token_count);
    return 0;
}

int32_t mercan_tokenize(
    mercan_model * model,
    const char * text,
    size_t text_len,
    bool add_special,
    bool parse_special,
    mercan_token * out_tokens,
    int32_t capacity) {
    g_last_error.clear();
    if (!model || (!model->impl && !model->generic) || !text) {
        set_error("invalid tokenize arguments");
        return 0;
    }
    if (model->tokenizer && model->tokenizer->encode) {
        return model->tokenizer->encode(model->tokenizer_state, text, text_len, add_special, parse_special, out_tokens, capacity);
    }
    if (!model->impl) { set_error("generic model requires an external tokenizer encode callback"); return 0; }
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    if (!vocab) {
        set_error("model vocabulary is unavailable");
        return 0;
    }
    return llama_tokenize(
        vocab,
        text,
        static_cast<int32_t>(text_len),
        reinterpret_cast<llama_token *>(out_tokens),
        capacity,
        add_special,
        parse_special);
}

int32_t mercan_decode(mercan_context * ctx, const mercan_token * tokens, int32_t n_tokens) {
    g_last_error.clear();
    if (!ctx || (!ctx->impl && !ctx->generic) || !tokens || n_tokens <= 0) {
        set_error("invalid decode arguments");
        return -1;
    }

    int rc = 0;
    if (ctx->generic) {
        std::string e;
        rc = mercan_generic_decode(ctx->generic, tokens, n_tokens, e);
        if (rc != 0) set_error(e);
    } else {
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(reinterpret_cast<const llama_token *>(tokens)),
            n_tokens);
        rc = llama_decode(ctx->impl, batch);
        if (rc != 0) set_error("llama_decode failed with code " + std::to_string(rc));
    }

    if (rc == 0) {
        ctx->token_history.insert(ctx->token_history.end(), tokens, tokens + n_tokens);
    }
    return rc;
}

mercan_token mercan_sample_next(mercan_context * ctx, mercan_sampler_params params) {
    g_last_error.clear();
    if (!ctx || !ctx->model) {
        set_error("invalid sampler context");
        return -1;
    }

    const float * logits = mercan_logits(ctx);
    const int32_t n_vocab = mercan_vocab_size(ctx->model);
    if (!logits || n_vocab <= 0) {
        set_error("no logits available for sampling");
        return -1;
    }

    struct candidate {
        mercan_token id;
        float logit;
        double weight;
    };

    std::unordered_set<mercan_token> repeated;
    if (params.repeat_penalty > 0.0f && params.repeat_penalty != 1.0f && params.repeat_last_n != 0) {
        const size_t history_size = ctx->token_history.size();
        const size_t window = params.repeat_last_n < 0
            ? history_size
            : std::min(history_size, static_cast<size_t>(params.repeat_last_n));
        const size_t begin = history_size - window;
        for (size_t i = begin; i < history_size; ++i) repeated.insert(ctx->token_history[i]);
    }

    std::vector<candidate> candidates;
    candidates.reserve(static_cast<size_t>(n_vocab));
    for (int32_t i = 0; i < n_vocab; ++i) {
        float score = logits[i];
        if (repeated.find(i) != repeated.end() && params.repeat_penalty > 0.0f && params.repeat_penalty != 1.0f) {
            score = score <= 0.0f ? score * params.repeat_penalty : score / params.repeat_penalty;
        }
        candidates.push_back({i, score, 0.0});
    }

    auto by_logit = [](const candidate & a, const candidate & b) { return a.logit > b.logit; };

    if (params.temperature <= 0.0f) {
        return std::max_element(
            candidates.begin(),
            candidates.end(),
            [](const candidate & a, const candidate & b) { return a.logit < b.logit; }
        )->id;
    }

    const int32_t requested_top_k = params.top_k <= 0 ? n_vocab : params.top_k;
    const size_t top_k = static_cast<size_t>(std::max<int32_t>(1, std::min<int32_t>(requested_top_k, n_vocab)));
    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(), by_logit);
    candidates.resize(top_k);

    const float max_logit = candidates.front().logit;
    const double inv_temp = 1.0 / static_cast<double>(std::max(params.temperature, 1e-6f));
    for (auto & c : candidates) {
        c.weight = std::exp((static_cast<double>(c.logit) - max_logit) * inv_temp);
    }

    if (params.min_p > 0.0f && params.min_p < 1.0f && candidates.size() > 1) {
        const double threshold = static_cast<double>(params.min_p) * candidates.front().weight;
        size_t keep = 1;
        while (keep < candidates.size() && candidates[keep].weight >= threshold) ++keep;
        candidates.resize(keep);
    }

    if (params.top_p > 0.0f && params.top_p < 1.0f && candidates.size() > 1) {
        double total = 0.0;
        for (const auto & c : candidates) total += c.weight;
        if (total > 0.0 && std::isfinite(total)) {
            double cumulative = 0.0;
            size_t keep = 0;
            for (; keep < candidates.size(); ++keep) {
                cumulative += candidates[keep].weight / total;
                if (cumulative >= params.top_p) {
                    ++keep;
                    break;
                }
            }
            candidates.resize(std::max<size_t>(1, std::min(keep, candidates.size())));
        }
    }

    if (!ctx->rng_seeded || ctx->rng_seed_value != params.seed) {
        if (params.seed == MERCAN_DEFAULT_SEED) {
            std::random_device rd;
            std::seed_seq seq{rd(), rd(), rd(), rd()};
            ctx->rng.seed(seq);
        } else {
            ctx->rng.seed(params.seed);
        }
        ctx->rng_seed_value = params.seed;
        ctx->rng_seeded = true;
    }

    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (const auto & c : candidates) weights.push_back(c.weight);
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(ctx->rng)].id;
}

const float * mercan_logits(mercan_context * ctx) {
    if (!ctx) return nullptr;
    if (ctx->generic) return mercan_generic_logits(ctx->generic);
    return ctx->impl ? llama_get_logits_ith(ctx->impl, -1) : nullptr;
}

int32_t mercan_vocab_size(mercan_model * model) {
    if (!model) return 0;
    if (model->generic) return mercan_generic_vocab_size(model->generic);
    if (!model->impl) return 0;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_n_tokens(vocab) : 0;
}

uint32_t mercan_context_size(mercan_context * ctx) {
    if (!ctx) return 0;
    if (ctx->generic) return mercan_generic_context_size(ctx->generic);
    return ctx->impl ? llama_n_ctx(ctx->impl) : 0;
}

mercan_token mercan_bos_token(mercan_model * model) {
    if (!model) return -1;
    if (model->generic) return mercan_generic_bos_token(model->generic);
    if (!model->impl) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_bos(vocab) : -1;
}

mercan_token mercan_eos_token(mercan_model * model) {
    if (!model) return -1;
    if (model->generic) return mercan_generic_eos_token(model->generic);
    if (!model->impl) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_eos(vocab) : -1;
}

bool mercan_token_is_eog(mercan_model * model, mercan_token token) {
    if (!model || token < 0) return false;
    if (model->generic) return token == mercan_generic_eos_token(model->generic);
    if (!model->impl) return false;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_is_eog(vocab, static_cast<llama_token>(token)) : false;
}

mercan_token mercan_pad_token(mercan_model * model) {
    if (!model) return -1;
    if (model->generic) return mercan_generic_pad_token(model->generic);
    if (!model->impl) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_pad(vocab) : -1;
}

int32_t mercan_token_to_piece(
    mercan_model * model,
    mercan_token token,
    char * out,
    int32_t capacity,
    bool special) {
    if (!model || (!model->impl && !model->generic)) return 0;
    if (model->tokenizer && model->tokenizer->decode_piece) {
        return model->tokenizer->decode_piece(model->tokenizer_state, token, out, capacity, special);
    }
    if (!model->impl) return 0;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    if (!vocab) return 0;
    return llama_token_to_piece(vocab, token, out, capacity, 0, special);
}

} // extern "C"
