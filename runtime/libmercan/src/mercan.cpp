#include "mercan.h"
#include "mercan_arch.h"
#include "mercan_tokenizer.h"
#include "mercan_internal.hpp"
#include "generic_backend.hpp"
#include "llama.h"
#include "llama-model.h"
#include "device_info.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

struct mercan_model {
    llama_model * impl = nullptr;
    mercan_generic_model * generic = nullptr;
    const mercan_architecture_v1 * architecture = nullptr;
    const mercan_tokenizer_v1 * tokenizer = nullptr;
    void * tokenizer_state = nullptr;
    std::string architecture_name;
    std::string tokenizer_name;
    std::vector<ggml_backend_dev_t> selected_devices;
    std::string device = "cpu";
    std::string placement;
};

struct mercan_context {
    llama_context * impl = nullptr;
    mercan_generic_context * generic = nullptr;
    mercan_model * model = nullptr;
};

static thread_local std::string g_last_error;

static void set_error(const std::string & s) {
    g_last_error = s;
}

// Internal bridge so generation entry points use the same thread-local error.
void mercan_generation_set_error(const std::string & s) { set_error(s); }

static void mercan_llama_log(enum ggml_log_level level, const char * text, void *) {
    const char * verbose = std::getenv("MERCAN_VERBOSE");
    const bool show_all = verbose && *verbose && std::strcmp(verbose, "0") != 0;
    if (show_all || level == GGML_LOG_LEVEL_ERROR) {
        if (text) std::fputs(text, stderr);
    }
}

extern "C" {

const char * mercan_version(void) {
    return "0.3.0";
}

uint32_t mercan_abi_version(void) {
    return MERCAN_ABI_VERSION;
}

size_t mercan_model_params_size(void) {
    return sizeof(mercan_model_params);
}

size_t mercan_context_params_size(void) {
    return sizeof(mercan_context_params);
}

const char * mercan_last_error(void) {
    return g_last_error.c_str();
}

uint32_t mercan_device_api_version(void) { return MERCAN_DEVICE_API_VERSION; }
bool mercan_cuda_available_v1(void) {
    try { return !mercan_devices::cuda().empty(); }
    catch (...) { return false; }
}
const char * mercan_runtime_devices_json_v1(void) {
    static thread_local std::string snapshot;
    try {
#ifdef MERCAN_CUDA_BUILD
        const char * compiled = "true";
#else
        const char * compiled = "false";
#endif
        snapshot = std::string("{\"api_version\":1,\"cuda_compiled\":") + compiled +
            ",\"cuda_available\":" + (mercan_cuda_available_v1() ? "true" : "false") + ",\"backends\":[";
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            if (i) snapshot += ',';
            snapshot += mercan_devices::quote(ggml_backend_reg_name(ggml_backend_reg_get(i)));
        }
        snapshot += "],\"devices\":[";
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            size_t available = 0, total = 0;
            ggml_backend_dev_memory(dev, &available, &total);
            if (i) snapshot += ',';
            snapshot += "{\"name\":" + mercan_devices::quote(ggml_backend_dev_name(dev)) +
                ",\"description\":" + mercan_devices::quote(ggml_backend_dev_description(dev)) +
                ",\"backend\":" + mercan_devices::quote(mercan_devices::backend(dev)) +
                ",\"type\":" + mercan_devices::quote(mercan_devices::kind(dev)) +
                ",\"memory_free\":" + std::to_string(available) + ",\"memory_total\":" + std::to_string(total) + "}";
        }
        snapshot += "]}";
        return snapshot.c_str();
    } catch (const std::exception & e) { set_error(e.what()); }
    catch (...) { set_error("device discovery failed"); }
    return nullptr;
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

static mercan_model * load_model(const char * path, mercan_model_params params, const char * device) {
    g_last_error.clear();
    if (!path || !*path) {
        set_error("model path is empty");
        return nullptr;
    }
    try {
        mercan_ensure_builtin_plugins();
        const std::string requested = device ? device : "legacy";
        if (device && requested != "cpu" && requested != "cuda" && requested != "auto") {
            set_error("device must be cpu, cuda, or auto"); return nullptr;
        }
        if (params.n_gpu_layers < -1) { set_error("gpu_layers must be >= -1"); return nullptr; }
        auto selected = mercan_devices::cuda();
        if (requested == "auto") {
            if (selected.empty() && params.n_gpu_layers != 0) {
                set_error("GPU offload explicitly requested but no usable CUDA device is available"); return nullptr;
            }
            if (selected.empty()) params.n_gpu_layers = 0;
            else if (!params.n_gpu_layers) params.n_gpu_layers = -1;
        } else if (requested == "cuda") {
            if (!params.n_gpu_layers) { set_error("device=cuda conflicts with gpu_layers=0"); return nullptr; }
#ifndef MERCAN_CUDA_BUILD
            set_error("CUDA requested but this libmercan was built without CUDA. Install a CUDA build or use device=cpu/auto.");
            return nullptr;
#else
            if (selected.empty()) { set_error("CUDA requested but no usable CUDA device was found. Check the NVIDIA driver and device visibility."); return nullptr; }
#endif
        } else if (requested == "cpu" && params.n_gpu_layers != 0) {
            set_error("device=cpu conflicts with GPU offload"); return nullptr;
        }
        if (!params.n_gpu_layers) selected.clear();
        else if (requested == "legacy") {
            selected.clear();
            auto dev = mercan_devices::first_gpu();
            if (!dev) { set_error("GPU offload requested but this runtime has no usable GPU device"); return nullptr; }
            selected.push_back(dev);
        }
        if (params.n_gpu_layers != 0 && selected.empty()) {
            set_error("GPU offload requested but no compatible device is available"); return nullptr;
        }
        selected.push_back(nullptr); // null-terminated list retained by the model

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
            gm = mercan_generic_model_load(path, arch, params, generic_error, selected[0]);
            if (!gm) { set_error(generic_error); return nullptr; }
        } else {
            llama_model_params lp = llama_model_default_params();
            lp.n_gpu_layers = params.n_gpu_layers;
            lp.devices = device ? selected.data() : nullptr;
            lp.split_mode = LLAMA_SPLIT_MODE_LAYER;
            lp.load_mode = params.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
            lp.check_tensors = params.check_tensors;
            lm = llama_model_load_from_file(path, lp);
            if (!lm) {
                set_error(std::string("backend failed to load ") + arch->name + " model: " + path);
                return nullptr;
            }
        }

        // Inspect actual weight-buffer ownership instead of reporting requested
        // settings as if they were effective device placement.
        std::vector<ggml_backend_dev_t> actual;
        bool host_weights = false, gpu_weights = false, only_cuda = true;
        auto remember = [&](ggml_backend_dev_t dev) {
            if (!dev) return;
            if (std::find(actual.begin(), actual.end(), dev) == actual.end()) actual.push_back(dev);
            if (mercan_devices::gpu(dev)) {
                gpu_weights = true;
                only_cuda = only_cuda && std::strcmp(mercan_devices::backend(dev), "CUDA") == 0;
            } else host_weights = true;
        };
        if (gm) remember(mercan_generic_model_device(gm));
        else for (const auto & entry : lm->tensors_by_name) {
            auto * tensor = entry.second;
            while (tensor && !tensor->buffer && tensor->view_src) tensor = tensor->view_src;
            if (!tensor || !tensor->buffer) continue;
            auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            if (ggml_backend_buffer_is_host(tensor->buffer)) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            remember(dev);
        }
        if (params.n_gpu_layers != 0 && (!gpu_weights || (requested == "cuda" && !only_cuda))) {
            if (lm) llama_model_free(lm);
            if (gm) mercan_generic_model_free(gm);
            set_error("GPU requested but model weights were not placed on the requested GPU backend"); return nullptr;
        }
        if (actual.empty()) remember(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        const std::string effective = gpu_weights ? (only_cuda ? "cuda" : "gpu") : "cpu";
        std::string placement = "{\"device\":" + mercan_devices::quote(effective) +
            ",\"placement\":" + mercan_devices::quote(gpu_weights ? (host_weights ? "mixed" : "gpu") : "cpu") +
            ",\"gpu_layers_requested\":" + std::to_string(params.n_gpu_layers) + ",\"devices\":[";
        for (size_t i = 0; i < actual.size(); ++i) {
            if (i) placement += ',';
            placement += "{\"name\":" + mercan_devices::quote(ggml_backend_dev_name(actual[i])) +
                ",\"backend\":" + mercan_devices::quote(mercan_devices::backend(actual[i])) + "}";
        }
        placement += "]}";

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
        out->selected_devices = std::move(selected);
        out->device = effective;
        out->placement = std::move(placement);
        return out;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown error while loading model");
        return nullptr;
    }
}

mercan_model * mercan_model_load(const char * path, mercan_model_params params) {
    return load_model(path, params, nullptr);
}
mercan_model * mercan_model_load_device_v1(const char * path, mercan_model_params params, const char * device) {
    if (!device) { set_error("device is required"); return nullptr; }
    return load_model(path, params, device);
}
const char * mercan_model_device_v1(const mercan_model * model) {
    return model ? model->device.c_str() : "unknown";
}
const char * mercan_model_device_json_v1(const mercan_model * model) {
    return model ? model->placement.c_str() : nullptr;
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
    if (ctx->generic) {
        std::string e; const int rc = mercan_generic_decode(ctx->generic, tokens, n_tokens, e); if (rc != 0) set_error(e); return rc;
    }
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(reinterpret_cast<const llama_token *>(tokens)),
        n_tokens);
    const int rc = llama_decode(ctx->impl, batch);
    if (rc != 0) set_error("llama_decode failed with code " + std::to_string(rc));
    return rc;
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
