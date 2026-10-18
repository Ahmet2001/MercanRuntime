#include "generic_backend.hpp"

#include "mercan_graph.h"
#include "mercan_kv.h"
#include "mercan_tensor.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct generic_tensor_catalog {
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::string last_error;
};

struct generic_layer_cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    int64_t head_dim = 0;
    int64_t n_kv_heads = 0;
    enum ggml_type k_type = GGML_TYPE_COUNT;
    enum ggml_type v_type = GGML_TYPE_COUNT;
};

struct generic_cache_update {
    int32_t layer = -1;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct generic_graph_state {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * embedding = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * hidden = nullptr;

    ggml_backend_t cache_backend = nullptr;
    uint32_t n_past = 0;
    uint32_t n_ctx = 0;
    uint32_t n_tokens = 0;
    mercan_tensor_handle_v1 attention_mask = MERCAN_TENSOR_NONE_V1;
    std::vector<generic_layer_cache> * layer_caches = nullptr;
    std::vector<generic_cache_update> updates;
    std::string last_error;
};

struct generic_kv_batch {
    mercan_tensor_handle_v1 positions = MERCAN_TENSOR_NONE_V1;
    mercan_tensor_handle_v1 mask = MERCAN_TENSOR_NONE_V1;
    int64_t window = 0;
    std::string last_error;
};

static ggml_tensor * from_handle(mercan_tensor_handle_v1 h) {
    return reinterpret_cast<ggml_tensor *>(static_cast<uintptr_t>(h));
}
static mercan_tensor_handle_v1 to_handle(ggml_tensor * t) {
    return static_cast<mercan_tensor_handle_v1>(reinterpret_cast<uintptr_t>(t));
}

static generic_tensor_catalog * catalog(mercan_tensor_resolver_v1 * r) {
    return r ? static_cast<generic_tensor_catalog *>(r->userdata) : nullptr;
}
static int tensor_declare(mercan_tensor_resolver_v1 * r, const char * name, uint32_t flags) {
    auto * c = catalog(r);
    if (!c || !name || !*name) return -1;
    if ((flags & MERCAN_TENSOR_REQUIRED_V1) && c->tensors.find(name) == c->tensors.end()) {
        c->last_error = std::string("required tensor is missing: ") + name;
        return -1;
    }
    return 0;
}
static mercan_tensor_handle_v1 tensor_by_name(mercan_tensor_resolver_v1 * r, const char * name) {
    auto * c = catalog(r);
    if (!c || !name) return MERCAN_TENSOR_NONE_V1;
    auto it = c->tensors.find(name);
    return it == c->tensors.end() ? MERCAN_TENSOR_NONE_V1 : to_handle(it->second);
}
static mercan_tensor_handle_v1 tensor_require(mercan_tensor_resolver_v1 * r, const char * name) {
    auto h = tensor_by_name(r, name);
    if (h == MERCAN_TENSOR_NONE_V1) {
        if (auto * c = catalog(r)) c->last_error = std::string("required tensor lookup failed: ") + (name ? name : "<null>");
    }
    return h;
}
static int tensor_has(mercan_tensor_resolver_v1 * r, const char * name) {
    return tensor_by_name(r, name) != MERCAN_TENSOR_NONE_V1;
}
static const char * tensor_error(mercan_tensor_resolver_v1 * r) {
    auto * c = catalog(r);
    return c ? c->last_error.c_str() : "invalid resolver";
}
static size_t tensor_decl_count(mercan_tensor_resolver_v1 *) { return 0; }
static const char * tensor_decl_name(mercan_tensor_resolver_v1 *, size_t) { return nullptr; }
static uint32_t tensor_decl_flags(mercan_tensor_resolver_v1 *, size_t) { return 0; }
static const mercan_tensor_api_v1 TENSOR_API = {
    MERCAN_TENSOR_ABI_VERSION, sizeof(mercan_tensor_api_v1), tensor_declare, tensor_by_name, tensor_require,
    tensor_has, tensor_error, tensor_decl_count, tensor_decl_name, tensor_decl_flags
};

static generic_kv_batch * kv_batch(mercan_kv_resolver_v1 * r) {
    return r ? static_cast<generic_kv_batch *>(r->userdata) : nullptr;
}
static mercan_kv_cache_handle_v1 kv_cache_by_kind(mercan_kv_resolver_v1 * r, mercan_kv_cache_kind_v1 kind) {
    auto * b = kv_batch(r);
    if (!b) return MERCAN_KV_CACHE_NONE_V1;
    if (kind == MERCAN_KV_CACHE_KIND_BASE_V1) return 1;
    if (kind == MERCAN_KV_CACHE_KIND_SLIDING_WINDOW_V1 && b->window > 0) return 2;
    b->last_error = "requested KV cache kind is unavailable";
    return MERCAN_KV_CACHE_NONE_V1;
}
static mercan_kv_cache_kind_v1 kv_kind(mercan_kv_resolver_v1 *, mercan_kv_cache_handle_v1 h) {
    if (h == 1) return MERCAN_KV_CACHE_KIND_BASE_V1;
    if (h == 2) return MERCAN_KV_CACHE_KIND_SLIDING_WINDOW_V1;
    return MERCAN_KV_CACHE_KIND_UNKNOWN_V1;
}
static int64_t kv_window(mercan_kv_resolver_v1 * r, mercan_kv_cache_handle_v1 h) {
    auto * b = kv_batch(r);
    if (!b) return -1;
    if (h == 1) return 0;
    if (h == 2 && b->window > 0) return b->window;
    return -1;
}
static mercan_tensor_handle_v1 kv_k_indices(mercan_kv_resolver_v1 * r, mercan_kv_cache_handle_v1 h) {
    auto * b = kv_batch(r);
    return b && (h == 1 || h == 2) ? b->positions : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 kv_v_indices(mercan_kv_resolver_v1 * r, mercan_kv_cache_handle_v1 h) {
    return kv_k_indices(r, h);
}
static mercan_tensor_handle_v1 kv_mask(mercan_kv_resolver_v1 * r, mercan_kv_cache_handle_v1 h) {
    auto * b = kv_batch(r);
    return b && (h == 1 || h == 2) ? b->mask : MERCAN_TENSOR_NONE_V1;
}
static const char * kv_error(mercan_kv_resolver_v1 * r) {
    auto * b = kv_batch(r);
    return b ? b->last_error.c_str() : "invalid KV resolver";
}
static const mercan_kv_api_v1 KV_API = {
    MERCAN_KV_ABI_VERSION, sizeof(mercan_kv_api_v1), kv_cache_by_kind, kv_kind, kv_window,
    kv_k_indices, kv_v_indices, kv_mask, kv_error
};

static generic_graph_state * gs(mercan_graph_builder_v1 * b) {
    return b ? static_cast<generic_graph_state *>(b->userdata) : nullptr;
}
static int64_t g_dim(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 h, uint32_t a) {
    auto * t = from_handle(h);
    return t && a < GGML_MAX_DIMS ? t->ne[a] : 0;
}
static size_t g_stride(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 h, uint32_t a) {
    auto * t = from_handle(h);
    return t && a < GGML_MAX_DIMS ? t->nb[a] : 0;
}
static mercan_tensor_handle_v1 g_get_rows(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 i) {
    auto * s = gs(b); auto * x = from_handle(a); auto * y = from_handle(i);
    return s && s->ctx && x && y ? to_handle(ggml_get_rows(s->ctx, x, y)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_cast(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a) {
    auto * s = gs(b); auto * x = from_handle(a);
    return s && s->ctx && x ? to_handle(ggml_cast(s->ctx, x, GGML_TYPE_F32)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_swiglu(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) {
    auto * s = gs(b); auto * x = from_handle(a); auto * y = from_handle(c);
    return s && s->ctx && x && y ? to_handle(ggml_swiglu_split(s->ctx, x, y)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_view2(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, int64_t n0, int64_t n1, size_t nb1, size_t off) {
    auto * s = gs(b); auto * x = from_handle(a);
    return s && s->ctx && x ? to_handle(ggml_view_2d(s->ctx, x, n0, n1, nb1, off)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_mul(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) {
    auto * s = gs(b); auto * x = from_handle(a); auto * y = from_handle(c);
    return s && s->ctx && x && y ? to_handle(ggml_mul(s->ctx, x, y)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_add(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) {
    auto * s = gs(b); auto * x = from_handle(a); auto * y = from_handle(c);
    return s && s->ctx && x && y ? to_handle(ggml_add(s->ctx, x, y)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_concat(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c, int32_t d) {
    auto * s = gs(b); auto * x = from_handle(a); auto * y = from_handle(c);
    return s && s->ctx && x && y ? to_handle(ggml_concat(s->ctx, x, y, d)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_matmul(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) {
    auto * s = gs(b); auto * x = from_handle(a); auto * y = from_handle(c);
    return s && s->ctx && x && y ? to_handle(ggml_mul_mat(s->ctx, x, y)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_rms(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, float eps) {
    auto * s = gs(b); auto * x = from_handle(a);
    return s && s->ctx && x ? to_handle(ggml_rms_norm(s->ctx, x, eps)) : MERCAN_TENSOR_NONE_V1;
}
static mercan_tensor_handle_v1 g_rope(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 pos,
                                      mercan_tensor_handle_v1 freq, int32_t nd, mercan_rope_mode_v1 mode, int32_t ctx,
                                      float base, float scale, float ext, float attn, float bf, float bs) {
    auto * s = gs(b); auto * x = from_handle(a); auto * p = from_handle(pos); auto * f = from_handle(freq);
    int m = mode == MERCAN_ROPE_MODE_NORMAL_V1 ? GGML_ROPE_TYPE_NORMAL : mode == MERCAN_ROPE_MODE_NEOX_V1 ? GGML_ROPE_TYPE_NEOX : -1;
    return s && s->ctx && x && p && m >= 0 ? to_handle(ggml_rope_ext(s->ctx, x, p, f, nd, m, ctx, base, scale, ext, attn, bf, bs)) : MERCAN_TENSOR_NONE_V1;
}

static bool ensure_layer_cache(generic_graph_state * s, int32_t layer, ggml_tensor * k, ggml_tensor * v,
                               int64_t head_dim, int64_t n_kv_heads) {
    if (!s || !s->layer_caches || !s->cache_backend || layer < 0 || head_dim <= 0 || n_kv_heads <= 0 || s->n_ctx == 0) {
        if (s) s->last_error = "invalid generic KV cache request";
        return false;
    }
    if (static_cast<size_t>(layer) >= s->layer_caches->size()) s->layer_caches->resize(static_cast<size_t>(layer) + 1);
    auto & lc = (*s->layer_caches)[static_cast<size_t>(layer)];
    if (lc.ctx) {
        if (lc.head_dim != head_dim || lc.n_kv_heads != n_kv_heads || lc.k_type != k->type || lc.v_type != v->type) {
            s->last_error = "generic KV cache tensor shape/type changed across decode calls";
            return false;
        }
        return true;
    }

    ggml_init_params cp{};
    cp.mem_size = 2u * 1024u * 1024u;
    cp.mem_buffer = nullptr;
    cp.no_alloc = true;
    lc.ctx = ggml_init(cp);
    if (!lc.ctx) { s->last_error = "generic KV cache metadata allocation failed"; return false; }
    lc.k = ggml_new_tensor_3d(lc.ctx, k->type, head_dim, n_kv_heads, s->n_ctx);
    lc.v = ggml_new_tensor_3d(lc.ctx, v->type, head_dim, n_kv_heads, s->n_ctx);
    if (!lc.k || !lc.v) { s->last_error = "generic KV cache tensor creation failed"; return false; }
    ggml_set_name(lc.k, "mercan.generic.k_cache");
    ggml_set_name(lc.v, "mercan.generic.v_cache");
    lc.buffer = ggml_backend_alloc_ctx_tensors(lc.ctx, s->cache_backend);
    if (!lc.buffer) { s->last_error = "generic KV cache backend allocation failed"; return false; }
    ggml_backend_buffer_set_usage(lc.buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_clear(lc.buffer, 0);
    lc.head_dim = head_dim;
    lc.n_kv_heads = n_kv_heads;
    lc.k_type = k->type;
    lc.v_type = v->type;
    return true;
}

static mercan_tensor_handle_v1 g_attention(mercan_graph_builder_v1 * b,
                                            mercan_tensor_handle_v1 qh,
                                            mercan_tensor_handle_v1 kh,
                                            mercan_tensor_handle_v1 vh,
                                            mercan_tensor_handle_v1 out_weight_h,
                                            mercan_tensor_handle_v1 out_bias_h,
                                            mercan_tensor_handle_v1 out_scale_h,
                                            float kq_scale,
                                            int32_t layer_index) {
    auto * s = gs(b);
    auto * q = from_handle(qh); auto * k = from_handle(kh); auto * v = from_handle(vh);
    if (!s || !s->ctx || !s->graph || !q || !k || !v || s->attention_mask == MERCAN_TENSOR_NONE_V1) return MERCAN_TENSOR_NONE_V1;

    const bool q3 = q->ne[2] > 1;
    const bool k3 = k->ne[2] > 1;
    const bool v3 = v->ne[2] > 1;
    const int64_t head_dim = q->ne[0];
    const int64_t n_heads = q3 ? q->ne[1] : 1;
    const int64_t n_tokens = q3 ? q->ne[2] : q->ne[1];
    const int64_t n_kv_heads = k3 ? k->ne[1] : 1;
    const int64_t k_tokens = k3 ? k->ne[2] : k->ne[1];
    const int64_t v_kv_heads = v3 ? v->ne[1] : 1;
    const int64_t v_tokens = v3 ? v->ne[2] : v->ne[1];
    if (head_dim <= 0 || n_heads <= 0 || n_tokens <= 0 || k->ne[0] != head_dim || v->ne[0] != head_dim ||
        k_tokens != n_tokens || v_tokens != n_tokens || v_kv_heads != n_kv_heads || n_heads % n_kv_heads != 0) {
        s->last_error = "generic self-attention received incompatible Q/K/V shapes";
        return MERCAN_TENSOR_NONE_V1;
    }
    if (static_cast<uint64_t>(s->n_past) + static_cast<uint64_t>(n_tokens) > s->n_ctx) {
        s->last_error = "generic KV cache capacity exceeded";
        return MERCAN_TENSOR_NONE_V1;
    }
    if (!ensure_layer_cache(s, layer_index, k, v, head_dim, n_kv_heads)) return MERCAN_TENSOR_NONE_V1;
    auto & lc = (*s->layer_caches)[static_cast<size_t>(layer_index)];
    ggml_tensor * mask = from_handle(s->attention_mask);
    if (!mask || mask->ne[0] != static_cast<int64_t>(s->n_past) + n_tokens || mask->ne[1] != n_tokens) {
        s->last_error = "generic self-attention mask shape mismatch";
        return MERCAN_TENSOR_NONE_V1;
    }

    auto current_head = [&](ggml_tensor * t, bool three_d, int64_t h) -> ggml_tensor * {
        if (!three_d) return t;
        return ggml_view_2d(s->ctx, t, head_dim, n_tokens, t->nb[2], static_cast<size_t>(h) * t->nb[1]);
    };
    auto cached_head = [&](ggml_tensor * cache, int64_t h) -> ggml_tensor * {
        return ggml_view_2d(s->ctx, cache, head_dim, s->n_past, cache->nb[2], static_cast<size_t>(h) * cache->nb[1]);
    };

    ggml_tensor * merged = nullptr;
    const int64_t group = n_heads / n_kv_heads;
    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t kvh = h / group;
        ggml_tensor * q_head = current_head(q, q3, h);
        ggml_tensor * k_cur = current_head(k, k3, kvh);
        ggml_tensor * v_cur = current_head(v, v3, kvh);
        ggml_tensor * k_all = k_cur;
        ggml_tensor * v_all = v_cur;
        if (s->n_past > 0) {
            ggml_tensor * kp = cached_head(lc.k, kvh);
            ggml_tensor * vp = cached_head(lc.v, kvh);
            k_all = ggml_concat(s->ctx, kp, k_cur, 1);
            v_all = ggml_concat(s->ctx, vp, v_cur, 1);
        }
        ggml_tensor * scores = ggml_mul_mat(s->ctx, k_all, q_head);
        if (kq_scale != 1.0f) scores = ggml_scale(s->ctx, scores, kq_scale);
        scores = ggml_add(s->ctx, scores, mask);
        ggml_tensor * probs = ggml_soft_max(s->ctx, scores);
        // ggml_mul_mat(A,B) computes A^T*B and requires A itself to be non-transposed.
        // Compute probs^T * V^T -> [n_tokens, head_dim], then transpose to [head_dim, n_tokens].
        ggml_tensor * vt = ggml_cont(s->ctx, ggml_transpose(s->ctx, v_all));
        ggml_tensor * head_rows = ggml_mul_mat(s->ctx, probs, vt);
        ggml_tensor * head_out = ggml_cont(s->ctx, ggml_transpose(s->ctx, head_rows));
        merged = merged ? ggml_concat(s->ctx, merged, head_out, 0) : head_out;
    }

    if (!merged) return MERCAN_TENSOR_NONE_V1;
    // K/V are copied into the persistent runtime cache after graph execution.
    // Mark them as retained graph outputs so the scheduler allocator cannot reuse their buffers first.
    ggml_set_output(k);
    ggml_set_output(v);
    if (auto * w = from_handle(out_weight_h)) merged = ggml_mul_mat(s->ctx, w, merged);
    if (auto * bias = from_handle(out_bias_h)) merged = ggml_add(s->ctx, merged, bias);
    if (auto * scale = from_handle(out_scale_h)) merged = ggml_mul(s->ctx, merged, scale);
    s->updates.push_back({layer_index, k, v});
    return to_handle(merged);
}

static int g_output(mercan_graph_builder_v1 * b, mercan_graph_output_kind_v1 k, mercan_tensor_handle_v1 h) {
    auto * s = gs(b); auto * t = from_handle(h);
    if (!s || !t) return -1;
    if (k == MERCAN_GRAPH_OUTPUT_EMBEDDING_V1) s->embedding = t;
    else if (k == MERCAN_GRAPH_OUTPUT_LOGITS_V1) s->logits = t;
    else if (k == MERCAN_GRAPH_OUTPUT_HIDDEN_V1) s->hidden = t;
    else return -1;
    ggml_set_output(t);
    return 0;
}
static int g_finalize(mercan_graph_builder_v1 * b, mercan_tensor_handle_v1 h) {
    auto * s = gs(b); auto * t = from_handle(h);
    if (!s || !s->graph || !t) return -1;
    ggml_build_forward_expand(s->graph, t);
    return 0;
}
static const mercan_graph_api_v1 GRAPH_API = {
    MERCAN_GRAPH_ABI_VERSION, sizeof(mercan_graph_api_v1), g_dim, g_stride, g_get_rows, g_cast, g_swiglu,
    g_view2, g_mul, g_add, g_concat, g_matmul, g_rms, g_rope, g_attention, g_output, g_finalize
};

static int key_i64(const gguf_context * g, const char * key, int fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return static_cast<int>(gguf_get_val_u32(g, id));
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return static_cast<int>(gguf_get_val_u64(g, id));
        case GGUF_TYPE_INT64:  return static_cast<int>(gguf_get_val_i64(g, id));
        default: return fallback;
    }
}
static int metadata_has(const mercan_metadata_v1 * m, const char * k) {
    auto * g = m ? static_cast<const gguf_context *>(m->userdata) : nullptr;
    return g && k && gguf_find_key(g, k) >= 0;
}
static const char * metadata_str(const mercan_metadata_v1 * m, const char * k) {
    auto * g = m ? static_cast<const gguf_context *>(m->userdata) : nullptr;
    if (!g || !k) return nullptr;
    auto id = gguf_find_key(g, k);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : nullptr;
}
static int64_t metadata_i64(const mercan_metadata_v1 * m, const char * k, int64_t f) {
    auto * g = m ? static_cast<const gguf_context *>(m->userdata) : nullptr;
    return g ? key_i64(g, k, static_cast<int>(f)) : f;
}
static double metadata_f64(const mercan_metadata_v1 * m, const char * k, double f) {
    auto * g = m ? static_cast<const gguf_context *>(m->userdata) : nullptr;
    if (!g || !k) return f;
    auto id = gguf_find_key(g, k);
    if (id < 0) return f;
    auto t = gguf_get_kv_type(g, id);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(g, id);
    if (t == GGUF_TYPE_FLOAT64) return gguf_get_val_f64(g, id);
    return static_cast<double>(key_i64(g, k, static_cast<int>(f)));
}

} // namespace

struct mercan_generic_model {
    const mercan_architecture_v1 * architecture = nullptr;
    gguf_context * gguf = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_fallback = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    generic_tensor_catalog catalog;
    mercan_tensor_resolver_v1 resolver{};
    mercan_metadata_v1 metadata{};
    int32_t vocab_size = 0;
    mercan_token bos = -1, eos = -1, pad = -1;
    int64_t sliding_window = 0;
};

struct mercan_generic_context {
    mercan_generic_model * model = nullptr;
    mercan_context_params params{};
    ggml_backend_sched_t sched = nullptr;
    ggml_context * graph_ctx = nullptr;
    std::vector<float> logits;
    std::vector<generic_layer_cache> layer_caches;
    uint32_t n_past = 0;
};

ggml_backend_dev_t mercan_generic_model_device(const mercan_generic_model * model) {
    return model && model->backend ? ggml_backend_get_device(model->backend) : nullptr;
}

mercan_generic_model * mercan_generic_model_load(const char * path, const mercan_architecture_v1 * architecture,
                                                  mercan_model_params params, std::string & error,
                                                  ggml_backend_dev_t preferred) {
    auto m = std::make_unique<mercan_generic_model>();
    m->architecture = architecture;
    gguf_init_params ip{}; ip.no_alloc = true; ip.ctx = &m->weights_ctx;
    m->gguf = gguf_init_from_file(path, ip);
    if (!m->gguf || !m->weights_ctx) { error = "generic backend failed to read GGUF tensors"; return nullptr; }

    ggml_backend_dev_t dev = nullptr;
    if (params.n_gpu_layers != 0) {
        dev = preferred ? preferred : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) { error = "GPU offload requested but no compatible GPU device is available"; return nullptr; }
    } else dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) { error = "generic backend found no ggml backend device"; return nullptr; }
    m->backend = ggml_backend_dev_init(dev, nullptr);
    if (!m->backend) { error = "generic backend device initialization failed"; return nullptr; }
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) { error = "generic GPU backend requires a CPU fallback device"; return nullptr; }
        m->cpu_fallback = ggml_backend_dev_init(cpu_dev, nullptr);
        if (!m->cpu_fallback) { error = "generic CPU fallback initialization failed"; return nullptr; }
    }
    m->weights_buffer = ggml_backend_alloc_ctx_tensors(m->weights_ctx, m->backend);
    if (!m->weights_buffer) { error = "generic backend failed to allocate model weights"; return nullptr; }
    ggml_backend_buffer_set_usage(m->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = ggml_fopen(path, "rb");
    if (!f) { error = "generic backend could not reopen model data"; return nullptr; }
    const size_t data_off = gguf_get_data_offset(m->gguf);
    std::vector<unsigned char> tmp;
    const int64_t nt = gguf_get_n_tensors(m->gguf);
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(m->gguf, i);
        ggml_tensor * t = ggml_get_tensor(m->weights_ctx, name);
        if (!t) { std::fclose(f); error = std::string("generic tensor metadata missing: ") + name; return nullptr; }
        const size_t sz = gguf_get_tensor_size(m->gguf, i);
        tmp.resize(sz);
        if (std::fseek(f, static_cast<long>(data_off + gguf_get_tensor_offset(m->gguf, i)), SEEK_SET) != 0 ||
            std::fread(tmp.data(), 1, sz, f) != sz) {
            std::fclose(f); error = std::string("generic tensor read failed: ") + name; return nullptr;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, sz);
        m->catalog.tensors.emplace(name, t);
    }
    std::fclose(f);
    ggml_backend_synchronize(m->backend);

    m->resolver = {MERCAN_TENSOR_ABI_VERSION, sizeof(mercan_tensor_resolver_v1), &m->catalog, &TENSOR_API};
    m->metadata = {MERCAN_ARCH_ABI_VERSION, sizeof(mercan_metadata_v1), m->gguf, metadata_has, metadata_str, metadata_i64, metadata_f64};
    ggml_tensor * te = ggml_get_tensor(m->weights_ctx, "token_embd.weight");
    if (!te || te->ne[1] <= 0) { error = "generic backend requires token_embd.weight"; return nullptr; }
    m->vocab_size = static_cast<int32_t>(te->ne[1]);
    m->bos = key_i64(m->gguf, "mercan.tokenizer.bos_token_id", -1);
    m->eos = key_i64(m->gguf, "mercan.tokenizer.eos_token_id", -1);
    m->pad = key_i64(m->gguf, "mercan.tokenizer.pad_token_id", -1);
    m->sliding_window = key_i64(m->gguf, "mercan.attention.sliding_window", 0);
    return m.release();
}

void mercan_generic_model_free(mercan_generic_model * m) {
    if (!m) return;
    if (m->weights_buffer) ggml_backend_buffer_free(m->weights_buffer);
    if (m->cpu_fallback) ggml_backend_free(m->cpu_fallback);
    if (m->backend) ggml_backend_free(m->backend);
    if (m->gguf) gguf_free(m->gguf);
    if (m->weights_ctx) ggml_free(m->weights_ctx);
    delete m;
}

mercan_generic_context * mercan_generic_context_create(mercan_generic_model * m, mercan_context_params p, std::string & error) {
    if (!m || !m->backend || p.n_ctx == 0) { error = "invalid generic model/context parameters"; return nullptr; }
    auto * c = new mercan_generic_context();
    c->model = m; c->params = p;
    ggml_backend_t bs[2] = {m->backend, m->cpu_fallback};
    const int nb = m->cpu_fallback ? 2 : 1;
    c->sched = ggml_backend_sched_new(bs, nullptr, nb, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!c->sched) { delete c; error = "generic scheduler creation failed"; return nullptr; }
    return c;
}

void mercan_generic_context_free(mercan_generic_context * c) {
    if (!c) return;
    if (c->sched) { ggml_backend_sched_reset(c->sched); ggml_backend_sched_free(c->sched); }
    if (c->graph_ctx) ggml_free(c->graph_ctx);
    for (auto & lc : c->layer_caches) {
        if (lc.buffer) ggml_backend_buffer_free(lc.buffer);
        if (lc.ctx) ggml_free(lc.ctx);
    }
    delete c;
}

bool mercan_generic_context_truncate(mercan_generic_context * c, uint32_t keep) {
    if (!c || keep > c->n_past) return false;
    // Cache positions beyond n_past are ignored and overwritten on the next decode.
    c->n_past = keep;
    c->logits.clear();
    return true;
}

int32_t mercan_generic_decode(mercan_generic_context * c, const mercan_token * tokens, int32_t n, std::string & error) {
    if (!c || !tokens || n <= 0) { error = "invalid generic decode arguments"; return -1; }
    if (static_cast<uint64_t>(c->n_past) + static_cast<uint64_t>(n) > c->params.n_ctx) {
        error = "generic decode exceeds context capacity"; return -1;
    }
    if (c->graph_ctx) {
        ggml_backend_sched_reset(c->sched);
        ggml_free(c->graph_ctx);
        c->graph_ctx = nullptr;
    }
    ggml_init_params gp{}; gp.mem_size = 64u * 1024u * 1024u; gp.mem_buffer = nullptr; gp.no_alloc = true;
    c->graph_ctx = ggml_init(gp);
    if (!c->graph_ctx) { error = "generic graph metadata allocation failed"; return -1; }
    generic_graph_state st{};
    st.ctx = c->graph_ctx;
    st.graph = ggml_new_graph_custom(c->graph_ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    st.cache_backend = c->model->backend;
    st.n_past = c->n_past;
    st.n_ctx = c->params.n_ctx;
    st.n_tokens = static_cast<uint32_t>(n);
    st.layer_caches = &c->layer_caches;
    if (!st.graph) { error = "generic graph creation failed"; return -1; }

    const int64_t total = static_cast<int64_t>(c->n_past) + n;
    ggml_init_params ip{}; ip.mem_size = 2u * 1024u * 1024u; ip.mem_buffer = nullptr; ip.no_alloc = true;
    ggml_context * input_ctx = ggml_init(ip);
    if (!input_ctx) { error = "generic input metadata allocation failed"; return -1; }
    ggml_tensor * inp = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n); ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, total, n); ggml_set_input(mask);
    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx, c->model->backend);
    if (!input_buffer) { ggml_free(input_ctx); error = "generic input backend allocation failed"; return -1; }
    auto cleanup_inputs = [&]() { ggml_backend_buffer_free(input_buffer); ggml_free(input_ctx); };

    ggml_backend_tensor_set(inp, tokens, 0, sizeof(mercan_token) * static_cast<size_t>(n));
    std::vector<int32_t> positions(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(c->n_past) + i;
    ggml_backend_tensor_set(pos, positions.data(), 0, sizeof(int32_t) * static_cast<size_t>(n));

    std::vector<float> mask_data(static_cast<size_t>(total) * static_cast<size_t>(n));
    const int64_t window = c->model->sliding_window;
    for (int32_t q = 0; q < n; ++q) {
        const int64_t abs_q = static_cast<int64_t>(c->n_past) + q;
        const int64_t min_k = window > 0 ? std::max<int64_t>(0, abs_q - window + 1) : 0;
        for (int64_t key = 0; key < total; ++key) {
            const bool allowed = key >= min_k && key <= abs_q;
            mask_data[static_cast<size_t>(q) * static_cast<size_t>(total) + static_cast<size_t>(key)] =
                allowed ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, sizeof(float) * mask_data.size());

    generic_kv_batch kvb{};
    kvb.positions = to_handle(pos);
    kvb.mask = to_handle(mask);
    kvb.window = window;
    mercan_kv_resolver_v1 kvr{MERCAN_KV_ABI_VERSION, sizeof(mercan_kv_resolver_v1), &kvb, &KV_API};
    st.attention_mask = to_handle(mask);

    mercan_graph_builder_v1 b{MERCAN_GRAPH_ABI_VERSION, sizeof(mercan_graph_builder_v1), &st, &GRAPH_API, &c->model->resolver, &kvr};
    mercan_arch_graph_invocation_v1 inv{};
    inv.abi_version = MERCAN_ARCH_GRAPH_INVOCATION_ABI_VERSION;
    inv.struct_size = sizeof(inv);
    inv.metadata = &c->model->metadata;
    inv.builder = &b;
    inv.input_tokens = to_handle(inp);
    inv.input_positions = to_handle(pos);
    inv.output_ids = MERCAN_TENSOR_NONE_V1;
    inv.n_tokens = n;
    inv.n_outputs = n;
    char e[512] = {};
    if (mercan_arch_build_graph_v1(c->model->architecture, &inv, e, sizeof(e)) != 0) {
        error = std::string("external graph build failed: ") + (e[0] ? e : (st.last_error.empty() ? "unknown error" : st.last_error));
        cleanup_inputs(); return -1;
    }
    if (!st.logits || !st.graph) { error = "external architecture did not publish logits/finalize graph"; cleanup_inputs(); return -1; }
    if (st.logits->type != GGML_TYPE_F32) { error = "generic backend currently requires F32 logits"; cleanup_inputs(); return -1; }
    if (!ggml_backend_sched_alloc_graph(c->sched, st.graph)) { error = "generic scheduler graph allocation failed"; cleanup_inputs(); return -1; }
    enum ggml_status rc = ggml_backend_sched_graph_compute(c->sched, st.graph);
    if (rc != GGML_STATUS_SUCCESS) { error = std::string("generic graph compute failed: ") + ggml_status_to_string(rc); cleanup_inputs(); return -1; }
    ggml_backend_sched_synchronize(c->sched);

    for (const auto & u : st.updates) {
        if (u.layer < 0 || static_cast<size_t>(u.layer) >= c->layer_caches.size() || !u.k || !u.v) {
            error = "generic KV cache update metadata is invalid"; cleanup_inputs(); return -1;
        }
        auto & lc = c->layer_caches[static_cast<size_t>(u.layer)];
        const size_t kb = ggml_nbytes(u.k);
        const size_t vb = ggml_nbytes(u.v);
        std::vector<unsigned char> ktmp(kb), vtmp(vb);
        ggml_backend_tensor_get(u.k, ktmp.data(), 0, kb);
        ggml_backend_tensor_get(u.v, vtmp.data(), 0, vb);
        const size_t koff = static_cast<size_t>(c->n_past) * lc.k->nb[2];
        const size_t voff = static_cast<size_t>(c->n_past) * lc.v->nb[2];
        ggml_backend_tensor_set(lc.k, ktmp.data(), koff, kb);
        ggml_backend_tensor_set(lc.v, vtmp.data(), voff, vb);
    }
    ggml_backend_synchronize(c->model->backend);

    const int64_t vocab = st.logits->ne[0];
    if (vocab <= 0) { error = "generic logits have invalid shape"; cleanup_inputs(); return -1; }
    c->logits.resize(static_cast<size_t>(vocab));
    const size_t off = static_cast<size_t>(std::max<int64_t>(1, st.logits->ne[1]) - 1) * st.logits->nb[1];
    ggml_backend_tensor_get(st.logits, c->logits.data(), off, sizeof(float) * static_cast<size_t>(vocab));
    c->n_past += static_cast<uint32_t>(n);
    cleanup_inputs();
    return 0;
}

const float * mercan_generic_logits(mercan_generic_context * c) {
    return c && !c->logits.empty() ? c->logits.data() : nullptr;
}
int32_t mercan_generic_vocab_size(const mercan_generic_model * m) { return m ? m->vocab_size : 0; }
uint32_t mercan_generic_context_size(const mercan_generic_context * c) { return c ? c->params.n_ctx : 0; }
mercan_token mercan_generic_bos_token(const mercan_generic_model * m) { return m ? m->bos : -1; }
mercan_token mercan_generic_eos_token(const mercan_generic_model * m) { return m ? m->eos : -1; }
mercan_token mercan_generic_pad_token(const mercan_generic_model * m) { return m ? m->pad : -1; }
