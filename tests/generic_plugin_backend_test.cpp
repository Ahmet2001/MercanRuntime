#include "mercan.h"
#include "mercan_plugin.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

static void fill_zero(ggml_tensor * t) {
    std::fill_n(static_cast<float *>(t->data), ggml_nelements(t), 0.0f);
}
static void fill_one(ggml_tensor * t) {
    std::fill_n(static_cast<float *>(t->data), ggml_nelements(t), 1.0f);
}
static void fill_identity(ggml_tensor * t) {
    fill_zero(t);
    float * p = static_cast<float *>(t->data);
    const int64_t n = std::min(t->ne[0], t->ne[1]);
    for (int64_t i = 0; i < n; ++i) p[i * t->ne[0] + i] = 1.0f;
}

static bool make_model(const char * path, int history_token = 63) {
    constexpr int D = 4;
    constexpr int FF = 4;
    constexpr int VOCAB = 256;
    constexpr int CTX = 16;
    ggml_init_params p{}; p.mem_size = 8 * 1024 * 1024; p.no_alloc = false;
    ggml_context * c = ggml_init(p); if (!c) return false;

    auto t1 = [&](const char * name, int64_t n0) {
        ggml_tensor * t = ggml_new_tensor_1d(c, GGML_TYPE_F32, n0); ggml_set_name(t, name); return t;
    };
    auto t2 = [&](const char * name, int64_t n0, int64_t n1) {
        ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, n0, n1); ggml_set_name(t, name); return t;
    };

    ggml_tensor * emb = t2("token_embd.weight", D, VOCAB); fill_zero(emb);
    float * e = static_cast<float *>(emb->data);
    e[65 * D + 0] = 1.0f; // A
    e[66 * D + 1] = 1.0f; // B

    ggml_tensor * pos = t2("position_embd.weight", D, CTX); fill_zero(pos);
    ggml_tensor * an = t1("blk.0.attn_norm.weight", D); fill_one(an);
    ggml_tensor * wq = t2("blk.0.attn_q.weight", D, D); fill_identity(wq);
    ggml_tensor * wk = t2("blk.0.attn_k.weight", D, D); fill_zero(wk);
    float * kp = static_cast<float *>(wk->data);
    kp[0 * D + 1] = 1.0f; // swap dims 0/1; symmetric so transpose convention is irrelevant
    kp[1 * D + 0] = 1.0f;
    kp[2 * D + 2] = 1.0f;
    kp[3 * D + 3] = 1.0f;
    ggml_tensor * wv = t2("blk.0.attn_v.weight", D, D); fill_identity(wv);
    ggml_tensor * wo = t2("blk.0.attn_output.weight", D, D); fill_zero(wo);
    float * op = static_cast<float *>(wo->data);
    op[0 * D + 0] = 4.0f;
    op[1 * D + 1] = 1.0f;
    op[2 * D + 2] = 1.0f;
    op[3 * D + 3] = 1.0f;

    ggml_tensor * fn = t1("blk.0.ffn_norm.weight", D); fill_one(fn);
    ggml_tensor * gate = t2("blk.0.ffn_gate.weight", D, FF); fill_zero(gate);
    ggml_tensor * up = t2("blk.0.ffn_up.weight", D, FF); fill_zero(up);
    ggml_tensor * down = t2("blk.0.ffn_down.weight", FF, D); fill_zero(down);
    ggml_tensor * on = t1("output_norm.weight", D); fill_one(on);
    ggml_tensor * out = t2("output.weight", D, VOCAB); fill_zero(out);
    float * outp = static_cast<float *>(out->data);
    outp[history_token * D + 0] = 10.0f; // history-sensitive path, '?' or EOS
    outp[33 * D + 1] = 10.0f; // B-alone path => '!'

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "mercan.format", "mercan");
    gguf_set_val_i64(g, "mercan.format_version", 1);
    gguf_set_val_i64(g, "mercan.runtime_abi", 1);
    gguf_set_val_str(g, "general.architecture", "anka");
    gguf_set_val_str(g, "mercan.tokenizer.type", "anka-byte");
    gguf_set_val_i64(g, "mercan.tokenizer.eos_token_id", 255);
    gguf_set_val_i64(g, "anka.block_count", 1);
    gguf_set_val_f32(g, "anka.rms_norm_eps", 1e-5f);
    gguf_set_val_f32(g, "anka.attention.scale", 0.5f);

    gguf_add_tensor(g, emb); gguf_add_tensor(g, pos); gguf_add_tensor(g, an);
    gguf_add_tensor(g, wq); gguf_add_tensor(g, wk); gguf_add_tensor(g, wv); gguf_add_tensor(g, wo);
    gguf_add_tensor(g, fn); gguf_add_tensor(g, gate); gguf_add_tensor(g, up); gguf_add_tensor(g, down);
    gguf_add_tensor(g, on); gguf_add_tensor(g, out);
    const bool ok = gguf_write_to_file(g, path, false);
    gguf_free(g); ggml_free(c); return ok;
}

static int best_token(const float * logits, int vocab) {
    return logits ? static_cast<int>(std::max_element(logits, logits + vocab) - logits) : -1;
}

int main(int argc, char ** argv) {
    auto fail = [](const char * m) { std::cerr << "FAIL: " << m << "\n"; return 10; };
    if (argc != 3 && argc != 4) return fail("usage");
    if (!make_model(argv[2])) return fail("model write");
    const int prc = mercan_plugin_load_v1(argv[1]);
    if (prc != 0) { std::cerr << mercan_plugin_last_error_v1() << "\n"; return fail("plugin load"); }
    mercan_backend_init();

    auto mp = mercan_model_default_params();
    mp.n_gpu_layers = (argc == 4 && std::strcmp(argv[3], "gpu") == 0) ? -1 : 0;
    if (mp.n_gpu_layers && !mercan_cuda_available_v1()) {
        std::cout << "No usable CUDA device; skipping GPU execution test\n";
        mercan_backend_free();
        return 77;
    }
    mercan_model * m = mercan_model_load(argv[2], mp);
    if (m) {
        const char * actual = mercan_model_device_v1(m);
        const bool on_cpu = std::strcmp(actual, "cpu") == 0;
        if (on_cpu != (mp.n_gpu_layers == 0) || !mercan_model_device_json_v1(m)) return 51;
    }
    if (!m) { std::cerr << mercan_last_error() << "\n"; return fail("model load"); }
    if (std::strcmp(mercan_model_architecture(m), "anka") != 0) return fail("architecture");
    if (std::strcmp(mercan_model_tokenizer(m), "anka-byte") != 0) return fail("tokenizer");
    if (mercan_vocab_size(m) != 256) return fail("vocab");

    mercan_token toks[8] = {};
    const int32_t n = mercan_tokenize(m, "AB", 2, false, false, toks, 8);
    if (n != 2 || toks[0] != 65 || toks[1] != 66) return fail("encode");
    auto cp = mercan_context_default_params(); cp.n_ctx = 16; cp.n_batch = 8; cp.n_threads = 2; cp.n_threads_batch = 2;

    mercan_context * ctx = mercan_context_create(m, cp);
    if (!ctx) { std::cerr << mercan_last_error() << "\n"; return fail("context"); }
    if (mercan_decode(ctx, toks, 2) != 0) { std::cerr << mercan_last_error() << "\n"; return fail("prefill"); }
    const int prefill_best = best_token(mercan_logits(ctx), 256);
    if (prefill_best != 63) { std::cerr << "prefill best=" << prefill_best << "\n"; return fail("prefill attention prediction"); }

    const mercan_token b = 66;
    if (mercan_decode(ctx, &b, 1) != 0) { std::cerr << mercan_last_error() << "\n"; return fail("cached decode"); }
    const int cached_best = best_token(mercan_logits(ctx), 256);
    if (cached_best != 63) { std::cerr << "cached best=" << cached_best << "\n"; return fail("KV cache reuse prediction"); }

    mercan_context * fresh = mercan_context_create(m, cp);
    if (!fresh) return fail("fresh context");
    if (mercan_decode(fresh, &b, 1) != 0) { std::cerr << mercan_last_error() << "\n"; return fail("fresh B decode"); }
    const int fresh_best = best_token(mercan_logits(fresh), 256);
    if (fresh_best != 33) { std::cerr << "fresh best=" << fresh_best << "\n"; return fail("fresh-context prediction"); }
    if (fresh_best == cached_best) return fail("KV history did not affect logits");

    char ch = 0;
    if (mercan_token_to_piece(m, cached_best, &ch, 1, false) != 1 || ch != '?') return fail("decode piece");
    mercan_context_free(fresh);
    mercan_context_free(ctx);

    // Verify the high-level native loop against the tiny real plugin model.
    mercan_generation_params_v1 gp{};
    mercan_generation_default_params_v1(&gp, sizeof(gp));
    gp.max_tokens = 2; gp.temperature = 0; gp.repeat_penalty = 1;
    auto collect = [](const char * bytes, size_t size, void * data) -> int {
        static_cast<std::string *>(data)->append(bytes, size); return 1;
    };
    auto * request = mercan_generate_begin_v1(m, "AB", 2, cp, &gp);
    if (!request) { std::cerr << mercan_last_error() << "\n"; return fail("generation begin"); }
    std::string generated;
    if (mercan_generation_run_v1(request, collect, &generated) != 0 || generated != "??") return fail("native generation");
    if (mercan_generation_finish_reason_v1(request) != MERCAN_GENERATION_LIMIT) return fail("limit reason");
    if (mercan_generation_run_v1(request, collect, &generated) != -1) return fail("repeated generation run");
    mercan_generation_free_v1(request);

    request = mercan_generate_begin_v1(m, "AB", 2, cp, &gp);
    if (!request) return fail("cancellation begin");
    mercan_generation_cancel_v1(request);
    generated.clear();
    if (mercan_generation_run_v1(request, collect, &generated) != 1 || !generated.empty()) return fail("pre-cancel");
    if (mercan_generation_finish_reason_v1(request) != MERCAN_GENERATION_CANCELLED) return fail("cancel reason");
    mercan_generation_free_v1(request);

    request = mercan_generate_begin_v1(m, "AB", 2, cp, &gp);
    if (!request) return fail("callback cancellation begin");
    auto cancel = [](const char *, size_t, void *) -> int { return 0; };
    if (mercan_generation_run_v1(request, cancel, nullptr) != 1) return fail("callback cancellation");
    mercan_generation_free_v1(request);
    gp.max_tokens = 100;
    if (mercan_generate_begin_v1(m, "AB", 2, cp, &gp)) return fail("context overflow accepted");

    const std::string stop_path = std::string(argv[2]) + ".stop.mercan";
    if (!make_model(stop_path.c_str(), 255)) return fail("stop model write");
    auto * stop_model = mercan_model_load(stop_path.c_str(), mp);
    if (!stop_model) return fail("stop model load");
    gp.max_tokens = 2;
    request = mercan_generate_begin_v1(stop_model, "AB", 2, cp, &gp);
    if (!request) return fail("stop generation begin");
    generated.clear();
    if (mercan_generation_run_v1(request, collect, &generated) != 0 || !generated.empty()) return fail("stop token emitted");
    if (mercan_generation_finish_reason_v1(request) != MERCAN_GENERATION_STOP) return fail("stop reason");
    mercan_generation_free_v1(request);
    mercan_model_free(stop_model);
    std::remove(stop_path.c_str());

    auto * session = mercan_session_create_v1(m, cp);
    if (!session) return fail("session create");
    auto session_turn = [&](const char * prompt) {
        auto * turn = mercan_session_generate_begin_v1(session, prompt, std::strlen(prompt), &gp);
        if (!turn) return false;
        std::string cached, cold;
        bool ok = mercan_generation_run_v1(turn, collect, &cached) == 0;
        ok = ok && mercan_generation_commit_session_v1(turn) == 0;
        mercan_generation_free_v1(turn);
        auto * baseline = mercan_generate_begin_v1(m, prompt, std::strlen(prompt), cp, &gp);
        if (!baseline) return false;
        ok = ok && mercan_generation_run_v1(baseline, collect, &cold) == 0 && cached == cold;
        mercan_generation_free_v1(baseline);
        return ok;
    };
    if (!session_turn("AB")) return fail("first cached turn");
    request = mercan_session_generate_begin_v1(session, "AB??B", 5, &gp);
    if (!request) return fail("session pre-cancel begin");
    mercan_generation_cancel_v1(request);
    if (mercan_generation_run_v1(request, collect, &generated) != 1) return fail("session pre-cancel");
    mercan_generation_free_v1(request);
    if (std::string(mercan_session_stats_json_v1(session)).find("\"cached_tokens\":3") == std::string::npos)
        return fail("untouched cache discarded");
    if (!session_turn("AB??B")) return fail("continued cached turn");
    std::string stats = mercan_session_stats_json_v1(session);
    if (stats.find("\"reused_tokens\":3") == std::string::npos ||
        stats.find("\"context_allocations\":1") == std::string::npos) return fail("prefix was not reused");
    if (!session_turn("BB")) return fail("changed prefix differs from cold output");
    stats = mercan_session_stats_json_v1(session);
    if (stats.find("\"reused_tokens\":0") == std::string::npos ||
        stats.find("\"context_allocations\":1") == std::string::npos) return fail("prefix truncation recreated context");
    request = mercan_session_generate_begin_v1(session, "BB!B", 4, &gp);
    if (!request) return fail("session cancelled turn begin");
    if (mercan_session_reset_v1(session) != -1 || mercan_session_free_v1(session) != -1 ||
        mercan_session_generate_begin_v1(session, "B", 1, &gp)) return fail("busy session accepted operation");
    if (mercan_generation_run_v1(request, cancel, nullptr) != 1 ||
        mercan_generation_commit_session_v1(request) != -1) return fail("session cancellation committed");
    mercan_generation_free_v1(request);
    stats = mercan_session_stats_json_v1(session);
    if (stats.find("\"cached_tokens\":0") == std::string::npos) return fail("cancelled cache retained");
    if (!session_turn("AB")) return fail("session restart after cancellation");
    if (mercan_session_reset_v1(session) != 0) return fail("session reset");
    stats = mercan_session_stats_json_v1(session);
    if (stats.find("\"cached_tokens\":0") == std::string::npos) return fail("reset retained tokens");
    if (!session_turn("AB")) return fail("session resume after reset");
    request = mercan_session_generate_begin_v1(session, "AB??B", 5, &gp);
    if (!request || mercan_generation_run_v1(request, collect, &generated) != 0) return fail("uncommitted session turn");
    mercan_generation_free_v1(request);
    if (std::string(mercan_session_stats_json_v1(session)).find("\"cached_tokens\":0") == std::string::npos)
        return fail("uncommitted cache retained");
    if (mercan_session_free_v1(session) != 0) return fail("session free");
    mercan_model_free(m);
    mercan_backend_free();
    std::remove(argv[2]);
    std::cout << "GENERIC_TRANSFORMER_KV_OK prefill=" << prefill_best
              << " cached=" << cached_best << " fresh=" << fresh_best
              << " backend=" << (mp.n_gpu_layers != 0 ? "gpu" : "cpu") << "\n";
    return 0;
}
