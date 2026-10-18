#include "mercan.h"
#include "generation_sampling.hpp"
#include <atomic>
#include <climits>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

void mercan_generation_set_error(const std::string &);

struct mercan_generation {
    mercan_model * model = nullptr;
    mercan_context * context = nullptr;
    mercan_context_params context_params{};
    mercan_generation_params_v1 params{};
    std::vector<mercan_token> prompt;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};
    std::atomic<int> finish{MERCAN_GENERATION_PENDING};
    ~mercan_generation() { mercan_context_free(context); }
};

namespace {
std::runtime_error native_error(const char * fallback) {
    const char * e = mercan_last_error();
    return std::runtime_error(e && *e ? e : fallback);
}
mercan_generation_params_v1 defaults() {
    return {sizeof(mercan_generation_params_v1), 256, 0.7f, 40, 0.9f, 1.15f, 64, 0, 0};
}
void validate(const mercan_generation_params_v1 & p, mercan_context_params cp) {
    if (p.struct_size < sizeof(p)) throw std::invalid_argument("generation params struct_size is too small");
    if (p.max_tokens <= 0 || p.top_k <= 0 || p.repeat_last_n < 0 || p.use_seed > 1 ||
        !std::isfinite(p.temperature) || p.temperature < 0 ||
        !std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1 ||
        !std::isfinite(p.repeat_penalty) || p.repeat_penalty <= 0)
        throw std::invalid_argument("invalid generation sampling parameters");
    if (!cp.n_ctx || !cp.n_batch || cp.n_ctx > INT32_MAX || cp.n_batch > INT32_MAX ||
        cp.n_threads < 0 || cp.n_threads_batch < 0)
        throw std::invalid_argument("invalid generation context parameters");
}
std::vector<mercan_token> tokenize(mercan_model * m, const char * text, size_t size) {
    if (!text || !size || size > INT32_MAX) throw std::invalid_argument("prompt must contain 1..INT32_MAX bytes");
    int32_t n = mercan_tokenize(m, text, size, false, true, nullptr, 0);
    if (!n || n == INT32_MIN) throw native_error("prompt tokenization failed");
    size_t capacity = static_cast<size_t>(n < 0 ? -n : n);
    for (int retry = 0; retry < 3; ++retry) {
        std::vector<mercan_token> out(capacity);
        n = mercan_tokenize(m, text, size, false, true, out.data(), static_cast<int32_t>(capacity));
        if (n > 0 && static_cast<size_t>(n) <= capacity) { out.resize(n); return out; }
        if (n >= 0 || n == INT32_MIN) break;
        capacity = static_cast<size_t>(-n);
    }
    throw native_error("prompt tokenization failed");
}
mercan_token special(mercan_model * m, const char * marker) {
    mercan_token t = -1;
    return mercan_tokenize(m, marker, std::strlen(marker), false, true, &t, 1) == 1 ? t : -1;
}
std::string token_piece(mercan_model * m, mercan_token token) {
    char local[64];
    int32_t n = mercan_token_to_piece(m, token, local, sizeof(local), false);
    if (n >= 0 && n <= static_cast<int32_t>(sizeof(local))) return std::string(local, n);
    if (n < 0 && n != INT32_MIN) {
        std::string out(static_cast<size_t>(-n), '\0');
        n = mercan_token_to_piece(m, token, &out[0], static_cast<int32_t>(out.size()), false);
        if (n >= 0 && static_cast<size_t>(n) <= out.size()) { out.resize(n); return out; }
    }
    throw native_error("token_to_piece failed");
}
std::string header(const char * name, const char * fallback) {
    std::string s = name ? name : fallback;
    if (s.empty() || s.find_first_of("\r\n") != std::string::npos ||
        s.find("<|") != std::string::npos || s.find("|>") != std::string::npos ||
        s.find_first_not_of(" \t") == std::string::npos)
        throw std::invalid_argument("invalid chat role header");
    return s;
}
std::string format_chat(const mercan_chat_message_v1 * messages, size_t count,
                        const char * system, const char * user, const char * assistant) {
    if (!messages || !count) throw std::invalid_argument("at least one chat message is required");
    const std::string sh = header(system, "sistem"), uh = header(user, "kullanici"), ah = header(assistant, "asistan");
    std::string out;
    for (size_t i = 0; i < count; ++i) {
        const auto & message = messages[i];
        if (!message.role || (!message.content && message.content_len)) throw std::invalid_argument("invalid chat message");
        std::string role;
        if (std::strcmp(message.role, "system") == 0) role = sh;
        else if (std::strcmp(message.role, "user") == 0) role = uh;
        else if (std::strcmp(message.role, "assistant") == 0) role = ah;
        else throw std::invalid_argument("unsupported chat role");
        if (message.content_len > INT32_MAX) throw std::invalid_argument("chat message is too long");
        std::string content(message.content ? message.content : "", message.content_len);
        size_t pos = 0;
        while ((pos = content.find("<|", pos)) != std::string::npos) {
            content.replace(pos, 2, "<\xe2\x80\x8b|"); pos += 5;
        }
        out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        if (out.size() > INT32_MAX) throw std::invalid_argument("chat prompt is too long");
    }
    return out + "<|im_start|>" + ah + "\n";
}
}

extern "C" {
uint32_t mercan_generation_api_version(void) { return MERCAN_GENERATION_API_VERSION; }
size_t mercan_generation_params_size_v1(void) { return sizeof(mercan_generation_params_v1); }
int mercan_generation_default_params_v1(mercan_generation_params_v1 * out, size_t size) {
    mercan_generation_set_error("");
    if (!out || size < sizeof(*out)) { mercan_generation_set_error("generation params buffer is too small"); return -1; }
    *out = defaults(); return 0;
}

mercan_generation * mercan_generate_begin_v1(
    mercan_model * model, const char * prompt, size_t length, mercan_context_params cp,
    const mercan_generation_params_v1 * params) {
    mercan_generation_set_error("");
    try {
        if (!model) throw std::invalid_argument("invalid model");
        if (params && params->struct_size < sizeof(*params)) throw std::invalid_argument("generation params struct_size is too small");
        const auto p = params ? *params : defaults();
        validate(p, cp);
        auto g = std::make_unique<mercan_generation>();
        g->model = model; g->params = p; g->context_params = cp;
        g->prompt = tokenize(model, prompt, length);
        if (g->prompt.size() + static_cast<size_t>(p.max_tokens) > cp.n_ctx)
            throw std::invalid_argument("prompt + max_tokens exceeds n_ctx");
        g->context = mercan_context_create(model, cp);
        if (!g->context) throw native_error("context creation failed");
        if (g->prompt.size() + static_cast<size_t>(p.max_tokens) > mercan_context_size(g->context))
            throw std::invalid_argument("prompt + max_tokens exceeds native context size");
        return g.release();
    } catch (const std::exception & e) { mercan_generation_set_error(e.what()); }
    catch (...) { mercan_generation_set_error("unknown generation initialization error"); }
    return nullptr;
}

mercan_generation * mercan_chat_begin_v1(
    mercan_model * model, const mercan_chat_message_v1 * messages, size_t count,
    const char * system, const char * user, const char * assistant,
    mercan_context_params cp, const mercan_generation_params_v1 * params) {
    mercan_generation_set_error("");
    try {
        const auto prompt = format_chat(messages, count, system, user, assistant);
        return mercan_generate_begin_v1(model, prompt.data(), prompt.size(), cp, params);
    } catch (const std::exception & e) { mercan_generation_set_error(e.what()); }
    catch (...) { mercan_generation_set_error("unknown chat initialization error"); }
    return nullptr;
}

int mercan_generation_run_v1(mercan_generation * g, mercan_generation_callback_v1 callback, void * user) {
    mercan_generation_set_error("");
    if (!g) { mercan_generation_set_error("invalid generation handle"); return -1; }
    if (g->started.exchange(true)) { mercan_generation_set_error("generation handle has already run"); return -1; }
    try {
        const auto cancelled = [&]() {
            if (!g->cancelled.load()) return false;
            g->finish = MERCAN_GENERATION_CANCELLED; return true;
        };
        for (size_t i = 0; i < g->prompt.size();) {
            if (cancelled()) return 1;
            const int32_t n = static_cast<int32_t>(std::min<size_t>(g->context_params.n_batch, g->prompt.size() - i));
            if (mercan_decode(g->context, g->prompt.data() + i, n) != 0) throw native_error("prompt decode failed");
            i += n;
        }
        std::vector<mercan_token> stops;
        for (auto t : {mercan_eos_token(g->model), mercan_pad_token(g->model),
                       special(g->model, "<|im_start|>"), special(g->model, "<|im_end|>")})
            if (t >= 0) stops.push_back(t);
        const auto & p = g->params;
        std::mt19937 rng(p.use_seed ? p.seed : std::random_device{}());
        std::vector<mercan_token> recent; // generated output only, matching CLI policy
        const int32_t vocab = mercan_vocab_size(g->model);
        for (int32_t i = 0; i < p.max_tokens; ++i) {
            if (cancelled()) return 1;
            auto token = mercan_generation_detail::sample(mercan_logits(g->context), vocab, p, recent, stops, rng);
            if (std::find(stops.begin(), stops.end(), token) != stops.end()) { g->finish = MERCAN_GENERATION_STOP; return 0; }
            const auto piece = token_piece(g->model, token);
            if (callback && !callback(piece.data(), piece.size(), user)) { g->cancelled = true; cancelled(); return 1; }
            if (cancelled()) return 1;
            if (p.repeat_last_n) {
                recent.push_back(token);
                if (recent.size() > static_cast<size_t>(p.repeat_last_n)) recent.erase(recent.begin());
            }
            if (i + 1 < p.max_tokens && mercan_decode(g->context, &token, 1) != 0) throw native_error("generation decode failed");
        }
        g->finish = MERCAN_GENERATION_LIMIT; return 0;
    } catch (const std::exception & e) { mercan_generation_set_error(e.what()); }
    catch (...) { mercan_generation_set_error("unknown generation error"); }
    g->finish = MERCAN_GENERATION_ERROR; return -1;
}
void mercan_generation_cancel_v1(mercan_generation * g) { if (g) g->cancelled = true; }
int mercan_generation_finish_reason_v1(const mercan_generation * g) { return g ? g->finish.load() : MERCAN_GENERATION_ERROR; }
void mercan_generation_free_v1(mercan_generation * g) { delete g; }
}
