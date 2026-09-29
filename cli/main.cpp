#include "mercan.h"
#include "mercan_arch.h"
#include "mercan_graph.h"
#include "mercan_kv.h"
#include "mercan_plugin.h"
#include "mercan_tensor.h"
#include "mercan_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static constexpr const char * VERSION = "0.1.2";

static void die(const std::string & message) {
    std::cerr << "mercan: " << message << "\n";
    std::exit(1);
}

static std::string cache_root() {
    if (const char * p = std::getenv("MERCAN_CACHE")) return p;
    if (const char * p = std::getenv("XDG_CACHE_HOME")) return (fs::path(p) / "mercan").string();
    if (const char * p = std::getenv("HOME")) return (fs::path(p) / ".cache" / "mercan").string();
    return ".mercan-cache";
}

static bool safe_hf_component(const std::string & s, bool allow_slash) {
    if (s.empty()) return false;
    for (unsigned char c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') continue;
        if (allow_slash && c == '/') continue;
        return false;
    }
    return true;
}

struct hf_spec {
    std::string repo;
    std::string filename = "model.mercan";
};

static hf_spec parse_hf_spec(const std::string & input) {
    hf_spec out;
    const auto colon = input.find(':');
    out.repo = colon == std::string::npos ? input : input.substr(0, colon);
    if (colon != std::string::npos) out.filename = input.substr(colon + 1);

    const auto slash = out.repo.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= out.repo.size() ||
        out.repo.find('/', slash + 1) != std::string::npos) {
        die("Hugging Face model must be owner/repo or owner/repo:filename.mercan");
    }
    if (!safe_hf_component(out.repo, true) || !safe_hf_component(out.filename, false)) {
        die("invalid characters in Hugging Face model reference");
    }
    if (out.filename.size() < 7 || out.filename.substr(out.filename.size() - 7) != ".mercan") {
        die("Hugging Face artifact must end in .mercan");
    }
    return out;
}

static fs::path pull_hf(const hf_spec & spec, bool force = false) {
    fs::path dst = fs::path(cache_root()) / "huggingface" / spec.repo / spec.filename;
    if (!force && fs::exists(dst) && fs::file_size(dst) > 0) {
        std::cout << "Using cached model: " << dst << "\n";
        return dst;
    }

    fs::create_directories(dst.parent_path());
    fs::path tmp = dst;
    tmp += ".part";
    std::error_code ec;
    fs::remove(tmp, ec);

    const std::string url = "https://huggingface.co/" + spec.repo + "/resolve/main/" + spec.filename + "?download=true";
    std::cout << "Pulling " << spec.repo << "/" << spec.filename << " from Hugging Face...\n";

#ifdef _WIN32
    const std::string cmd = "curl.exe -fL --retry 3 --connect-timeout 20 -o \"" + tmp.string() + "\" \"" + url + "\"";
#else
    const std::string cmd = "curl -fL --retry 3 --connect-timeout 20 -o '" + tmp.string() + "' '" + url + "'";
#endif
    const int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::exists(tmp) || fs::file_size(tmp) == 0) {
        fs::remove(tmp, ec);
        die("download failed; make sure curl is installed and the Hugging Face repository is public");
    }
    fs::rename(tmp, dst, ec);
    if (ec) {
        fs::remove(dst, ec);
        ec.clear();
        fs::rename(tmp, dst, ec);
    }
    if (ec) die("could not move downloaded model into cache: " + ec.message());

    std::cout << "Saved: " << dst << "\n";
    return dst;
}

static fs::path resolve_model(const std::string & input) {
    fs::path p(input);
    if (fs::exists(p)) return fs::absolute(p);
    return pull_hf(parse_hf_spec(input));
}

// ---- Web search (CLI-side, unconditional) ----
// This model has no tool-calling ability, so when --web-search is on, the CLI itself
// performs the search and fetches the top pages before the prompt ever reaches the model;
// the model just sees the results as plain text. Best-effort HTML scraping: no API key.

static std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string url_encode(const std::string & s) {
    static const char * hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[(c >> 4) & 0xF];
            out += hex[c & 0xF];
        }
    }
    return out;
}

// GET via curl, captured through a pipe. Bounded to avoid unbounded memory use on huge pages.
static std::string http_get(const std::string & url) {
#ifdef _WIN32
    const std::string cmd = "curl.exe -s -L --max-time 15 -A \"Mozilla/5.0\" \"" + url + "\" 2>nul";
#else
    const std::string cmd = "curl -s -L --max-time 15 -A 'Mozilla/5.0' " + shell_quote(url) + " 2>/dev/null";
#endif
    std::string out;
    FILE * pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    char buf[4096];
    const size_t cap = 2 * 1024 * 1024; // 2 MiB safety cap per request
    size_t n;
    while (out.size() < cap && (n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        out.append(buf, n);
    }
    pclose(pipe);
    return out;
}

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static std::string strip_html(const std::string & html) {
    std::string s = html;
    auto remove_block = [&](const std::string & tag) {
        const std::string open = "<" + tag;
        const std::string close = "</" + tag + ">";
        size_t pos = 0;
        while ((pos = s.find(open, pos)) != std::string::npos) {
            const size_t end_open = s.find('>', pos);
            if (end_open == std::string::npos) { s.erase(pos); break; }
            const size_t close_pos = s.find(close, end_open);
            if (close_pos == std::string::npos) { s.erase(pos); break; }
            s.erase(pos, close_pos + close.size() - pos);
        }
    };
    remove_block("script");
    remove_block("style");

    std::string out;
    out.reserve(s.size());
    bool in_tag = false;
    for (char c : s) {
        if (c == '<') { in_tag = true; continue; }
        if (c == '>') { in_tag = false; out += ' '; continue; }
        if (!in_tag) out += c;
    }

    replace_all(out, "&nbsp;", " ");
    replace_all(out, "&amp;", "&");
    replace_all(out, "&quot;", "\"");
    replace_all(out, "&#39;", "'");
    replace_all(out, "&#x27;", "'");
    replace_all(out, "&lt;", "<");
    replace_all(out, "&gt;", ">");

    std::string collapsed;
    collapsed.reserve(out.size());
    bool last_space = false;
    for (char c : out) {
        const bool is_space = std::isspace(static_cast<unsigned char>(c)) != 0;
        if (is_space) {
            if (!last_space) collapsed += ' ';
            last_space = true;
        } else {
            collapsed += c;
            last_space = false;
        }
    }
    const size_t b = collapsed.find_first_not_of(' ');
    if (b == std::string::npos) return {};
    const size_t e = collapsed.find_last_not_of(' ');
    return collapsed.substr(b, e - b + 1);
}

struct web_result {
    std::string title;
    std::string url;
    std::string snippet;
};

static void append_utf8(std::string & out, unsigned int cp) {
    if (cp <= 0x7F) {
        out += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

static int hex4(const std::string & s, size_t pos) {
    if (pos + 4 > s.size()) return -1;
    int v = 0;
    for (size_t i = 0; i < 4; ++i) {
        const char c = s[pos + i];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return -1;
        v = (v << 4) | d;
    }
    return v;
}

// Reads a JSON string value for "key": "..." out of a raw object slice (Python's json.dumps
// spacing), handling \uXXXX escapes (incl. surrogate pairs) and the usual backslash escapes.
// No general JSON parser needed for this flat shape.
static std::string json_extract_string(const std::string & obj, const std::string & key) {
    size_t pos = obj.find("\"" + key + "\"");
    if (pos == std::string::npos) return {};
    pos = obj.find(':', pos);
    if (pos == std::string::npos) return {};
    ++pos;
    while (pos < obj.size() && std::isspace(static_cast<unsigned char>(obj[pos]))) ++pos;
    if (pos >= obj.size() || obj[pos] != '"') return {};
    ++pos;

    std::string out;
    while (pos < obj.size() && obj[pos] != '"') {
        if (obj[pos] == '\\' && pos + 1 < obj.size()) {
            const char e = obj[pos + 1];
            if (e == 'u') {
                const int cp = hex4(obj, pos + 2);
                if (cp < 0) { pos += 2; continue; }
                pos += 6;
                if (cp >= 0xD800 && cp <= 0xDBFF && pos + 1 < obj.size() && obj[pos] == '\\' && obj[pos + 1] == 'u') {
                    const int low = hex4(obj, pos + 2);
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        const unsigned int combined = 0x10000 + ((static_cast<unsigned int>(cp) - 0xD800) << 10) + (static_cast<unsigned int>(low) - 0xDC00);
                        append_utf8(out, combined);
                        pos += 6;
                        continue;
                    }
                }
                append_utf8(out, static_cast<unsigned int>(cp));
                continue;
            }
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                default: out += e; break;
            }
            pos += 2;
            continue;
        }
        out += obj[pos];
        ++pos;
    }
    return out;
}

// Finds the matching closing '}' for the object opening at s[start], respecting string
// literals so braces inside quoted text don't throw off the depth count.
static size_t json_object_end(const std::string & s, size_t start) {
    int depth = 0;
    bool in_string = false;
    for (size_t i = start; i < s.size(); ++i) {
        const char c = s[i];
        if (in_string) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == '{') ++depth;
        else if (c == '}') { --depth; if (depth == 0) return i; }
    }
    return std::string::npos;
}

// Queries a self-hosted SearXNG instance's JSON API (open source, no API key). Public
// instances tend to sit behind bot-detection that blocks unauthenticated scripts, so this
// is meant to point at an instance the user runs themselves (e.g. via Docker, on their own
// network) rather than a public one.
static std::vector<web_result> web_search_query(const std::string & query, int max_results, const std::string & searxng_base) {
    std::vector<web_result> results;
    const std::string url = searxng_base + "/search?q=" + url_encode(query) + "&format=json";
    const std::string body = http_get(url);
    if (body.empty()) return results;

    const size_t results_key = body.find("\"results\"");
    if (results_key == std::string::npos) return results;
    const size_t bracket = body.find('[', results_key);
    if (bracket == std::string::npos) return results;

    size_t pos = bracket + 1;
    while (static_cast<int>(results.size()) < max_results) {
        const size_t obj_start = body.find('{', pos);
        if (obj_start == std::string::npos) break;
        const size_t obj_end = json_object_end(body, obj_start);
        if (obj_end == std::string::npos) break;
        const std::string obj = body.substr(obj_start, obj_end - obj_start + 1);

        const std::string title = json_extract_string(obj, "title");
        const std::string result_url = json_extract_string(obj, "url");
        const std::string content = json_extract_string(obj, "content");
        if (!title.empty() && !result_url.empty()) {
            results.push_back({title, result_url, content});
        }
        pos = obj_end + 1;
    }
    return results;
}

static std::string fetch_page_excerpt(const std::string & url, size_t max_chars) {
    const std::string html = http_get(url);
    if (html.empty()) return {};
    std::string text = strip_html(html);
    if (text.size() > max_chars) text.resize(max_chars);
    return text;
}

// Mandatory when --web-search is on: always searches and fetches, regardless of what the
// model would have "wanted" to do, since this model can't decide that for itself.
static std::string build_web_context(
    const std::string & query, int max_results, size_t max_chars_per_page, const std::string & searxng_base) {
    std::cerr << "[web-search] araniyor: \"" << query << "\" (" << searxng_base << ")\n";
    const auto results = web_search_query(query, max_results, searxng_base);
    if (results.empty()) {
        std::cerr << "[web-search] sonuc bulunamadi (SearXNG'e ulasilamadi mi? --web-search-url ile kontrol et), aramasiz devam ediliyor.\n";
        return {};
    }

    // Deliberately avoids bracket/tag-style markup (e.g. "[FOO]") around the sources: a small
    // model tends to mimic whatever formatting it just saw in its own input, and previous
    // wording here got echoed back verbatim as fake section labels in the model's answer.
    std::ostringstream oss;
    oss << "Web arama sonuclari:\n\n";
    int idx = 1;
    for (const auto & r : results) {
        std::cerr << "[web-search] okunuyor (" << idx << "/" << results.size() << "): " << r.url << "\n";
        std::string excerpt = fetch_page_excerpt(r.url, max_chars_per_page);
        if (excerpt.empty()) excerpt = r.snippet;
        oss << idx << ") " << r.title << " (" << r.url << ")\n" << excerpt << "\n\n";
        ++idx;
    }
    oss << "Yukaridaki kaynaklara dayanarak soruyu bir kere, net ve kisa sekilde cevapla. "
        << "Arama yapma surecini anlatma, sadece cevabi yaz. "
        << "Kaynaklarda bilgi yoksa bunu soyle, tahmin yurutme.";
    return oss.str();
}

// parse_special=true so structural markers like "<|im_start|>"/"<|im_end|>" in the
// formatted prompt resolve to their real control token ids instead of being split as text.
static std::vector<mercan_token> tokenize(mercan_model * model, const std::string & text) {
    int32_t n = mercan_tokenize(model, text.data(), text.size(), false, true, nullptr, 0);
    if (n == 0 && !text.empty()) die(std::string("tokenization failed: ") + mercan_last_error());
    if (n < 0) n = -n;
    std::vector<mercan_token> tokens(static_cast<size_t>(std::max(1, n)));
    int32_t got = mercan_tokenize(model, text.data(), text.size(), false, true, tokens.data(), static_cast<int32_t>(tokens.size()));
    if (got < 0) {
        tokens.resize(static_cast<size_t>(-got));
        got = mercan_tokenize(model, text.data(), text.size(), false, true, tokens.data(), static_cast<int32_t>(tokens.size()));
    }
    if (got < 0) die("token buffer sizing failed");
    tokens.resize(static_cast<size_t>(got));
    return tokens;
}

// Resolves a single structural/special token (e.g. "<|im_start|>") to its id.
// Returns -1 if the text does not map to exactly one special token in this model's vocab.
static mercan_token special_token_id(mercan_model * model, const std::string & text) {
    int32_t n = mercan_tokenize(model, text.data(), text.size(), false, true, nullptr, 0);
    if (n < 0) n = -n;
    if (n != 1) return -1;
    mercan_token tok = -1;
    int32_t got = mercan_tokenize(model, text.data(), text.size(), false, true, &tok, 1);
    if (got != 1) return -1;
    return tok;
}

static std::string token_piece(mercan_model * model, mercan_token token) {
    char small[64];
    int32_t n = mercan_token_to_piece(model, token, small, sizeof(small), false);
    if (n >= 0 && n <= static_cast<int32_t>(sizeof(small))) return std::string(small, small + n);
    if (n < 0) {
        std::vector<char> buf(static_cast<size_t>(-n));
        n = mercan_token_to_piece(model, token, buf.data(), static_cast<int32_t>(buf.size()), false);
        if (n >= 0) return std::string(buf.data(), buf.data() + n);
    }
    return {};
}

static mercan_token sample_token(
    const float * raw_logits, int32_t n_vocab, float temperature, int top_k, float top_p,
    float repeat_penalty, const std::vector<mercan_token> & recent_tokens, std::mt19937 & rng) {
    if (!raw_logits || n_vocab <= 0) die("no logits available");

    std::vector<float> adjusted(raw_logits, raw_logits + n_vocab);
    if (repeat_penalty != 1.0f) {
        for (mercan_token t : recent_tokens) {
            if (t < 0 || t >= n_vocab) continue;
            float & v = adjusted[static_cast<size_t>(t)];
            v = v > 0.0f ? v / repeat_penalty : v * repeat_penalty;
        }
    }
    const float * logits = adjusted.data();

    if (temperature <= 0.0f) {
        return static_cast<mercan_token>(std::max_element(logits, logits + n_vocab) - logits);
    }

    top_k = std::max(1, std::min(top_k, n_vocab));
    std::vector<std::pair<float, int32_t>> ranked;
    ranked.reserve(static_cast<size_t>(n_vocab));
    for (int32_t i = 0; i < n_vocab; ++i) ranked.emplace_back(logits[i], i);
    std::partial_sort(ranked.begin(), ranked.begin() + top_k, ranked.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });
    ranked.resize(static_cast<size_t>(top_k));

    const float max_logit = ranked.front().first;
    std::vector<double> weights;
    weights.reserve(ranked.size());
    double total = 0.0;
    for (const auto & x : ranked) {
        const double w = std::exp((static_cast<double>(x.first) - max_logit) / temperature);
        weights.push_back(w);
        total += w;
    }

    if (top_p > 0.0f && top_p < 1.0f && total > 0.0) {
        double cumulative = 0.0;
        size_t keep = 0;
        for (; keep < weights.size(); ++keep) {
            cumulative += weights[keep] / total;
            if (cumulative >= top_p) { ++keep; break; }
        }
        keep = std::max<size_t>(1, std::min(keep, weights.size()));
        ranked.resize(keep);
        weights.resize(keep);
    }

    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return static_cast<mercan_token>(ranked[dist(rng)].second);
}

static constexpr const char * DEFAULT_SYSTEM_PROMPT =
    "Sen Mercan'sın, MercanAI tarafından Türkçe konuşan kullanıcılar için geliştirilmiş "
    "bir yapay zeka asistanısın. Sorulara açık, doğru ve anlaşılır bir Türkçeyle yanıt "
    "verirsin; emin olmadığın konularda bunu açıkça belirtir, tahmini bilgiyi gerçek "
    "gibi sunmazsın. Kısa sorulara kısa ve net, ayrıntı gerektiren sorulara ise "
    "gerektiği kadar açıklayıcı cevaplar verirsin. Kullanıcıya karşı saygılı, sabırlı "
    "ve yardımsever bir tutum sergilersin; zararlı, yasa dışı veya güvenlik açısından "
    "tehlikeli isteklerde bulunulduğunda kibarca reddeder, bunun yerine güvenli ve "
    "yapıcı bir alternatif sunmaya çalışırsın.";

struct run_options {
    std::string model;
    std::string prompt;
    std::string system = DEFAULT_SYSTEM_PROMPT;
    std::string role_system = "sistem";
    std::string role_user = "kullanici";
    std::string role_assistant = "asistan";
    int max_tokens = 256;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.15f;
    int repeat_last_n = 64;
    bool web_search = false;
    int web_results = 3;
    size_t web_chars_per_page = 1500;
    std::string web_search_url = "http://localhost:8080";
    int threads = 0;
    std::vector<std::string> plugins;
#ifdef MERCAN_CUDA_BUILD
    int gpu_layers = -1;
#else
    int gpu_layers = 0;
#endif
};

static std::string generate(mercan_model * model, const run_options & opt, const std::string & formatted_prompt, bool stream_to_stdout = true) {
    mercan_context_params cp = mercan_context_default_params();
    if (opt.threads > 0) cp.n_threads = cp.n_threads_batch = opt.threads;
    mercan_context * ctx = mercan_context_create(model, cp);
    if (!ctx) die(std::string("context creation failed: ") + mercan_last_error());

    auto prompt_tokens = tokenize(model, formatted_prompt);
    if (prompt_tokens.empty()) {
        mercan_context_free(ctx);
        die("prompt produced no tokens");
    }

    const int batch_size = 512;
    for (size_t i = 0; i < prompt_tokens.size();) {
        const int n = static_cast<int>(std::min<size_t>(batch_size, prompt_tokens.size() - i));
        if (mercan_decode(ctx, prompt_tokens.data() + i, n) != 0) {
            const std::string err = mercan_last_error();
            mercan_context_free(ctx);
            die("prompt decode failed: " + err);
        }
        i += static_cast<size_t>(n);
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::string text;
    const int32_t n_vocab = mercan_vocab_size(model);
    const mercan_token eos = mercan_eos_token(model);
    const mercan_token message_start = special_token_id(model, "<|im_start|>");
    const mercan_token message_end = special_token_id(model, "<|im_end|>");

    const size_t history_cap = std::max(0, opt.repeat_last_n);
    std::vector<mercan_token> recent_tokens;
    if (history_cap > 0) {
        const size_t tail = std::min(history_cap, prompt_tokens.size());
        recent_tokens.assign(prompt_tokens.end() - static_cast<long>(tail), prompt_tokens.end());
    }

    for (int i = 0; i < opt.max_tokens; ++i) {
        mercan_token next = sample_token(
            mercan_logits(ctx), n_vocab, opt.temperature, opt.top_k, opt.top_p,
            opt.repeat_penalty, recent_tokens, rng);
        if (next == eos || next == message_start || next == message_end) break;

        const std::string piece = token_piece(model, next);
        text += piece;
        if (stream_to_stdout) std::cout << piece << std::flush;

        if (history_cap > 0) {
            recent_tokens.push_back(next);
            if (recent_tokens.size() > history_cap) recent_tokens.erase(recent_tokens.begin());
        }

        if (mercan_decode(ctx, &next, 1) != 0) break;
    }

    mercan_context_free(ctx);
    return text;
}

static void print_help() {
    std::cout
        << "Mercan CLI " << VERSION << "\n\n"
        << "Usage:\n"
        << "  mercan run <model.mercan|owner/repo[:file.mercan]> [options]\n"
        << "  mercan pull <owner/repo[:file.mercan]>\n"
        << "  mercan arch list\n"
        << "  mercan tokenizer list\n"
        << "  mercan graph abi\n"
        << "  mercan tensor abi\n"
        << "  mercan kv abi\n"
        << "  mercan plugin list\n"
        << "  mercan plugin load <library>\n"
        << "  mercan --version\n\n"
        << "Run options:\n"
        << "  -p, --prompt TEXT       single-shot prompt (otherwise interactive)\n"
        << "  -s, --system TEXT       system prompt prepended to the conversation\n"
        << "                          (defaults to Mercan's built-in system prompt; pass an empty string to disable it)\n"
        << "  --role-system NAME      chat role header for system turns (default: sistem)\n"
        << "  --role-user NAME        chat role header for user turns (default: kullanici)\n"
        << "  --role-assistant NAME   chat role header for assistant turns (default: asistan)\n"
        << "  -n, --max-tokens N      maximum generated tokens (default 256)\n"
        << "  --temperature F         sampling temperature (default 0.7; 0 = greedy)\n"
        << "  --top-k N               top-k sampling (default 40)\n"
        << "  --top-p F               nucleus cutoff (default 0.9)\n"
        << "  --repeat-penalty F      penalty applied to recently used tokens (default 1.15; 1.0 = off)\n"
        << "  --repeat-last-n N       how many recent tokens the penalty looks at (default 64; 0 = off)\n"
        << "  --web-search            mandatory web search before every prompt (model has no tool-calling;\n"
        << "                          the CLI searches and fetches pages itself and hands the model the text).\n"
        << "                          Requires a SearXNG instance (open source, no API key); self-host it,\n"
        << "                          e.g. via Docker, and point --web-search-url at it.\n"
        << "  --web-search-url URL    SearXNG base URL (default http://localhost:8080)\n"
        << "  --web-results N         how many pages to fetch per search (default 3)\n"
        << "  --web-chars N           max characters read from each fetched page (default 1500)\n"
        << "  -t, --threads N         CPU threads\n"
        << "  --plugin PATH           load external architecture/tokenizer plugin\n"
        << "  --gpu-layers N          GPU layers (-1 = all; CUDA build defaults to -1)\n\n"
        << "Examples:\n"
        << "  mercan run model.mercan\n"
        << "  mercan run Ahmet2001/Mercan-0.8B-SFT\n"
        << "  mercan run Ahmet2001/Mercan-0.8B-SFT:model-q4.mercan -p \"Merhaba\"\n"
        << "  mercan run model.mercan -s \"Sen yardımsever bir asistansın.\" -p \"Merhaba\"\n"
        << "  mercan run model.mercan --web-search -p \"Bugün hava durumu nasıl?\"\n";
}

static void load_plugin_or_die(const std::string & path) {
    const int rc = mercan_plugin_load_v1(path.c_str());
    if (rc < 0) die(std::string("plugin load failed: ") + mercan_plugin_last_error_v1());
}

static void load_plugins_from_env() {
    const char * raw = std::getenv("MERCAN_PLUGINS");
    if (!raw || !*raw) return;
    const char separator =
#ifdef _WIN32
        ';';
#else
        ':';
#endif
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, separator)) {
        if (!item.empty()) load_plugin_or_die(item);
    }
}

static int command_run(int argc, char ** argv) {
    if (argc < 3) die("missing model; try 'mercan run --help'");
    run_options opt;
    opt.model = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * flag) -> std::string {
            if (++i >= argc) die(std::string("missing value for ") + flag);
            return argv[i];
        };
        if (a == "-p" || a == "--prompt") opt.prompt = need(a.c_str());
        else if (a == "-s" || a == "--system") opt.system = need(a.c_str());
        else if (a == "--role-system") opt.role_system = need(a.c_str());
        else if (a == "--role-user") opt.role_user = need(a.c_str());
        else if (a == "--role-assistant") opt.role_assistant = need(a.c_str());
        else if (a == "-n" || a == "--max-tokens") opt.max_tokens = std::stoi(need(a.c_str()));
        else if (a == "--temperature") opt.temperature = std::stof(need(a.c_str()));
        else if (a == "--top-k") opt.top_k = std::stoi(need(a.c_str()));
        else if (a == "--top-p") opt.top_p = std::stof(need(a.c_str()));
        else if (a == "--repeat-penalty") opt.repeat_penalty = std::stof(need(a.c_str()));
        else if (a == "--repeat-last-n") opt.repeat_last_n = std::stoi(need(a.c_str()));
        else if (a == "--web-search") opt.web_search = true;
        else if (a == "--web-search-url") opt.web_search_url = need(a.c_str());
        else if (a == "--web-results") opt.web_results = std::stoi(need(a.c_str()));
        else if (a == "--web-chars") opt.web_chars_per_page = static_cast<size_t>(std::stoul(need(a.c_str())));
        else if (a == "-t" || a == "--threads") opt.threads = std::stoi(need(a.c_str()));
        else if (a == "--plugin") opt.plugins.push_back(need(a.c_str()));
        else if (a == "--gpu-layers") opt.gpu_layers = std::stoi(need(a.c_str()));
        else if (a == "-h" || a == "--help") { print_help(); return 0; }
        else die("unknown option: " + a);
    }

    for (const auto & plugin : opt.plugins) load_plugin_or_die(plugin);
    fs::path model_path = resolve_model(opt.model);
    mercan_backend_init();
    mercan_model_params mp = mercan_model_default_params();
    mp.n_gpu_layers = opt.gpu_layers;
#ifdef MERCAN_CUDA_BUILD
    std::cout << "Backend: CUDA";
    if (opt.gpu_layers < 0) std::cout << " (all layers)";
    else std::cout << " (" << opt.gpu_layers << " GPU layers)";
    std::cout << "\n";
#else
    std::cout << "Backend: CPU";
    if (opt.gpu_layers != 0) std::cout << " (CUDA backend not included in this build)";
    std::cout << "\n";
#endif
    std::cout << "Loading " << model_path << "...\n";
    mercan_model * model = mercan_model_load(model_path.string().c_str(), mp);
    if (!model) die(std::string("model load failed: ") + mercan_last_error());

    const std::string system_prefix = opt.system.empty()
        ? std::string()
        : "<|im_start|>" + opt.role_system + "\n" + opt.system + "<|im_end|>\n";

    if (!opt.prompt.empty()) {
        std::string user_content = opt.prompt;
        if (opt.web_search) {
            const std::string context = build_web_context(opt.prompt, opt.web_results, opt.web_chars_per_page, opt.web_search_url);
            if (!context.empty()) user_content = context + "\n\nKullanicinin sorusu: " + opt.prompt;
        }
        const std::string formatted = system_prefix + "<|im_start|>" + opt.role_user + "\n" + user_content + "<|im_end|>\n<|im_start|>" + opt.role_assistant + "\n";
        generate(model, opt, formatted);
        std::cout << "\n";
    } else {
        std::cout << "Mercan ready. Type /exit to quit.\n\n";
        std::string line;
        std::string transcript = system_prefix;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line == "/exit" || line == "/quit") break;
            if (line.empty()) continue;
            std::string user_content = line;
            if (opt.web_search) {
                const std::string context = build_web_context(line, opt.web_results, opt.web_chars_per_page, opt.web_search_url);
                if (!context.empty()) user_content = context + "\n\nKullanicinin sorusu: " + line;
            }
            transcript += "<|im_start|>" + opt.role_user + "\n" + user_content + "<|im_end|>\n<|im_start|>" + opt.role_assistant + "\n";
            const std::string answer = generate(model, opt, transcript);
            std::cout << "\n\n";
            transcript += answer + "<|im_end|>\n";
        }
    }

    mercan_model_free(model);
    mercan_backend_free();
    return 0;
}

static int command_arch(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[2]) != "list") {
        die("usage: mercan arch list");
    }
    const size_t count = mercan_arch_count_v1();
    std::cout << "Mercan Architecture SDK ABI " << MERCAN_ARCH_ABI_VERSION << "\n";
    for (size_t i = 0; i < count; ++i) {
        const mercan_architecture_v1 * arch = mercan_arch_at_v1(i);
        if (!arch) continue;
        std::cout << arch->name;
        if (arch->display_name && *arch->display_name) std::cout << "\t" << arch->display_name;
        if (arch->default_tokenizer && *arch->default_tokenizer) std::cout << "\ttokenizer=" << arch->default_tokenizer;
        if (arch->flags & MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES) std::cout << "\tgraph-abi-v1";
        if (arch->flags & MERCAN_ARCH_TENSOR_ABI_V1) std::cout << "\ttensor-abi-v1";
        if (arch->flags & MERCAN_ARCH_KV_ABI_V1) std::cout << "\tkv-abi-v1";
        if (arch->flags & MERCAN_ARCH_GRAPH_CALLBACK_V1) std::cout << "\tgraph-callback-v1";
        if (arch->flags & MERCAN_ARCH_BUILTIN) std::cout << "\tbuiltin";
        std::cout << "\n";
    }
    return 0;
}

static int command_graph(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[2]) != "abi") {
        die("usage: mercan graph abi");
    }
    std::cout << "Mercan Graph ABI " << MERCAN_GRAPH_ABI_VERSION << "\n"
              << "tensor_handles=opaque\n"
              << "primitives=get_rows,cast_f32,swiglu_split,view_2d,mul,add,concat,matmul,rms_norm,rope_ext,self_attention\n"
              << "attention=runtime-owned-mask-and-cache\n"
              << "outputs=set_output,finalize\n"
              << "tensor_resolver=tensor-abi-v1\n";
    return 0;
}


static int command_plugin(int argc, char ** argv) {
    if (argc < 3) die("usage: mercan plugin <list|load PATH>");
    const std::string action = argv[2];
    if (action == "load") {
        if (argc != 4) die("usage: mercan plugin load <library>");
        load_plugin_or_die(argv[3]);
    } else if (action != "list") {
        die("usage: mercan plugin <list|load PATH>");
    }
    std::cout << "Mercan Plugin ABI " << MERCAN_PLUGIN_ABI_VERSION << "\n";
    for (size_t i = 0; i < mercan_plugin_count_v1(); ++i) {
        const char * name = mercan_plugin_name_v1(i);
        const char * version = mercan_plugin_version_v1(i);
        const char * path = mercan_plugin_path_v1(i);
        std::cout << (name ? name : "<unnamed>");
        if (version && *version) std::cout << "\tversion=" << version;
        if (path && *path) std::cout << "\t" << path;
        std::cout << "\n";
    }
    return 0;
}

static int command_kv(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[2]) != "abi") {
        die("usage: mercan kv abi");
    }
    std::cout << "Mercan KV Cache ABI " << MERCAN_KV_ABI_VERSION << "\n"
              << "handles=opaque\n"
              << "views=base,sliding-window\n"
              << "batch_tensors=k_indices,v_indices,attention_mask\n"
              << "mutation=runtime-owned\n";
    return 0;
}

static int command_tensor(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[2]) != "abi") {
        die("usage: mercan tensor abi");
    }
    std::cout << "Mercan Tensor ABI " << MERCAN_TENSOR_ABI_VERSION << "\n"
              << "handles=opaque\n"
              << "lookup=tensor_by_name,require_tensor\n"
              << "declarations=required,optional,weight,state\n";
    return 0;
}

static int command_tokenizer(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[2]) != "list") {
        die("usage: mercan tokenizer list");
    }
    const size_t count = mercan_tokenizer_count_v1();
    std::cout << "Mercan Tokenizer SDK ABI " << MERCAN_TOKENIZER_ABI_VERSION << "\n";
    for (size_t i = 0; i < count; ++i) {
        const mercan_tokenizer_v1 * tok = mercan_tokenizer_at_v1(i);
        if (!tok) continue;
        std::cout << tok->name;
        if (tok->display_name && *tok->display_name) std::cout << "\t" << tok->display_name;
        if (tok->flags & MERCAN_TOKENIZER_BACKEND_MANAGED) std::cout << "\tbackend-managed";
        if (tok->flags & MERCAN_TOKENIZER_BUILTIN) std::cout << "\tbuiltin";
        std::cout << "\n";
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { print_help(); return 0; }
    const std::string cmd = argv[1];
    load_plugins_from_env(); // MERCAN_PLUGINS
    if (cmd == "--version" || cmd == "version") {
        std::cout << "mercan " << VERSION << " (libmercan " << mercan_version()
#ifdef MERCAN_CUDA_BUILD
                  << ", backend cuda)\n";
#else
                  << ", backend cpu)\n";
#endif
        return 0;
    }
    if (cmd == "-h" || cmd == "--help" || cmd == "help") { print_help(); return 0; }
    if (cmd == "arch") return command_arch(argc, argv);
    if (cmd == "tokenizer") return command_tokenizer(argc, argv);
    if (cmd == "graph") return command_graph(argc, argv);
    if (cmd == "tensor") return command_tensor(argc, argv);
    if (cmd == "kv") return command_kv(argc, argv);
    if (cmd == "plugin") return command_plugin(argc, argv);
    if (cmd == "pull") {
        if (argc < 3) die("missing Hugging Face repository");
        const auto p = pull_hf(parse_hf_spec(argv[2]), argc > 3 && std::string(argv[3]) == "--force");
        std::cout << p << "\n";
        return 0;
    }
    if (cmd == "run") return command_run(argc, argv);
    die("unknown command: " + cmd);
    return 1;
}
