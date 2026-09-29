#include "mercan.h"
#include "mercan_arch.h"
#include "mercan_graph.h"
#include "mercan_kv.h"
#include "mercan_plugin.h"
#include "mercan_tensor.h"
#include "mercan_tokenizer.h"

#include <algorithm>
#include <cmath>
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

// NedoLM/Mercan chat-control IDs. The textual aliases are <|im_start|> and
// <|im_end|>, while the training roles are Turkish: kullanici / asistan.
static constexpr mercan_token NEDOLM_MESSAGE_START_TOKEN = 32000;
static constexpr mercan_token NEDOLM_MESSAGE_END_TOKEN = 32001;

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

static std::vector<mercan_token> tokenize(mercan_model * model, const std::string & text) {
    int32_t n = mercan_tokenize(model, text.data(), text.size(), false, false, nullptr, 0);
    if (n == 0 && !text.empty()) die(std::string("tokenization failed: ") + mercan_last_error());
    if (n < 0) n = -n;
    std::vector<mercan_token> tokens(static_cast<size_t>(std::max(1, n)));
    int32_t got = mercan_tokenize(model, text.data(), text.size(), false, false, tokens.data(), static_cast<int32_t>(tokens.size()));
    if (got < 0) {
        tokens.resize(static_cast<size_t>(-got));
        got = mercan_tokenize(model, text.data(), text.size(), false, false, tokens.data(), static_cast<int32_t>(tokens.size()));
    }
    if (got < 0) die("token buffer sizing failed");
    tokens.resize(static_cast<size_t>(got));
    return tokens;
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

struct run_options {
    std::string model;
    std::string prompt;
    int max_tokens = 256;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.10f;
    int repeat_last_n = 64;
    int threads = 0;
    std::vector<std::string> plugins;
#ifdef MERCAN_CUDA_BUILD
    int gpu_layers = -1;
#else
    int gpu_layers = 0;
#endif
};

static std::string generate(mercan_model * model, const run_options & opt, const std::string & formatted_prompt) {
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

    std::string text;
    const mercan_token eos = mercan_eos_token(model);
    mercan_sampler_params sampler = mercan_sampler_default_params();
    sampler.temperature = opt.temperature;
    sampler.top_k = opt.top_k;
    sampler.top_p = opt.top_p;
    sampler.min_p = opt.min_p;
    sampler.repeat_penalty = opt.repeat_penalty;
    sampler.repeat_last_n = opt.repeat_last_n;

    for (int i = 0; i < opt.max_tokens; ++i) {
        mercan_token next = mercan_sample_next(ctx, sampler);
        if (next < 0) {
            const std::string err = mercan_last_error();
            mercan_context_free(ctx);
            die("sampling failed: " + err);
        }

        // Stop structural chat tokens before they are converted to text or fed
        // back into the decoder. This prevents <|im_end|>/<|im_start|> leakage.
        if (next == eos ||
            next == NEDOLM_MESSAGE_END_TOKEN ||
            next == NEDOLM_MESSAGE_START_TOKEN) {
            break;
        }

        const std::string piece = token_piece(model, next);
        text += piece;

        // Compatibility fallback for artifacts/tokenizers that surface a
        // textual alias as ordinary pieces instead of the structural ID.
        const auto end_stop = text.find("<|im_end|>");
        const auto start_stop = text.find("<|im_start|>");
        const auto stop = std::min(end_stop, start_stop);
        if (stop != std::string::npos) {
            text.resize(stop);
            break;
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
        << "  -n, --max-tokens N      maximum generated tokens (default 256)\n"
        << "  --temperature F         sampling temperature (default 0.7; 0 = greedy)\n"
        << "  --top-k N               top-k sampling (default 40)\n"
        << "  --top-p F               nucleus cutoff (default 0.95)\n"
        << "  --min-p F               minimum probability ratio (default 0.05)\n"
        << "  --repeat-penalty F      repetition penalty (default 1.10)\n"
        << "  --repeat-last-n N       repetition window (default 64)\n"
        << "  -t, --threads N         CPU threads\n"
        << "  --plugin PATH           load external architecture/tokenizer plugin\n"
        << "  --gpu-layers N          GPU layers (-1 = all; CUDA build defaults to -1)\n\n"
        << "Examples:\n"
        << "  mercan run model.mercan\n"
        << "  mercan run Ahmet2001/Mercan-0.8B-SFT\n"
        << "  mercan run Ahmet2001/Mercan-0.8B-SFT:model-q4.mercan -p \"Merhaba\"\n";
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
        else if (a == "-n" || a == "--max-tokens") opt.max_tokens = std::stoi(need(a.c_str()));
        else if (a == "--temperature") opt.temperature = std::stof(need(a.c_str()));
        else if (a == "--top-k") opt.top_k = std::stoi(need(a.c_str()));
        else if (a == "--top-p") opt.top_p = std::stof(need(a.c_str()));
        else if (a == "--min-p") opt.min_p = std::stof(need(a.c_str()));
        else if (a == "--repeat-penalty") opt.repeat_penalty = std::stof(need(a.c_str()));
        else if (a == "--repeat-last-n") opt.repeat_last_n = std::stoi(need(a.c_str()));
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

    if (!opt.prompt.empty()) {
        const std::string formatted = "<|im_start|>kullanici\n" + opt.prompt + "<|im_end|>\n<|im_start|>asistan\n";
        std::cout << generate(model, opt, formatted) << "\n";
    } else {
        std::cout << "Mercan ready. Type /exit to quit.\n\n";
        std::string line;
        std::string transcript;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line == "/exit" || line == "/quit") break;
            if (line.empty()) continue;
            transcript += "<|im_start|>kullanici\n" + line + "<|im_end|>\n<|im_start|>asistan\n";
            const std::string answer = generate(model, opt, transcript);
            std::cout << answer << "\n\n";
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
