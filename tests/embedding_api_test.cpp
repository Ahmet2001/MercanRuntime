#include "mercan.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

static bool error_contains(const char * message) {
    return std::strstr(mercan_last_error(), message) != nullptr;
}

// Without a model only argument validation is checked. Set
// MERCAN_EMBED_TEST_MODEL to a .mercan file to also run a real embedding.
int main() {
    mercan_backend_init();
    if (mercan_embedding_api_version() != MERCAN_EMBEDDING_API_VERSION) return 1;
    auto cp = mercan_context_default_params();
    if (mercan_embedder_create_v1(nullptr, cp, MERCAN_POOLING_MEAN) || !error_contains("invalid model")) return 2;
    float out[4] = {};
    if (mercan_embed_v1(nullptr, "x", 1, out, 4) != -1 || !error_contains("invalid embed")) return 3;
    if (mercan_embedder_size_v1(nullptr) != 0) return 4;
    mercan_embedder_free_v1(nullptr);

    const char * path = std::getenv("MERCAN_EMBED_TEST_MODEL");
    if (!path || !*path) { std::cout << "embedding API checks passed (no model)\n"; return 0; }
    auto * model = mercan_model_load_device_v1(path, mercan_model_default_params(), "cpu");
    if (!model) { std::cerr << mercan_last_error() << '\n'; return 10; }
    cp.n_ctx = 256;
    if (mercan_embedder_create_v1(model, cp, 9) || !error_contains("pooling")) return 11;
    for (int pooling : {MERCAN_POOLING_MEAN, MERCAN_POOLING_LAST}) {
        auto * embedder = mercan_embedder_create_v1(model, cp, pooling);
        if (!embedder) { std::cerr << mercan_last_error() << '\n'; return 12; }
        const int32_t n = mercan_embedder_size_v1(embedder);
        std::vector<float> a(n), b(n);
        const char * text = "Ankara Türkiye'nin başkentidir.";
        if (mercan_embed_v1(embedder, text, std::strlen(text), a.data(), n - 1) != -1 ||
            !error_contains("capacity")) return 13;
        if (mercan_embed_v1(embedder, text, std::strlen(text), a.data(), n) != n) return 14;
        // A reused context must not leak state between texts.
        if (mercan_embed_v1(embedder, "başka bir metin", 16, b.data(), n) != n) return 15;
        if (mercan_embed_v1(embedder, text, std::strlen(text), b.data(), n) != n) return 16;
        double norm = 0;
        for (int32_t i = 0; i < n; ++i) {
            if (!std::isfinite(a[i]) || std::fabs(a[i] - b[i]) > 1e-4f) return 17;
            norm += a[i] * a[i];
        }
        if (norm <= 0) return 18;
        mercan_embedder_free_v1(embedder);
    }
    mercan_model_free(model);
    mercan_backend_free();
    std::cout << "embedding API checks passed\n";
}
