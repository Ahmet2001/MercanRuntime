#include "mercan.h"
#include "../runtime/libmercan/src/generation_sampling.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

#define CHECK(expr) do { if (!(expr)) { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; return 1; } } while (0)

int main() {
    mercan_generation_params_v1 p{};
    CHECK(mercan_generation_api_version() == 1);
    CHECK(mercan_generation_params_size_v1() == sizeof(p));
    CHECK(mercan_generation_default_params_v1(&p, sizeof(p) - 1) == -1);
    CHECK(mercan_generation_default_params_v1(&p, sizeof(p)) == 0);
    CHECK(p.struct_size == sizeof(p));
    CHECK(mercan_generate_begin_v1(nullptr, "hello", 5, mercan_context_default_params(), &p) == nullptr);
    CHECK(std::strlen(mercan_last_error()) > 0);
    CHECK(mercan_chat_begin_v1(nullptr, nullptr, 0, nullptr, nullptr, nullptr,
                              mercan_context_default_params(), &p) == nullptr);
    CHECK(mercan_generation_run_v1(nullptr, nullptr, nullptr) == -1);
    mercan_generation_cancel_v1(nullptr);
    mercan_generation_free_v1(nullptr);
    std::mt19937 rng(42);
    const float logits[] = {0, 4, 3};
    p.temperature = 0; p.repeat_penalty = 2;
    // A repeated output token is penalized once, never once per occurrence.
    const float close_logits[] = {0, 4, 1.5f};
    CHECK(mercan_generation_detail::sample(close_logits, 3, p, {1, 1}, {}, rng) == 1);
    CHECK(mercan_generation_detail::sample(logits, 3, p, {1}, {}, rng) == 2);
    CHECK(mercan_generation_detail::sample(logits, 3, p, {1}, {1}, rng) == 1);
    const float invalid[] = {NAN, -INFINITY};
    bool failed = false;
    try { mercan_generation_detail::sample(invalid, 2, p, {}, {}, rng); }
    catch (const std::exception &) { failed = true; }
    CHECK(failed);
    p.temperature = 1; p.top_k = 1; p.top_p = 1; p.repeat_penalty = 1;
    CHECK(mercan_generation_detail::sample(logits, 3, p, {}, {}, rng) == 1);
    p.top_k = 3; p.top_p = 0.01f;
    CHECK(mercan_generation_detail::sample(logits, 3, p, {}, {}, rng) == 1);
    std::mt19937 a(7), b(7);
    p.top_p = 1;
    for (int i = 0; i < 100; ++i)
        CHECK(mercan_generation_detail::sample(logits, 3, p, {}, {}, a) ==
              mercan_generation_detail::sample(logits, 3, p, {}, {}, b));
    std::cout << "GENERATION_API_OK\n";
    return 0;
}
