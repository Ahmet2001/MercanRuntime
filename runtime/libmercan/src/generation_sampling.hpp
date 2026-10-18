#pragma once
#include "mercan.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace mercan_generation_detail {
inline mercan_token sample(
    const float * logits, int32_t vocab, const mercan_generation_params_v1 & p,
    const std::vector<mercan_token> & recent, const std::vector<mercan_token> & stops,
    std::mt19937 & rng) {
    if (!logits || vocab <= 0) throw std::runtime_error("no logits available");
    std::vector<double> scores(logits, logits + vocab);
    for (double & v : scores) if (std::isnan(v)) v = -std::numeric_limits<double>::infinity();
    if (p.repeat_penalty != 1.0f) {
        auto distinct = recent;
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        for (auto t : distinct) {
            if (t < 0 || t >= vocab || std::find(stops.begin(), stops.end(), t) != stops.end()) continue;
            scores[t] = scores[t] > 0 ? scores[t] / p.repeat_penalty : scores[t] * p.repeat_penalty;
        }
    }
    const auto best = std::max_element(scores.begin(), scores.end());
    if (*best == -std::numeric_limits<double>::infinity()) throw std::runtime_error("all logits are invalid");
    if (p.temperature == 0) return static_cast<mercan_token>(best - scores.begin());
    std::vector<int32_t> ids(vocab);
    for (int32_t i = 0; i < vocab; ++i) ids[i] = i;
    const size_t k = static_cast<size_t>(std::min(p.top_k, vocab));
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int32_t a, int32_t b) {
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    ids.resize(k);
    if (scores[ids[0]] == std::numeric_limits<double>::infinity()) {
        ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int32_t t) { return scores[t] != std::numeric_limits<double>::infinity(); }), ids.end());
        return ids[std::uniform_int_distribution<size_t>(0, ids.size() - 1)(rng)];
    }
    std::vector<double> weights;
    double total = 0;
    for (auto t : ids) {
        const double w = std::exp((scores[t] - scores[ids[0]]) / p.temperature);
        weights.push_back(w); total += w;
    }
    if (p.top_p < 1) {
        double mass = 0;
        size_t keep = 0;
        do { mass += weights[keep++] / total; } while (keep < weights.size() && mass < p.top_p);
        ids.resize(keep); weights.resize(keep);
    }
    // Explicit conversion avoids implementation-specific discrete_distribution.
    total = 0; for (double w : weights) total += w;
    double target = (static_cast<double>(rng()) / 4294967296.0) * total;
    for (size_t i = 0; i < weights.size(); ++i) {
        target -= weights[i]; if (target < 0) return ids[i];
    }
    return ids.back();
}
}
