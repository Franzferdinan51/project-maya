// src/metal/sampler.hpp - maya-metal's token pick, the CUDA engine's sampler chain on the host (src/kernels/cuda/
// sampler.cu, its reference in src/kernels/sampler_parity.cpp): logit bias, penalties, top_k (ties to the lower id),
// top_p over the top_k list in double, min_p, temperature, one Philox draw at (seed, counter).  Greedy: the argmax of
// the biased, penalized logits.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace maya::metal {

struct Sampling {
    bool greedy = true;             // no sampler key: greedy (generate.cpp o.greedy)
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float temperature = 1.0f;
    int min_keep = 1;
    uint64_t seed = 0;
    uint64_t counter = 0;
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    std::vector<std::pair<int32_t, float>> bias;
};

float philox_uniform(uint64_t seed, uint64_t counter);
// hist: the tokens the state was fed (the prompt, then what was written); the window is its last pen_n
int32_t pick(const std::vector<float>& logits, const Sampling& p, const std::vector<int32_t>& hist);

// Reusable scratch for the serve loops' picks: the working copy of the logits and the
// top-k index list, kept across calls so a decode loop allocates neither per token.
struct PickScratch {
    std::vector<float> s;
    std::vector<int32_t> ids;
};

// The serve loops' pick: one pass copies the raw logits into the scratch (checking
// finiteness as it goes) and the pick runs in place on it - the same tokens as pick()
// above for the same inputs.  Returns -1 when any raw logit is non-finite; the serve
// loops treat that as fatal, exactly as their old separate pre-scan did.
int32_t pick(const float* logits, size_t n, const Sampling& p, const std::vector<int32_t>& hist,
             PickScratch& sc);

}  // namespace maya::metal
