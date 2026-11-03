// src/metal-qwen/stubs/sampler.cpp - Metal implementation for
// include/strata/kernels/sampler.hpp (was the stage-1 stub).
//
// GREEDY PATH IS REAL (Stage 2's gate): sample_tokens with p.greedy or
// temperature <= 0 runs sampler.metal's sampler_greedy_k - biased, penalized
// argmax, ties to the lowest index, one threadgroup of 1024 per row, exactly
// the CUDA sampler_greedy_kernel's semantics.
//
// THE SAMPLED PATHS STILL STOP LOUDLY, by design: the CUDA chain's top_p and
// draw stages run in double precision and Apple GPUs expose no FP64 in
// Metal. A float transcription would produce plausible but different draws
// - the worst kind of wrong - so sample_tokens' sampled path, sample_draft
// and sample_verify abort with a named message until a float port is
// validated against sampler_parity.
#include "strata/kernels/sampler.hpp"

#include <cstdio>
#include <cstdlib>

#include "../kernels/dense_host.hpp"

namespace strata::kernels {
namespace {

struct GreedyArgs {
    int32_t n_vocab, history_len, plen, has_history, has_bias;
    float penalty_repeat, penalty_freq, penalty_present;
};

[[noreturn]] void sampled_not_ported(const char* who) {
    std::fprintf(stderr,
                 "strata metal: %s: the sampled (temperature) sampler chain is not ported yet - its "
                 "top_p/draw stage is double-precision in the CUDA TU and Metal has no FP64 on Apple "
                 "GPUs; a float port must be validated against sampler_parity first\n",
                 who);
    std::abort();
}

}  // namespace

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::abort();
    }
    if (!(p.greedy || p.temperature <= 0.0f)) sampled_not_ported("sample_tokens");
    GreedyArgs a{};
    a.n_vocab = n_vocab;
    a.history_len = history ? history_len : 0;
    a.plen = p.penalty_last_n < a.history_len ? p.penalty_last_n : a.history_len;
    if (a.plen < 0) a.plen = 0;
    a.has_history = history ? 1 : 0;
    a.has_bias = p.logit_bias ? 1 : 0;
    a.penalty_repeat = p.penalty_repeat;
    a.penalty_freq = p.penalty_freq;
    a.penalty_present = p.penalty_present;
    const metal::KernelArg args[] = {metal::buf(logits),
                                     metal::buf(history ? history : (const int*) logits),
                                     metal::buf(p.logit_bias ? p.logit_bias : logits), metal::buf(out),
                                     metal::val(a)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "sampler_greedy_k", (uint32_t) n_tokens * 1024,
                              1024, args, 5, stream == nullptr);
}

void sample_draft(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, SpecList* q, int* out,
                  void* stream) {
    (void) logits; (void) n_tokens; (void) n_vocab; (void) p; (void) q; (void) out; (void) stream;
    sampled_not_ported("sample_draft");
}

void sample_verify(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, const SpecList* q,
                   const int* draft, int* out, void* stream, const int* history, int history_len) {
    (void) logits; (void) n_tokens; (void) n_vocab; (void) p; (void) q; (void) draft; (void) out;
    (void) stream; (void) history; (void) history_len;
    sampled_not_ported("sample_verify");
}

}  // namespace strata::kernels
