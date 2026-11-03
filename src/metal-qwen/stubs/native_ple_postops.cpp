// src/metal-qwen/stubs/native_ple_postops.cpp - Metal implementation for
// include/strata/kernels/native_ple_postops.hpp (was the stage-1 stub).
// Stage order and kernel shapes follow native_ple_postops.cu: rms_rep norms
// (1024 threads per stream-row), the 512-thread gate with eight ordered
// partial lanes, broadcast, the second norm, then conv+residual in one
// kernel. The host validation (nonnull/aligned spans, overlap rules with the
// two documented aliases) is ported from the CUDA TU.
#include "strata/kernels/native_ple_postops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "../kernels/dense_host.hpp"
#include "strata/kernels/ngram.hpp"

namespace strata::kernels {
namespace {
constexpr int N = NG_N_EMBD, H = NG_HC, D = NG_HC_DIM, HISTORY = NG_HIST;

struct Span {
    const void* p;
    size_t bytes;
    size_t alignment;
};
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}

void rms_rep(const float* in, const float* gamma, float* out, int rows, void* stream) {
    const metal::KernelArg args[] = {metal::buf(in), metal::buf(gamma), metal::buf(out)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "ple_rms_rep_k", (uint32_t) rows * 1024, 1024,
                              args, 3, false);
}
void gate512(const float* key, const float* query, float* gate, int rows, void* stream) {
    const float scale = 1.0f / std::sqrt(float(N));
    const metal::KernelArg args[] = {metal::buf(key), metal::buf(query), metal::buf(gate), metal::val(scale)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "ple_gate512_k", (uint32_t) rows * 512, 512,
                              args, 4, false);
}
}  // namespace

void native_ple_postops(const float* projected_key, const float* hidden, const float* value,
                        const float* history, const PleWeights& w, const NativePlePostopsBuffers& b,
                        void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key, D * 4, 4}, {hidden, D * 4, 4}, {value, N * 4, 4},
                           {history, HISTORY * D * 4, 4}, {w.norm_key, D * 4, 4}, {w.norm_query, D * 4, 4},
                           {w.norm_conv, D * 4, 4}, {w.conv1d_f16, 4 * D * 2, 2}};
    const Span outputs[] = {{b.key, D * 4, 4}, {b.query, D * 4, 4}, {b.gate, H * 4, 4},
                            {b.gated, D * 4, 4}, {b.normalized, D * 4, 4}, {b.conv, D * 4, 4},
                            {b.result, D * 4, 4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    rms_rep(projected_key, w.norm_key, b.key, H, stream);
    rms_rep(hidden, w.norm_query, b.query, H, stream);
    gate512(b.key, b.query, b.gate, H, stream);
    {
        const metal::KernelArg args[] = {metal::buf(value), metal::buf(b.gate), metal::buf(b.gated)};
        dense_detail::launch_sync(dense_detail::stream_of(stream), "ple_bcast_k", D, 256, args, 3, false);
    }
    rms_rep(b.gated, w.norm_conv, b.normalized, H, stream);
    {
        const metal::KernelArg args[] = {metal::buf(history), metal::buf(b.normalized), metal::buf(w.conv1d_f16),
                                         metal::buf(hidden), metal::buf(b.gated), metal::buf(b.conv),
                                         metal::buf(b.result)};
        dense_detail::launch_sync(dense_detail::stream_of(stream), "ple_conv_res_k", D, 256, args, 7, false);
    }
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history,
                              const PleWeights& w, float* query_norm, float* gated, float* gate, int T,
                              void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    const int rows = T * H;
    rms_rep(key, w.norm_key, key, rows, stream);
    rms_rep(hidden, w.norm_query, query_norm, rows, stream);
    gate512(key, query_norm, gate, rows, stream);
    const uint64_t total = (uint64_t) T * D;
    {
        const metal::KernelArg args[] = {metal::buf(value), metal::buf(gate), metal::buf(gated), metal::val(total)};
        dense_detail::launch_sync(dense_detail::stream_of(stream), "ple_bcast_batch_k", (uint32_t) total, 256,
                                  args, 4, false);
    }
    rms_rep(gated, w.norm_conv, query_norm, rows, stream);
    {
        const metal::KernelArg args[] = {metal::buf(history), metal::buf(query_norm), metal::buf(w.conv1d_f16),
                                         metal::buf(hidden), metal::buf(gated), metal::val(total)};
        dense_detail::launch_sync(dense_detail::stream_of(stream), "ple_conv_res_batch_k", (uint32_t) total, 256,
                                  args, 6, false);
    }
    {
        const metal::KernelArg args[] = {metal::buf(history), metal::buf(query_norm), metal::val(T)};
        dense_detail::launch_sync(dense_detail::stream_of(stream), "ple_hist_batch_k", D, 256, args, 3, false);
    }
}

}  // namespace strata::kernels
