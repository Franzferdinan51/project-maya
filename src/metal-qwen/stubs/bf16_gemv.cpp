// src/metal-qwen/stubs/bf16_gemv.cpp - Metal implementation for
// include/strata/kernels/bf16_gemv.hpp (was the stage-1 stub). bf16_gemv
// keeps the naive in-order accumulation; bf16_gemv_split and the MMVF pair
// keep their CUDA reduction shapes (threadgroup / warp sums), so neither is
// bit-identical to bf16_gemv - the header says the same of the CUDA pair.
// The MMVF host validation and adaptive block-size choice are ported from
// native_bf16.cu verbatim.
#include "strata/kernels/bf16_gemv.hpp"

#include <stdexcept>

#include "../kernels/dense_host.hpp"

namespace strata::kernels {

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    const uint32_t ni = (uint32_t) n_in, no = (uint32_t) n_out;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(w), metal::buf(y), metal::val(ni), metal::val(no)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "bf16_gemv_k", no, 128, args, 5, false);
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    const uint32_t tpr = dense_detail::checked_tpr(threads_per_row, "bf16_gemv_split: bad threads_per_row");
    const uint32_t ni = (uint32_t) n_in;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(w), metal::buf(y), metal::val(ni), metal::val(tpr)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "bf16_gemv_split_k", (uint32_t) n_out * tpr, tpr,
                              args, 5, false);
}

namespace {
int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}
void validate_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || n_out > INT32_MAX || n_in > INT32_MAX || x == nullptr ||
        w == nullptr || y == nullptr || stream == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 || (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: even n_in, positive sizes, aligned pointers, explicit stream");
}
}  // namespace

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                         void* stream) {
    validate_mmvf(x, w, y, n_in, n_out, stream);
    const uint32_t ni = (uint32_t) n_in, bs = (uint32_t) mmvf_block_size(n_in);
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(w), metal::buf(y), metal::val(ni), metal::val(bs)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "bf16_mmvf_k", (uint32_t) n_out * bs, bs, args, 5,
                              false);
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) {
        // The CUDA TU's shortcut, with its validation: single-row MMVF (its own
        // checks require the explicit stream and alignments).
        bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream);
        return;
    }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    const uint32_t ni = (uint32_t) n_in, bs = (uint32_t) mmvf_block_size(n_in);
    const int32_t ldx32 = (int32_t) ldx, ldy32 = (int32_t) ldy;
    const metal::KernelArg args[] = {metal::buf(x),  metal::buf(w),  metal::buf(y),  metal::val(ni),
                                     metal::val(bs), metal::val(ldx32), metal::val(ldy32), metal::val(n_tok)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "bf16_mmvf_multi_k", (uint32_t) n_out * bs, bs,
                              args, 8, false);
}

}  // namespace strata::kernels
