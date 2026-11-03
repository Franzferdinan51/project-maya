// src/metal-qwen/stubs/s2_gemv_q8.cpp - Metal implementation for
// include/strata/kernels/s2_gemv_q8.hpp (was the stage-1 stub). Keeps the CUDA
// kernel's shape: one threadgroup per output row, threads striding quads with
// four accumulators, tree reduction through threadgroup memory.
#include "strata/kernels/s2_gemv_q8.hpp"

#include "../kernels/dense_host.hpp"

namespace strata::kernels {

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 64 != 0) dense_detail::bad_shape("s2_gemv_q8: n_in must be a multiple of 64");
    const uint32_t tpr = dense_detail::checked_tpr(threads_per_row, "s2_gemv_q8: bad threads_per_row");
    const uint32_t ni = (uint32_t) n_in;
    const metal::KernelArg args[] = {metal::buf(act), metal::buf(codes), metal::buf(scales), metal::buf(y),
                                     metal::val(ni), metal::val(tpr)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "s2_gemv_q8_k", (uint32_t) n_out * tpr, tpr, args,
                              6, stream == nullptr);
}

}  // namespace strata::kernels
