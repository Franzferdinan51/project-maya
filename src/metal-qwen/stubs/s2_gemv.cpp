// src/metal-qwen/stubs/s2_gemv.cpp - Metal implementation for
// include/strata/kernels/s2_gemv.hpp (was the stage-1 stub). One thread per
// output row with in-order accumulation, so the result matches the CUDA
// naive kernel's accumulation order; synchronizes before returning, as it did.
#include "strata/kernels/s2_gemv.hpp"

#include "../kernels/dense_host.hpp"

namespace strata::kernels {

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 64 != 0) dense_detail::bad_shape("s2_gemv: n_in is not a multiple of 64");
    const uint32_t ni = (uint32_t) n_in, no = (uint32_t) n_out;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(codes), metal::buf(scales), metal::buf(y),
                                     metal::val(ni), metal::val(no)};
    dense_detail::launch_sync(nullptr, "s2_gemv_k", no, 128, args, 6, true);
}

}  // namespace strata::kernels
