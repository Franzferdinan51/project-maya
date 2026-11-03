// src/metal-qwen/stubs/dequant_s2.cpp - Metal implementation for
// include/strata/kernels/dequant_s2.hpp (was the stage-1 stub).
// (code - 1) * scale in the integer domain, one thread per output element;
// the kernel launches on the default stream and this entry synchronizes
// before returning, exactly the CUDA TU's contract.
#include "strata/kernels/dequant_s2.hpp"

#include "../kernels/dense_host.hpp"

namespace strata::kernels {

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    const uint64_t nb = (uint64_t) n_blocks;
    const metal::KernelArg args[] = {metal::buf(codes), metal::buf(scales), metal::buf(out), metal::val(nb)};
    dense_detail::launch_sync(nullptr, "dequant_s2_k", (uint32_t) (nb * 64), 256, args, 4, true);
}

}  // namespace strata::kernels
