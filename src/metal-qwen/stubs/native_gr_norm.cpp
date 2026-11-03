// native_gr_norm.cpp - Metal launcher for pinned weighted F32 RMSNorm.
#include "strata/kernels/native_gr_norm.hpp"
#include "../runtime/metal_kernel.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace strata::kernels {
namespace { void chk(const void* p) { if (!p || reinterpret_cast<uintptr_t>(p) % 4)
    throw std::invalid_argument("native GR RMSNorm requires non-null four-byte aligned pointers"); } }
void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    chk(input); chk(gamma); chk(output);
    int block = n_cols < 1024 ? 256 : 1024;
    using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
    KernelArg a[] = {buf(input), buf(gamma), buf(output), val(n_cols), val(epsilon)};
    strata::metal::launch((cudaStream_t)stream, "native_gr_rms_norm_kernel",
                          (uint32_t)(n_rows * block), 1, 1, (uint32_t)block, 1, 1, a, 5);
}
}  // namespace strata::kernels
