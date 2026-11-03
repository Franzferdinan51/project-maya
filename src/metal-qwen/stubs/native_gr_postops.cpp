// native_gr_postops.cpp - Metal launchers for pinned GR post-operations.
#include "strata/kernels/native_gr_postops.hpp"
#include "../runtime/metal_kernel.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace strata::kernels {
namespace { constexpr int THREADS = 256;
using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
void chk(const void* p) { if (!p || reinterpret_cast<uintptr_t>(p) % alignof(float))
    throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers"); }
void shape(int n, int hc) { if (n <= 0 || hc <= 0 || uint64_t(n) * hc > uint64_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("native GR postops require positive bounded dimensions"); }
void launch(void* s, const char* fn, size_t n, const KernelArg* a, size_t na) {
    strata::metal::launch((cudaStream_t)s, fn, (uint32_t)n, 1, 1, THREADS, 1, 1, a, na); } }
void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    shape(hc_lr, hc); chk(lo);
    float scale = 1.0f / float(hc);
    KernelArg a[] = {buf(lo), val(hc_lr), val(scale)};
    launch(stream, "native_gr_down_silu_kernel", (size_t)hc_lr, a, 3);
}
void native_gr_pre_gated(const float* xn, float* gate, float* mixed, int n_embd, int hc, bool fused_layer, void* stream) {
    shape(n_embd, hc); chk(xn); chk(gate); chk(mixed);
    float scale = 1.0f / float(hc); int fused = fused_layer ? 1 : 0;
    KernelArg a[] = {buf(xn), buf(gate), buf(mixed), val(n_embd), val(hc), val(scale), val(fused)};
    launch(stream, "native_gr_pre_gated_kernel", (size_t)n_embd, a, 7);
}
void native_gr_post(const float* residual, const float* block_out, const float* inject, float* output,
                    int n_embd, int hc, void* stream) {
    shape(n_embd, hc); chk(residual); chk(block_out); chk(inject); chk(output);
    float scale = 1.0f / float(hc);
    KernelArg a[] = {buf(residual), buf(block_out), buf(inject), buf(output), val(n_embd), val(hc), val(scale)};
    launch(stream, "native_gr_post_kernel", size_t(n_embd) * hc, a, 7);
}
}  // namespace strata::kernels
