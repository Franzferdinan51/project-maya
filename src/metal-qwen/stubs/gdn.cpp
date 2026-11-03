// gdn.cpp - Metal host launchers for include/strata/kernels/gdn.hpp.
// Kernels in src/metal-qwen/kernels/gdn.metal; semantics from the CUDA unit.
#include "strata/kernels/gdn.hpp"
#include "../runtime/metal_kernel.hpp"
#include <cstdio>
#include <cstdlib>
namespace strata::kernels {
namespace { using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
void launch(void* s, const char* fn, uint32_t g, uint32_t b, const KernelArg* a, size_t n) {
    strata::metal::launch((cudaStream_t)s, fn, g, 1, 1, b, 1, 1, a, n);
} }
void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) { std::fprintf(stderr, "gdn_step: S exceeds 128\n"); std::exit(1); }
    int S = (int)s.S, hk = (int)s.h_k, hv = (int)s.h_v;
    KernelArg a[] = {buf(state), buf(q), buf(k), buf(v), buf(gate), buf(beta), buf(o), val(S), val(hk), val(hv)};
    launch(stream, "gdn_step_kernel", (uint32_t)(hv * S), 128, a, 10);
}
void gdn_conv_step(float* cs, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv, void* stream) {
    if (channels <= 0 || d_conv < 1) return;
    int C = (int)channels, dc = (int)d_conv;
    KernelArg a[] = {buf(cs), buf(x), buf(kW), buf(out), val(C), val(dc)};
    launch(stream, "gdn_conv_kernel", (uint32_t)C, 256, a, 6);
}
void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) { std::fprintf(stderr, "gdn_l2_norm: cols exceeds 1024\n"); std::exit(1); }
    int c = (int)cols;
    KernelArg a[] = {buf(x), val(c), val(eps)};
    // grid in threads: one 128-thread threadgroup per row
    strata::metal::launch((cudaStream_t)stream, "gdn_l2_kernel", (uint32_t)(rows * 128), 1, 1, 128, 1, 1, a, 3);
}
void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    if (!beta || h_v <= 0) return;
    int n = (int)h_v;
    KernelArg a[] = {buf(beta), val(n)};
    launch(stream, "gdn_beta_kernel", (uint32_t)n, 128, a, 2);
}
void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                  float eps, void* stream) {
    if (h_v <= 0 || S <= 0) return;
    int s = (int)S;
    KernelArg a[] = {buf(o), buf(z), buf(ssm_norm), buf(y), val(s), val(eps)};
    strata::metal::launch((cudaStream_t)stream, "gdn_out_norm_kernel", (uint32_t)(h_v * 128), 1, 1, 128, 1, 1, a, 6);
}
}  // namespace strata::kernels
