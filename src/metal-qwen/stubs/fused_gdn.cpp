// fused_gdn.cpp - Metal launchers for the fused GDN entries.
#include "strata/kernels/fused_gdn.hpp"
#include "../runtime/metal_kernel.hpp"
#include <cstdio>
#include <cstdlib>
namespace strata::kernels {
namespace { using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
[[noreturn]] void bad(const char* m) { std::fprintf(stderr, "%s\n", m); std::exit(1); } }
void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % 128 != 0 || qk_heads < 0 || qk_heads > channels / 128)
        bad("fused_gdn_conv_l2: invalid arguments");
    KernelArg a[] = {buf(history), buf(qkv), buf(conv_w), buf(h), val(qk_heads), val(eps)};
    strata::metal::launch((cudaStream_t)stream, "fused_gdn_conv_l2_kernel", (uint32_t)channels, 1, 1, 128, 1, 1, a, 6);
}
void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0)
        bad("fused_gdn_ab: invalid arguments");
    KernelArg a[] = {buf(x), buf(w_alpha), buf(w_beta), buf(dt), buf(ssm_a), buf(gate), buf(beta), val(n_embd), val(h_v)};
    strata::metal::launch((cudaStream_t)stream, "fused_gdn_ab_kernel", (uint32_t)(2 * h_v), 1, 1, 1, 1, 1, a, 9);
}
void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k)
        bad("fused_gdn_step_norm: invalid arguments");
    KernelArg a[] = {buf(state), buf(q), buf(k), buf(v), buf(gate), buf(beta), buf(z), buf(gamma), val(eps), buf(y), val(h_k), val(h_v)};
    strata::metal::launch((cudaStream_t)stream, "fused_gdn_step_norm_kernel", (uint32_t)(h_v * 128), 1, 1, 128, 1, 1, a, 12);
}
}  // namespace strata::kernels
