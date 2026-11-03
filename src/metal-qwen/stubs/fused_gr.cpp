// fused_gr.cpp - Metal launchers for the fused hyper-connection read.
// xn is recomputed on the fly from R' / w_norm / rs instead of staged in
// threadgroup memory (the CUDA version stages 10240 floats = 40 KB); the
// math and per-output accumulation axes are unchanged.
#include "strata/kernels/fused_gr.hpp"
#include "../runtime/metal_kernel.hpp"
#include <cstdio>
#include <cstdlib>
namespace strata::kernels {
namespace { using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
void L(void* s, const char* fn, uint32_t g, uint32_t b, const KernelArg* a, size_t n) {
    strata::metal::launch((cudaStream_t)s, fn, g, 1, 1, b, 1, 1, a, n); }
const float* or_dummy(const float* p, const float* fallback) { return p ? p : fallback; } }
bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == 2560 && hc == 4 && hc_lr == 320;
}
void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n"); std::exit(1); }
    const int N = 2560, HC = 4, LR = 320;
    int apply = a.apply ? 1 : 0;
    const float* bo = or_dummy(a.bo_prev, a.R);
    const float* inj = or_dummy(a.inj_prev, a.rs /* any valid float buffer; unread when !apply */);
    { KernelArg k[] = {buf(a.R), buf(bo), buf(inj), buf(a.rs), val(N), val(a.eps), val(apply), val(HC)};
      L(stream, "fused_gr_norm_kernel", (uint32_t)(HC * 256), 256, k, 8); }
    { int epi = 1;
      KernelArg k[] = {buf(a.R), buf(bo), buf(inj), buf(a.w_norm), buf(a.rs), buf(a.w_down), buf(a.lo),
                       val(N), val(HC), val(apply), val(epi)};
      L(stream, "fused_gr_gemv_kernel", (uint32_t)(LR * 256), 256, k, 11); }
    if (a.w_inject) { int epi = 0;
      KernelArg k[] = {buf(a.R), buf(bo), buf(inj), buf(a.w_norm), buf(a.rs), buf(a.w_inject), buf(a.inject_out),
                       val(N), val(HC), val(apply), val(epi)};
      L(stream, "fused_gr_gemv_kernel", (uint32_t)(HC * 256), 256, k, 11); }
    { float* rout = a.R_out ? a.R_out : const_cast<float*>(a.R);
      KernelArg k[] = {buf(a.R), buf(rout), buf(bo), buf(inj), buf(a.w_norm), buf(a.rs), buf(a.w_up), buf(a.lo),
                       buf(a.mixed), val(N), val(HC), val(LR), val(apply)};
      L(stream, "fused_gr_up_kernel", (uint32_t)N, 256, k, 13); }
}
void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf, int stamp_i0) {
    (void)stamp_buf; (void)stamp_i0; // telemetry stamps: no Metal equivalent yet
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n"); std::exit(1); }
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed ||
            (x.w_inject && !x.inject_out) || (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) ||
            x.w_down != a[0].w_down || x.w_up != a[0].w_up || x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t); std::exit(1); }
    }
    // Per-token launches: trivially bitwise identical to fused_gr_read per
    // token (the CUDA multi-kernel's stated contract); weights are re-read
    // per token, so this is a correctness-first version, not the P6 speedup.
    for (int t = 0; t < n_tok; ++t) fused_gr_read(a[t], stream);
}
}  // namespace strata::kernels
