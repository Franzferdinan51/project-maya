// gr.cpp - Metal launchers for the gated residual (gr_read / gr_write).
// Composition follows src/kernels/cuda/gr.cu: norm -> down -> (silu) -> gate
// -> mean -> inject, with the same three activation contracts. The native
// MMVF contract composes this workstream's native_gr_* entries plus the
// generic BF16 GEMV in gr.metal (fp32 activations, bf16 weights).
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "../runtime/metal_kernel.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
namespace strata::kernels {
namespace { bool g_fp32_act = false; bool g_native_mmvf = false;
using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
void L(void* s, const char* fn, uint32_t g, uint32_t b, const KernelArg* a, size_t n) {
    strata::metal::launch((cudaStream_t)s, fn, g, 1, 1, b, 1, 1, a, n); }
// tg-per-row BF16 GEMV over a float activation buffer (act_mode 0) or the
// bf16 activation buffer (act_mode 1, passed as act_b; act_f unused dummy).
void gemv(const float* act_f, const uint16_t* act_b, const uint16_t* w, float* out,
          int n_in, int rows, int act_mode, int epilogue, int hc, void* stream) {
    const float* af = act_f ? act_f : (const float*)act_b;
    const uint16_t* ab = act_b ? act_b : (const uint16_t*)act_f;
    const int zero = 0;  // named local: KernelArg::val binds by reference (a literal would dangle)
    KernelArg a[] = {buf(af), buf(ab), buf(w), buf(out), val(n_in), val(act_mode), val(zero), val(epilogue), val(hc)};
    L(stream, "gr_gemv_kernel", (uint32_t)(rows * 256), 256, a, 9);
} }
void gr_set_fp32_activations(bool enabled) { g_fp32_act = enabled; }
void gr_set_native_mmvf(bool enabled) { g_native_mmvf = enabled; }
size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t)s.hc * (size_t)s.n_embd;
    const size_t sz[5] = {hc_dim * sizeof(float), hc_dim * sizeof(uint16_t),
                          (size_t)s.hc_lr * sizeof(uint16_t), hc_dim * sizeof(float),
                          (size_t)s.hc_lr * sizeof(float)};
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) { al[k] = (sz[k] + 15) & ~(size_t)15; bytes += al[k]; }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*)base; void* ptr[5];
        for (int k = 0; k < 5; ++k) { ptr[k] = p; p += al[k]; }
        out.xn = (float*)ptr[0]; out.xq = (uint16_t*)ptr[1]; out.lq = (uint16_t*)ptr[2];
        out.gated = (float*)ptr[3]; out.lo = (float*)ptr[4];
    }
    return bytes;
}
void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (!ws.xn || !ws.xq || !ws.lq || !ws.gated || !ws.lo) {
        std::fprintf(stderr, "gr_read: GrWorkspace is not initialised (see gr_workspace_init)\n"); std::exit(1); }
    if (ws.bytes < gr_workspace_bytes(s)) {
        std::fprintf(stderr, "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu\n", ws.bytes, gr_workspace_bytes(s)); std::exit(1); }
    const int n_embd = (int)s.n_embd, hc = (int)s.hc, hc_lr = (int)s.hc_lr, hc_dim = (int)(s.hc * s.n_embd);
    const bool use_native = g_native_mmvf;
    const bool use_fp32 = g_fp32_act || use_native;
    if (use_native) {
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);
        gemv(ws.xn, nullptr, w_down, ws.lo, hc_dim, hc_lr, 0, 0, hc, stream);
        native_gr_down_silu(ws.lo, hc_lr, hc, stream);
        gemv(ws.lo, nullptr, w_up, ws.gated, hc_lr, hc_dim, 0, 0, hc, stream);
        native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);
    } else if (use_fp32) {
        int fp = 1;
        KernelArg a[] = {buf(R), buf(w_norm), buf(ws.xn), buf(ws.xq), val(n_embd), val(eps), val(fp)};
        L(stream, "gr_norm_kernel", (uint32_t)(hc * 256), 256, a, 7);
        gemv(ws.xn, nullptr, w_down, ws.lo, hc_dim, hc_lr, 0, 1, hc, stream);
        const int zero = 0;  // named local: KernelArg::val binds by reference
        KernelArg g[] = {buf(ws.lo), buf(ws.lq), buf(w_up), buf(ws.xn), buf(ws.gated), val(hc_lr), val(zero)};
        L(stream, "gr_gate_kernel", (uint32_t)(hc_dim * 256), 256, g, 7);
    } else {
        int fp = 0;
        KernelArg a[] = {buf(R), buf(w_norm), buf(ws.xn), buf(ws.xq), val(n_embd), val(eps), val(fp)};
        L(stream, "gr_norm_kernel", (uint32_t)(hc * 256), 256, a, 7);
        KernelArg d[] = {buf(ws.xq), buf(w_down), buf(ws.lq), val(hc_dim), val(hc)};
        L(stream, "gr_down_bf16_kernel", (uint32_t)(hc_lr * 256), 256, d, 5);
        const int one = 1;  // named local: KernelArg::val binds by reference
        KernelArg g[] = {buf(ws.lo), buf(ws.lq), buf(w_up), buf(ws.xn), buf(ws.gated), val(hc_lr), val(one)};
        L(stream, "gr_gate_kernel", (uint32_t)(hc_dim * 256), 256, g, 7);
    }
    if (!use_native) {
        KernelArg m[] = {buf(ws.gated), buf(mixed), val(n_embd), val(hc)};
        L(stream, "gr_mean_kernel", (uint32_t)n_embd, 256, m, 4);
    }
    if (w_inject != nullptr) {
        if (use_native) gemv(ws.xn, nullptr, w_inject, inject, hc_dim, hc, 0, 0, hc, stream);
        else if (use_fp32) gemv(ws.xn, nullptr, w_inject, inject, hc_dim, hc, 0, 0, hc, stream);
        else gemv(nullptr, ws.xq, w_inject, inject, hc_dim, hc, 1, 0, hc, stream);
    }
}
void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    if (g_native_mmvf) { native_gr_post(R, block_out, inject, R_out, (int)s.n_embd, (int)s.hc, stream); return; }
    int n_embd = (int)s.n_embd, hc = (int)s.hc;
    KernelArg a[] = {buf(R), buf(block_out), buf(inject), buf(R_out), val(n_embd), val(hc)};
    L(stream, "gr_write_kernel", (uint32_t)(hc * n_embd), 256, a, 6);
}
}  // namespace strata::kernels
