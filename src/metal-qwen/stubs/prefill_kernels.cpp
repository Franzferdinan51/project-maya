// src/metal-qwen/stubs/prefill_kernels.cpp - Metal host launchers for
// include/strata/prefill/kernels.hpp. Kernels in
// src/metal-qwen/kernels/prefill.metal; semantics from src/prefill/kernels.cu
// (see src/metal-qwen/PREFILL-NOTES.md for the entry-by-entry inventory).
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/router_top10.hpp"

#include "../runtime/metal_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

using strata::metal::buf;
using strata::metal::val;
using strata::metal::KernelArg;

void L(void* s, const char* fn, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz,
       const KernelArg* a, size_t n) {
    strata::metal::launch((cudaStream_t) s, fn, gx, gy, gz, bx, by, bz, a, n);
}
void L1(void* s, const char* fn, uint32_t total, const KernelArg* a, size_t n) {
    L(s, fn, total, 1, 1, 256, 1, 1, a, n);
}

// A tracked allocation bound in place of an optional (null) kernel argument;
// the kernel never dereferences it (the flags / presence arguments gate the
// access), it only has to resolve through the runtime's allocation map.
const void* dummy() {
    static void* p = [] {
        void* q = nullptr;
        if (cudaMalloc(&q, 4096) != cudaSuccess) {
            std::fprintf(stderr, "prefill launchers: dummy allocation failed\n");
            std::abort();
        }
        return q;
    }();
    return p;
}
template <typename T> const T* or_dummy(const T* p) { return p ? p : (const T*) dummy(); }

constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int64_t HV = 48, C = 10240;

}  // namespace

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(R), buf(w_norm), val(eps), buf(xn), buf(xn16)};
    L(stream, "pf_gr_norm_kernel", (uint32_t) (T * HC * 256), 1, 1, 256, 1, 1, a, 5);
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(R), buf(w_norm), val(eps), buf(rs), buf(xn16)};
    L(stream, "pf_gr_norm_rs_kernel", (uint32_t) (T * HC * 256), 1, 1, 256, 1, 1, a, 5);
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed,
              uint16_t* mixed16, int64_t T, void* stream, uint16_t* mixed_h) {
    if (T <= 0) return;
    const int flags = (mixed16 ? 1 : 0) | (mixed_h ? 2 : 0);
    KernelArg a[] = {buf(R), buf(rs), buf(w_norm), buf(gated), buf(mixed), buf(or_dummy(mixed16)),
                     buf(or_dummy(mixed_h)), val(T), val(flags)};
    L1(stream, "pf_gr_mix_r_kernel", (uint32_t) (T * N), a, 9);
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next,
                      float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(R), buf(bo), buf(inj), val(inj_ld), buf(w_norm_next), val(eps), buf(rs), buf(xn16)};
    L(stream, "pf_gr_write_norm_rs_kernel", (uint32_t) (T * HC * 256), 1, 1, 256, 1, 1, a, 8);
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    if (T <= 0) return;
    const int64_t n = T * LR;
    KernelArg a[] = {buf(lo), buf(lo16), val(n)};
    L1(stream, "pf_gr_silu_kernel", (uint32_t) n, a, 3);
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    if (T <= 0) return;
    const int flags = (mixed16 ? 1 : 0) | (mixed_h ? 2 : 0);
    KernelArg a[] = {buf(xn), buf(gated), buf(mixed), buf(or_dummy(mixed16)), buf(or_dummy(mixed_h)), val(T),
                     val(flags)};
    L1(stream, "pf_gr_mix_kernel", (uint32_t) (T * N), a, 7);
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(R), buf(bo), buf(inj), val(inj_ld), val(T)};
    L1(stream, "pf_gr_write_kernel", (uint32_t) (T * D), a, 5);
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(e), buf(R), val(T)};
    L1(stream, "pf_gr_broadcast_kernel", (uint32_t) (T * D), a, 3);
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T,
               void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(ab), buf(dt), buf(ssm_a), buf(gate), buf(beta), val(T)};
    L1(stream, "pf_gdn_gates_kernel", (uint32_t) (T * HV), a, 6);
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    if (T <= 0) return;
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;  // the old walk (A/B)
    if (serial || T <= 64) {
        KernelArg a[] = {buf(history), buf(qkv), buf(conv_w), buf(h), val(T)};
        L1(stream, "pf_gdn_conv_serial_kernel", (uint32_t) C, a, 5);
    } else {
        const uint32_t tiles = (uint32_t) ((T + 63) / 64);
        KernelArg a[] = {buf(history), buf(qkv), buf(conv_w), buf(h), val(T)};
        L(stream, "pf_gdn_conv_tiled_kernel", (uint32_t) C, tiles, 1, 128, 1, 1, a, 5);
        KernelArg b[] = {buf(history), buf(qkv), val(T)};
        L1(stream, "pf_gdn_conv_hist_kernel", (uint32_t) C, b, 3);
    }
    KernelArg c[] = {buf(h), val(eps)};
    L(stream, "pf_gdn_l2_kernel", 32 * 128, (uint32_t) T, 1, 128, 1, 1, c, 2);
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    if (T <= 0) return;
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;  // the one-block-per-head kernel (A/B)
    if (serial) {
        KernelArg a[] = {buf(state), buf(h), buf(gate), buf(beta), buf(z), buf(gamma), val(eps), buf(y), buf(y16),
                         val(T)};
        // HV threadgroups of (128, 4) threads
        L(stream, "pf_gdn_rec_kernel", (uint32_t) (HV * 128), 4, 1, 128, 4, 1, a, 10);
    } else {
        KernelArg a[] = {buf(state), buf(h), buf(gate), buf(beta), buf(y), val(T)};
        // HV * 4 threadgroups of (32, 4) threads
        L(stream, "pf_gdn_rec_cols_kernel", (uint32_t) (HV * 4 * 32), 4, 1, 32, 4, 1, a, 6);
        KernelArg b[] = {buf(z), buf(gamma), val(eps), buf(y), buf(y16)};
        L(stream, "pf_gdn_out_norm_kernel", (uint32_t) (T * 128), (uint32_t) HV, 1, 128, 1, 1, b, 5);
    }
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (T <= 0) return;
    if (n_expert == 512 || n_expert == 256) {
        const int reg = n_expert == 512 ? 16 : 8;
        KernelArg a[] = {buf(logits), buf(ids), buf(weights), val(T), val(reg)};
        L(stream, "pf_route_kernel", (uint32_t) (((T + 7) / 8) * 256), 1, 1, 256, 1, 1, a, 5);
    } else {
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    }
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    const int half_flag = 0;
    KernelArg a[] = {buf(blob), buf(gu16), buf(down16), val(half_flag)};
    L1(stream, "pf_blob_dequant_kernel", (uint32_t) (1280 * 640 + 2560 * 160), a, 4);
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    const int half_flag = 1;
    KernelArg a[] = {buf(blob), buf(gu16), buf(down16), val(half_flag)};
    L1(stream, "pf_blob_dequant_kernel", (uint32_t) (1280 * 640 + 2560 * 160), a, 4);
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(gu), buf(h16), val(n)};
    L1(stream, "pf_swiglu_il_kernel", (uint32_t) (n * 640), a, 3);
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(g), buf(u), buf(h16), val(n)};
    L1(stream, "pf_swiglu_pair_kernel", (uint32_t) (n * 640), a, 4);
}
void swiglu_split(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(gu), buf(h16), val(n)};
    L1(stream, "pf_swiglu_split_kernel", (uint32_t) (n * 640), a, 3);
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(dst), buf(src), val(n)};
    L1(stream, "pf_copy_i32_kernel", (uint32_t) n, a, 3);
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width,
                   void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(x16), buf(src), buf(dst16), val(n), val(width)};
    L1(stream, "pf_gather_rows16_kernel", (uint32_t) (n * width), a, 5);
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg,
                 float* bo, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(Dm), buf(slot), buf(w), buf(shared), buf(sg), buf(bo), val(T)};
    L1(stream, "pf_moe_combine_kernel", (uint32_t) (T * N), a, 7);
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    KernelArg a[] = {buf(x), buf(w), val(cols), val(ld), val(eps)};
    L(stream, "pf_rms_rows_kernel", (uint32_t) (rows * 256), 1, 1, 256, 1, 1, a, 5);
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base,
          void* stream) {
    if (T <= 0) return;
    const float theta_scale = std::pow(freq_base, -2.0f / 64.0f);
    const int32_t* mtab = strata::kernels::mrope_table();
    const int has = mtab ? 1 : 0;
    KernelArg a[] = {buf(x), val(heads), val(dim), val(ld), val(pos0), val(theta_scale),
                     buf(or_dummy(mtab)), val(has)};
    L(stream, "pf_rope_kernel", (uint32_t) (T * heads * 32), 1, 1, 32, 1, 1, a, 8);
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(q_full), buf(q), val(T)};
    L1(stream, "pf_split_q_kernel", (uint32_t) (T * 24 * 256), a, 3);
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    if (T <= 0) return;
    KernelArg a[] = {buf(attn), buf(q_full), buf(out16), val(T)};
    L1(stream, "pf_gate_attn_kernel", (uint32_t) (T * 24 * 256), a, 4);
}
void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table,
               int64_t page_size, uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale,
               uint16_t* v_scale, void* stream, const strata::kernels::KvHostPools* host,
               const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    using HP = strata::kernels::KvHostPools;
    const HP empty{};
    const HP& hs = host ? *host : empty;
    const HP& st = stage ? *stage : empty;
    // flags: 1 f16 pool present | 2 host f16 | 4 host int8 | 8 stage f16 | 16 stage int8
    const int flags = (k_pool ? 1 : 0) | (hs.k_pool ? 2 : 0) | (hs.k_q ? 4 : 0) | (st.k_pool ? 8 : 0) |
                      (st.k_q ? 16 : 0);
    KernelArg a[] = {buf(K),
                     buf(V),
                     buf(page_table),
                     val(pos0),
                     val(page_size),
                     buf(or_dummy(k_pool)),
                     buf(or_dummy(v_pool)),
                     buf(or_dummy(k_q)),
                     buf(or_dummy(v_q)),
                     buf(or_dummy(k_scale)),
                     buf(or_dummy(v_scale)),
                     buf(or_dummy(hs.k_pool)),
                     buf(or_dummy(hs.v_pool)),
                     buf(or_dummy(hs.k_q)),
                     buf(or_dummy(hs.v_q)),
                     buf(or_dummy(hs.k_scale)),
                     buf(or_dummy(hs.v_scale)),
                     buf(or_dummy(st.k_pool)),
                     buf(or_dummy(st.v_pool)),
                     buf(or_dummy(st.k_q)),
                     buf(or_dummy(st.v_q)),
                     buf(or_dummy(st.k_scale)),
                     buf(or_dummy(st.v_scale)),
                     val(flags)};
    L(stream, "pf_kv_append_kernel", (uint32_t) (T * 64), 2, 8, 64, 1, 1, a, 24);
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(x), buf(y), val(n)};
    L1(stream, "pf_to_f16_kernel", (uint32_t) n, a, 3);
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(x), buf(y), val(n)};
    L1(stream, "pf_to_bf16_kernel", (uint32_t) n, a, 3);
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    KernelArg a[] = {buf(x), buf(y), val(n)};
    L1(stream, "pf_round_f16_kernel", (uint32_t) n, a, 3);
}

}  // namespace strata::prefill
