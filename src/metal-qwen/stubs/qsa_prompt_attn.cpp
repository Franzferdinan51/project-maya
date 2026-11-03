// src/metal-qwen/stubs/qsa_prompt_attn.cpp - Metal implementation for
// include/strata/kernels/qsa_prompt_attn.hpp (workstream qsa).
//
// The entry is a capability probe by design: false (nothing launched)
// when it cannot serve the request, and the caller falls back to
// qsa_decode_attn_batch. Metal serves the same modes as the CUDA tensor
// path - fp16 pools, int8, and the hybrid int8-K/q4-V - and refuses pure
// q4_0 pools and non-24/2/256 geometry with false, exactly as the header
// promises. The math runs through the FP32 chunk + merge kernels in
// src/metal-qwen/kernels/qsa_attn.metal (the CUDA prompt kernel's FP16
// hi/lo tensor-core evaluation differs by rounding only), using a
// grow-only scratch allocated through the compat runtime, reused across
// query blocks of 64 (the entry has no caller scratch, unlike the decode
// entries).

#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include "../runtime/metal_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int HD = 256;
constexpr int G = 12;
constexpr int CHUNK = 64;
constexpr int64_t kQueriesPerBlock = 64;

float* prompt_scratch(int64_t floats) {
    static float* scratch = nullptr;
    static int64_t have = 0;
    if (floats > have) {
        float* fresh = nullptr;
        if (cudaMalloc((void**) &fresh, (size_t) floats * sizeof(float)) != cudaSuccess) return nullptr;
        scratch = fresh;
        have = floats;
    }
    return scratch;
}
}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 ||
        !ids || !steps || !pools.page_table)
        return false;
    const int mode = pools.k_q != nullptr && pools.v_q4 != nullptr ? 3 : (pools.k_q != nullptr ? 1 : 0);
    if (mode == 3 && !pools.k_scale) return false;
    if (mode == 1 && (!pools.v_q || !pools.k_scale || !pools.v_scale)) return false;
    if (mode == 0 && (!pools.k_pool || !pools.v_pool)) return false;

    const int chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const int64_t stride = (int64_t) qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = prompt_scratch(stride * kQueriesPerBlock);
    if (scratch == nullptr) return false;
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) chunks * s.n_head;
    const int kv_heads = (int) s.n_head_kv, page_size = (int) s.page_size, n_head = (int) s.n_head;
    const int cap_i = (int) cap;
    const long stride_l = (long) stride;
    const float scale = 1.0f / sqrtf((float) HD);
    const void* ph = pools.page_table;
    cudaStream_t st = (cudaStream_t) stream;
    for (int64_t q0 = 0; q0 < n_q; q0 += kQueriesPerBlock) {
        const int64_t nb = n_q - q0 < kQueriesPerBlock ? n_q - q0 : kQueriesPerBlock;
        const float* qb = q + (size_t) q0 * s.n_head * HD;
        const int32_t* idsb = ids + (size_t) q0 * cap;
        const int32_t* stepsb = steps + (size_t) q0 * kStepCount;
        float* attnb = attn + (size_t) q0 * s.n_head * HD;
        const ::strata::metal::KernelArg args[] = {
            ::strata::metal::buf(qb),
            ::strata::metal::buf(pools.k_pool ? (const void*) pools.k_pool : ph),
            ::strata::metal::buf(pools.v_pool ? (const void*) pools.v_pool : ph),
            ::strata::metal::buf(pools.k_q ? (const void*) pools.k_q : ph),
            ::strata::metal::buf(pools.v_q ? (const void*) pools.v_q : ph),
            ::strata::metal::buf(pools.k_scale ? (const void*) pools.k_scale : ph),
            ::strata::metal::buf(pools.v_scale ? (const void*) pools.v_scale : ph),
            ::strata::metal::buf(pools.k_q4 ? (const void*) pools.k_q4 : ph),
            ::strata::metal::buf(pools.v_q4 ? (const void*) pools.v_q4 : ph),
            ::strata::metal::buf(pools.page_table),
            ::strata::metal::buf(idsb),
            ::strata::metal::buf(stepsb),
            ::strata::metal::buf(part_acc),
            ::strata::metal::buf(part_m),
            ::strata::metal::buf(part_l),
            ::strata::metal::val(mode),
            ::strata::metal::val(kv_heads),
            ::strata::metal::val(page_size),
            ::strata::metal::val(scale),
            ::strata::metal::val(chunks),
            ::strata::metal::val(cap_i),
            ::strata::metal::val(stride_l)};
        ::strata::metal::launch(st, "strata_qsa_attn_chunk", (uint32_t) chunks * 256, (uint32_t) s.n_head_kv,
                                (uint32_t) nb, 256, 1, 1, args, sizeof(args) / sizeof(args[0]));
        const ::strata::metal::KernelArg margs[] = {::strata::metal::buf(part_acc), ::strata::metal::buf(part_m),
                                                    ::strata::metal::buf(part_l), ::strata::metal::buf(attnb),
                                                    ::strata::metal::val(chunks), ::strata::metal::val(n_head),
                                                    ::strata::metal::val(stride_l)};
        ::strata::metal::launch(st, "strata_qsa_attn_merge", (uint32_t) s.n_head * 256, (uint32_t) nb, 1, 256,
                                1, 1, margs, sizeof(margs) / sizeof(margs[0]));
    }
    return true;
}
}  // namespace strata::kernels
