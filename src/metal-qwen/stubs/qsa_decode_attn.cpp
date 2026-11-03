// src/metal-qwen/stubs/qsa_decode_attn.cpp - Metal implementation for
// include/strata/kernels/qsa_decode_attn.hpp (workstream qsa).
//
// qsa_decode_attn_scratch_floats is a pure host function, ported for real
// (64 cells per chunk; per chunk n_head partial outputs plus a max and a
// sum per head; 64 floats of slack). The device entries launch the chunk
// and merge kernels in src/metal-qwen/kernels/qsa_attn.metal with the
// same scratch layout as CUDA: per query, [acc: chunks*n_head*256]
// [m: chunks*n_head] [l: chunks*n_head] at qsa_decode_attn_scratch_floats
// stride. Geometry and pool-completeness validation match the CUDA TU.

#include "strata/kernels/qsa_decode_attn.hpp"

#include "../runtime/metal_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int HD = 256;     // head_dim
constexpr int G = 12;       // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;   // cells per block

int kv_mode_of(const QsaAttnPools& pools) {
    return pools.k_q4 != nullptr ? 2
           : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3 : (pools.k_q != nullptr ? 1 : 0));
}

// Launches chunk + merge for n_q queries whose steps are `steps` (n_q
// consecutive step records). `cap_ids` is the per-query ids stride (0 for
// the single-step form, as on CUDA) and `stride` the per-query scratch
// stride in floats (0 for the single-step form).
void launch_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                 int64_t cap_ids, int64_t stride, const QsaShapes& s, float* scratch, float* attn, int64_t n_q,
                 void* stream) {
    const int mode = kv_mode_of(pools);
    const int kv_heads = (int) s.n_head_kv, page_size = (int) s.page_size, n_head = (int) s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const long stride_l = (long) stride;
    // The chunk count derives from the selection capacity `cap` in both
    // forms; `cap` doubles as the per-query ids stride (harmless in the
    // step form, whose only query index is 0 - CUDA passes 0 there).
    const int chunks = (int) ((cap_ids + CHUNK - 1) / CHUNK);
    // GRFOLD fix: KernelArg::val binds by reference (see metal_kernel.hpp -
    // the value must stay alive until launch returns), so cap must be a
    // named local. Passing val((int) cap_ids) bound a temporary that died
    // at the end of the args-array statement; the chunk kernel then read a
    // garbage ids stride and every query row z >= 1 indexed its selection
    // out of the ids buffer (in practice: all-zero ids, cell 0 for every
    // cell - uniform attention over the first cell). Row 0 was immune
    // because its ids offset is 0 under any stride.
    const int cap_i = (int) cap_ids;
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) chunks * s.n_head;
    const void* ph = pools.page_table;  // placeholder for unset pools (never read in their mode)
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(q),
        ::strata::metal::buf(pools.k_pool ? (const void*) pools.k_pool : ph),
        ::strata::metal::buf(pools.v_pool ? (const void*) pools.v_pool : ph),
        ::strata::metal::buf(pools.k_q ? (const void*) pools.k_q : ph),
        ::strata::metal::buf(pools.v_q ? (const void*) pools.v_q : ph),
        ::strata::metal::buf(pools.k_scale ? (const void*) pools.k_scale : ph),
        ::strata::metal::buf(pools.v_scale ? (const void*) pools.v_scale : ph),
        ::strata::metal::buf(pools.k_q4 ? (const void*) pools.k_q4 : ph),
        ::strata::metal::buf(pools.v_q4 ? (const void*) pools.v_q4 : ph),
        ::strata::metal::buf(pools.page_table),
        ::strata::metal::buf(ids),
        ::strata::metal::buf(steps),
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
    ::strata::metal::launch((cudaStream_t) stream, "strata_qsa_attn_chunk", (uint32_t) chunks * 256,
                            (uint32_t) s.n_head_kv, (uint32_t) n_q, 256, 1, 1, args,
                            sizeof(args) / sizeof(args[0]));
    const ::strata::metal::KernelArg margs[] = {::strata::metal::buf(part_acc), ::strata::metal::buf(part_m),
                                                ::strata::metal::buf(part_l), ::strata::metal::buf(attn),
                                                ::strata::metal::val(chunks), ::strata::metal::val(n_head),
                                                ::strata::metal::val(stride_l)};
    ::strata::metal::launch((cudaStream_t) stream, "strata_qsa_attn_merge", (uint32_t) s.n_head * 256,
                            (uint32_t) n_q, 1, 256, 1, 1, margs, sizeof(margs) / sizeof(margs[0]));
}
}  // namespace

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = kv_mode_of(pools);
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4)
                                     : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    // The CUDA step form runs the shared kernel with cap = 0 / stride = 0
    // template defaults; the chunk count still derives from `cap`, which
    // the MSL kernel receives separately from the ids stride.
    launch_attn(q, pools, ids, step, cap, 0, s, scratch, attn, 1, stream);
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q,
                           void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int64_t stride = (int64_t) qsa_decode_attn_scratch_floats(cap, s);
    launch_attn(q, pools, ids, steps, cap, stride, s, scratch, attn, n_q, stream);
}
}  // namespace strata::kernels
