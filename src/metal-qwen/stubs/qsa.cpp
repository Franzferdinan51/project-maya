// src/metal-qwen/stubs/qsa.cpp - Metal implementation for
// include/strata/kernels/qsa.hpp (workstream qsa).
//
// qsa_step_fill is a pure host function, ported verbatim. The device
// entries launch the kernels in src/metal-qwen/kernels/qsa.metal. The
// non-step wrappers mirror the CUDA ones: they fill a host step, upload it
// to a lazily allocated device scratch (through the compat runtime, the
// same shape as the CUDA TU's cudaMalloc/cudaMemcpy), and call the step
// entry - so the launches stay constant-shaped and graph-friendly.

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"

#include "../runtime/metal_kernel.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "qsa: %s\n", what);
    std::exit(1);
}
void validate(const QsaShapes& s, const char* who) {
    if (s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.idx_dim <= 0 || s.idx_n_head <= 0 ||
        s.idx_block < 2 || s.page_size < 1) {
        std::fprintf(stderr, "qsa: %s: geometry is not set up\n", who);
        std::exit(1);
    }
    if (s.n_head % s.n_head_kv != 0) {
        std::fprintf(stderr, "qsa: %s: n_head %lld is not a multiple of n_head_kv %lld\n", who,
                     (long long) s.n_head, (long long) s.n_head_kv);
        std::exit(1);
    }
    if (s.head_dim % 4 != 0) {
        std::fprintf(stderr, "qsa: %s: head_dim %lld must be a multiple of 4 (the gather copies uint2)\n", who,
                     (long long) s.head_dim);
        std::exit(1);
    }
    if (s.n_rot <= 0 || s.n_rot % 2 != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim) {
        std::fprintf(stderr, "qsa: %s: n_rot %lld must be even and <= head_dim %lld and idx_dim %lld\n", who,
                     (long long) s.n_rot, (long long) s.head_dim, (long long) s.idx_dim);
        std::exit(1);
    }
    if (s.idx_n_head > 32) {
        std::fprintf(stderr, "qsa: %s: idx_n_head %lld > 32 (one warp per indexer head)\n", who,
                     (long long) s.idx_n_head);
        std::exit(1);
    }
}
inline unsigned grid_for(long long n, int threads) { return (unsigned) ((n + threads - 1) / threads); }

void launch(const char* fn, void* stream, uint32_t grid_x, uint32_t block_x,
            const ::strata::metal::KernelArg* args, size_t n_args) {
    ::strata::metal::launch((cudaStream_t) stream, fn, grid_x, 1, 1, block_x, 1, 1, args, n_args);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "qsa: %s (synchronise): %s\n", fn, cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

// ---- step upload (the CUDA TU's scratch, through the compat runtime) ----
int32_t* step_scratch() {
    static int32_t* d_step = nullptr;
    if (d_step == nullptr) {
        if (cudaMalloc((void**) &d_step, qsa_step_bytes()) != cudaSuccess) {
            std::fprintf(stderr, "qsa: step upload: cudaMalloc failed\n");
            std::exit(1);
        }
    }
    return d_step;
}
void step_upload_raw(const int32_t* h_step) {
    int32_t* d = step_scratch();
    if (cudaMemcpy(d, h_step, qsa_step_bytes(), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "qsa: step upload: cudaMemcpy failed\n");
        std::exit(1);
    }
}
const int32_t* step_upload(int64_t pos, int64_t n_kv_hint, const QsaShapes& s) {
    int32_t h[kStepCount];
    qsa_step_fill(h, pos, s);
    if (n_kv_hint >= 0 && n_kv_hint != (int64_t) h[kStepNKv]) {
        std::fprintf(stderr, "qsa: step upload: n_kv %lld disagrees with pos+1 = %d\n", (long long) n_kv_hint,
                     h[kStepNKv]);
        std::exit(1);
    }
    step_upload_raw(h);
    return step_scratch();
}
const int32_t* step_upload_width(int64_t width, const QsaShapes& s) {
    int32_t h[kStepCount];
    for (int i = 0; i < kStepCount; ++i) h[i] = 0;
    h[kStepWidth] = (int32_t) width;
    (void) s;
    step_upload_raw(h);
    return step_scratch();
}
}  // namespace

void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) fail("qsa_step_fill: pos < 0");
    // One place computes all four, exactly as the CUDA host function: n_kv,
    // n_bid and width are derived from the single position.
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                    const KvHostPools* host) {
    validate(s, "kv_append");
    if (step == nullptr) fail("kv_append: step is null");
    const long long n = s.n_head_kv * s.head_dim;
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const int has_host = (host != nullptr && host->k_pool != nullptr) ? 1 : 0;
    const void* host_k = has_host ? (const void*) host->k_pool : (const void*) k_pool;
    const void* host_v = has_host ? (const void*) host->v_pool : (const void*) v_pool;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(k_pool), ::strata::metal::buf(v_pool), ::strata::metal::buf(page_table),
        ::strata::metal::buf(step),   ::strata::metal::buf(kcur),   ::strata::metal::buf(vcur),
        ::strata::metal::buf(host_k), ::strata::metal::buf(host_v), ::strata::metal::val(has_host),
        ::strata::metal::val(kv_heads), ::strata::metal::val(head_dim), ::strata::metal::val(page_size)};
    launch("strata_kv_append", stream, grid_for(n, 128) * 128, 128, args, sizeof(args) / sizeof(args[0]));
}

void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (step == nullptr) fail("qsa_index: step is null");
    if (max_blocks <= 0) fail("qsa_index: max_blocks must be positive");
    const int idx_n_head = (int) s.idx_n_head, idx_dim = (int) s.idx_dim;
    const int64_t r = s.idx_block;
    const int has_bias = bias != nullptr ? 1 : 0;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(pooled), ::strata::metal::buf(q_idx),
        ::strata::metal::buf(has_bias ? (const void*) bias : (const void*) pooled),
        ::strata::metal::buf(step),   ::strata::metal::buf(cell_scores), ::strata::metal::val(has_bias),
        ::strata::metal::val(idx_n_head), ::strata::metal::val(idx_dim), ::strata::metal::val(r)};
    const uint32_t block = 32u * (uint32_t) idx_n_head;
    launch("strata_qsa_index", stream, (uint32_t) max_blocks * block, block, args,
           sizeof(args) / sizeof(args[0]));
}

void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream) {
    validate(s, "topk_512");
    if (step == nullptr) fail("topk_512: step is null");
    const int64_t width_max = qsa_selection_width(kTopkMaxCells, s);
    if (cap < width_max) {
        std::fprintf(stderr, "qsa: topk_512: cap %lld < the largest possible selection width %lld\n",
                     (long long) cap, (long long) width_max);
        std::exit(1);
    }
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(cell_scores), ::strata::metal::buf(step),
                                               ::strata::metal::buf(ids)};
    launch("strata_topk_512", stream, 256, 256, args, sizeof(args) / sizeof(args[0]));
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (step == nullptr) fail("kv_gather: step is null");
    if (max_ids <= 0) return;
    const int kv_heads = (int) s.n_head_kv, head_dim = (int) s.head_dim, page_size = (int) s.page_size;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(k_pool), ::strata::metal::buf(v_pool), ::strata::metal::buf(page_table),
        ::strata::metal::buf(ids),    ::strata::metal::buf(step),   ::strata::metal::buf(k_scratch),
        ::strata::metal::buf(v_scratch), ::strata::metal::val(kv_heads), ::strata::metal::val(head_dim),
        ::strata::metal::val(page_size)};
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    launch("strata_kv_gather", stream, grid_for(total, 256) * 256, 256, args, sizeof(args) / sizeof(args[0]));
}

void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch,
                     const int32_t* step, int64_t max_ids, const QsaShapes& s, float* attn, float* weights,
                     void* stream) {
    validate(s, "qsa_attend");
    if (step == nullptr) fail("qsa_attend: step is null");
    if (max_ids <= 0) fail("qsa_attend: max_ids must be positive");
    // The MSL kernel's score array is sized for the real geometry's largest
    // selection width (2051); the CUDA kernel sizes shared memory from
    // max_ids dynamically. No caller can select more than the width, so a
    // larger capacity is refused loudly rather than silently truncated.
    if (max_ids > 2051) {
        std::fprintf(stderr, "qsa: qsa_attend: max_ids %lld over the Metal kernel's 2051 score capacity\n",
                     (long long) max_ids);
        std::exit(1);
    }
    const int n_head = (int) s.n_head, n_head_kv = (int) s.n_head_kv, head_dim = (int) s.head_dim;
    const int has_weights = weights != nullptr ? 1 : 0;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(q), ::strata::metal::buf(k_scratch), ::strata::metal::buf(v_scratch),
        ::strata::metal::buf(step), ::strata::metal::buf(attn),
        ::strata::metal::buf(has_weights ? (const void*) weights : (const void*) attn),
        ::strata::metal::val(has_weights), ::strata::metal::val(n_head), ::strata::metal::val(n_head_kv),
        ::strata::metal::val(head_dim)};
    launch("strata_qsa_attend", stream, (uint32_t) (s.n_head * s.head_dim), (uint32_t) s.head_dim, args,
           sizeof(args) / sizeof(args[0]));
}

void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos, const float* kcur,
               const float* vcur, const QsaShapes& s, void* stream) {
    if (pos < 0) {
        validate(s, "kv_append");
        fail("kv_append: pos < 0");
    }
    kv_append_step(k_pool, v_pool, page_table, step_upload(pos, -1, s), kcur, vcur, s, stream);
}

void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    validate(s, "indexer_key_append");
    if (pos_dev == nullptr) fail("indexer_key_append: pos_dev is null");
    if (b.tail == nullptr || b.dead == nullptr || b.pooled == nullptr || b.block_pos == nullptr)
        fail("indexer_key_append: the indexer buffers are not all set (tail/dead/pooled/block_pos)");
    const int32_t* tab = mrope_table();
    const int has_mtab = tab != nullptr ? 1 : 0;
    const int idx_dim = (int) s.idx_dim, r = (int) s.idx_block, n_rot = (int) s.n_rot;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(raw),     ::strata::metal::buf(pos_dev), ::strata::metal::buf(w_k_norm),
        ::strata::metal::buf(b.tail),  ::strata::metal::buf(b.dead),  ::strata::metal::buf(b.pooled),
        ::strata::metal::buf(b.block_pos),
        ::strata::metal::buf(cos_tab), ::strata::metal::buf(sin_tab),
        ::strata::metal::buf(has_mtab ? (const void*) tab : (const void*) raw),
        ::strata::metal::val(has_mtab), ::strata::metal::val(pos_base), ::strata::metal::val(eps),
        ::strata::metal::val(idx_dim),  ::strata::metal::val(r),        ::strata::metal::val(n_rot)};
    launch("strata_indexer_key_append", stream, (uint32_t) idx_dim, (uint32_t) idx_dim, args,
           sizeof(args) / sizeof(args[0]));
}

void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s,
               int64_t n_kv, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (n_bid < 0 || n_kv <= 0) fail("qsa_index: n_bid < 0 or n_kv <= 0");
    if ((n_kv / s.idx_block) != n_bid) {
        std::fprintf(stderr, "qsa: qsa_index: n_bid %lld is not n_kv %lld / r %lld\n", (long long) n_bid,
                     (long long) n_kv, (long long) s.idx_block);
        std::exit(1);
    }
    qsa_index_step(pooled, q_idx, bias, s, step_upload(n_kv - 1, n_kv, s), n_bid + 1, cell_scores, stream);
}

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids,
              void* stream) {
    validate(s, "topk_512");
    if (n_kv <= 0) return;
    if (n_kv > kTopkMaxCells) {
        std::fprintf(stderr, "qsa: topk_512: n_kv %lld > kTopkMaxCells %lld (one block; phase 3 replaces this)\n",
                     (long long) n_kv, (long long) kTopkMaxCells);
        std::exit(1);
    }
    topk_512_step(cell_scores, s, cap, step_upload(n_kv - 1, n_kv, s), ids, stream);
}

void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (n_ids <= 0) return;
    kv_gather_step(k_pool, v_pool, page_table, ids, step_upload_width(n_ids, s), n_ids, s, k_scratch, v_scratch,
                   stream);
}

void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream) {
    validate(s, "qsa_attend");
    if (n_ids < 0) fail("qsa_attend: n_ids < 0");
    if (n_ids == 0) {
        qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(0, s), qsa_selection_width(kTopkMaxCells, s),
                        s, attn, weights, stream);
        return;
    }
    qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(n_ids, s), n_ids, s, attn, weights, stream);
}

void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    const long long n = s.n_head * s.head_dim;
    const int n_head = (int) s.n_head, head_dim = (int) s.head_dim;
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(attn), ::strata::metal::buf(q_full),
                                               ::strata::metal::buf(out), ::strata::metal::val(n_head),
                                               ::strata::metal::val(head_dim)};
    launch("strata_qsa_gate_apply", stream, grid_for(n, 256) * 256, 256, args, sizeof(args) / sizeof(args[0]));
}

void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    const long long n = s.n_head * s.head_dim;
    const int n_head = (int) s.n_head, head_dim = (int) s.head_dim;
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(attn), ::strata::metal::buf(q_full),
                                               ::strata::metal::buf(out), ::strata::metal::val(n_head),
                                               ::strata::metal::val(head_dim)};
    launch("strata_qsa_gate_apply_f32", stream, grid_for(n, 256) * 256, 256, args,
           sizeof(args) / sizeof(args[0]));
}
}  // namespace strata::kernels
