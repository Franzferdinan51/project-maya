// src/metal-qwen/stubs/qsa_select.cpp - Metal implementation for
// include/strata/kernels/qsa_select.hpp (workstream qsa).
//
// Kernels in src/metal-qwen/kernels/qsa_score.metal. qsa_block_scores_tc is
// the tensor-core entry on CUDA (RISKS.md section 4 probe family): Metal
// has no TF32 path to fall back from, so it runs the same FP32 score kernel
// as qsa_block_scores and answers true when the geometry is supported,
// false (nothing launched) when it is not - the contract its caller uses.

#include "strata/kernels/qsa_select.hpp"

#include "../runtime/metal_kernel.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;

void launch_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                         int64_t nq, int64_t max_blocks, float* scores, void* stream, int64_t reach) {
    const int64_t reach_l = reach, max_blocks_l = max_blocks;
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(pooled), ::strata::metal::buf(dead),
                                               ::strata::metal::buf(q_idx), ::strata::metal::buf(steps),
                                               ::strata::metal::buf(scores), ::strata::metal::val(max_blocks_l),
                                               ::strata::metal::val(reach_l)};
    ::strata::metal::launch((cudaStream_t) stream, "strata_qsa_block_scores",
                            (uint32_t) (nq * reach), 1, 1, 256, 1, 1, args, sizeof(args) / sizeof(args[0]));
}
int64_t reach_of(int64_t max_blocks, int64_t active_blocks) {
    return active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
}
void launch_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                       int32_t* ids, void* stream) {
    const int64_t max_blocks_l = max_blocks, cap_l = cap;
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(scores), ::strata::metal::buf(steps),
                                               ::strata::metal::buf(ids), ::strata::metal::val(max_blocks_l),
                                               ::strata::metal::val(cap_l)};
    // Grid in threads: one 256-thread threadgroup per query.
    ::strata::metal::launch((cudaStream_t) stream, "strata_qsa_block_topk", (uint32_t) (nq * 256), 1, 1, 256,
                            1, 1, args, sizeof(args) / sizeof(args[0]));
}
}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                      int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                      int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // A block past a query's n_bid returns at once: the grid need only reach
    // the batch's largest n_bid (perf-review C-1).
    launch_block_scores(pooled, dead, q_idx, steps, nq, max_blocks, scores, stream,
                        reach_of(max_blocks, active_blocks));
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                         int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                         int64_t active_blocks) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * 16) return false;
    launch_block_scores(pooled, dead, q_idx, steps, nq, max_blocks, scores, stream,
                        reach_of(max_blocks, active_blocks));
    return true;
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    launch_block_topk(scores, steps, nq, max_blocks, cap, ids, stream);
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                        int64_t cap, const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    launch_block_topk(scores, steps, nq, max_blocks, cap, ids, stream);
}
}  // namespace strata::kernels
