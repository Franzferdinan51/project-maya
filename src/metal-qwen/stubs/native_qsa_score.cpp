// src/metal-qwen/stubs/native_qsa_score.cpp - Metal implementation for
// include/strata/kernels/native_qsa_score.hpp (workstream qsa).
//
// The enabled flag is pure host state, ported for real (RISKS.md section 4,
// which names this pair explicitly). The kernel
// (src/metal-qwen/kernels/qsa_score.metal) implements the ordered scalar-F32
// semantics of the CUDA fallback path (full-precision dots, ReLU per head,
// heads added in order) rather than the TF32 tensor-core rounding of the
// primary CUDA path; validation matches the CUDA TU.

#include "strata/kernels/native_qsa_score.hpp"

#include "../runtime/metal_kernel.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> g_enabled{false};
constexpr int D = 128, HEADS = 4, R = 4;

struct Span {
    const void* p;
    size_t n;
};
void validate(Span s) {
    const auto p = reinterpret_cast<uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
}  // namespace

void native_qsa_score_set_enabled(bool value) { g_enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_score_enabled() { return g_enabled.load(std::memory_order_relaxed); }

void native_qsa_score(const float* pooled, const float* query, const float* bias, const QsaShapes& s,
                      const int32_t* step, int64_t max_blocks, int64_t max_cells, float* cells, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_n_head != HEADS || s.idx_block != R || s.idx_top_k != 2048 ||
        max_cells < 1 || max_cells > INT32_MAX - 3 || max_blocks != max_cells / R + 1)
        throw std::invalid_argument(
            "native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[] = {{pooled, size_t(max_blocks) * D * 4},
                          {query, HEADS * D * 4},
                          {step, kStepCount * 4},
                          {cells, size_t(max_cells) * 4},
                          {bias, bias ? size_t(max_blocks) * 4 : 0}};
    const int count = bias ? 5 : 4;
    for (int i = 0; i < count; ++i) validate(spans[i]);
    for (int i = 0; i < count; ++i)
        for (int j = i + 1; j < count; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA score spans overlap");
    const int has_bias = bias != nullptr ? 1 : 0;
    const int max_cells_i = (int) max_cells;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(pooled), ::strata::metal::buf(query),
        ::strata::metal::buf(bias ? (const void*) bias : (const void*) pooled),
        ::strata::metal::buf(step),   ::strata::metal::buf(cells),
        ::strata::metal::val(has_bias), ::strata::metal::val(max_cells_i)};
    // One thread per block row; rows past the step's n_bid return at once.
    ::strata::metal::launch((cudaStream_t) stream, "strata_native_qsa_score", (uint32_t) max_blocks, 1, 1, 256,
                            1, 1, args, sizeof(args) / sizeof(args[0]));
}
}  // namespace strata::kernels
