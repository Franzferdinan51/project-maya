// src/metal-qwen/stubs/native_qsa_indexer.cpp - Metal implementation for
// include/strata/kernels/native_qsa_indexer.hpp (workstream qsa).
//
// The enabled flag is pure host state, ported for real (RISKS.md section 4).
// Kernels in src/metal-qwen/kernels/qsa_indexer.metal; validation and the
// host-side float theta_scale match the CUDA TU. The batch entry launches
// first/blocks/tail in the CUDA order (the blocks kernel reads the tail the
// tail kernel is about to refresh, so the order is semantic).

#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"

#include "../runtime/metal_kernel.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> g_enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;

struct Span {
    const void* p;
    size_t n;
};
void validate(Span s) {
    const auto p = reinterpret_cast<uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
void launch(const char* fn, void* stream, uint32_t grid_x, uint32_t block_x,
            const ::strata::metal::KernelArg* args, size_t n_args) {
    ::strata::metal::launch((cudaStream_t) stream, fn, grid_x, 1, 1, block_x, 1, 1, args, n_args);
}
}  // namespace

void native_qsa_indexer_set_enabled(bool value) { g_enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return g_enabled.load(std::memory_order_relaxed); }

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, float freq_base, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || max_cells < 1 ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument(
            "native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, "
            "valid frequency and explicit stream");
    const Span spans[] = {{raw, D * 4},
                          {relative_pos_device, 4},
                          {gamma, D * 4},
                          {b.tail, (R - 1) * D * 4},
                          {b.dead, D * 4},
                          {b.pooled, size_t(max_cells / R + 1) * D * 4},
                          {b.block_pos, 4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i)
        for (int j = i + 1; j < 7; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    const int32_t* tab = mrope_table();
    const int has_mtab = tab != nullptr ? 1 : 0;
    const void* tab_arg = has_mtab ? (const void*) tab : (const void*) raw;
    const int max_cells_i = (int) max_cells;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(raw),  ::strata::metal::buf(relative_pos_device), ::strata::metal::buf(gamma),
        ::strata::metal::buf(b.tail), ::strata::metal::buf(b.dead),            ::strata::metal::buf(b.pooled),
        ::strata::metal::buf(b.block_pos), ::strata::metal::buf(tab_arg),      ::strata::metal::val(has_mtab),
        ::strata::metal::val(pos_base),    ::strata::metal::val(epsilon),      ::strata::metal::val(theta_scale),
        ::strata::metal::val(max_cells_i)};
    launch("strata_native_indexer_append", stream, THREADS, THREADS, args, sizeof(args) / sizeof(args[0]));
}

void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base,
                                     const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, float freq_base, void* stream) {
    if (n <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || p0 < 0 || p0 + n > max_cells ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument("native QSA indexer (batch): bad geometry, positions or parameters");
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    const int32_t* tab = mrope_table();
    const int has_mtab = tab != nullptr ? 1 : 0;
    const void* tab_arg = has_mtab ? (const void*) tab : (const void*) raw;
    if (p0 == 0) {
        const ::strata::metal::KernelArg args[] = {::strata::metal::buf(raw), ::strata::metal::buf(gamma),
                                                   ::strata::metal::buf(b.dead), ::strata::metal::buf(b.pooled),
                                                   ::strata::metal::val(epsilon),
                                                   ::strata::metal::val(theta_scale)};
        launch("strata_native_indexer_append_first", stream, THREADS, THREADS, args,
               sizeof(args) / sizeof(args[0]));
    }
    // Completed blocks: those whose last cell (4b+3) lies in [p0, p0 + n).
    const int64_t first = p0 <= R - 1 ? 0 : (p0 - (R - 1) + R - 1) / R;
    const int64_t hi = p0 + n - 1 >= R - 1 ? (p0 + n - 1 - (R - 1)) / R : -1;
    if (hi >= first) {
        const ::strata::metal::KernelArg args[] = {
            ::strata::metal::buf(raw),   ::strata::metal::buf(gamma),  ::strata::metal::buf(b.tail),
            ::strata::metal::buf(b.dead), ::strata::metal::buf(b.pooled), ::strata::metal::buf(b.block_pos),
            ::strata::metal::buf(tab_arg), ::strata::metal::val(has_mtab), ::strata::metal::val(p0),
            ::strata::metal::val(pos_base), ::strata::metal::val(epsilon), ::strata::metal::val(theta_scale),
            ::strata::metal::val(first), ::strata::metal::val(hi)};
        launch("strata_native_indexer_append_blocks", stream, (uint32_t) ((hi - first + 1) * THREADS), THREADS,
               args, sizeof(args) / sizeof(args[0]));
    }
    {
        const ::strata::metal::KernelArg args[] = {::strata::metal::buf(raw), ::strata::metal::buf(b.tail),
                                                   ::strata::metal::val(n), ::strata::metal::val(p0)};
        launch("strata_native_indexer_append_tail", stream, (R - 1) * D, D, args,
               sizeof(args) / sizeof(args[0]));
    }
}
}  // namespace strata::kernels
