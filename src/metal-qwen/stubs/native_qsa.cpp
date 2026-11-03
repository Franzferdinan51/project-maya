// src/metal-qwen/stubs/native_qsa.cpp - Metal implementation for
// include/strata/kernels/native_qsa.hpp (workstream qsa).
//
// The enabled flag is pure host state, ported for real (RISKS.md section 4).
// The kernels live in src/metal-qwen/kernels/native_qsa.metal; validation
// (throws on bad spans) matches the CUDA TU.

#include "strata/kernels/native_qsa.hpp"

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

size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || (uint64_t) cols * rows > (uint64_t) std::numeric_limits<int>::max())
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return (size_t) cols * rows;
}
bool valid(const void* ptr, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, size_t in_bytes, const float* weight, size_t weight_bytes, float* output,
             void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument(
            "native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
}  // namespace

void native_qsa_set_enabled(bool value) { g_enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return g_enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols,
                                  int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, (size_t) n_cols * 4, output, stream);
    const uint32_t block = n_cols < 1024 ? 256 : 1024;
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(input), ::strata::metal::buf(gamma),
                                               ::strata::metal::buf(output), ::strata::metal::val(n_cols),
                                               ::strata::metal::val(epsilon)};
    // Grid is in threads: one threadgroup per row.
    ::strata::metal::launch((cudaStream_t) stream, "strata_native_qsa_rms_norm_weighted_eps",
                            block * (uint32_t) n_rows, 1, 1, block, 1, 1, args,
                            sizeof(args) / sizeof(args[0]));
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output, int n_head, int head_dim,
                           void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(attn), ::strata::metal::buf(q_full),
                                               ::strata::metal::buf(output), ::strata::metal::val(n_head),
                                               ::strata::metal::val(head_dim)};
    ::strata::metal::launch((cudaStream_t) stream, "strata_native_qsa_gate_apply", (uint32_t) count, 1, 1, 256,
                            1, 1, args, sizeof(args) / sizeof(args[0]));
}
}  // namespace strata::kernels
