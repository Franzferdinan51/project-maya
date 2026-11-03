// src/metal-qwen/stubs/native_moe.cpp - Metal implementation for include/strata/kernels/native_moe.hpp.
// Replaces the stage-1 loud stub. Host launcher over the MSL kernels in
// src/metal-qwen/kernels/moe.metal via strata::metal::launch; semantics mirror
// src/kernels/cuda/native_moe.cu (see that file and the header for the contract).

#include "strata/kernels/native_moe.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "../runtime/metal_kernel.hpp"

namespace strata::kernels {
namespace {
std::atomic<bool>& flag() {
    static std::atomic<bool> f{false};
    return f;
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
struct Args {
    int32_t n_embd, k, has_shared;
};
void launch_combine(const float* parts, const float* weights, const float* shared, float* output,
                    int64_t n_embd, int64_t k, int n_tok, void* stream) {
    const Args a{(int32_t) n_embd, (int32_t) k, shared != nullptr ? 1 : 0};
    // A null shared pointer is never read (has_shared == 0); pass parts to keep the
    // argument list well-formed.
    const metal::KernelArg args[] = {metal::buf(parts), metal::buf(weights),
                                     metal::buf(shared ? shared : parts), metal::buf(output), metal::val(a)};
    metal::launch((cudaStream_t) stream, "native_combine_k", (uint32_t) n_embd, (uint32_t) n_tok, 1, 256, 1, 1,
                  args, 5);
}
}  // namespace

// Probe pair (RISKS.md S4): pure host state, ported for real (a std::atomic<bool> in
// native_moe.cu; the same shape here).
void native_moe_combine_set_enabled(bool enabled) { flag().store(enabled, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return flag().load(std::memory_order_relaxed); }

void native_moe_combine(const float* parts, const float* weights, const float* shared, float* output,
                        int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes) ||
        (shared && !valid_span(shared, row_bytes)) || overlap(output, row_bytes, parts, part_bytes) ||
        overlap(output, row_bytes, weights, weight_bytes) ||
        (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    launch_combine(parts, weights, shared, output, n_embd, k, 1, stream);
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    launch_combine(parts, weights, shared, output, n_embd, k, n_tok, stream);
}
}  // namespace strata::kernels
