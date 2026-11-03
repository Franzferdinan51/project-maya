// src/metal-qwen/stubs/native_router.cpp - Metal implementation for include/strata/kernels/native_router.hpp.
// Replaces the stage-1 loud stub. Host launcher over the MSL kernels in
// src/metal-qwen/kernels/moe.metal via strata::metal::launch; semantics mirror
// src/kernels/cuda/native_router.cu (see that file and the header for the contract).

#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>

#include "../runtime/metal_kernel.hpp"

namespace strata::kernels {
namespace {
std::atomic<bool>& flag() {
    static std::atomic<bool> f{false};
    return f;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void launch_rows(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    const metal::KernelArg args[] = {metal::buf(logits), metal::buf(ids), metal::buf(weights)};
    metal::launch((cudaStream_t) stream, "native_router_k", (uint32_t) n_tok * 32u, 1, 1, 32, 1, 1, args, 3);
}
}  // namespace

// Probe pair (RISKS.md S4): pure host state, ported for real - the flag lives in
// native_router.cu as a std::atomic<bool>; the same shape here.
void native_router_set_enabled(bool enabled) { flag().store(enabled, std::memory_order_relaxed); }
bool native_router_enabled() { return flag().load(std::memory_order_relaxed); }

void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4) ||
        overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4) ||
        overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    launch_rows(logits, ids, weights, 1, stream);
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) ||
        !valid(ids, (size_t) n_tok * 10 * 4) || !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument(
            "native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    launch_rows(logits, ids, weights, n_tok, stream);
}
}  // namespace strata::kernels
