// native_gdn.cpp - Metal host launcher for the pinned native GDN step.
#include "strata/kernels/native_gdn.hpp"
#include "../runtime/metal_kernel.hpp"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace strata::kernels {
namespace { std::atomic<bool> g_enabled{false};
bool valid_span(const void* p, size_t bytes) {
    auto a = reinterpret_cast<uintptr_t>(p); return p && a % 4 == 0 && bytes <= UINTPTR_MAX - a; }
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an; } }
void native_gdn_set_enabled(bool v) { g_enabled.store(v, std::memory_order_relaxed); }
bool native_gdn_enabled() { return g_enabled.load(std::memory_order_relaxed); }
void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream) {
    constexpr int S = 128;
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 || shape.h_v > 65535 ||
        shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t sb = size_t(S) * S * size_t(shape.h_v) * 4, qkb = size_t(S) * size_t(shape.h_k) * 4,
                 ob = size_t(S) * size_t(shape.h_v) * 4, hb = size_t(shape.h_v) * 4;
    if (!valid_span(state, sb) || !valid_span(output, ob) || overlap(state, sb, output, ob))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* in[] = {q, k, v, gate, beta}; const size_t by[] = {qkb, qkb, ob, hb, hb};
    for (int i = 0; i < 5; ++i)
        if (!valid_span(in[i], by[i]) || overlap(state, sb, in[i], by[i]) || overlap(output, ob, in[i], by[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    int hk = (int)shape.h_k, hv = (int)shape.h_v;
    float scale = 1.0f / sqrtf(float(S));
    using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
    KernelArg a[] = {buf(state), buf(q), buf(k), buf(v), buf(gate), buf(beta), buf(output), val(hk), val(hv), val(scale)};
    strata::metal::launch((cudaStream_t)stream, "native_gdn_step_kernel", (uint32_t)(hv * S), 1, 1, 128, 1, 1, a, 10);
}
}  // namespace strata::kernels
