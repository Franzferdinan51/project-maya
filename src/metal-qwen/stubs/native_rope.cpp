// src/metal-qwen/stubs/native_rope.cpp - Metal implementation for
// include/strata/kernels/native_rope.hpp (workstream qsa).
//
// The enabled flag is pure host state, ported for real (RISKS.md section 4).
// native_rope_apply launches strata_native_rope_apply from
// src/metal-qwen/kernels/rope.metal; argument validation and the host-side
// float theta_scale match the CUDA TU exactly.

#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"

#include "../runtime/metal_kernel.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> g_enabled{false};
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
}  // namespace

void native_rope_set_enabled(bool enabled) { g_enabled.store(enabled, std::memory_order_relaxed); }
bool native_rope_enabled() { return g_enabled.load(std::memory_order_relaxed); }

void native_rope_apply(const float* x, float* out, int rows, int head_dim, int n_rot, float freq_base,
                       const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || n_rot != 64 || !std::isfinite(freq_base) ||
        freq_base <= 1.0f || reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument(
            "native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base and explicit "
            "stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    // Match the CUDA host side: float powf for the scale, device pow per pair.
    const float theta_scale = powf(freq_base, -2.0f / n_rot);
    const int32_t* tab = mrope_table();
    const int has_mtab = tab != nullptr ? 1 : 0;
    const void* tab_arg = has_mtab ? (const void*) tab : (const void*) positions;
    const int width = head_dim;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(x),         ::strata::metal::buf(out), ::strata::metal::buf(positions),
        ::strata::metal::buf(tab_arg),   ::strata::metal::val(has_mtab), ::strata::metal::val(rows),
        ::strata::metal::val(width),     ::strata::metal::val(n_rot),    ::strata::metal::val(theta_scale)};
    const uint32_t grid_x = (uint32_t) ((head_dim / 2 + 127) / 128) * 128;
    ::strata::metal::launch((cudaStream_t) stream, "strata_native_rope_apply", grid_x, (uint32_t) rows, 1, 128,
                            1, 1, args, sizeof(args) / sizeof(args[0]));
}
}  // namespace strata::kernels
