// src/metal-qwen/stubs/native_flash_attn.cpp - Metal implementation for
// include/strata/kernels/native_flash_attn.hpp (workstream qsa).
//
// Launches strata_native_flash_attn_short_step from
// src/metal-qwen/kernels/native_flash_attn.metal; validation (throws)
// matches the CUDA TU.

#include "strata/kernels/native_flash_attn.hpp"

#include "../runtime/metal_kernel.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
struct Span {
    const void* p;
    size_t n, alignment;
};
void validate_spans(const Span* spans, int count) {
    for (int i = 0; i < count; ++i) {
        const auto a = reinterpret_cast<uintptr_t>(spans[i].p);
        if (!a || a % spans[i].alignment || spans[i].n > UINTPTR_MAX - a)
            throw std::invalid_argument("native FlashAttention requires nonnull aligned bounded spans");
        for (int j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<uintptr_t>(spans[j].p);
            if (a < b + spans[j].n && b < a + spans[i].n)
                throw std::invalid_argument("native FlashAttention requires disjoint buffers");
        }
    }
}
}  // namespace

void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* step,
                                  int64_t capacity, int max_context, const QsaShapes& shapes, float* output,
                                  int32_t* status, const uint16_t* mask, void* stream) {
    if (!stream || shapes.n_head != 24 || shapes.n_head_kv != 2 || shapes.head_dim != 256 ||
        shapes.idx_block != 4 || shapes.idx_top_k < 256 || capacity < 256 ||
        (uint64_t) capacity > std::numeric_limits<size_t>::max() / 1024 || max_context < 1 ||
        max_context > 256)
        throw std::invalid_argument(
            "native FlashAttention supports only Q24x256/KV2x256, capacity>=256 and context1..256 on an "
            "explicit stream");
    const size_t kv_bytes = (size_t) capacity * 1024;
    const Span spans[] = {{q, 24 * 256 * 4, 4}, {k, kv_bytes, 2}, {v, kv_bytes, 2},
                          {step, kStepCount * 4, 4}, {output, 24 * 256 * 4, 4},
                          {status, 4, 4}, {mask, 256 * 2, 2}};
    validate_spans(spans, mask ? 7 : 6);
    const int has_mask = mask != nullptr ? 1 : 0;
    const ::strata::metal::KernelArg args[] = {
        ::strata::metal::buf(q), ::strata::metal::buf(k), ::strata::metal::buf(v), ::strata::metal::buf(step),
        ::strata::metal::buf(mask ? (const void*) mask : (const void*) k), ::strata::metal::buf(output),
        ::strata::metal::buf(status), ::strata::metal::val(has_mask), ::strata::metal::val(max_context)};
    ::strata::metal::launch((cudaStream_t) stream, "strata_native_flash_attn_short_step", 24 * 256, 1, 1, 256,
                            1, 1, args, sizeof(args) / sizeof(args[0]));
}
}  // namespace strata::kernels
