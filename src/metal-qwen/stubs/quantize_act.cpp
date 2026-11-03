// src/metal-qwen/stubs/quantize_act.cpp - Metal implementation for
// include/strata/kernels/quantize_act.hpp (was the stage-1 stub). One thread
// per block, the CUDA kernels' shapes; see misc.metal for the one documented
// delta (q8_0's double division becomes float division in MSL).
#include "strata/kernels/quantize_act.hpp"

#include "../kernels/dense_host.hpp"

namespace strata::kernels {
namespace {
void launch1(const char* fn, const void* a, void* b, int64_t n_elems, int block_elems, void* stream) {
    const uint64_t nb = (uint64_t) (n_elems / block_elems);
    const metal::KernelArg args[] = {metal::buf(a), metal::buf(b), metal::val(nb)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), fn, (uint32_t) nb, 128, args, 3,
                              stream == nullptr);
}
}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % 32 != 0) dense_detail::bad_shape("quantize_q8_0: n is not a multiple of 32");
    launch1("quantize_q8_0_k", x, blocks, n, 32, stream);
}

void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % 32 != 0) dense_detail::bad_shape("quantize_q8_0_scaled: n is not a multiple of 32");
    if (scales == nullptr) dense_detail::bad_shape("quantize_q8_0_scaled: scales is null");
    const uint64_t nb = (uint64_t) (n / 32);
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(blocks), metal::buf(scales), metal::val(nb)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "quantize_q8_0_scaled_k", (uint32_t) nb, 128,
                              args, 4, stream == nullptr);
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    launch1("dequant_q8_0_k", blocks, x, n, 32, stream);
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % 256 != 0) dense_detail::bad_shape("quantize_q8_K: n is not a multiple of 256");
    launch1("quantize_q8_K_k", x, blocks, n, 256, stream);
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    launch1("dequant_q8_K_k", blocks, x, n, 256, stream);
}

}  // namespace strata::kernels
