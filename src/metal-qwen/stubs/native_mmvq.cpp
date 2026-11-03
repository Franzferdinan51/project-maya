// src/metal-qwen/stubs/native_mmvq.cpp - Metal implementation for
// include/strata/kernels/native_mmvq.hpp (was the stage-1 stub).
//
// Probe / pure-host entries (native_q8_1_bytes, the multi-exact flag pair,
// native_mmvq_supported, native_mmvq_weight_bytes) are ported from the CUDA
// TU's host arithmetic. The kernel entries run ONE generic MSL kernel
// (dense.metal's mmvq_k): Q8_1 quantization first (quantize_q8_1_k, the CUDA
// kernel's formula), then a per-(row, column) sequential dot of dequantized
// weight x dequantized Q8_1 activation. DELTA vs the CUDA MMVQ: the CUDA
// kernels use integer dp4a dots with the K formats' scale/min decomposition
// grouped per sub-block; this port multiplies the same dequantized values in
// float and accumulates in element order, so results differ only by float
// rounding (measured in the dense parity harness report). The multi-column
// layout is always the "exact" one - every column is computed exactly as a
// single-column call - whatever native_mmvq_multi_exact() says; the flag pair
// is kept for the engine's contract, and the upstream (non-exact) multi
// layout is not reproduced. The i-quant types dispatch to iq_mmvq, as in CUDA.
#include "strata/kernels/native_mmvq.hpp"

#include <limits>
#include <stdexcept>

#include "../kernels/dense_host.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {
namespace {
bool g_multi_exact = true;  // the CUDA TU's default (measured-layout rule in native_mmvq.cu)

void validate_shape(int n_in, int ncols, int block_elems = 32) {
    if (n_in <= 0 || n_in % block_elems != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (ncols < 1 || ncols > 8) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0)
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
int block_elems_of(int type) {
    switch (type) {
    case 2: case 6: case 8: case 20: return 32;
    case 42: return 64;
    default: return 256;
    }
}
void mmvq_launch(int type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                 void* stream) {
    validate_shape(n_in, ncols, block_elems_of(type));
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const uint32_t ni = (uint32_t) n_in, no = (uint32_t) n_out;
    const metal::KernelArg args[] = {metal::buf(weights), metal::buf(x_q8_1), metal::buf(y), metal::val(type),
                                     metal::val(ni),        metal::val(no)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "mmvq_k", no * (uint32_t) ncols, 128, args, 6,
                              false);
}
void f32_launch(int type, const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in,
                int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, block_elems_of(type));
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    mmvq_launch(type, weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}
}  // namespace

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    // Pure host arithmetic, ported from the CUDA TU: one 36-byte block_q8_1 per 32
    // elements per column, with the CUDA validation contract (throws on bad shapes).
    if (n_in <= 0 || n_in % 32 != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (ncols < 1 || ncols > 8) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    return std::size_t(ncols) * std::size_t(n_in / 32) * 36;
}

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

bool native_mmvq_supported(int ggml_type) noexcept {
    // The CUDA TU's list, verbatim: the ten native formats plus the i-quant types it
    // dispatches to iq_mmvq (19 is handled by native_mmvq_weight_bytes but is NOT in
    // the CUDA supported list - kept that way here).
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    // Pure host arithmetic, ported from the CUDA TU (block geometry table + validation).
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    if (n_in <= 0 || n_in % block_elems != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out))
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    return row_bytes * std::size_t(n_out);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(x_q8_1)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "quantize_q8_1_k",
                              (uint32_t) (n_in * ncols), 32, args, 2, false);
}

#define STRATA_MMVQ_PAIR(NAME, TYPE)                                                                     void NAME##_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,                              int ncols, void* stream) {                                                              mmvq_launch(TYPE, weights, x_q8_1, y, n_in, n_out, ncols, stream);                               }                                                                                                    void NAME##_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in,                         int n_out, int ncols, void* stream) {                                                    f32_launch(TYPE, weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);                       }
STRATA_MMVQ_PAIR(native_q5_k, 13)
STRATA_MMVQ_PAIR(native_q2_0, 42)
STRATA_MMVQ_PAIR(native_q3_k, 11)
STRATA_MMVQ_PAIR(native_iq4_xs, 23)
STRATA_MMVQ_PAIR(native_q4_k, 12)
STRATA_MMVQ_PAIR(native_q6_k, 14)
STRATA_MMVQ_PAIR(native_q4_0, 2)
STRATA_MMVQ_PAIR(native_q5_0, 6)
STRATA_MMVQ_PAIR(native_q8_0, 8)
STRATA_MMVQ_PAIR(native_iq4_nl, 20)
#undef STRATA_MMVQ_PAIR

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out,
                 int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

}  // namespace strata::kernels
