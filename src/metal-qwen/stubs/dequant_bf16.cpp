// src/metal-qwen/stubs/dequant_bf16.cpp - Metal implementation for
// include/strata/kernels/dequant_bf16.hpp (was the stage-1 stub). One thread
// per output element over the generic block dequantizer in dense.metal's
// dq_elem (transcribed from this TU's CUDA group32 expressions); the CUDA
// TU's fast paths (Q8_0 / K-format f16 vector stores) are bandwidth variants
// of the same arithmetic and are not ported - the generic kernel is the
// correctness path. The i-quant-only types route to iq_dequant_* as in CUDA.
#include "strata/kernels/dequant_bf16.hpp"

#include "../kernels/dense_host.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {

bool dequant_bf16_supported(int ggml_type) noexcept {
    // The CUDA TU's geometry table: Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL,
    // IQ4_XS and Q2_0. (The i-quant-only types route through iq_dequant_* instead and
    // are not in this table in the CUDA TU either.)
    switch (ggml_type) {
    case 2: case 6: case 8: case 20: case 11: case 12: case 13: case 14: case 23: case 42: return true;
    default: return false;
    }
}

namespace {
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }

void launch_dequant(int mode, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                    void* out, void* stream) {
    int be = 0, bb = 0;
    switch (ggml_type) {
    case 2: be = 32; bb = 18; break;
    case 6: be = 32; bb = 22; break;
    case 8: be = 32; bb = 34; break;
    case 20: be = 32; bb = 18; break;
    case 11: be = 256; bb = 110; break;
    case 12: be = 256; bb = 144; break;
    case 13: be = 256; bb = 176; break;
    case 14: be = 256; bb = 210; break;
    case 23: be = 256; bb = 136; break;
    case 42: be = 64; bb = 18; break;
    default: break;
    }
    if (!be || cols % be != 0 || rows <= 0)
        dense_detail::bad_shape("dequant_bf16: unsupported type or shape");
    const uint64_t row_bytes = (uint64_t) (cols / be * bb);
    const uint64_t r0 = (uint64_t) row0, total = (uint64_t) (rows * cols);
    const uint32_t c = (uint32_t) cols;
    const int m = mode, t = ggml_type;
    const metal::KernelArg args[] = {metal::buf(blocks), metal::buf(out),     metal::val(t),
                                     metal::val(m),    metal::val(row_bytes), metal::val(r0),
                                     metal::val(total), metal::val(c)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "dequant_generic_k", (uint32_t) total, 256, args,
                              8, false);
}
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols),
                       rows * cols, out, stream);
        return;
    }
    launch_dequant(1, ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols),
                       rows * cols, out, stream);
        return;
    }
    launch_dequant(2, ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols),
                       rows * cols, out, stream);
        return;
    }
    launch_dequant(0, ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
