// src/metal-qwen/stubs/s_gemv.cpp - Metal implementation for
// include/strata/kernels/s_gemv.hpp (was the stage-1 stub), including the
// s2_gemv_quads / s2_gemv_fast entries whose CUDA TUs live in this family.
// The sequential entries keep the CUDA naive accumulation order; the split
// entries reduce through threadgroup memory and, like the CUDA split, are
// not bit-identical to the sequential ones. s2_gemv_fast's shared-memory
// activation staging is a bandwidth variant of s2_gemv_quads' arithmetic;
// the Metal port computes the same values without the staging (documented
// delta: performance only, stage_x currently selects nothing).
#include "strata/kernels/s_gemv.hpp"

#include "../kernels/dense_host.hpp"

namespace strata::kernels {
namespace {

struct FormArgs {
    int32_t code_bits, code_bias, group_elems, codebook, has_offset;
};
FormArgs form_args(const SForm& f) {
    return FormArgs{f.code_bits, f.code_bias, f.group_elems, (int) f.codebook, f.has_offset ? 1 : 0};
}
// When the form has no offset plane, the offset argument must still be a real
// buffer for the runtime's allocation map: pass the scales buffer (never read,
// has_offset gates every load in the kernel).
const float* offset_or_dummy(const float* offset, const float* scales) { return offset ? offset : scales; }

void gemv_seq(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
              int64_t n_in, int64_t n_out, const SForm& form, void* stream, bool sync) {
    const uint32_t ni = (uint32_t) n_in, no = (uint32_t) n_out;
    const FormArgs fa = form_args(form);
    const metal::KernelArg args[] = {metal::buf(x),     metal::buf(codes), metal::buf(scales),
                                     metal::buf(offset_or_dummy(offset, scales)), metal::buf(y),
                                     metal::val(ni),    metal::val(no),    metal::val(fa)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "s_gemv_k", no, 128, args, 8, sync);
}

void gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row, void* stream, bool sync) {
    const uint32_t tpr = dense_detail::checked_tpr(threads_per_row, "s_gemv_split: bad threads_per_row");
    const uint32_t ni = (uint32_t) n_in;
    const FormArgs fa = form_args(form);
    const metal::KernelArg args[] = {metal::buf(x),     metal::buf(codes), metal::buf(scales),
                                     metal::buf(offset_or_dummy(offset, scales)), metal::buf(y),
                                     metal::val(ni),    metal::val(tpr),   metal::val(fa)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "s_gemv_split_k", (uint32_t) n_out * tpr, tpr,
                              args, 8, sync);
}

void s2_split(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
              int64_t n_out, int threads_per_row) {
    if (n_in % 64 != 0) dense_detail::bad_shape("s2 split GEMV: n_in is not a multiple of 64");
    const uint32_t tpr = dense_detail::checked_tpr(threads_per_row, "s2 split GEMV: bad threads_per_row");
    const uint32_t ni = (uint32_t) n_in;
    const metal::KernelArg args[] = {metal::buf(x),  metal::buf(codes), metal::buf(scales),
                                     metal::buf(y),  metal::val(ni),    metal::val(tpr)};
    dense_detail::launch_sync(nullptr, "s2_gemv_split_k", (uint32_t) n_out * tpr, tpr, args, 6, true);
}

void gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, const float* offset, float* y,
             int64_t n_in, int64_t n_out, const SForm& form, int act_kind, bool split, void* stream) {
    const uint32_t ni = (uint32_t) n_in;
    const FormArgs fa = form_args(form);
    if (!split) {
        const uint32_t no = (uint32_t) n_out;
        const metal::KernelArg args[] = {metal::buf(act),   metal::buf(codes), metal::buf(scales),
                                         metal::buf(offset_or_dummy(offset, scales)), metal::buf(y),
                                         metal::val(ni),    metal::val(no),    metal::val(fa),
                                         metal::val(act_kind)};
        dense_detail::launch_sync(dense_detail::stream_of(stream), "s_gemv_q8_k", no, 128, args, 9, false);
        return;
    }
    // The CUDA split form is warp-per-row: 32 threads.
    const uint32_t tpr = 32;
    const metal::KernelArg args[] = {metal::buf(act),   metal::buf(codes), metal::buf(scales),
                                     metal::buf(offset_or_dummy(offset, scales)), metal::buf(y),
                                     metal::val(ni),    metal::val(tpr),   metal::val(fa),
                                     metal::val(act_kind)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "s_gemv_q8_split_k", (uint32_t) n_out * tpr, tpr,
                              args, 9, false);
}

}  // namespace

void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form) {
    if (n_in <= 0 || n_out <= 0) return;
    gemv_seq(x, codes, scales, offset, y, n_in, n_out, form, nullptr, true);
}

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
                  int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    gemv_split(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, nullptr, true);
}

void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                   int64_t n_out, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    s2_split(x, codes, scales, y, n_in, n_out, threads_per_row);
}

void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                  int64_t n_out, int threads_per_row, bool stage_x) {
    (void) stage_x;  // see the file header: staging is a performance variant only
    if (n_in <= 0 || n_out <= 0) return;
    s2_split(x, codes, scales, y, n_in, n_out, threads_per_row);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    gemv_split(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, stream, false);
}

void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    gemv_q8(x_q8k, codes, scales, offset, y, n_in, n_out, form, 1, false, stream);
}

void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    gemv_q8(x_q8k, codes, scales, offset, y, n_in, n_out, form, 1, true, stream);
}

void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 32 != 0) dense_detail::bad_shape("s_gemv_q8_0_split: n_in must be a multiple of 32");
    gemv_q8(x_q8_0, codes, scales, offset, y, n_in, n_out, form, 0, true, stream);
}

}  // namespace strata::kernels
