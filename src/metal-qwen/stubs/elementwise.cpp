// src/metal-qwen/stubs/elementwise.cpp - Metal implementation for
// include/strata/kernels/elementwise.hpp (was the stage-1 stub). Kernels in
// misc.metal; the doorbell/copy entries keep their mapped-memory semantics
// (on Metal the mapped regions are shared MTLBuffers, so the kernel copies
// are plain loads/stores, and the ring is still a device-side increment so
// captured graphs advance it per replay - the runtime's closure capture is
// what replays them).
#include "strata/kernels/elementwise.hpp"

#include "../kernels/dense_host.hpp"
#include "../runtime/metal_internal.hpp"

namespace strata::kernels {
namespace {
void launch_n(const char* fn, const metal::KernelArg* args, size_t n_args, uint64_t n, void* stream,
              uint32_t block = 256) {
    dense_detail::launch_sync(dense_detail::stream_of(stream), fn, (uint32_t) n, block, args, n_args, false);
}
}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets, int64_t n,
                      int code_bits, int code_bias, int group_elems, float* out, void* stream) {
    if (n <= 0) return;
    const int32_t has_off = offsets ? 1 : 0;
    const metal::KernelArg args[] = {metal::buf(codes), metal::buf(scales),
                                     metal::buf(offsets ? offsets : scales), metal::buf(out),
                                     metal::val(n), metal::val(code_bits), metal::val(code_bias),
                                     metal::val(group_elems), metal::val(has_off)};
    launch_n("embedding_gather_k", args, 9, (uint64_t) n, stream);
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    const uint64_t total = (uint64_t) (n_tokens * h_v);
    const uint32_t hv = (uint32_t) h_v;
    const metal::KernelArg args[] = {metal::buf(alpha), metal::buf(dt), metal::buf(ssm_a), metal::buf(gate),
                                     metal::val(total), metal::val(hv)};
    launch_n("gdn_gate_k", args, 6, total, stream);
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    const uint64_t nn = (uint64_t) n;
    const metal::KernelArg args[] = {metal::buf(x), metal::val(nn), metal::val(s)};
    launch_n("scale_inplace_k", args, 3, nn, stream);
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const uint64_t nn = (uint64_t) n;
    const metal::KernelArg args[] = {metal::buf(dst), metal::buf(src), metal::val(nn)};
    launch_n("add_inplace_k", args, 3, nn, stream);
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    const uint64_t nn = (uint64_t) n;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(y), metal::val(nn)};
    launch_n("f32_to_f16_bulk_k", args, 3, nn, stream);
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    const uint64_t nn = (uint64_t) n;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(y), metal::val(nn)};
    launch_n("f32_to_bf16_bulk_k", args, 3, nn, stream);
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const uint64_t nn = (uint64_t) n;
    const metal::KernelArg args[] = {metal::buf(x), metal::val(nn)};
    launch_n("silu_inplace_k", args, 2, nn, stream);
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    const uint32_t c = (uint32_t) cols;
    const int32_t has_w = w ? 1 : 0;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(w ? w : x), metal::val(c), metal::val(eps),
                                     metal::val(has_w)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "rms_norm_weighted_k", (uint32_t) rows * 256,
                              256, args, 5, false);
}

void doorbell_ring(uint32_t* d_seq, void* stream) {
    const metal::KernelArg args[] = {metal::buf(d_seq)};
    launch_n("doorbell_ring_k", args, 1, 1, stream, 1);
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    // The session interleave's wait, in the wait_flag_ge form (VSTALL-NOTES.md):
    // a held-commit gate at this stream position, released by the runtime's
    // watcher when *d_flag == *d_seq. (The in-kernel spin it replaces could
    // never observe the host's flag write.)
    metal_internal::stream_wait_doorbell(dense_detail::stream_of(stream), d_flag, d_seq);
}

void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const uint64_t nn = (uint64_t) n;
    const metal::KernelArg args[] = {metal::buf(dst), metal::buf(src), metal::val(nn)};
    launch_n("copy_f32_k", args, 3, nn, stream);
}

void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0 || width <= 0) return;
    const uint64_t wd = (uint64_t) width, total = (uint64_t) (rows * width);
    const metal::KernelArg args[] = {metal::buf(dst), metal::buf(src), metal::buf(hit_rows), metal::buf(count),
                                     metal::val(wd), metal::val(total)};
    launch_n("copy_rows_mapped_k", args, 6, total, stream);
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                      float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    const uint64_t nn = (uint64_t) n, kk = (uint64_t) k;
    const metal::KernelArg args[] = {metal::buf(x),     metal::buf(ids),  metal::buf(weights), metal::buf(x_out),
                                     metal::buf(ids_out), metal::buf(weights_out), metal::buf(d_seq),
                                     metal::val(nn),    metal::val(kk)};
    launch_n("doorbell_publish_k", args, 9, nn > kk ? nn : kk, stream);
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const uint64_t nn = (uint64_t) n;
    const metal::KernelArg args[] = {metal::buf(dst), metal::buf(src), metal::val(nn)};
    launch_n("copy_i32_k", args, 3, nn, stream);
}

}  // namespace strata::kernels
