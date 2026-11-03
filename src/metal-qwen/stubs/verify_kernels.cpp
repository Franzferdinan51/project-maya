// src/metal-qwen/stubs/verify_kernels.cpp - Metal implementation for
// include/strata/kernels/verify_kernels.hpp (was the stage-1 stub). GPU
// kernels in verify.metal.
//   - resident_plan is a device kernel (resident_plan_k): it only writes
//     pointer VALUES as integers, so it runs in stream order with no host
//     sync. (The first port ran it on the host after a stream sync - inside
//     the verify body that sync deadlocks against the body's own flag-spin
//     kernels, the same failure as the grouped launcher's sync; see
//     SYNCFIX-NOTES.md in this directory.)
//   - fetch_blobs is a deliberate no-op here: on Metal the PCIe staging
//     copy is performed by the pool-side DMA path (Verifier::fetch_dma,
//     driven by expert_source's plan publish) before flag B is raised, and
//     rebase_ptrs then points the plan at staging in stream order - the
//     body-side host memcpy the CUDA kernel replaces would need a mid-body
//     sync to know its inputs, which is the same deadlock.
//   - gpu_stamp: Metal exposes no GPU timestamp register to kernels, so the
//     value is the host's monotonic clock in ns, written by a 1-thread
//     kernel in stream order (the CUDA value is the GPU globaltimer in ns).
// gdn_ab_multi note: the CUDA kernel accumulates 8-wide vector chunks per
// lane; the port accumulates per element in the same lane stride - the
// partial-sum tree differs, so results can differ from CUDA by summation
// rounding (documented, like the split GEMVs).
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "../kernels/dense_host.hpp"
#include "../runtime/metal_internal.hpp"
#include "strata/metal_compat/cuda_runtime.h"

namespace strata::kernels {
namespace {

[[noreturn]] void bad(const char* who) {
    std::fprintf(stderr, "%s: invalid arguments\n", who);
    std::abort();
}
}  // namespace

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % 128 != 0 || n_tok < 1 || n_tok > kVerifyMaxT)
        bad("gdn_conv_l2_multi");
    const metal::KernelArg args[] = {metal::buf(history), metal::buf(qkv), metal::buf(conv_w), metal::buf(h),
                                     metal::val(channels), metal::val(qk_heads), metal::val(eps),
                                     metal::val(t_begin)};
    metal::launch(dense_detail::stream_of(stream), "gdn_conv_l2_k", (uint32_t) channels, (uint32_t) n_tok, 1, 128, 1,
                  1, args, 8, 0);
    if (stream == nullptr) cudaDeviceSynchronize();
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    const metal::KernelArg args[] = {metal::buf(history), metal::buf(qkv), metal::val(channels), metal::buf(n_keep)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "gdn_conv_commit_k", (uint32_t) channels, 256, args, 4,
                              stream == nullptr);
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt,
                  const float* ssm_a, float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) bad("gdn_ab_multi");
    struct AbArgs {
        int32_t n, h_v, T;
    } a{n_embd, h_v, n_tok};
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(w_alpha), metal::buf(w_beta), metal::buf(dt),
                                     metal::buf(ssm_a), metal::buf(gate), metal::buf(beta), metal::val(a)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "gdn_ab_k", (uint32_t) (2 * h_v) * 32, 32, args, 8,
                              stream == nullptr);
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT)
        bad("gdn_step_norm_multi");
    struct StepArgs {  // mirrors GdnStepArgs in verify.metal, field for field
        int32_t C;
        float eps;
        int32_t h_k, h_v, T, has_nkeep, t_out_begin;
    } a{conv_channels, eps, h_k, h_v, n_tok, n_keep ? 1 : 0, t_out_begin};
    const void* nk = n_keep ? (const void*) n_keep : (const void*) state;
    const metal::KernelArg args[] = {metal::buf(state), metal::buf(h), metal::buf(gate), metal::buf(beta),
                                     metal::buf(z), metal::buf(gamma), metal::buf(y), metal::buf(nk), metal::val(a)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "gdn_step_norm_k", (uint32_t) h_v * 512, 512, args, 9,
                              stream == nullptr);
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    // Not a kernel: on this backend a device spin on a host-raised flag can
    // never observe the raise (VSTALL-NOTES.md), so the wait is encoded as a
    // held-commit gate at this exact stream position by the runtime, whose
    // watcher polls the flag word host-side. A null stream keeps its old
    // contract (the wait is on the default stream, then a device sync).
    metal_internal::stream_wait_flag(dense_detail::stream_of(stream), flag, value);
    if (stream == nullptr) cudaDeviceSynchronize();
}

void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const uint64_t ns = (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
    const metal::KernelArg args[] = {metal::buf(buf), metal::val(i), metal::val(ns)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "stamp_k", 1, 1, args, 3, stream == nullptr);
}

void resid_hash(const float* x, int64_t n, unsigned long long* out, int slot, void* stream) {
    if (!x || !out || n < 1 || slot < 0) bad("resid_hash");
    const int nn = (int) n;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(out), metal::val(nn), metal::val(slot)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "resid_hash_k", 256, 256, args, 4,
                              stream == nullptr);
}

void resid_sample(const float* x, int64_t n, float* out, int slot, int nsamp, void* stream) {
    if (!x || !out || n < nsamp || slot < 0 || nsamp < 1) bad("resid_sample");
    const int stride = (int) (n / nsamp);
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(out), metal::val(stride), metal::val(slot),
                                     metal::val(nsamp)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "resid_sample_k", 64, 64, args, 5,
                              stream == nullptr);
}

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    // One-thread device kernel (resident_plan_k in verify.metal) performing
    // the host layout's exact integer operations in stream order. No sync:
    // the kernel is queue-ordered after the router kernels that produced
    // `ids` and before the conditional copies that consume the plan, which
    // is precisely the ordering the captured body needs. slot_off is bound
    // to a dummy buffer when null (has_off gates every load of it).
    struct ResidentPlanArgs {
        int32_t n_entries, k, n_expert, has_off;
        uint64_t base;
        int64_t blob, capx;
        uint32_t ring;
    };
    const ResidentPlanArgs a{n_entries, k, n_expert, slot_off ? 1 : 0,
                             (uint64_t) (uintptr_t) cache_base, blob, capx, ring};
    const metal::KernelArg args[] = {metal::buf(ids), metal::buf(res_layer),
                                     metal::buf(slot_off ? slot_off : (const unsigned long long*) res_layer),
                                     metal::buf(plan), metal::buf(skip), metal::val(a)};
    metal::launch(dense_detail::stream_of(stream), "resident_plan_k", 1, 1, 1, 1, 1, 1, args, 6);
}

void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    // The wait_flag_ge gate, with the device-plan escape: also released when
    // *skip == value (the watcher reads both words; see VSTALL-NOTES.md).
    metal_internal::stream_wait_flag_or(dense_detail::stream_of(stream), flag, value, skip);
    if (stream == nullptr) cudaDeviceSynchronize();
}

void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip,
                                 uint32_t value, void* stream) {
    if (n <= 0) return;
    const int nn = (int) n;
    const metal::KernelArg args[] = {metal::buf(dst), metal::buf(src), metal::val(nn), metal::buf(skip),
                                     metal::val(value)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "copy_i32_unless_k", 128, 128, args, 5,
                              stream == nullptr);
}

void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const int64_t n4 = n / 4;
    const metal::KernelArg args[] = {metal::buf(dst), metal::buf(src), metal::val(n4), metal::buf(skip),
                                     metal::val(value)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "copy_or_zero_k", (uint32_t) n4, 256, args, 5,
                              stream == nullptr);
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    struct EmbArgs {
        int64_t n;
        int32_t code_bits, code_bias, group_elems, has_offsets;
        uint64_t row_codes, row_groups;
    } a{n, code_bits, code_bias, group_elems, offsets ? 1 : 0, row_codes, row_groups};
    const void* of = offsets ? (const void*) offsets : (const void*) scales;
    const metal::KernelArg args[] = {metal::buf(codes), metal::buf(scales), metal::buf(of), metal::buf(tokens),
                                     metal::buf(out), metal::val(a)};
    metal::launch(dense_detail::stream_of(stream), "embedding_gather_dev_k", (uint32_t) n, (uint32_t) n_tok, 1, 256,
                  1, 1, args, 6, 0);
    if (stream == nullptr) cudaDeviceSynchronize();
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(R), metal::val(n_embd), metal::val(hc)};
    metal::launch(dense_detail::stream_of(stream), "broadcast_streams_k", (uint32_t) (n_embd * hc), (uint32_t) n_tok,
                  1, 256, 1, 1, args, 4, 0);
    if (stream == nullptr) cudaDeviceSynchronize();
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const metal::KernelArg args[] = {metal::buf(dst), metal::buf(src), metal::val(stride), metal::buf(index),
                                     metal::val(n)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "copy_indexed_k", (uint32_t) n, 256, args, 5,
                              stream == nullptr);
}

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap,
                 void* stream) {
    // Deliberate no-op on Metal (argument validation kept). The CUDA entry
    // is a device kernel copying *n blobs from device-alias addresses into
    // staging, in stream order inside the verify body. The Metal staging
    // copy is owned by the pool side instead: expert_source's plan publish
    // hands the fetch list to Verifier::fetch_dma for every pcie mode on
    // this backend (see expert_source.cpp), which copies host->staging on
    // the copy stream and only then raises flag B - and the body's
    // wait_flag_ge(flag B) precedes this call, so by the time any later
    // kernel runs the blobs are already in staging, and the rebase_ptrs
    // call that follows points the plan at them. A host memcpy here would
    // have to read src / *n, whose values only exist once the body's plan
    // copy has executed - i.e. it would need the mid-body stream sync that
    // deadlocks the body against its own flag spins.
    (void) src;
    (void) n;
    (void) dst;
    (void) stream;
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) {
        std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n");
        std::abort();
    }
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    const uint64_t b = (uint64_t) (uintptr_t) base;
    const metal::KernelArg args[] = {metal::buf(ptr), metal::buf(n), metal::val(b), metal::val(blob_bytes)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "rebase_ptrs_k", 128, 128, args, 4,
                              stream == nullptr);
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok,
                           void* stream) {
    const metal::KernelArg args[] = {metal::buf(h), metal::buf(e), metal::buf(R), metal::val(n_embd),
                                     metal::val(hc)};
    metal::launch(dense_detail::stream_of(stream), "add_streams_broadcast_k", (uint32_t) (n_embd * hc),
                  (uint32_t) n_tok, 1, 256, 1, 1, args, 5, 0);
    if (stream == nullptr) cudaDeviceSynchronize();
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) {
        std::fprintf(stderr, "ident_hits: n out of range\n");
        std::abort();
    }
    const metal::KernelArg args[] = {metal::buf(ids), metal::val(n), metal::buf(slot), metal::buf(dst),
                                     metal::buf(count)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "ident_hits_k", 1024, 1024, args, 5,
                              stream == nullptr);
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    struct MtpArgs {
        int64_t stride;
        int32_t j, has_out, has_probs;
    } a{R_stride, j, out ? 1 : 0, (probs && out_p) ? 1 : 0};
    const void* ob = out ? (const void*) out : (const void*) tok_dst;
    const void* pb = probs ? (const void*) probs : (const void*) R_src;
    const void* opb = out_p ? (const void*) out_p : (const void*) R_dst;
    const metal::KernelArg args[] = {metal::buf(R_src), metal::buf(ids), metal::buf(row_dev), metal::buf(R_dst),
                                     metal::buf(tok_dst), metal::buf(ob), metal::buf(pb), metal::buf(opb),
                                     metal::val(a)};
    const uint32_t total = (uint32_t) std::max<int64_t>(R_stride, 1);
    dense_detail::launch_sync(dense_detail::stream_of(stream), "mtp_select_k", total, 256, args, 9,
                              stream == nullptr);
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    const char* fn;
    int64_t row_e;
    if (row_bytes % 16 == 0) {
        fn = "gather_rows16_k";
        row_e = row_bytes / 16;
    } else if (row_bytes % 4 == 0) {
        fn = "gather_rows4_k";
        row_e = row_bytes / 4;
    } else {
        fn = "gather_rows1_k";
        row_e = row_bytes;
    }
    const metal::KernelArg args[] = {metal::buf(src), metal::val(row_e), metal::buf(ids), metal::val(n),
                                     metal::buf(dst)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), fn, (uint32_t) (n * row_e), 256, args, 5,
                              stream == nullptr);
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    const metal::KernelArg args[] = {metal::buf(ids), metal::buf(table), metal::val(n)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "map_ids_k", 64, 64, args, 3, stream == nullptr);
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    const metal::KernelArg args[] = {metal::buf(logits), metal::val(n_vocab), metal::buf(ids), metal::buf(probs)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "row_top_prob_k", (uint32_t) n_rows * 1024, 1024,
                              args, 4, stream == nullptr);
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    const metal::KernelArg args[] = {metal::buf(steps), metal::val(window), metal::buf(ids), metal::val(ids_stride)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "window_ids_k", (uint32_t) n * 256, 256, args, 4,
                              stream == nullptr);
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    const metal::KernelArg args[] = {metal::buf(cells), metal::val(n), metal::buf(steps)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "dense_steps_k", 64, 64, args, 3, stream == nullptr);
}

}  // namespace strata::kernels
