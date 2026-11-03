// src/metal-qwen/stubs/kv_stream.cpp - Metal host launchers for the KV
// streaming map (clock eviction over paged slots); semantics transcribed from
// src/kernels/cuda/kv_stream.cu. Signatures match
// include/strata/kernels/kv_stream.hpp exactly.
#include "strata/kernels/kv_stream.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/metal_compat/cuda_runtime.h"

#include "../runtime/metal_kernel.hpp"

namespace strata::kernels {
namespace {

// The per-block byte runs of the (up to four) pool arrays, as in the CUDA TU:
// block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Mirrors the MSL structs in kernels/kv.metal.
struct KvCopyArgs {
    int32_t len[4];
    int32_t n_runs;
    int32_t n_tg;
};
struct KvResolveArgs {
    int32_t n_q, cap, page_size;
    int64_t n_slots;
};

void check_copy(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream metal: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}


void launch(const char* fn, void* stream, uint32_t grid_x, uint32_t block_x, const metal::KernelArg* args,
            size_t n_args) {
    metal::launch((cudaStream_t) stream, fn, grid_x, 1, 1, block_x, 1, 1, args, n_args);
}
}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    const int64_t n = std::max<int64_t>({m.n_blocks, m.n_slots, kKvCtlInts});
    const metal::KernelArg args[] = {metal::buf(m.page_table), metal::buf(m.slot_block), metal::buf(m.slot_stamp),
                                     metal::buf(m.slot_ref),   metal::buf(m.ctl),       metal::val(m.n_blocks),
                                     metal::val(m.n_slots)};
    launch("kv_reset_k", stream, (uint32_t) n, 256, args, 7);
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    const metal::KernelArg args[] = {metal::buf(page_table), metal::val(n_blocks), metal::val(n_slots)};
    launch("kv_ring_k", stream, (uint32_t) std::max<int64_t>(n_blocks, 1), 256, args, 3);
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream metal: a block's scale run must be a multiple of 16 bytes\n");
        std::abort();
    }
    const KvResolveArgs ra{(int32_t) n_q, (int32_t) cap, (int32_t) s.page_size, m.n_slots};
    const metal::KernelArg rargs[] = {metal::buf(m.page_table), metal::buf(m.slot_block), metal::buf(m.slot_stamp),
                                      metal::buf(m.slot_ref),   metal::buf(m.ctl),       metal::buf(m.miss_block),
                                      metal::buf(m.miss_slot),  metal::buf(ids),         metal::buf(steps),
                                      metal::val(ra)};
    launch("kv_resolve_k", stream, 1024, 1024, rargs, 10);
    // Copy the missed blocks from the host copy into their slots.
    const Runs r = runs_of(slots, host, fmt, s);
    KvCopyArgs ca{};
    ca.n_runs = r.n;
    ca.n_tg = 96;
    for (int i = 0; i < 4; ++i) ca.len[i] = i < r.n ? r.len[i] : 0;
    const void* srcs[4];
    void* dsts[4];
    for (int i = 0; i < 4; ++i) {
        srcs[i] = i < r.n ? (const void*) r.src[i] : (const void*) m.page_table;
        dsts[i] = i < r.n ? (void*) r.dst[i] : (void*) m.page_table;
    }
    const metal::KernelArg cargs[] = {metal::buf(srcs[0]), metal::buf(dsts[0]), metal::buf(srcs[1]), metal::buf(dsts[1]),
                                      metal::buf(srcs[2]), metal::buf(dsts[2]), metal::buf(srcs[3]), metal::buf(dsts[3]),
                                      metal::buf(m.miss_block), metal::buf(m.miss_slot), metal::buf(m.ctl),
                                      metal::val(ca)};
    launch("kv_copy_k", stream, 96 * 128, 128, cargs, 12);
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);
        for (int a = 0; a < r.n; ++a)
            check_copy(cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                       cudaMemcpyDefault, (cudaStream_t) stream),
                       "ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        check_copy(cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                                   (cudaStream_t) stream),
                   "stage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
