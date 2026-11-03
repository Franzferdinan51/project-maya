// src/metal-qwen/stubs/kv_q4.cpp - Metal host launchers for the Q4_0 KV pool
// family and FWHT-256; semantics transcribed from src/kernels/cuda/kv_q4.cu.
// Signatures match include/strata/kernels/kv_q4.hpp exactly (stream, then
// host pointer).
#include "strata/kernels/kv_q4.hpp"

#include "strata/kernels/kv_stream.hpp"  // KvHostPools definition

#include <cstdio>
#include <cstdlib>

#include "../runtime/metal_kernel.hpp"

namespace strata::kernels {
namespace {

void need(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "kv_q4 metal: %s\n", what);
        std::abort();
    }
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "kv_q4 metal: %s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::abort();
    }
}

// Mirrors the MSL KvQ4Args in kernels/kv.metal.
struct KvQ4Args {
    int32_t heads, head_dim, page_size, has_host, has_stage;
    int64_t pos0;
};


void launch(const char* fn, void* stream, uint32_t grid_x, uint32_t block_x, const metal::KernelArg* args,
            size_t n_args) {
    metal::launch((cudaStream_t) stream, fn, grid_x, 1, 1, block_x, 1, 1, args, n_args);
}
}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const metal::KernelArg args[] = {metal::buf(src), metal::buf(dst), metal::val(n_rows)};
    launch("fwht256_k", stream, (uint32_t) (n_rows * 32), 32, args, 3);
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    need_256(s, "kv_append_q4");
    const bool has_host = host != nullptr && host->k_q4 != nullptr;
    const uint8_t* hk = has_host ? host->k_q4 : k_q4;
    const uint8_t* hv = has_host ? host->v_q4 : v_q4;
    const KvQ4Args a{(int32_t) s.n_head_kv, (int32_t) s.head_dim, (int32_t) s.page_size, has_host ? 1 : 0, 0, 0};
    const metal::KernelArg args[] = {metal::buf(k_q4), metal::buf(v_q4), metal::buf(page_table), metal::buf(step),
                                     metal::buf(kcur), metal::buf(vcur), metal::buf(hk),        metal::buf(hv),
                                     metal::val(a)};
    const uint32_t groups = (uint32_t) (2 * s.n_head_kv * (s.head_dim / QK4_0));
    launch("kv_append_q4_k", stream, groups * 32, 32, args, 9);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                  const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const bool has_host = host != nullptr && host->k_q4 != nullptr;
    const bool has_stage = stage != nullptr && stage->k_q4 != nullptr;
    const uint8_t* hk = has_host ? host->k_q4 : k_q4;
    const uint8_t* hv = has_host ? host->v_q4 : v_q4;
    const uint8_t* sk = has_stage ? stage->k_q4 : k_q4;
    const uint8_t* sv = has_stage ? stage->v_q4 : v_q4;
    const KvQ4Args a{(int32_t) s.n_head_kv, (int32_t) s.head_dim, (int32_t) s.page_size, has_host ? 1 : 0,
                     has_stage ? 1 : 0, pos0};
    const metal::KernelArg args[] = {metal::buf(k_q4), metal::buf(v_q4), metal::buf(page_table), metal::buf(K),
                                     metal::buf(V),    metal::buf(hk),   metal::buf(hv),        metal::buf(sk),
                                     metal::buf(sv),   metal::val(a)};
    const uint32_t groups = (uint32_t) (T * 2 * s.n_head_kv * (s.head_dim / QK4_0));
    launch("kv_append_q4_batch_k", stream, groups * 32, 32, args, 10);
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const KvQ4Args a{(int32_t) s.n_head_kv, (int32_t) s.head_dim, (int32_t) s.page_size, 0, 0, 0};
    const metal::KernelArg args[] = {metal::buf(k_q4), metal::buf(v_q4), metal::buf(page_table), metal::buf(ids),
                                     metal::buf(step), metal::buf(k_scratch), metal::buf(v_scratch), metal::val(a)};
    launch("kv_gather_q4_k", stream, (uint32_t) (max_ids * s.n_head_kv * s.head_dim), 256, args, 8);
}

}  // namespace strata::kernels
