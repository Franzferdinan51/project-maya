// src/metal-qwen/stubs/kv_q8.cpp - Metal host launchers for the int8 KV pool
// family; semantics transcribed from src/kernels/cuda/kv_q8.cu. Signatures
// match include/strata/kernels/kv_q8.hpp exactly (stream, then host pointer).
#include "strata/kernels/kv_q8.hpp"

#include <cstdio>
#include <cstdlib>

#include "../runtime/metal_kernel.hpp"

namespace strata::kernels {
namespace {

void need(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "kv_q8 metal: %s\n", what);
        std::abort();
    }
}

// Mirrors the MSL KvQ8Args in kernels/kv.metal.
struct KvQ8Args {
    int32_t heads, head_dim, page_size, has_host;
};

void validate(const QsaShapes& s) {
    need(s.head_dim % KV_Q8_GROUP == 0 && s.n_head_kv > 0 && s.page_size > 0,
         "head_dim must be a multiple of 64 with positive heads/page_size");
}


void launch(const char* fn, void* stream, uint32_t grid_x, uint32_t block_x, const metal::KernelArg* args,
            size_t n_args) {
    metal::launch((cudaStream_t) stream, fn, grid_x, 1, 1, block_x, 1, 1, args, n_args);
}
}  // namespace

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s);
    const bool has_host = host != nullptr && host->k_q != nullptr;
    const int8_t* hk = has_host ? host->k_q : k_q;
    const int8_t* hv = has_host ? host->v_q : v_q;
    const uint16_t* hks = has_host ? host->k_scale : k_scale;
    const uint16_t* hvs = has_host ? host->v_scale : v_scale;
    const KvQ8Args a{(int32_t) s.n_head_kv, (int32_t) s.head_dim, (int32_t) s.page_size, has_host ? 1 : 0};
    const metal::KernelArg args[] = {metal::buf(k_q),   metal::buf(v_q),   metal::buf(k_scale), metal::buf(v_scale),
                                     metal::buf(page_table), metal::buf(step), metal::buf(kcur), metal::buf(vcur),
                                     metal::buf(hk),    metal::buf(hv),    metal::buf(hks),     metal::buf(hvs),
                                     metal::val(a)};
    const uint32_t groups = (uint32_t) (2 * s.n_head_kv * (s.head_dim / KV_Q8_GROUP));
    launch("kv_append_q8_k", stream, groups * 64, 64, args, 13);
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s);
    if (max_ids <= 0) return;
    const KvQ8Args a{(int32_t) s.n_head_kv, (int32_t) s.head_dim, (int32_t) s.page_size, 0};
    const metal::KernelArg args[] = {metal::buf(k_q), metal::buf(v_q), metal::buf(k_scale), metal::buf(v_scale),
                                     metal::buf(page_table), metal::buf(ids), metal::buf(step),
                                     metal::buf(k_scratch), metal::buf(v_scratch), metal::val(a)};
    launch("kv_gather_q8_k", stream, (uint32_t) (max_ids * s.n_head_kv * s.head_dim), 256, args, 10);
}

}  // namespace strata::kernels
