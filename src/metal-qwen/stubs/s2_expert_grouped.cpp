// src/metal-qwen/stubs/s2_expert_grouped.cpp - Metal implementation for
// include/strata/kernels/s2_expert_grouped.hpp. Replaces the stage-1 loud stub
// (moe_hit_grouped_s2_cpu_order excepted - see its note below).
//
// The composition mirrors src/kernels/cuda/s2_expert_grouped.cu: ONE launch per
// projection (gu -> swiglu -> the intermediate's quantize_q8_0[_scaled] host
// entry from the dense family -> down), scratch carved from the caller's arena
// with moe_hit_grouped_scratch_bytes' layout, no allocation. The MSL kernels
// are in kernels/s2_expert.metal; the fixed blob geometry (H 2560, FF 640) is
// the blob layout's, as in the CUDA TU.

#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../runtime/metal_kernel.hpp"
#include "strata/kernels/quantize_act.hpp"

namespace strata::kernels {
namespace {

constexpr int64_t H = 2560;
constexpr int64_t FF = 640;

struct Scratch {
    float* gate_up;
    uint8_t* h_q8_0;
    float* h_scales;
    uint64_t gu_bytes, q8_bytes;
};

Scratch carve(void* scratch, int64_t cap) {
    Scratch s;
    s.gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    s.q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    s.gate_up = (float*) scratch;
    s.h_q8_0 = (uint8_t*) scratch + s.gu_bytes;
    s.h_scales = (float*) ((uint8_t*) scratch + s.gu_bytes + s.q8_bytes);
    return s;
}

struct GuArgs {
    int32_t n_hits;
    int64_t blob_bytes;
    int32_t has_scales, use_count, tok_div;
};
struct DownArgs {
    int32_t n_hits;
    int64_t blob_bytes;
    int32_t has_scales, use_count;
};

void launch_gu(const uint8_t* blob, const int32_t* slot, const int32_t* dst, const int32_t* d_count,
               const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int64_t cap, int64_t blob_bytes,
               int tok_div, void* stream) {
    const GuArgs a{(int32_t) cap, blob_bytes, x_scales != nullptr ? 1 : 0, d_count != nullptr ? 1 : 0, tok_div};
    const metal::KernelArg args[] = {metal::buf(blob),
                                     metal::buf(slot),
                                     metal::buf(dst ? dst : slot),
                                     metal::buf(d_count ? d_count : slot),
                                     metal::buf(x_q8_0),
                                     metal::buf(x_scales ? x_scales : (const float*) x_q8_0),
                                     metal::buf(gate_up),
                                     metal::val(a)};
    const uint32_t sg = (uint32_t) (cap * 2 * FF);
    metal::launch((cudaStream_t) stream, "s2_gu_k", sg * 32u, 1, 1, 256, 1, 1, args, 8);
}
void launch_swiglu(float* gate_up, int64_t pairs, void* stream) {
    const metal::KernelArg args[] = {metal::buf(gate_up), metal::val(pairs)};
    metal::launch((cudaStream_t) stream, "s2_swiglu_k", (uint32_t) pairs, 1, 1, 256, 1, 1, args, 2);
}
void launch_down(const uint8_t* blob, const int32_t* slot, const int32_t* dst, const int32_t* d_count,
                 const uint8_t* h_q8_0, const float* h_scales, float* out, int64_t cap, int64_t blob_bytes,
                 void* stream) {
    const DownArgs a{(int32_t) cap, blob_bytes, h_scales != nullptr ? 1 : 0, d_count != nullptr ? 1 : 0};
    const metal::KernelArg args[] = {metal::buf(blob),
                                     metal::buf(slot),
                                     metal::buf(dst),
                                     metal::buf(d_count ? d_count : slot),
                                     metal::buf(h_q8_0),
                                     metal::buf(h_scales ? h_scales : (const float*) h_q8_0),
                                     metal::buf(out),
                                     metal::val(a)};
    const uint32_t sg = (uint32_t) (cap * H);
    metal::launch((cudaStream_t) stream, "s2_down_k", sg * 32u, 1, 1, 256, 1, 1, args, 8);
}

// The shared 4-step body; cap is the layout size (host count or capacity).
void grouped_body(const uint8_t* blob, const int32_t* slot, const int32_t* dst, const int32_t* d_count,
                  int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0, const float* x_scales, void* scratch,
                  float* out, void* stream, int tok_div) {
    Scratch s = carve(scratch, cap);
    launch_gu(blob, slot, dst, d_count, x_q8_0, x_scales, s.gate_up, cap, blob_bytes, tok_div, stream);
    launch_swiglu(s.gate_up, cap * FF, stream);
    if (x_scales != nullptr)
        quantize_q8_0_scaled(s.gate_up, s.h_q8_0, s.h_scales, cap * FF, stream);
    else
        quantize_q8_0(s.gate_up, s.h_q8_0, cap * FF, stream);
    launch_down(blob, slot, dst, d_count, s.h_q8_0, x_scales != nullptr ? s.h_scales : nullptr, out, cap,
                blob_bytes, stream);
}

}  // namespace

// Pure host arithmetic (a *_bytes probe in the RISKS.md S4 sense): the exact
// formula from s2_expert_grouped.cu - gate/up activations (n_hits x 2 x n_ff
// f32), the quantized intermediate (n_hits x n_ff/32 x 34 B), TWO fp32 scale
// regions for it (n_hits x n_ff/32), and the input's correction region
// (n_embd/32 f32); each 16-byte aligned.
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    grouped_body(blob_base, slot_index, dst_index, nullptr, n_hits, blob_bytes, x_q8_0, x_scales, scratch, out,
                 stream, 0);
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot,
                    int32_t* dst, int32_t* count, void* stream) {
    if (k < 1 || k > 32) {
        std::fprintf(stderr, "moe_hit_select: k must be 1..32\n");
        std::exit(1);
    }
    struct Args {
        int32_t n, n_expert;
    };
    const Args a{k, n_expert};
    const metal::KernelArg args[] = {metal::buf(ids), metal::buf(res_row), metal::buf(slot), metal::buf(dst),
                                     metal::buf(count), metal::val(a)};
    metal::launch((cudaStream_t) stream, "moe_hit_select_k", 1, 1, 1, 1, 1, 1, args, 6);
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    grouped_body(blob_base, slot_index, dst_index, d_count, cap, blob_bytes, x_q8_0, x_scales, scratch, out,
                 stream, 0);
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    struct Args {
        int32_t n_embd;
    };
    const Args a{(int32_t) n_embd};
    const metal::KernelArg args[] = {metal::buf(parts), metal::buf(hit_out), metal::buf(dst), metal::buf(count),
                                     metal::val(a)};
    metal::launch((cudaStream_t) stream, "moe_hit_add_k", (uint32_t) n_embd, (uint32_t) cap, 1, 256, 1, 1, args,
                  5);
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    Scratch s = carve(scratch, cap_entries);
    struct GrpArgs {
        int32_t cap_entries, has_scales;
    };
    const GrpArgs a{(int32_t) cap_entries, x_scales != nullptr ? 1 : 0};
    {
        const metal::KernelArg args[] = {metal::buf(grp_ptr),
                                         metal::buf(grp_start),
                                         metal::buf(n_groups),
                                         metal::buf(ent_tok),
                                         metal::buf(x_q8_0),
                                         metal::buf(x_scales ? x_scales : (const float*) x_q8_0),
                                         metal::buf(s.gate_up),
                                         metal::val(a)};
        const uint32_t sg = (uint32_t) (cap_entries * 2 * FF);
        metal::launch((cudaStream_t) stream, "s2_gu_grouped_k", sg * 32u, 1, 1, 256, 1, 1, args, 8);
    }
    launch_swiglu(s.gate_up, cap_entries * FF, stream);
    if (x_scales != nullptr)
        quantize_q8_0_scaled(s.gate_up, s.h_q8_0, s.h_scales, cap_entries * FF, stream);
    else
        quantize_q8_0(s.gate_up, s.h_q8_0, cap_entries * FF, stream);
    {
        const GrpArgs ad{(int32_t) cap_entries, x_scales != nullptr ? 1 : 0};
        const metal::KernelArg args[] = {metal::buf(grp_ptr),
                                         metal::buf(grp_start),
                                         metal::buf(n_groups),
                                         metal::buf(ent_dst),
                                         metal::buf(s.h_q8_0),
                                         metal::buf(x_scales != nullptr ? (const float*) s.h_scales
                                                                          : (const float*) s.h_q8_0),
                                         metal::buf(out),
                                         metal::val(ad)};
        const uint32_t sg = (uint32_t) (cap_entries * H);
        metal::launch((cudaStream_t) stream, "s2_down_grouped_k", sg * 32u, 1, 1, 256, 1, 1, args, 8);
    }
}

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) {
        std::fprintf(stderr, "moe_group_resident: n must be 1..128\n");
        std::exit(1);
    }
    struct Args {
        int32_t n, k_per_tok;
        uint64_t base;
        int64_t blob;
    };
    const Args a{n, k_per_tok, (uint64_t) (uintptr_t) base, blob};
    const metal::KernelArg args[] = {metal::buf(ids),     metal::buf(grp_ptr), metal::buf(grp_start),
                                     metal::buf(counts),  metal::buf(ent_dst), metal::buf(ent_tok),
                                     metal::val(a)};
    metal::launch((cudaStream_t) stream, "moe_group_resident_k", 1, 1, 1, 1, 1, 1, args, 7);
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot,
                          int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 128) {
        std::fprintf(stderr, "moe_hit_select_multi: n must be 1..128\n");
        std::exit(1);
    }
    struct Args {
        int32_t n, n_expert;
    };
    const Args a{n, n_expert};
    const metal::KernelArg args[] = {metal::buf(ids), metal::buf(res_row), metal::buf(slot), metal::buf(dst),
                                     metal::buf(count), metal::val(a)};
    metal::launch((cudaStream_t) stream, "moe_hit_select_k", 1, 1, 1, 1, 1, 1, args, 6);
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    grouped_body(blob_base, slot_index, dst_index, d_count, cap, blob_bytes, x_q8_0, x_scales, scratch, out,
                 stream, k_per_token);
}

// NOT PORTED, still loud on purpose: the experimental CPU-order variant exists
// to reproduce the CPU pool's VNNI lane order bit-for-bit (eight-lane shuffle
// pattern 4,1,2, accurate expf, separate correction term). A float
// approximation would defeat its only purpose - it is the parity diagnostic
// for hit-vs-miss agreement, not a production path (expert_source.cpp picks
// it only under its CPU-order mode). Port it only with a bit-exactness plan.
void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                 const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                 const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                 const float* x_scales, float* gate_up_trace) {
    (void) blob_base; (void) slot_index; (void) dst_index; (void) n_hits; (void) blob_bytes;
    (void) x_q8_0; (void) scratch; (void) out; (void) stream; (void) x_scales; (void) gate_up_trace;
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::kernels::moe_hit_grouped_s2_cpu_order\n");
    std::abort();
}

}  // namespace strata::kernels
