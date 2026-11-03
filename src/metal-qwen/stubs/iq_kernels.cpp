// src/metal-qwen/stubs/iq_kernels.cpp - Metal implementation for
// include/strata/kernels/iq_kernels.hpp (was the stage-1 stub).
//
// The RISKS §4 probes are pure host answers, converted first (the engine
// probes before it runs - NativeEmbed::load calls iq_supported during pack
// load): iq_supported, iq_row_bytes, native_expert_layout and
// native_expert_scratch_bytes carry the CUDA TU's tables verbatim, and
// gemm_iq_f16_grouped_supported honestly answers false (no grouped WMMA
// port; gemm_iq_f16_grouped itself also returns false so callers fall
// back). The kernel entries are implemented in dense.metal for the types
// dq_elem decodes; entries for the still-unported grid i-quants stop loudly
// by name, as stage 1 required of real work not yet done.
// native_expert_grouped is implemented at the bottom of this TU over
// kernels/native_expert.metal (the grouped-expert workstream; see
// GROUPED-NOTES.md in this directory).

#include "strata/kernels/iq_kernels.hpp"

#include "../kernels/dense_host.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
// Types dense.metal's dq_elem decodes among the is_iq set (the remaining grid
// i-quants - IQ1_S/M, IQ2_XXS/XS/S, IQ3_S - are not among them yet; IQ3_XXS is).
bool iq_covered(int t) { return t == 11 || t == 18 || t == 20 || t == 23 || t == 42; }

void iq_dequant_launch(int mode, int ggml_type, const void* src, int64_t n, void* dst, void* stream) {
    if (n % 256 != 0 || !iq_covered(ggml_type)) {
        std::fprintf(stderr, "strata metal: iq_dequant: type %d not yet ported or bad n\n", ggml_type);
        std::abort();
    }
    const uint64_t rb = (uint64_t) iq_row_bytes(ggml_type, 256);
    const uint64_t r0 = 0, total = (uint64_t) n;
    const uint32_t cols = 256;
    const metal::KernelArg args[] = {metal::buf(src), metal::buf(dst),     metal::val(ggml_type),
                                     metal::val(mode), metal::val(rb),     metal::val(r0),
                                     metal::val(total), metal::val(cols)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "dequant_generic_k", (uint32_t) total, 256, args,
                              8, false);
}
}  // namespace
bool iq_supported(int ggml_type) noexcept {
    // The CUDA TU's is_iq list, verbatim.
    return ggml_type == 16 || ggml_type == 17 || ggml_type == 18 || ggml_type == 19 || ggml_type == 20 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 23 || ggml_type == 29 || ggml_type == 42 ||
           ggml_type == 11;
}
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept {
    // ggml block byte sizes, as the CUDA TU's sizeof(block_*) table: q4_0 18, q5_0 22,
    // q8_0 34, q4_K 144, q5_K 176, q6_K 210, iq2_xxs 66, iq2_xs 74, iq3_xxs 98,
    // iq4_nl 18, iq3_s 110, iq2_s 82, iq1_m 56, iq1_s 50, iq4_xs 136, q3_K 110, q2_0 18.
    // (iq3_xxs is 98 = f16 + 96 qs bytes per ggml-common's static_assert; an earlier
    // transcription here said 50 - iq1_s's size - and is fixed.)
    switch (ggml_type) {
        case 2: return (size_t) (n / 32) * 18;
        case 6: return (size_t) (n / 32) * 22;
        case 8: return (size_t) (n / 32) * 34;
        case 12: return (size_t) (n / 256) * 144;
        case 13: return (size_t) (n / 256) * 176;
        case 14: return (size_t) (n / 256) * 210;
        case 16: return (size_t) (n / 256) * 66;
        case 17: return (size_t) (n / 256) * 74;
        case 18: return (size_t) (n / 256) * 98;
        case 20: return (size_t) (n / 32) * 18;
        case 21: return (size_t) (n / 256) * 110;
        case 22: return (size_t) (n / 256) * 82;
        case 29: return (size_t) (n / 256) * 56;
        case 19: return (size_t) (n / 256) * 50;
        case 23: return (size_t) (n / 256) * 136;
        case 11: return (size_t) (n / 256) * 110;
        case 42: return (size_t) (n / 64) * 18;
        default: return 0;
    }
}
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    // Same block_q8_1 image as native_quantize_q8_1 (the CUDA TU's two quantizers
    // produce the same 36-byte blocks), so the same MSL kernel serves both.
    const int64_t n = n_rows * n_cols;
    if (n <= 0) return;
    const metal::KernelArg args[] = {metal::buf(x), metal::buf(y)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "quantize_q8_1_k", (uint32_t) n, 32, args, 2,
                              false);
}
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    // Covered by the generic MMVQ kernel for the formats dense.metal's dq_elem
    // decodes (Q3_K, Q5_K, IQ4_NL, IQ4_XS, Q2_0 - the CUDA iq_mmvq switch's
    // non-grid types - plus IQ3_XXS, whose grid codebook dq_elem now carries
    // and which the Qwen native path's PLE keys use). The remaining grid
    // i-quants (IQ1_S/M, IQ2_XXS/XS/S, IQ3_S) still stop loudly here.
    switch (ggml_type) {
    case 11: case 13: case 18: case 20: case 23: case 42: break;
    default:
        std::fprintf(stderr, "strata metal: iq_mmvq: grid i-quant type %d not yet ported\n", ggml_type);
        std::abort();
    }
    const uint32_t ni = (uint32_t) n_in, no = (uint32_t) n_out;
    const metal::KernelArg args[] = {metal::buf(w), metal::buf(x_q8_1), metal::buf(y), metal::val(ggml_type),
                                     metal::val(ni), metal::val(no)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "mmvq_k", no * (uint32_t) ncols, 128, args, 6,
                              false);
}
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream) {
    iq_dequant_launch(2, ggml_type, src, n, dst, stream);
}
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream) {
    iq_dequant_launch(0, ggml_type, src, n, dst, stream);
}
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !iq_covered(ggml_type)) {
        std::fprintf(stderr, "strata metal: iq_embed_rows: type %d not yet ported or bad n_embd\n", ggml_type);
        std::abort();
    }
    const uint64_t rb = (uint64_t) row_bytes;
    const uint32_t ne = (uint32_t) n_embd;
    const metal::KernelArg args[] = {metal::buf(table), metal::buf(tokens), metal::buf(out),
                                     metal::val(ggml_type), metal::val(rb), metal::val(ne)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "iq_embed_k", (uint32_t) (n_tok * n_embd), 256,
                              args, 6, false);
}
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream) {
    if (!iq_covered(ggml_type) && ggml_type != 13) {
        std::fprintf(stderr, "strata metal: iq_dequant_gu_f16: type %d not yet ported\n", ggml_type);
        std::abort();
    }
    const uint64_t rb = (uint64_t) iq_row_bytes(ggml_type, n_embd);
    const uint32_t ne = (uint32_t) n_embd, rows2 = (uint32_t) (2 * n_ff);
    const metal::KernelArg args[] = {metal::buf(gate), metal::buf(up),     metal::buf(dst),
                                     metal::val(ggml_type), metal::val(rb), metal::val(ne),
                                     metal::val(rows2)};
    dense_detail::launch_sync(dense_detail::stream_of(stream), "iq_gu_k", rows2 * ne, 256, args, 7, false);
}
bool gemm_iq_f16_grouped(int ggml_type, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream,
                         const int32_t* ids, const GroupedWeightTable* indirect, bool down) {
    (void) ggml_type; (void) W; (void) expert_bytes; (void) n_out; (void) K; (void) X; (void) ldx;
    (void) bounds; (void) n_experts; (void) max_rows; (void) Y; (void) ldy; (void) stream;
    (void) ids; (void) indirect; (void) down;
    return false;  // honest: the grouped WMMA launch is not ported; callers fall back by design
}
bool gemm_iq_f16_grouped_supported(int dev) {
    (void) dev;
    return false;  // honest probe: the grouped WMMA launch has no Metal port yet
}
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    // Pure host arithmetic, ported from the CUDA TU.
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff) {
    // Pure host arithmetic, ported from the CUDA TU (36-byte block_q8_1).
    const size_t f = (size_t) cap_entries * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap_entries * (size_t) (n_ff / 32) * 36 + 255) & ~(size_t) 255);
}
namespace {
// Shared by both grouped launchers below: the type/geometry gate (the
// intersection of the CUDA launcher's switches with the formats dq_elem
// decodes: gate/up {Q5_K 13, IQ3_XXS 18, IQ4_XS 23, Q2_0 42} of its
// {16,17,18,21,22,23,29,19,42,13}, down {IQ3_XXS 18, IQ4_NL 20, IQ4_XS 23,
// Q2_0 42} - its whole switch; the remaining grid i-quants stop loudly, the
// same line iq_mmvq draws) and the scratch carving, byte-identical to the
// CUDA TU (and to native_expert_scratch_bytes above).
struct GroupedScratch {
    float *gate, *up, *h;
    void* hq;
};
GroupedScratch grouped_validate_carve(const NativeExpertLayout& L, int64_t cap_entries, void* scratch,
                                      const char* who) {
    const bool gu_ok = L.gu_type == 13 || L.gu_type == 18 || L.gu_type == 23 || L.gu_type == 42;
    const bool d_ok = L.d_type == 18 || L.d_type == 20 || L.d_type == 23 || L.d_type == 42;
    if (!gu_ok || !d_ok || L.n_embd <= 0 || L.n_ff <= 0 || L.n_embd % 32 != 0 || L.n_ff % 32 != 0) {
        std::fprintf(stderr,
                     "strata metal: %s: gate/up type %d / down type %d not yet ported "
                     "or bad geometry (n_embd %lld, n_ff %lld)\n",
                     who, L.gu_type, L.d_type, (long long) L.n_embd, (long long) L.n_ff);
        std::abort();
    }
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    GroupedScratch sc;
    sc.gate = (float*) scratch;
    sc.up = (float*) ((uint8_t*) scratch + fa);
    sc.h = (float*) ((uint8_t*) scratch + 2 * fa);
    sc.hq = (void*) ((uint8_t*) scratch + 3 * fa);
    return sc;
}
// The two middle stages, launched once over all entries by both launchers
// (swiglu into the third scratch region, its q8_1 quantization into the
// fourth - the shared quantize_q8_1_k, as in quantize_q8_1_rows).
void grouped_middle(cudaStream_t s, const GroupedScratch& sc, int64_t cap_entries, int64_t n_ff,
                    float swiglu_limit) {
    const long long nh = (long long) cap_entries * n_ff;
    {
        const metal::KernelArg args[] = {metal::buf(sc.gate), metal::buf(sc.up), metal::buf(sc.h),
                                         metal::val(swiglu_limit), metal::val(nh)};
        metal::launch(s, "native_swiglu_k", (uint32_t) nh, 1, 1, 256, 1, 1, args, 5);
    }
    {
        const metal::KernelArg args[] = {metal::buf(sc.h), metal::buf(sc.hq)};
        metal::launch(s, "quantize_q8_1_k", (uint32_t) nh, 1, 1, 32, 1, 1, args, 2);
    }
}
}  // namespace

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream, float swiglu_limit) {
    // The grouped native experts, composed exactly as the CUDA TU composes
    // them (src/kernels/cuda/iq_kernels.cu): gate/up dots into the scratch's
    // first two regions, swiglu into the third, q8_1 quantization of the
    // intermediate into the fourth, then the down dots into `out`. The MSL
    // kernels are in kernels/native_expert.metal (plus the shared
    // quantize_q8_1_k).
    //
    // This launcher reads the group plan on the host, after a stream sync -
    // which is only legal for callers that run OUTSIDE a captured body,
    // where the stream's in-flight work completes without the host's help.
    // Its remaining callers are exactly those: RemoteExperts::launch (a
    // host-side pool thread; its finish() syncs the same stream by design)
    // and the parity runner. Captured-body callers (the verify window) must
    // use native_expert_grouped_based below: inside a body this sync waits
    // for the stream's in-flight command buffers, which include flag-spin
    // kernels that only the post-body host loop can release - the sync
    // deadlocks the body against itself (the grouped workstream's first
    // end-to-end hang).
    if (cap_groups <= 0 || cap_entries <= 0) return;
    const GroupedScratch sc = grouped_validate_carve(L, cap_entries, scratch, "native_expert_grouped");
    float* gate = sc.gate;
    float* up = sc.up;
    void* hq = sc.hq;
    cudaStream_t s = dense_detail::stream_of(stream);
    // The group plan is read on the host, after a stream sync: grp_ptr's
    // values are raw contents addresses of other allocations, which are
    // host values in this backend - an MSL kernel cannot dereference one
    // (the same reason verify_kernels.cpp executes fetch_blobs and
    // resident_plan on the host). Each group then launches with its blob as
    // a genuine buffer argument. The cost is one sync per call; the CUDA
    // original is a single launch that reads the counts on the device.
    if (s) cudaStreamSynchronize(s);
    else cudaDeviceSynchronize();
    int32_t ng = *n_groups;
    if (ng < 0) ng = 0;
    if (ng > cap_groups) ng = (int32_t) cap_groups;
    struct GuArgs {
        int32_t type, n_embd, n_ff, n_entries;
        uint64_t gu_row, up_off;
        int32_t e0, pad;
    };
    struct DownArgs {
        int32_t type, n_ff, n_embd, n_entries;
        uint64_t d_row, down_off;
        int32_t e0, pad;
    };
    const uint32_t gx_gu = (uint32_t) (((2 * L.n_ff + 7) / 8) * 256);
    const uint32_t gx_dn = (uint32_t) (((L.n_embd + 7) / 8) * 256);
    for (int32_t g = 0; g < ng; ++g) {  // gate/up, one launch per group
        const int32_t e0 = grp_start[g], e1 = grp_start[g + 1];
        if (e1 <= e0) continue;
        const GuArgs a{L.gu_type, (int32_t) L.n_embd, (int32_t) L.n_ff, e1 - e0,
                       (uint64_t) L.gu_row, (uint64_t) L.up_off, e0, 0};
        const void* blob = (const void*) (uintptr_t) grp_ptr[g];
        const metal::KernelArg args[] = {metal::buf(blob), metal::buf(ent_tok), metal::buf(x_q8_1),
                                         metal::buf(gate), metal::buf(up),      metal::val(a)};
        metal::launch(s, "native_gu_one_k", gx_gu, 1, 1, 256, 1, 1, args, 6);
    }
    grouped_middle(s, sc, cap_entries, L.n_ff, swiglu_limit);
    for (int32_t g = 0; g < ng; ++g) {  // down, one launch per group
        const int32_t e0 = grp_start[g], e1 = grp_start[g + 1];
        if (e1 <= e0) continue;
        const DownArgs a{L.d_type, (int32_t) L.n_ff, (int32_t) L.n_embd, e1 - e0,
                         (uint64_t) L.d_row, (uint64_t) L.down_off, e0, 0};
        const void* blob = (const void*) (uintptr_t) grp_ptr[g];
        const metal::KernelArg args[] = {metal::buf(blob), metal::buf(ent_dst), metal::buf(hq),
                                         metal::buf(out),  metal::val(a)};
        metal::launch(s, "native_down_one_k", gx_dn, 1, 1, 256, 1, 1, args, 5);
    }
}

void native_expert_grouped_based(const NativeExpertLayout& L, const void* blob_base, uint64_t blob_span,
                                 const unsigned long long* grp_ptr, const int32_t* grp_start,
                                 const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                                 int64_t cap_groups, int64_t cap_entries, const void* x_q8_1, void* scratch,
                                 float* out, void* stream, float swiglu_limit) {
    // The captured-body form of the grouped native experts: the CUDA
    // launcher's structure, faithfully - ONE launch per stage over a fixed
    // grid whose y dimension is cap_groups, the kernels reading n_groups /
    // grp_start / grp_ptr on the device (native_gu_grouped_k /
    // native_down_grouped_k in kernels/native_expert.metal). No host sync,
    // no host readback, at any point: the only stream state this launcher
    // touches is appending three command buffers to the stream's queue, in
    // order, so it cannot wait on work that a later host action must
    // release - the self-deadlock of the host-plan form cannot recur.
    //
    // The one Metal-forced difference from CUDA: a blob is addressed as an
    // offset inside the bound base allocation (grp_ptr[g] - blob_base,
    // computed on the device from the pointer VALUE), because an MSL kernel
    // cannot dereference a contents address belonging to another buffer.
    // The caller guarantees every group's blob lies inside the base
    // allocation (the verify window: the expert-cache arena for the VRAM
    // groups, the staging region for the PCIe groups); the kernels skip a
    // group whose pointer falls outside the span rather than reading wild.
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (blob_base == nullptr) {
        std::fprintf(stderr, "strata metal: native_expert_grouped_based: null blob base\n");
        std::abort();
    }
    const GroupedScratch sc = grouped_validate_carve(L, cap_entries, scratch, "native_expert_grouped_based");
    cudaStream_t s = dense_detail::stream_of(stream);
    const uint64_t base_value = (uint64_t) (uintptr_t) blob_base;
    struct GuGroupedArgs {
        int32_t type, n_embd, n_ff, pad0;
        uint64_t gu_row, up_off, base_value, base_span, blob_bytes;
    };
    struct DownGroupedArgs {
        int32_t type, n_ff, n_embd, pad0;
        uint64_t d_row, down_off, base_value, base_span, blob_bytes;
    };
    const uint32_t gx_gu = (uint32_t) (((2 * L.n_ff + 7) / 8) * 256);
    const uint32_t gx_dn = (uint32_t) (((L.n_embd + 7) / 8) * 256);
    {   // gate/up: grid (row blocks, cap_groups), the CUDA ggu shape
        const GuGroupedArgs a{L.gu_type, (int32_t) L.n_embd, (int32_t) L.n_ff, 0,
                              (uint64_t) L.gu_row, (uint64_t) L.up_off, base_value, blob_span,
                              (uint64_t) L.bytes};
        const metal::KernelArg args[] = {metal::buf(blob_base), metal::buf(grp_ptr), metal::buf(grp_start),
                                         metal::buf(n_groups),  metal::buf(ent_tok), metal::buf(x_q8_1),
                                         metal::buf(sc.gate),   metal::buf(sc.up),    metal::val(a)};
        metal::launch(s, "native_gu_grouped_k", gx_gu, (uint32_t) cap_groups, 1, 256, 1, 1, args, 9);
    }
    grouped_middle(s, sc, cap_entries, L.n_ff, swiglu_limit);
    {   // down: grid (row blocks, cap_groups), the CUDA gd shape
        const DownGroupedArgs a{L.d_type, (int32_t) L.n_ff, (int32_t) L.n_embd, 0,
                                (uint64_t) L.d_row, (uint64_t) L.down_off, base_value, blob_span,
                                (uint64_t) L.bytes};
        const metal::KernelArg args[] = {metal::buf(blob_base), metal::buf(grp_ptr), metal::buf(grp_start),
                                         metal::buf(n_groups),  metal::buf(ent_dst), metal::buf(sc.hq),
                                         metal::buf(out),       metal::val(a)};
        metal::launch(s, "native_down_grouped_k", gx_dn, (uint32_t) cap_groups, 1, 256, 1, 1, args, 8);
    }
}
}  // namespace strata::kernels

