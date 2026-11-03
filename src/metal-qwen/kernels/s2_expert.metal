// s2_expert.metal - the grouped S2 (Q2_0) GPU expert tier. Blob geometry is
// fixed by the expert blob layout (see s2_expert_grouped.hpp / the CUDA source):
// H = 2560 inputs, FF = 640, gate/up rows interleaved (row-slot 2r = gate r,
// 2r+1 = up r), 640 B of 2-bit codes per gate/up row, 160 B per down row, one
// fp16 scale per 64 elements. The arithmetic is the CUDA kernel's:
//   acc += d_w * d_x * (sum_j c_j*x_j - sum_j x_j)
// over int8 activation chunks, integer-exact per chunk. d_x comes from the
// caller's fp32 x_scales when supplied, else the block's fp16 d - the two
// contracts the header spells out. Summation across lanes is a simd tree, not
// the CPU's order (the same contract the CUDA kernel carries).

constant int S2_H = 2560;
constant int S2_FF = 640;
constant int S2_ROW_GU = 640;    // H/4 bytes of codes
constant int S2_ROW_D = 160;     // FF/4
constant int S2_SC_GU = 40;      // H/64 scales
constant int S2_SC_D = 10;       // FF/64
constant ulong S2_O_D_CODES = 819200ul;    // 2*FF*ROW_GU
constant ulong S2_O_GU_SCALES = 1228800ul; // + H*ROW_D
constant ulong S2_O_D_SCALES = 1331200ul;  // + 2*FF*SC_GU*2

// One 32-element chunk's integer dot pair (s = sum c*x, hx = sum x).
inline void s2_chunk(device const uchar* codes8, device const uchar* xb, thread int& s, thread int& hx) {
    s = 0;
    hx = 0;
    for (int j = 0; j < 8; ++j) {
        const uint cb = codes8[j];
        const int cw[4] = {int(cb & 3u), int((cb >> 2) & 3u), int((cb >> 4) & 3u), int((cb >> 6) & 3u)};
        for (int k = 0; k < 4; ++k) {
            const int x = int(int8_t(xb[2 + 4 * j + k]));
            s += cw[k] * x;
            hx += x;
        }
    }
}

// A simdgroup's strided partial over the chunks of one row.
inline float s2_row_partial(device const uchar* codes, device const uchar* scales, device const uchar* x_q8_0,
                            device const float* x_scales, bool has_scales, int n_chunks, ushort lane) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        int s, hx;
        s2_chunk(codes + c * 8, x_q8_0 + c * 34, s, hx);
        const float dx = has_scales ? x_scales[c] : moe_f16_at(x_q8_0 + c * 34);
        const float dw = moe_f16_at(scales + (c >> 1) * 2);
        acc += dw * dx * float(s - hx);
    }
    return acc;
}

struct S2GuArgs {
    int n_hits;  // capacity in the device-count form
    long blob_bytes;
    int has_scales;
    int use_count;
    int tok_div;  // >0: hit h reads token dst[h]/tok_div's activation (multi)
};

kernel void s2_gu_k(device const uchar* blob_base [[buffer(0)]], device const int* slot_index [[buffer(1)]],
                    device const int* dst_index [[buffer(2)]], device const int* d_count [[buffer(3)]],
                    device const uchar* x_q8_0 [[buffer(4)]], device const float* x_scales [[buffer(5)]],
                    device float* gate_up [[buffer(6)]], constant S2GuArgs& a [[buffer(7)]],
                    uint tg_pos [[threadgroup_position_in_grid]], uint sg_in_tg [[simdgroup_index_in_threadgroup]],
                    uint tptg [[threads_per_threadgroup]], uint tpsg [[threads_per_simdgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint sgid = tg_pos * (tptg / tpsg) + sg_in_tg;
    const long slot = long(sgid);
    const long rows_per_hit = 2L * S2_FF;
    if (slot >= long(a.n_hits) * rows_per_hit) return;
    const int h = int(slot / rows_per_hit);
    if (a.use_count && h >= d_count[0]) return;
    const int i = int(slot % rows_per_hit);
    device const uchar* blob = blob_base + ulong(slot_index[h]) * ulong(a.blob_bytes);
    device const uchar* x = x_q8_0;
    device const float* xs = x_scales;
    if (a.tok_div > 0) {
        const int tok = dst_index[h] / a.tok_div;
        x += ulong(tok) * (S2_H / 32) * 34;
        xs += ulong(tok) * (S2_H / 32);
    }
    const float acc = s2_row_partial(blob + ulong(i) * S2_ROW_GU, blob + S2_O_GU_SCALES + ulong(i) * S2_SC_GU * 2,
                                     x, xs, a.has_scales != 0, S2_H / 32, lane);
    const float s = simd_sum(acc);
    if (lane != 0) return;
    // gate-major output: gate rows from 0, up rows from n_hits*FF
    const int r = i >> 1;
    const ulong base = (i & 1) ? (ulong(a.n_hits) * S2_FF + ulong(h) * S2_FF) : (ulong(h) * S2_FF);
    gate_up[base + r] = s;
}

kernel void s2_swiglu_k(device float* gate_up [[buffer(0)]], constant long& n_pairs [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (long(i) >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + exp(-g))) * u;
}

struct S2DownArgs {
    int n_hits;
    long blob_bytes;
    int has_scales;  // h_scales supplied for the intermediate
    int use_count;
};

kernel void s2_down_k(device const uchar* blob_base [[buffer(0)]], device const int* slot_index [[buffer(1)]],
                      device const int* dst_index [[buffer(2)]], device const int* d_count [[buffer(3)]],
                      device const uchar* h_q8_0 [[buffer(4)]], device const float* h_scales [[buffer(5)]],
                      device float* out [[buffer(6)]], constant S2DownArgs& a [[buffer(7)]],
                      uint tg_pos [[threadgroup_position_in_grid]], uint sg_in_tg [[simdgroup_index_in_threadgroup]],
                    uint tptg [[threads_per_threadgroup]], uint tpsg [[threads_per_simdgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint sgid = tg_pos * (tptg / tpsg) + sg_in_tg;
    const long row = long(sgid);
    if (row >= long(a.n_hits) * S2_H) return;
    const int h = int(row / S2_H);
    if (a.use_count && h >= d_count[0]) return;
    const int r = int(row % S2_H);
    device const uchar* blob = blob_base + ulong(slot_index[h]) * ulong(a.blob_bytes);
    device const uchar* xb = h_q8_0 + ulong(h) * (S2_FF / 32) * 34;
    device const float* hs = a.has_scales ? h_scales + ulong(h) * (S2_FF / 32) : nullptr;
    const float acc = s2_row_partial(blob + S2_O_D_CODES + ulong(r) * S2_ROW_D,
                                     blob + S2_O_D_SCALES + ulong(r) * S2_SC_D * 2, xb, hs, a.has_scales != 0,
                                     S2_FF / 32, lane);
    const float s = simd_sum(acc);
    if (lane == 0) out[ulong(dst_index[h]) * S2_H + r] = s;
}

// ---- grouped form (moe_grouped_s2): one blob pointer per group, entries are
// (token, dst) pairs. An entry's group is found by scanning grp_start (at most
// cap_entries long). NOTE: grp_ptr entries are raw contents addresses of
// shared buffers; converting them back to device pointers is the one place
// this backend relies on unified-memory addressing - flagged for the Mac
// build to confirm.
struct S2GrpArgs {
    int cap_entries;
    int has_scales;
};

inline int s2_group_of(device const int* grp_start, device const int* n_groups, int e) {
    const int ng = n_groups[0];
    for (int g = 0; g < ng; ++g)
        if (e >= grp_start[g] && e < grp_start[g + 1]) return g;
    return -1;
}

kernel void s2_gu_grouped_k(device const ulong* grp_ptr [[buffer(0)]], device const int* grp_start [[buffer(1)]],
                            device const int* n_groups [[buffer(2)]], device const int* ent_tok [[buffer(3)]],
                            device const uchar* x_q8_0 [[buffer(4)]], device const float* x_scales [[buffer(5)]],
                            device float* gate_up [[buffer(6)]], constant S2GrpArgs& a [[buffer(7)]],
                            uint tg_pos [[threadgroup_position_in_grid]], uint sg_in_tg [[simdgroup_index_in_threadgroup]],
                    uint tptg [[threads_per_threadgroup]], uint tpsg [[threads_per_simdgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint sgid = tg_pos * (tptg / tpsg) + sg_in_tg;
    const long slot = long(sgid);
    const long rows_per_entry = 2L * S2_FF;
    if (slot >= long(a.cap_entries) * rows_per_entry) return;
    const int e = int(slot / rows_per_entry);
    const int g = s2_group_of(grp_start, n_groups, e);
    if (g < 0) return;
    const int i = int(slot % rows_per_entry);
    device const uchar* blob = (device const uchar*)grp_ptr[g];
    device const uchar* x = x_q8_0 + ulong(ent_tok[e]) * (S2_H / 32) * 34;
    device const float* xs = x_scales + ulong(ent_tok[e]) * (S2_H / 32);
    const float acc = s2_row_partial(blob + ulong(i) * S2_ROW_GU, blob + S2_O_GU_SCALES + ulong(i) * S2_SC_GU * 2,
                                     x, xs, a.has_scales != 0, S2_H / 32, lane);
    const float s = simd_sum(acc);
    if (lane != 0) return;
    const int r = i >> 1;
    const ulong base = (i & 1) ? (ulong(a.cap_entries) * S2_FF + ulong(e) * S2_FF) : (ulong(e) * S2_FF);
    gate_up[base + r] = s;
}

kernel void s2_down_grouped_k(device const ulong* grp_ptr [[buffer(0)]], device const int* grp_start [[buffer(1)]],
                              device const int* n_groups [[buffer(2)]], device const int* ent_dst [[buffer(3)]],
                              device const uchar* h_q8_0 [[buffer(4)]], device const float* h_scales [[buffer(5)]],
                              device float* out [[buffer(6)]], constant S2GrpArgs& a [[buffer(7)]],
                              uint tg_pos [[threadgroup_position_in_grid]], uint sg_in_tg [[simdgroup_index_in_threadgroup]],
                    uint tptg [[threads_per_threadgroup]], uint tpsg [[threads_per_simdgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint sgid = tg_pos * (tptg / tpsg) + sg_in_tg;
    const long row = long(sgid);
    if (row >= long(a.cap_entries) * S2_H) return;
    const int e = int(row / S2_H);
    const int g = s2_group_of(grp_start, n_groups, e);
    if (g < 0) return;
    const int r = int(row % S2_H);
    device const uchar* blob = (device const uchar*)grp_ptr[g];
    device const uchar* xb = h_q8_0 + ulong(e) * (S2_FF / 32) * 34;
    device const float* hs = a.has_scales ? h_scales + ulong(e) * (S2_FF / 32) : nullptr;
    const float acc = s2_row_partial(blob + S2_O_D_CODES + ulong(r) * S2_ROW_D,
                                     blob + S2_O_D_SCALES + ulong(r) * S2_SC_D * 2, xb, hs, a.has_scales != 0,
                                     S2_FF / 32, lane);
    const float s = simd_sum(acc);
    if (lane == 0) out[ulong(ent_dst[e]) * S2_H + r] = s;
}

// ---- moe_group_resident: groups for the all-resident case, built serially on
// one thread (n <= 128): groups in first-appearance order, entries of a group
// in routing order - the CUDA kernel's result, computed its serial equivalent.
struct ResidentArgs {
    int n;
    int k_per_tok;
    ulong base;
    long blob;
};

kernel void moe_group_resident_k(device const int* ids [[buffer(0)]], device ulong* grp_ptr [[buffer(1)]],
                                 device int* grp_start [[buffer(2)]], device int* counts [[buffer(3)]],
                                 device int* ent_dst [[buffer(4)]], device int* ent_tok [[buffer(5)]],
                                 constant ResidentArgs& a [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    int gi = 0, acc = 0;
    for (int j = 0; j < a.n; ++j) {
        bool first = true;
        for (int t = 0; t < j; ++t)
            if (ids[t] == ids[j]) {
                first = false;
                break;
            }
        if (!first) continue;
        grp_ptr[gi] = a.base + ulong(ids[j]) * ulong(a.blob);
        grp_start[gi] = acc;
        int rank_base = acc;
        for (int t = j; t < a.n; ++t)
            if (ids[t] == ids[j]) {
                ent_dst[rank_base] = t;
                ent_tok[rank_base] = t / a.k_per_tok;
                ++rank_base;
            }
        acc = rank_base;
        ++gi;
    }
    grp_start[gi] = acc;
    counts[0] = gi;
    counts[1] = acc;
}
