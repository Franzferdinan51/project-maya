// native_expert.metal - the grouped native-format experts (the device side of
// strata::kernels::native_expert_grouped, include/strata/kernels/iq_kernels.hpp).
//
// Semantics transcribed from src/kernels/cuda/iq_kernels.cu. Two launcher
// shapes share the dots below: the plain entry passes a group's blob as a
// genuine buffer argument (its host launcher reads the group plan on the
// host - raw blob addresses are host values in this backend, so the kernels
// never chase a pointer value), each launch covering one group's entries
// [e0, e1); the based entry (native_gu_grouped_k / native_down_grouped_k,
// below) reads the plan on the device under the CUDA grid and addresses
// each blob as an offset inside one bound base allocation, which is the
// shape a captured body requires (no host sync is possible mid-body).
//   gate[r] = grow_r . x[ent_tok[e]],  up[r] = uprow_r . x[ent_tok[e]]
//   h       = swiglu(gate, up)   (plain, or GLM's clamped form at a finite limit)
//   out[ent_dst[e]] = down . q8_1(h)
// with the swiglu and the q8_1 quantization launched once over all entries by
// the host (the shared quantize_q8_1_k), between the two kernels here.
//
// The dots are the float form of the CUDA integer vec_dot_*_q8_1 products:
// dq_elem's dequantized weight times the dequantized q8_1 activation (stored
// f16 scale times the int8 code), summed per simdgroup lane in the CUDA
// row_dot stride order and reduced with the xor butterfly - the same
// float-form contract kernels/dense.metal's mmvq_k documents. For the types
// the host launcher admits (Q2_0, Q5_K, IQ4_XS, IQ3_XXS gate/up; IQ4_NL,
// IQ4_XS, IQ3_XXS, Q2_0 down) the per-element products are identical to the
// integer dots; only the summation grouping differs.

struct NativeGuArgs {
    int type;       // ggml type of the gate/up rows
    int n_embd;     // row length (activation length)
    int n_ff;       // gate/up rows per role
    int n_entries;  // the group's entry count (e1 - e0)
    ulong gu_row;   // bytes per gate/up row
    ulong up_off;   // byte offset of the up rows inside the blob
    int e0;         // the group's first entry
    int pad;
};

struct NativeDownArgs {
    int type;       // ggml type of the down rows
    int n_ff;       // row length (intermediate length)
    int n_embd;     // down rows
    int n_entries;  // the group's entry count (e1 - e0)
    ulong d_row;    // bytes per down row
    ulong down_off; // byte offset of the down rows inside the blob
    int e0;         // the group's first entry
    int pad;
};

// One row's dot against one q8_1 activation: lane-strided partial over the
// row's elements, then the warp reduction. xq points at the activation's
// first q8_1 block (36 bytes per 32 elements: f16 d, f16 sum, int8 qs[32]).
inline float native_row_dot(device const uchar* row, int type, device const uchar* xq, int n_in, ushort lane) {
    float acc = 0.0f;
    for (int i = int(lane); i < n_in; i += 32) {
        device const uchar* ab = xq + (i / 32) * 36;
        const float act = float(int(char(ab[4 + (i % 32)]))) * dq_ldf16(ab);
        acc += dq_elem(row, type, uint(i)) * act;
    }
    return moe_warp_sum_xor(acc);
}

// Grid: ceil(2*n_ff/8) threadgroups of 256 threads. One simdgroup (warp) per
// row, 8 rows per threadgroup - the CUDA GU_ROWS shape. The kernel loops the
// group's entries, as the CUDA kernel does.
kernel void native_gu_one_k(device const uchar* blob [[buffer(0)]], device const int* ent_tok [[buffer(1)]],
                            device const uchar* xq [[buffer(2)]], device float* gate [[buffer(3)]],
                            device float* up [[buffer(4)]], constant NativeGuArgs& a [[buffer(5)]],
                            uint tg_pos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const int row = int(tg_pos) * 8 + int(tid >> 5);  // 0 .. 2*n_ff
    if (row >= 2 * a.n_ff) return;
    const bool is_up = row >= a.n_ff;
    const int r = is_up ? row - a.n_ff : row;
    device const uchar* wr = blob + (is_up ? a.up_off : 0ul) + ulong(r) * a.gu_row;
    const ulong xb = ulong(a.n_embd / 32) * 36;  // activation bytes per token
    for (int e = a.e0; e < a.e0 + a.n_entries; ++e) {
        const float s = native_row_dot(wr, a.type, xq + ulong(ent_tok[e]) * xb, a.n_embd, ushort(lane));
        if (lane == 0) (is_up ? up : gate)[ulong(e) * ulong(a.n_ff) + r] = s;
    }
}

// h[i] = silu(gate[i]) * up[i], with GLM's clamp when the limit is finite:
// gate clamped above at the limit, up clamped to [-limit, limit], and the
// plain (unclamped) path otherwise - the CUDA swiglu_entries_kernel's two
// arms. exp() here is MSL's precise form, matching the finite arm's expf.
kernel void native_swiglu_k(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                            device float* h [[buffer(2)]], constant float& limit [[buffer(3)]],
                            constant long& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (long(gid) >= n) return;
    if (isinf(limit)) {
        const float g = gate[gid];
        h[gid] = (g / (1.0f + exp(-g))) * up[gid];
        return;
    }
    const float g = min(gate[gid], limit);
    const float u = min(max(up[gid], -limit), limit);
    h[gid] = (g / (1.0f + exp(-g))) * u;
}

// ---- grouped form (native_expert_grouped_based): the CUDA launcher's own
// shape. ONE launch over a fixed grid of (row blocks x cap_groups); the plan
// (n_groups / grp_start / grp_ptr) is read on the device, exactly as
// native_gu_kernel / native_down_kernel read it - no host readback, so this
// entry can run inside a captured body while earlier stream work is still in
// flight. A group's blob is addressed relative to the bound base buffer:
// grp_ptr[g] holds the blob's host contents address (a VALUE, never
// dereferenced); its offset from the base's contents address (base_value)
// selects the bytes inside the bound allocation. A pointer outside
// [base_value, base_value + base_span) or a blob that would run past the span
// contributes nothing (the CUDA kernel would dereference it; every real
// caller's plan satisfies the range by construction - the expert cache arena
// for the VRAM groups, the verifier's staging region for the PCIe groups).
struct NativeGuGroupedArgs {
    int type;
    int n_embd;
    int n_ff;
    int pad0;
    ulong gu_row;
    ulong up_off;
    ulong base_value;  // host contents address of the bound base buffer
    ulong base_span;   // bytes from base_value that are in range
    ulong blob_bytes;  // NativeExpertLayout.bytes
};

kernel void native_gu_grouped_k(device const uchar* base [[buffer(0)]], device const ulong* grp_ptr [[buffer(1)]],
                                device const int* grp_start [[buffer(2)]], device const int* n_groups [[buffer(3)]],
                                device const int* ent_tok [[buffer(4)]], device const uchar* xq [[buffer(5)]],
                                device float* gate [[buffer(6)]], device float* up [[buffer(7)]],
                                constant NativeGuGroupedArgs& a [[buffer(8)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int g = int(tg.y);
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e1 <= e0) return;
    const ulong ptr = grp_ptr[g];
    if (ptr < a.base_value) return;
    const ulong off = ptr - a.base_value;
    if (off > a.base_span || a.blob_bytes > a.base_span - off) return;
    device const uchar* blob = base + off;
    const int row = int(tg.x) * 8 + int(tid >> 5);  // 0 .. 2*n_ff
    if (row >= 2 * a.n_ff) return;
    const bool is_up = row >= a.n_ff;
    const int r = is_up ? row - a.n_ff : row;
    device const uchar* wr = blob + (is_up ? a.up_off : 0ul) + ulong(r) * a.gu_row;
    const ulong xb = ulong(a.n_embd / 32) * 36;  // activation bytes per token
    for (int e = e0; e < e1; ++e) {
        const float s = native_row_dot(wr, a.type, xq + ulong(ent_tok[e]) * xb, a.n_embd, ushort(lane));
        if (lane == 0) (is_up ? up : gate)[ulong(e) * ulong(a.n_ff) + r] = s;
    }
}

struct NativeDownGroupedArgs {
    int type;
    int n_ff;
    int n_embd;
    int pad0;
    ulong d_row;
    ulong down_off;
    ulong base_value;
    ulong base_span;
    ulong blob_bytes;
};

kernel void native_down_grouped_k(device const uchar* base [[buffer(0)]],
                                  device const ulong* grp_ptr [[buffer(1)]],
                                  device const int* grp_start [[buffer(2)]],
                                  device const int* n_groups [[buffer(3)]],
                                  device const int* ent_dst [[buffer(4)]], device const uchar* hq [[buffer(5)]],
                                  device float* out [[buffer(6)]], constant NativeDownGroupedArgs& a [[buffer(7)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const int g = int(tg.y);
    if (g >= n_groups[0]) return;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e1 <= e0) return;
    const ulong ptr = grp_ptr[g];
    if (ptr < a.base_value) return;
    const ulong off = ptr - a.base_value;
    if (off > a.base_span || a.blob_bytes > a.base_span - off) return;
    device const uchar* blob = base + off;
    const int r = int(tg.x) * 8 + int(tid >> 5);  // 0 .. n_embd
    if (r >= a.n_embd) return;
    device const uchar* wr = blob + a.down_off + ulong(r) * a.d_row;
    const ulong hb = ulong(a.n_ff / 32) * 36;  // intermediate bytes per entry
    for (int e = e0; e < e1; ++e) {
        const float s = native_row_dot(wr, a.type, hq + ulong(e) * hb, a.n_ff, ushort(lane));
        if (lane == 0) out[ulong(ent_dst[e]) * ulong(a.n_embd) + r] = s;
    }
}

// Grid: ceil(n_embd/8) threadgroups of 256 threads. One simdgroup per output
// row - the CUDA native_down_kernel shape.
kernel void native_down_one_k(device const uchar* blob [[buffer(0)]], device const int* ent_dst [[buffer(1)]],
                              device const uchar* hq [[buffer(2)]], device float* out [[buffer(3)]],
                              constant NativeDownArgs& a [[buffer(4)]],
                              uint tg_pos [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(tg_pos) * 8 + int(tid >> 5);  // 0 .. n_embd
    if (r >= a.n_embd) return;
    device const uchar* wr = blob + a.down_off + ulong(r) * a.d_row;
    const ulong hb = ulong(a.n_ff / 32) * 36;  // intermediate bytes per entry
    for (int e = a.e0; e < a.e0 + a.n_entries; ++e) {
        const float s = native_row_dot(wr, a.type, hq + ulong(e) * hb, a.n_ff, ushort(lane));
        if (lane == 0) out[ulong(ent_dst[e]) * ulong(a.n_embd) + r] = s;
    }
}
