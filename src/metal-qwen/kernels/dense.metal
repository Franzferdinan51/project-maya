// dense.metal - the dense family's MSL kernels: dequantizers, the S/S2/BF16 GEMVs,
// the Q8 activation GEMVs, Q8_1 activation quantization and the generic native
// MMVQ. Arithmetic transcribed from the tree's CUDA TUs (src/kernels/cuda) and
// checked against the src/artifact scalar reference; the sequential kernels keep
// the CUDA naive kernels' accumulation order (one thread per output row, elements
// in order), the split kernels reduce through threadgroup memory and are NOT
// bit-identical to the sequential ones - the same contract the CUDA TUs state.

struct SFormArgs {
    int code_bits;
    int code_bias;
    int group_elems;
    int codebook;  // 0 = Affine, 1 = Iq4Nl (strata::kernels::Codebook)
    int has_offset;
};

inline float s_decode(int code, constant SFormArgs& f) {
    if (f.codebook == 1) return kIq4NlF[code & 0xF];
    return float(code + f.code_bias);
}

inline float s_weight(device const uchar* codes, device const float* scales, device const float* offset,
                      constant SFormArgs& f, uint o, uint i, uint n_in) {
    uint epb = 8u / uint(f.code_bits);
    uint mask = (1u << uint(f.code_bits)) - 1u;
    uint code = (codes[o * (n_in / epb) + i / epb] >> ((i % epb) * uint(f.code_bits))) & mask;
    uint g = i / uint(f.group_elems);
    float w = s_decode(int(code), f) * scales[o * (n_in / uint(f.group_elems)) + g];
    if (f.has_offset) w += offset[o * (n_in / uint(f.group_elems)) + g];
    return w;
}

// ---- dequantizers ----------------------------------------------------------

kernel void dequant_s2_k(device const uchar* codes [[buffer(0)]], device const float* scales [[buffer(1)]],
                         device float* out [[buffer(2)]], constant ulong& n_blocks [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    ulong total = n_blocks * 64;
    if (gid >= total) return;
    ulong b = gid / 64;
    uint j = gid % 64;
    int code = (codes[b * 16 + j / 4] >> ((j % 4) * 2)) & 3;
    out[gid] = float(code - 1) * scales[b];
}

// mode: 0 = f32 out, 1 = bf16 bits out, 2 = f16 bits out (out element size follows)
kernel void dequant_generic_k(device const uchar* blocks [[buffer(0)]], device uchar* out [[buffer(1)]],
                              constant int& type [[buffer(2)]], constant int& mode [[buffer(3)]],
                              constant ulong& row_bytes [[buffer(4)]], constant ulong& row0 [[buffer(5)]],
                              constant ulong& total [[buffer(6)]], constant uint& cols [[buffer(7)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    ulong r = gid / cols, e = gid % cols;
    float v = dq_elem(blocks + (row0 + r) * row_bytes, type, uint(e));
    if (mode == 0) {
        ((device float*) out)[gid] = v;
    } else if (mode == 1) {
        ((device ushort*) out)[gid] = dq_f2bf(v);
    } else {
        ((device ushort*) out)[gid] = dq_f2h(v);
    }
}

// ---- S2 / S GEMVs ------------------------------------------------------------

kernel void s2_gemv_k(device const ushort* x [[buffer(0)]], device const uchar* codes [[buffer(1)]],
                      device const float* scales [[buffer(2)]], device float* y [[buffer(3)]],
                      constant uint& n_in [[buffer(4)]], constant uint& n_out [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= n_out) return;
    uint o = gid, nb = n_in / 64;
    device const uchar* c = codes + o * nb * 16;
    device const float* s = scales + o * nb;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = s[b];
        for (uint j = 0; j < 64; ++j) {
            int code = (c[b * 16 + j / 4] >> ((j % 4) * 2)) & 3;
            acc += float(code - 1) * d * float(as_type<half>(x[b * 64 + j]));
        }
    }
    y[o] = acc;
}

// One threadgroup per output row; partial[t] holds thread t's strided partial sum.
kernel void s2_gemv_split_k(device const ushort* x [[buffer(0)]], device const uchar* codes [[buffer(1)]],
                            device const float* scales [[buffer(2)]], device float* y [[buffer(3)]],
                            constant uint& n_in [[buffer(4)]], constant uint& tpr [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float partial[1024];
    uint o = tg, nb = n_in / 64;
    device const uchar* c = codes + o * nb * 16;
    device const float* s = scales + o * nb;
    float acc = 0.0f;
    for (uint idx = tid; idx < n_in; idx += tpr) {
        uint b = idx / 64, j = idx % 64;
        int code = (c[b * 16 + j / 4] >> ((j % 4) * 2)) & 3;
        acc += float(code - 1) * s[b] * float(as_type<half>(x[idx]));
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = tpr / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[o] = partial[0];
}

kernel void s_gemv_k(device const ushort* x [[buffer(0)]], device const uchar* codes [[buffer(1)]],
                     device const float* scales [[buffer(2)]], device const float* offset [[buffer(3)]],
                     device float* y [[buffer(4)]], constant uint& n_in [[buffer(5)]],
                     constant uint& n_out [[buffer(6)]], constant SFormArgs& f [[buffer(7)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n_out) return;
    float acc = 0.0f;
    for (uint i = 0; i < n_in; ++i)
        acc += s_weight(codes, scales, offset, f, gid, i, n_in) * float(as_type<half>(x[i]));
    y[gid] = acc;
}

kernel void s_gemv_split_k(device const ushort* x [[buffer(0)]], device const uchar* codes [[buffer(1)]],
                           device const float* scales [[buffer(2)]], device const float* offset [[buffer(3)]],
                           device float* y [[buffer(4)]], constant uint& n_in [[buffer(5)]],
                           constant uint& tpr [[buffer(6)]], constant SFormArgs& f [[buffer(7)]],
                           uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float partial[1024];
    float acc = 0.0f;
    for (uint i = tid; i < n_in; i += tpr)
        acc += s_weight(codes, scales, offset, f, tg, i, n_in) * float(as_type<half>(x[i]));
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = tpr / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[tg] = partial[0];
}

// Q8 activations against S-form weights. act_kind: 0 = Q8_0 blocks (34 B / 32,
// f16 scale at 0, qs at 2), 1 = Q8_K blocks (292 B / 256, f32 scale at 0, qs at 4).
inline float q8_act(device const uchar* act, int act_kind, uint i) {
    if (act_kind == 0) {
        device const uchar* b = act + (i / 32) * 34;
        return float(int(char(b[2 + (i % 32)]))) * dq_ldf16(b);
    }
    device const uchar* b = act + (i / 256) * 292;
    float d = as_type<float>(uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24));
    return float(int(char(b[4 + (i % 256)]))) * d;
}

kernel void s_gemv_q8_k(device const uchar* act [[buffer(0)]], device const uchar* codes [[buffer(1)]],
                        device const float* scales [[buffer(2)]], device const float* offset [[buffer(3)]],
                        device float* y [[buffer(4)]], constant uint& n_in [[buffer(5)]],
                        constant uint& n_out [[buffer(6)]], constant SFormArgs& f [[buffer(7)]],
                        constant int& act_kind [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n_out) return;
    float acc = 0.0f;
    for (uint i = 0; i < n_in; ++i) acc += s_weight(codes, scales, offset, f, gid, i, n_in) * q8_act(act, act_kind, i);
    y[gid] = acc;
}

kernel void s_gemv_q8_split_k(device const uchar* act [[buffer(0)]], device const uchar* codes [[buffer(1)]],
                              device const float* scales [[buffer(2)]], device const float* offset [[buffer(3)]],
                              device float* y [[buffer(4)]], constant uint& n_in [[buffer(5)]],
                              constant uint& tpr [[buffer(6)]], constant SFormArgs& f [[buffer(7)]],
                              constant int& act_kind [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float partial[1024];
    float acc = 0.0f;
    for (uint i = tid; i < n_in; i += tpr)
        acc += s_weight(codes, scales, offset, f, tg, i, n_in) * q8_act(act, act_kind, i);
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = tpr / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[tg] = partial[0];
}

// The CUDA s2_gemv_q8 shape: threadgroup per row, threads striding quads, four
// accumulators per thread (one per quad lane), then a tree reduction.
kernel void s2_gemv_q8_k(device const uchar* act [[buffer(0)]], device const uchar* codes [[buffer(1)]],
                         device const float* scales [[buffer(2)]], device float* y [[buffer(3)]],
                         constant uint& n_in [[buffer(4)]], constant uint& tpr [[buffer(5)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float partial[1024];
    uint o = tg;
    uint n_quads = n_in / 4;
    device const uchar* c = codes + o * n_quads;
    device const float* s = scales + o * (n_in / 64);
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (uint q = tid; q < n_quads; q += tpr) {
        uchar byte = c[q];
        float d = s[q >> 4];
        device const uchar* blk = act + ((q * 4) / 32) * 34;
        float dx = dq_ldf16(blk);
        int off = int((q * 4) % 32);
        a0 += float(int(byte & 3) - 1) * d * (float(int(char(blk[2 + off + 0]))) * dx);
        a1 += float(int((byte >> 2) & 3) - 1) * d * (float(int(char(blk[2 + off + 1]))) * dx);
        a2 += float(int((byte >> 4) & 3) - 1) * d * (float(int(char(blk[2 + off + 2]))) * dx);
        a3 += float(int((byte >> 6) & 3) - 1) * d * (float(int(char(blk[2 + off + 3]))) * dx);
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = tpr / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[o] = partial[0];
}

// ---- BF16 GEMVs ----------------------------------------------------------------

kernel void bf16_gemv_k(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant uint& n_in [[buffer(3)]],
                        constant uint& n_out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n_out) return;
    float acc = 0.0f;
    device const ushort* row = w + gid * n_in;
    for (uint i = 0; i < n_in; ++i) acc += dq_bf2f(x[i]) * dq_bf2f(row[i]);
    y[gid] = acc;
}

kernel void bf16_gemv_split_k(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant uint& n_in [[buffer(3)]],
                              constant uint& tpr [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float partial[1024];
    device const ushort* row = w + tg * n_in;
    float acc = 0.0f;
    for (uint i = tid; i < n_in; i += tpr) acc += dq_bf2f(x[i]) * dq_bf2f(row[i]);
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = tpr / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[tg] = partial[0];
}

// The pinned MMVF shape: threadgroup per row, threads striding element PAIRS with
// two ordered FMAs per pair (the CUDA kernel's ggml_cuda_mad order), warp sums then
// a cross-warp partial reduction.
kernel void bf16_mmvf_k(device const float* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant uint& n_in [[buffer(3)]],
                        constant uint& bs [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];
    device const ushort* row = w + tg * n_in;
    float acc = 0.0f;
    for (uint pair = tid; pair < n_in / 2; pair += bs) {
        acc = fma(dq_bf2f(row[2 * pair]), x[2 * pair], acc);
        acc = fma(dq_bf2f(row[2 * pair + 1]), x[2 * pair + 1], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) partials[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float p = (tid < bs / 32) ? partials[tid] : 0.0f;
        p = simd_sum(p);
        if (tid == 0) y[tg] = p;
    }
}

kernel void bf16_mmvf_multi_k(device const float* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                              device float* y [[buffer(2)]], constant uint& n_in [[buffer(3)]],
                              constant uint& bs [[buffer(4)]], constant int& ldx [[buffer(5)]],
                              constant int& ldy [[buffer(6)]], constant int& n_tok [[buffer(7)]],
                              uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8 * 8];
    device const ushort* row = w + tg * n_in;
    float acc[8];
    for (int k = 0; k < 8; ++k) acc[k] = 0.0f;
    for (uint pair = tid; pair < n_in / 2; pair += bs) {
        float w0 = dq_bf2f(row[2 * pair]), w1 = dq_bf2f(row[2 * pair + 1]);
        for (int k = 0; k < n_tok; ++k) {
            acc[k] = fma(w0, x[k * ldx + 2 * pair], acc[k]);
            acc[k] = fma(w1, x[k * ldx + 2 * pair + 1], acc[k]);
        }
    }
    for (int k = 0; k < n_tok; ++k) {
        float a = simd_sum(acc[k]);
        if (lane == 0) partials[k * 8 + sg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        for (int k = 0; k < n_tok; ++k) {
            float p = (tid < bs / 32) ? partials[k * 8 + tid] : 0.0f;
            p = simd_sum(p);
            if (tid == 0) y[k * ldy + tg] = p;
        }
    }
}

// ---- Q8_1 activation quantization + generic MMVQ --------------------------------

// One threadgroup of 32 per 32-element block: block_q8_1 is {f16 d, f16 sum, qs[32]}
// (36 bytes). d = min(amax/127, 65504), q = round(x/d) clamped to [-127, 127], the
// sum is the ORIGINAL inputs' sum clamped to f16 range - the CUDA kernel's contract.
kernel void quantize_q8_1_k(device const float* x [[buffer(0)]], device uchar* y [[buffer(1)]],
                            uint tg [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    float xi = x[tg * 32 + lane];
    float amax = simd_max(fabs(xi));
    float sum = simd_sum(xi);
    float d = min(amax / 127.0f, 65504.0f);
    int q = (amax == 0.0f) ? 0 : int(clamp(round(xi / d), -127.0f, 127.0f));
    device uchar* b = y + tg * 36;
    b[4 + lane] = uchar(q);
    if (lane == 0) {
        ushort db = dq_f2h(d), sb = dq_f2h(clamp(sum, -65504.0f, 65504.0f));
        b[0] = uchar(db & 0xFF);
        b[1] = uchar(db >> 8);
        b[2] = uchar(sb & 0xFF);
        b[3] = uchar(sb >> 8);
    }
}

// One thread per (output row, activation column); the reduction runs sequentially
// in element order over dequantized weights x dequantized Q8_1 activations. This is
// the float form of the CUDA integer-dot MMVQ: same products, different grouping
// of the K-format scale/min decomposition, so results differ from the CUDA kernel
// only by float rounding (documented in the dense workstream report).
kernel void mmvq_k(device const uchar* w [[buffer(0)]], device const uchar* xq [[buffer(1)]],
                   device float* y [[buffer(2)]], constant int& type [[buffer(3)]],
                   constant uint& n_in [[buffer(4)]], constant uint& n_out [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) {
    uint total = n_out;  // grid is n_out * ncols; column = gid / n_out
    uint col = gid / total, o = gid % total;
    uint be = 0, bb = 0;
    dq_geometry(type, be, bb);
    device const uchar* row = w + ulong(o) * (n_in / be) * bb;
    device const uchar* xc = xq + ulong(col) * (n_in / 32) * 36;
    float acc = 0.0f;
    for (uint i = 0; i < n_in; ++i) {
        device const uchar* ab = xc + (i / 32) * 36;
        float act = float(int(char(ab[4 + (i % 32)]))) * dq_ldf16(ab);
        acc += dq_elem(row, type, i) * act;
    }
    y[col * n_out + o] = acc;
}

// ---- IQ embedding / gate-up dequantizers ---------------------------------------

// out[t * n_embd + e] = dequant(table row tokens[t])[e]
kernel void iq_embed_k(device const uchar* table [[buffer(0)]], device const int* tokens [[buffer(1)]],
                       device float* out [[buffer(2)]], constant int& type [[buffer(3)]],
                       constant ulong& row_bytes [[buffer(4)]], constant uint& n_embd [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    uint t = gid / n_embd, e = gid % n_embd;
    device const uchar* row = table + ulong(tokens[t]) * row_bytes;
    out[gid] = dq_elem(row, type, e);
}

// Interleaved gate/up dequant to f16 bits: dst row 2r = gate row r, 2r+1 = up row r.
kernel void iq_gu_k(device const uchar* gate [[buffer(0)]], device const uchar* up [[buffer(1)]],
                    device ushort* dst [[buffer(2)]], constant int& type [[buffer(3)]],
                    constant ulong& row_bytes [[buffer(4)]], constant uint& n_embd [[buffer(5)]],
                    constant uint& rows2 [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    uint total = rows2 * n_embd;
    if (gid >= total) return;
    uint row2 = gid / n_embd, e = gid % n_embd;
    device const uchar* src = (row2 % 2 == 0) ? gate : up;
    device const uchar* row = src + ulong(row2 / 2) * row_bytes;
    dst[gid] = dq_f2h(dq_elem(row, type, e));
}
