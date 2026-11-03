// misc.metal - activation quantization (quantize_act family), the elementwise
// glue ops, and the control-vector apply kernel. Transcribed from
// src/kernels/cuda/quantize_act.cu, elementwise.cu and cvec.cu.

inline float f16_of(float f) { return dq_h2f(dq_f2h(f)); }

// ---- quantize_act -----------------------------------------------------------

// One thread per 32-element block, as the CUDA kernel. NOTE: the CUDA kernel
// divides in double before rint; MSL has no portable double path here, so the
// division is float - values can differ from the CUDA bytes only when x/d32
// sits within float-vs-double rounding of a .5 tie (documented delta).
kernel void quantize_q8_0_k(device const float* x [[buffer(0)]], device uchar* blocks [[buffer(1)]],
                            constant ulong& n_blocks [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n_blocks) return;
    device const float* xb = x + gid * 32;
    device uchar* out = blocks + gid * 34;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmax(amax, fabs(xb[i]));
    if (amax == 0.0f) {
        out[0] = 0;
        out[1] = 0;
        for (int i = 0; i < 32; ++i) out[2 + i] = 0;
        return;
    }
    float d32 = amax / 127.0f;
    ushort dbits = dq_f2h(d32);
    out[0] = uchar(dbits & 0xFF);
    out[1] = uchar(dbits >> 8);
    for (int i = 0; i < 32; ++i) {
        float q = rint(xb[i] / d32);
        q = clamp(q, -128.0f, 127.0f);
        out[2 + i] = uchar(char(q));
    }
}

kernel void quantize_q8_0_scaled_k(device const float* x [[buffer(0)]], device uchar* blocks [[buffer(1)]],
                                   device float* scales [[buffer(2)]], constant ulong& n_blocks [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= n_blocks) return;
    device const float* xb = x + gid * 32;
    device uchar* out = blocks + gid * 34;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmax(amax, fabs(xb[i]));
    float s = (amax > 0.0f) ? amax / 127.0f : 0.0f;
    float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    scales[gid] = s;
    ushort dbits = dq_f2h(s);
    out[0] = uchar(dbits & 0xFF);
    out[1] = uchar(dbits >> 8);
    for (int i = 0; i < 32; ++i) {
        float t = xb[i] * inv;
        float r = t + (t >= 0.0f ? 0.5f : -0.5f);
        int v = int(r);
        v = clamp(v, -127, 127);
        out[2 + i] = uchar(char(v));
    }
}

kernel void dequant_q8_0_k(device const uchar* blocks [[buffer(0)]], device float* x [[buffer(1)]],
                           constant ulong& n_blocks [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n_blocks) return;
    device const uchar* blk = blocks + gid * 34;
    float d = dq_ldf16(blk);
    for (int i = 0; i < 32; ++i) x[gid * 32 + i] = float(int(char(blk[2 + i]))) * d;
}

inline int nearest_int_m(float fval) {
    float val = fval + 12582912.0f;
    int i = as_type<int>(val);
    return (i & 0x007fffff) - 0x00400000;
}

kernel void quantize_q8_K_k(device const float* x [[buffer(0)]], device uchar* blocks [[buffer(1)]],
                            constant ulong& n_blocks [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n_blocks) return;
    device const float* xb = x + gid * 256;
    device uchar* out = blocks + gid * 292;
    float mx = 0.0f, amax = 0.0f;
    for (int j = 0; j < 256; ++j) {
        float ax = fabs(xb[j]);
        if (ax > amax) {
            amax = ax;
            mx = xb[j];
        }
    }
    if (amax == 0.0f) {
        ((device float*) out)[0] = 0.0f;
        for (int j = 0; j < 256; ++j) out[4 + j] = 0;
        for (int j = 0; j < 16; ++j) {
            out[260 + 2 * j] = 0;
            out[261 + 2 * j] = 0;
        }
        return;
    }
    float iscale = -127.0f / mx;
    for (int j = 0; j < 256; ++j) {
        float prod = iscale * xb[j];  // rounded to f32 before nearest_int, as the source does
        int v = nearest_int_m(prod);
        out[4 + j] = uchar(char(min(127, v)));  // MIN only - the source has no lower clamp
    }
    for (int j = 0; j < 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) sum += int(char(out[4 + j * 16 + ii]));
        short bs = short(sum);
        out[260 + 2 * j] = uchar(bs & 0xFF);
        out[261 + 2 * j] = uchar((bs >> 8) & 0xFF);
    }
    ((device float*) out)[0] = 1.0f / iscale;
}

kernel void dequant_q8_K_k(device const uchar* blocks [[buffer(0)]], device float* x [[buffer(1)]],
                           constant ulong& n_blocks [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n_blocks) return;
    device const uchar* blk = blocks + gid * 292;
    float d = as_type<float>(uint(blk[0]) | (uint(blk[1]) << 8) | (uint(blk[2]) << 16) | (uint(blk[3]) << 24));
    for (int i = 0; i < 256; ++i) x[gid * 256 + i] = float(int(char(blk[4 + i]))) * d;
}

// ---- elementwise --------------------------------------------------------------

kernel void embedding_gather_k(device const uchar* codes [[buffer(0)]], device const float* scales [[buffer(1)]],
                               device const float* offsets [[buffer(2)]], device float* out [[buffer(3)]],
                               constant long& n [[buffer(4)]], constant int& code_bits [[buffer(5)]],
                               constant int& code_bias [[buffer(6)]], constant int& group_elems [[buffer(7)]],
                               constant int& has_offset [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(n)) return;
    uint epb = 8u / uint(code_bits);
    uint mask = (1u << uint(code_bits)) - 1u;
    uint code = (codes[gid / epb] >> ((gid % epb) * uint(code_bits))) & mask;
    float v = float(int(code) + code_bias) * scales[gid / uint(group_elems)];
    if (has_offset) v = v + offsets[gid / uint(group_elems)];  // separate rounding, as the CPU decoder
    out[gid] = v;
}

kernel void gdn_gate_k(device const float* alpha [[buffer(0)]], device const float* dt [[buffer(1)]],
                       device const float* ssm_a [[buffer(2)]], device float* gate [[buffer(3)]],
                       constant ulong& total [[buffer(4)]], constant uint& h_v [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    uint h = gid % h_v;
    float z = alpha[gid] + dt[h];
    float sp = (z > 20.0f) ? z : log(1.0f + exp(z));  // ggml_compute_softplus_f32's large-x branch
    gate[gid] = sp * ssm_a[h];
}

kernel void scale_inplace_k(device float* x [[buffer(0)]], constant ulong& n [[buffer(1)]],
                            constant float& s [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    x[gid] *= s;
}

kernel void add_inplace_k(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
                          constant ulong& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] += src[gid];
}

kernel void f32_to_f16_bulk_k(device const float* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                              constant ulong& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    y[gid] = dq_f2h(x[gid]);
}

kernel void f32_to_bf16_bulk_k(device const float* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                               constant ulong& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    y[gid] = dq_f2bf(x[gid]);
}

kernel void silu_inplace_k(device float* x [[buffer(0)]], constant ulong& n [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    x[gid] = x[gid] / (1.0f + exp(-x[gid]));
}

// RMS norm over the last axis: one threadgroup of 256 per row.
kernel void rms_norm_weighted_k(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                                constant uint& cols [[buffer(2)]], constant float& eps [[buffer(3)]],
                                constant int& has_w [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device float* row = x + tg * cols;
    float ss = 0.0f;
    for (uint c = tid; c < cols; c += 256) ss += row[c] * row[c];
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += partial[i];
        partial[0] = 1.0f / sqrt(tot / float(cols) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partial[0];
    for (uint c = tid; c < cols; c += 256) row[c] = row[c] * inv * (has_w ? w[c] : 1.0f);
}

kernel void doorbell_ring_k(device uint* seq [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    *seq += 1;
    threadgroup_barrier(mem_flags::mem_device);
}

// Stream-ordered device-to-device copies for the runtime's async memcpy
// surface (GRFOLD): a host-side immediate copy reads the source at CALL time,
// while kernels already enqueued on the stream may still be writing it - the
// QSA q/gate split (cudaMemcpy2DAsync between the Q GEMV and the norm/RoPE
// kernels) corrupted multi-row verify windows exactly that way. Launched on
// the caller's stream, these copy at their stream position like any kernel.
kernel void memcpy_bytes_k(device uchar* dst [[buffer(0)]], device const uchar* src [[buffer(1)]],
                           constant ulong& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                           uint grid [[threads_per_grid]]) {
    for (ulong i = gid; i < n; i += grid) dst[i] = src[i];
}

kernel void memcpy2d_bytes_k(device uchar* dst [[buffer(0)]], device const uchar* src [[buffer(1)]],
                             constant ulong& dpitch [[buffer(2)]], constant ulong& spitch [[buffer(3)]],
                             constant ulong& width [[buffer(4)]], constant ulong& height [[buffer(5)]],
                             uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    const ulong total = width * height;
    for (ulong i = gid; i < total; i += grid) {
        const ulong r = i / width, c = i - r * width;
        dst[r * dpitch + c] = src[r * spitch + c];
    }
}

kernel void copy_f32_k(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
                       constant ulong& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = src[gid];
}

kernel void copy_i32_k(device int* dst [[buffer(0)]], device const int* src [[buffer(1)]],
                       constant ulong& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = src[gid];
}

kernel void copy_rows_mapped_k(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
                               device const int* hit_rows [[buffer(2)]], device const int* count [[buffer(3)]],
                               constant ulong& width [[buffer(4)]], constant ulong& total [[buffer(5)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    ulong r = gid / width;
    bool hit = false;
    for (int i = 0; i < *count; ++i)
        if (hit_rows[i] == int(r)) {
            hit = true;
            break;
        }
    dst[gid] = hit ? 0.0f : src[gid];
}

kernel void doorbell_publish_k(device const float* x [[buffer(0)]], device const int* ids [[buffer(1)]],
                               device const float* weights [[buffer(2)]], device float* x_out [[buffer(3)]],
                               device int* ids_out [[buffer(4)]], device float* weights_out [[buffer(5)]],
                               device uint* seq [[buffer(6)]], constant ulong& n [[buffer(7)]],
                               constant ulong& k [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) x_out[gid] = x[gid];
    if (gid < k) {
        ids_out[gid] = ids[gid];
        weights_out[gid] = weights[gid];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (gid == 0) *seq += 1;
}

// ---- cvec -----------------------------------------------------------------------
// One threadgroup (256 threads) per (stream c, token t), the CUDA kernel's shape:
// pending write first, then the mode-0 projection dot through a block reduction.
kernel void cvec_apply_k(device float* R [[buffer(0)]], device const float* dir [[buffer(1)]],
                         device const float* s_l [[buffer(2)]], device const int* on [[buffer(3)]],
                         constant int& mode [[buffer(4)]], constant long& layer [[buffer(5)]],
                         constant int& n [[buffer(6)]], constant int& hc [[buffer(7)]],
                         constant long& r_ld [[buffer(8)]], device const float* bo [[buffer(9)]],
                         constant long& bo_ld [[buffer(10)]], device const float* inj [[buffer(11)]],
                         constant long& inj_ld [[buffer(12)]], constant int& write [[buffer(13)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    int c = int(tg % uint(hc));
    long t = long(tg / uint(hc));
    device float* r = R + t * r_ld + long(c) * n;
    float s = s_l[layer];
    bool steer = (*on != 0) && s != 0.0f;
    if (!steer && !write) return;
    device const float* v = dir + layer * n;
    float wgt = write ? 2.0f * (1.0f / (1.0f + exp(-(inj[t * inj_ld + c] / float(hc))))) : 0.0f;
    float xv[16];
    float dot = 0.0f;
    for (int kk = 0; kk < 16; ++kk) {
        int d = int(tid) + kk * 256;
        if (d < n) {
            float val = r[d];
            if (write) val = fma(bo[t * bo_ld + d], wgt, val);
            xv[kk] = val;
            if (steer && mode == 0) dot = fma(val, v[d], dot);
        } else {
            xv[kk] = 0.0f;
        }
    }
    if (steer && mode == 0) {
        dot = simd_sum(dot);
        if (lane == 0) part[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            float p = (tid < 8) ? part[tid] : 0.0f;
            p = simd_sum(p);
            if (tid == 0) part[0] = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = part[0] * s;
    }
    for (int kk = 0; kk < 16; ++kk) {
        int d = int(tid) + kk * 256;
        if (d < n) {
            float val = xv[kk];
            if (steer) val = (mode == 0) ? fma(-dot, v[d], val) : val + v[d];
            r[d] = val;
        }
    }
}
