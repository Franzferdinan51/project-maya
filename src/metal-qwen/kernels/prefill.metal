// prefill.metal - the prompt-chunk kernels of include/strata/prefill/kernels.hpp.
// Semantics transcribed from src/prefill/kernels.cu (the CUDA TU): same
// constants (N 2560, HC 4, D 10240, LR 320, S 128, HK 16, HV 48, C 10240),
// same per-element arithmetic and, where a reduction's order is observable,
// the same order (pf_block_sum is the CUDA block_sum: xor-butterfly warp
// sums, warp leaders to threadgroup memory, first warp reduces). Optional
// outputs are bound to a dummy buffer by the host launcher and gated by the
// flags argument, as in the other families.

// block_sum of src/prefill/kernels.cu, for threadgroups of <= 1024 threads:
// every thread ends with the total.
inline float pf_block_sum(float v, threadgroup float* sh, uint tid, uint tcount) {
    v = moe_warp_sum_xor(v);
    uint lane = tid & 31u, sg = tid >> 5;
    if (lane == 0u) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nw = (tcount + 31u) >> 5;
    float t = (tid < nw) ? sh[tid] : 0.0f;
    if (sg == 0u) t = moe_warp_sum_xor(t);
    if (tid == 0u) sh[0] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}
// The 128-thread variant whose cross-warp total is part[0]+part[1]+part[2]+
// part[3] in that order (gdn_l2_kernel / gdn_out_norm_kernel in the CUDA TU).
inline float pf_warp_parts_sum(float v, threadgroup float* part, uint tid) {
    v = moe_warp_sum_xor(v);
    if ((tid & 31u) == 0u) part[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return part[0] + part[1] + part[2] + part[3];
}

// ---------------------------------------------------------------- hyper-connection
kernel void pf_gr_norm_kernel(device const float* R [[buffer(0)]],
                              device const float* w [[buffer(1)]],
                              constant float& eps [[buffer(2)]],
                              device float* xn [[buffer(3)]],
                              device ushort* xn16 [[buffer(4)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tcount [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const int N = 2560, HC = 4;
    int c = int(row % uint(HC));
    device const float* r = R + ulong(row) * N;
    float ss = 0.0f;
    for (uint d = tid; d < uint(N); d += tcount) ss += r[d] * r[d];
    float rs = rsqrt(pf_block_sum(ss, sh, tid, tcount) / float(N) + eps);
    for (uint d = tid; d < uint(N); d += tcount) {
        float v = r[d] * rs * w[c * N + int(d)];
        xn[ulong(row) * N + d] = v;
        xn16[ulong(row) * N + d] = dq_f2bf(v);
    }
}

kernel void pf_gr_norm_rs_kernel(device const float* R [[buffer(0)]],
                                 device const float* w [[buffer(1)]],
                                 constant float& eps [[buffer(2)]],
                                 device float* rs_out [[buffer(3)]],
                                 device ushort* xn16 [[buffer(4)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tcount [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    const int N = 2560, HC = 4;
    int c = int(row % uint(HC));
    device const float* r = R + ulong(row) * N;
    float ss = 0.0f;
    for (uint d = tid; d < uint(N); d += tcount) ss += r[d] * r[d];
    float rs = rsqrt(pf_block_sum(ss, sh, tid, tcount) / float(N) + eps);
    if (tid == 0u) rs_out[row] = rs;
    for (uint d = tid; d < uint(N); d += tcount)
        xn16[ulong(row) * N + d] = dq_f2bf(r[d] * rs * w[c * N + int(d)]);
}

kernel void pf_gr_mix_r_kernel(device const float* R [[buffer(0)]],
                               device const float* rs [[buffer(1)]],
                               device const float* w [[buffer(2)]],
                               device const float* g [[buffer(3)]],
                               device float* mixed [[buffer(4)]],
                               device ushort* mixed16 [[buffer(5)]],
                               device ushort* mixed_h [[buffer(6)]],
                               constant long& Tn [[buffer(7)]],
                               constant int& flags [[buffer(8)]],
                               uint i [[thread_position_in_grid]]) {
    const int N = 2560, HC = 4, D = 10240;
    if (long(i) >= Tn * N) return;
    int t = int(i) / N, d = int(i) % N;
    float s = 0.0f;
    for (int c = 0; c < HC; ++c) {
        ulong j = ulong(t) * D + ulong(c) * N + d;
        float x = R[j] * rs[t * HC + c] * w[c * N + d];
        s = fma(x, gdn_sigmoid(g[j]), s);
    }
    s /= float(HC);
    mixed[i] = s;
    if (flags & 1) mixed16[i] = dq_f2bf(s);
    if (flags & 2) mixed_h[i] = dq_f2h(s);
}

kernel void pf_gr_write_norm_rs_kernel(device float* R [[buffer(0)]],
                                       device const float* bo [[buffer(1)]],
                                       device const float* inj [[buffer(2)]],
                                       constant long& inj_ld [[buffer(3)]],
                                       device const float* w [[buffer(4)]],
                                       constant float& eps [[buffer(5)]],
                                       device float* rs_out [[buffer(6)]],
                                       device ushort* xn16 [[buffer(7)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint row [[threadgroup_position_in_grid]]) {
    threadgroup float sh[32];
    const int N = 2560, HC = 4;
    int t = int(row) / HC, c = int(row) % HC;
    device float* r = R + ulong(row) * N;
    float sc = 2.0f * gdn_sigmoid(inj[ulong(t) * inj_ld + c] / float(HC));
    float v[10];
    float ss = 0.0f;
    int k = 0;
    for (uint d = tid; d < uint(N); d += 256u, ++k) {
        float x = fma(bo[ulong(t) * N + d], sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    float rs = rsqrt(pf_block_sum(ss, sh, tid, 256u) / float(N) + eps);
    if (tid == 0u) rs_out[row] = rs;
    k = 0;
    for (uint d = tid; d < uint(N); d += 256u, ++k)
        xn16[ulong(row) * N + d] = dq_f2bf(v[k] * rs * w[c * N + int(d)]);
}

kernel void pf_gr_silu_kernel(device const float* lo [[buffer(0)]],
                              device ushort* lo16 [[buffer(1)]],
                              constant long& n [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (long(i) >= n) return;
    float x = lo[i] / 4.0f;  // HC
    lo16[i] = dq_f2bf(x / (1.0f + exp(-x)));
}

kernel void pf_gr_mix_kernel(device const float* xn [[buffer(0)]],
                             device const float* g [[buffer(1)]],
                             device float* mixed [[buffer(2)]],
                             device ushort* mixed16 [[buffer(3)]],
                             device ushort* mixed_h [[buffer(4)]],
                             constant long& Tn [[buffer(5)]],
                             constant int& flags [[buffer(6)]],
                             uint i [[thread_position_in_grid]]) {
    const int N = 2560, HC = 4, D = 10240;
    if (long(i) >= Tn * N) return;
    int t = int(i) / N, d = int(i) % N;
    float s = 0.0f;
    for (int c = 0; c < HC; ++c) {
        ulong j = ulong(t) * D + ulong(c) * N + d;
        s = fma(xn[j], gdn_sigmoid(g[j]), s);
    }
    s /= float(HC);
    mixed[i] = s;
    if (flags & 1) mixed16[i] = dq_f2bf(s);
    if (flags & 2) mixed_h[i] = dq_f2h(s);
}

kernel void pf_gr_write_kernel(device float* R [[buffer(0)]],
                               device const float* bo [[buffer(1)]],
                               device const float* inj [[buffer(2)]],
                               constant long& inj_ld [[buffer(3)]],
                               constant long& Tn [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
    const int N = 2560, HC = 4, D = 10240;
    if (long(i) >= Tn * D) return;
    int t = int(i) / D, c = (int(i) % D) / N, d = int(i) % N;
    R[i] = fma(bo[ulong(t) * N + d], 2.0f * gdn_sigmoid(inj[ulong(t) * inj_ld + c] / float(HC)), R[i]);
}

kernel void pf_gr_broadcast_kernel(device const float* e [[buffer(0)]],
                                   device float* R [[buffer(1)]],
                                   constant long& Tn [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    const int N = 2560, D = 10240;
    if (long(i) >= Tn * D) return;
    int t = int(i) / D, d = int(i) % N;
    R[i] = e[ulong(t) * N + d];
}

// ---------------------------------------------------------------- GDN
kernel void pf_gdn_gates_kernel(device const float* ab [[buffer(0)]],
                                device const float* dt [[buffer(1)]],
                                device const float* ssm_a [[buffer(2)]],
                                device float* gate [[buffer(3)]],
                                device float* beta [[buffer(4)]],
                                constant long& Tn [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
    const int HV = 48;
    if (long(i) >= Tn * HV) return;
    int t = int(i) / HV, h = int(i) % HV;
    float v = ab[ulong(t) * 2 * HV + h] + dt[h];
    gate[i] = gdn_softplus_log1p(v) * ssm_a[h];
    beta[i] = gdn_sigmoid(ab[ulong(t) * 2 * HV + HV + h]);
}

// The tiled causal conv (gdn_conv_tiled_kernel): thread (c, tile); the tile
// reads its 3 predecessors from the chunk (or the history before it); the
// same expression per element as the serial walk.
kernel void pf_gdn_conv_tiled_kernel(device const float* hist [[buffer(0)]],
                                     device const float* qkv [[buffer(1)]],
                                     device const float* w [[buffer(2)]],
                                     device float* h [[buffer(3)]],
                                     constant long& Tn [[buffer(4)]],
                                     uint3 pos [[thread_position_in_grid]],
                                     uint3 tgpos3 [[threadgroup_position_in_grid]]) {
    const int C = 10240, TILE = 64;
    const uint c = pos.x;
    const uint2 tgpos = tgpos3.xy;
    if (int(c) >= C) return;
    long t0 = long(tgpos.y) * TILE;
    if (t0 >= Tn) return;
    long t1 = min(t0 + TILE, Tn);
    // input(t) for t < 0 reads hist[c*3 + (t+3)]
    float v0 = (t0 - 3 >= 0) ? qkv[ulong(t0 - 3) * C + c] : hist[c * 3 + int(t0)];
    float v1 = (t0 - 2 >= 0) ? qkv[ulong(t0 - 2) * C + c] : hist[c * 3 + int(t0) + 1];
    float v2 = (t0 - 1 >= 0) ? qkv[ulong(t0 - 1) * C + c] : hist[c * 3 + int(t0) + 2];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (long t = t0; t < t1; ++t) {
        float x = qkv[ulong(t) * C + c];
        float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[ulong(t) * C + c] = s / (1.0f + exp(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}

kernel void pf_gdn_conv_hist_kernel(device float* hist [[buffer(0)]],
                                    device const float* qkv [[buffer(1)]],
                                    constant long& Tn [[buffer(2)]],
                                    uint c [[thread_position_in_grid]]) {
    const int C = 10240;
    if (int(c) >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        long t = Tn - 3 + k;
        v[k] = t >= 0 ? qkv[ulong(t) * C + c] : hist[c * 3 + int(t) + 3];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}

// The serial conv walk (STRATA_GDN_CONV_SERIAL arm and T <= 64 in the CUDA
// TU): one thread per channel walks the whole chunk, then writes history.
kernel void pf_gdn_conv_serial_kernel(device float* hist [[buffer(0)]],
                                      device const float* qkv [[buffer(1)]],
                                      device const float* w [[buffer(2)]],
                                      device float* h [[buffer(3)]],
                                      constant long& Tn [[buffer(4)]],
                                      uint c [[thread_position_in_grid]]) {
    const int C = 10240;
    if (int(c) >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (long t = 0; t < Tn; ++t) {
        float x = qkv[ulong(t) * C + c];
        float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[ulong(t) * C + c] = s / (1.0f + exp(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}

kernel void pf_gdn_l2_kernel(device float* h [[buffer(0)]],
                             constant float& eps [[buffer(1)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint2 grp [[threadgroup_position_in_grid]]) {
    threadgroup float part[4];
    const int S = 128, C = 10240;
    // grp.x = head (0..2*HK-1), grp.y = t
    device float* x = h + ulong(grp.y) * C + ulong(grp.x) * S;
    float v = x[tid];
    float ss = pf_warp_parts_sum(v * v, part, tid);
    x[tid] = v * rsqrt(ss + eps);
}

// The recurrence, column-split form (gdn_rec_cols_kernel): threadgroup
// (32 cols x 4 row-groups) per (head, column block); oc_out gets the
// pre-norm output, pf_gdn_out_norm_kernel finishes it.
kernel void pf_gdn_rec_cols_kernel(device float* state [[buffer(0)]],
                                   device const float* h [[buffer(1)]],
                                   device const float* gate [[buffer(2)]],
                                   device const float* beta [[buffer(3)]],
                                   device float* oc_out [[buffer(4)]],
                                   constant long& Tn [[buffer(5)]],
                                   uint3 tp3 [[thread_position_in_threadgroup]],
                                   uint3 blk3 [[threadgroup_position_in_grid]]) {
    const uint2 tp = tp3.xy;
    const uint blk = blk3.x;
    threadgroup float sk[128], sq[128], red[4 * 32];
    const int S = 128, HK = 16, HV = 48, C = 10240, CB = 32, NCB = 4, RG = 4, RPG = 32;
    int head = int(blk) / NCB, cb = int(blk) % NCB;
    int c = int(tp.x), rg = int(tp.y), tid = rg * CB + c, col = cb * CB + c;
    int qh = head % HK;
    float s[RPG];
    // state rows [rg*RPG, rg*RPG + RPG) of this head at column `col`
    ulong base = (ulong(rg * RPG) * HV + head) * S + col;
    const ulong rs = ulong(HV) * S;
    for (int r = 0; r < RPG; ++r) s[r] = state[base + ulong(r) * rs];
    for (long t = 0; t < Tn; ++t) {
        device const float* ht = h + ulong(t) * C;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float g = exp(gate[ulong(t) * HV + head]);
        float kv = 0.0f;
        for (int r = 0; r < RPG; ++r) kv = fma(s[r], sk[rg * RPG + r], kv);
        red[rg * CB + c] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kv_col = red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c];
        float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[ulong(t) * HV + head];
        float o = 0.0f;
        for (int r = 0; r < RPG; ++r) {
            s[r] = fma(g, s[r], sk[rg * RPG + r] * delta);
            o = fma(s[r], sq[rg * RPG + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg * CB + c] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0)
            oc_out[ulong(t) * HV * S + ulong(head) * S + col] =
                (red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c]) * rsqrt(float(S));
    }
    for (int r = 0; r < RPG; ++r) state[base + ulong(r) * rs] = s[r];
}

// The serial recurrence (gdn_rec_kernel, the STRATA_GDN_REC_HEADS arm):
// threadgroup (128 cols x 4 row-groups) per value head, output norm inline.
// Same per-column arithmetic and reduction order as the CUDA TU.
kernel void pf_gdn_rec_kernel(device float* state [[buffer(0)]],
                              device const float* h [[buffer(1)]],
                              device const float* gate [[buffer(2)]],
                              device const float* beta [[buffer(3)]],
                              device const float* z [[buffer(4)]],
                              device const float* gamma [[buffer(5)]],
                              constant float& eps [[buffer(6)]],
                              device float* y [[buffer(7)]],
                              device ushort* y16 [[buffer(8)]],
                              constant long& Tn [[buffer(9)]],
                              uint3 tp3 [[thread_position_in_threadgroup]],
                              uint3 head3 [[threadgroup_position_in_grid]]) {
    const uint2 tp = tp3.xy;
    const uint head = head3.x;
    threadgroup float sk[128], sq[128], red[4 * 128], wsum[16];
    const int S = 128, HK = 16, HV = 48, C = 10240, RG = 4, RPG = 32;
    int col = int(tp.x), rg = int(tp.y), tid = rg * S + col;
    int qh = int(head) % HK;
    float s[RPG];
    ulong base = (ulong(rg * RPG) * HV + head) * S + col;
    const ulong rs = ulong(HV) * S;
    for (int r = 0; r < RPG; ++r) s[r] = state[base + ulong(r) * rs];
    float g_col = gamma[col];
    for (long t = 0; t < Tn; ++t) {
        device const float* ht = h + ulong(t) * C;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float g = exp(gate[ulong(t) * HV + head]);
        float kv = 0.0f;
        for (int r = 0; r < RPG; ++r) kv = fma(s[r], sk[rg * RPG + r], kv);
        red[rg * S + col] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
        float delta = (ht[2 * HK * S + int(head) * S + col] - g * kv_col) * beta[ulong(t) * HV + head];
        float o = 0.0f;
        for (int r = 0; r < RPG; ++r) {
            s[r] = fma(g, s[r], sk[rg * RPG + r] * delta);
            o = fma(s[r], sq[rg * RPG + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg * S + col] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * rsqrt(float(S));
            sp = oc * oc;
        }
        sp = moe_warp_sum_xor(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0) {
            float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            float v = oc * rsqrt(ss / float(S) + eps) * g_col *
                      gdn_sigmoid(z[ulong(t) * HV * S + ulong(head) * S + col]);
            y[ulong(t) * HV * S + ulong(head) * S + col] = v;
            y16[ulong(t) * HV * S + ulong(head) * S + col] = dq_f2h(v);
        }
    }
    for (int r = 0; r < RPG; ++r) state[base + ulong(r) * rs] = s[r];
}

kernel void pf_gdn_out_norm_kernel(device const float* z [[buffer(0)]],
                                   device const float* gamma [[buffer(1)]],
                                   constant float& eps [[buffer(2)]],
                                   device float* y [[buffer(3)]],
                                   device ushort* y16 [[buffer(4)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint2 grp [[threadgroup_position_in_grid]]) {
    threadgroup float part[4];
    const int S = 128, HV = 48;
    // grp.x = t, grp.y = head; tid = col
    ulong at = ulong(grp.x) * HV * S + ulong(grp.y) * S + tid;
    float oc = y[at];
    float ss = pf_warp_parts_sum(oc * oc, part, tid);
    float v = oc * rsqrt(ss / float(S) + eps) * gamma[tid] * gdn_sigmoid(z[at]);
    y[at] = v;
    y16[at] = dq_f2h(v);
}

// ---------------------------------------------------------------- MoE
// route_kernel of the CUDA TU: one simdgroup per token, REG = n_expert / 32
// values per lane (16 for 512 experts, 8 for 256).
kernel void pf_route_kernel(device const float* logits [[buffer(0)]],
                            device int* ids [[buffer(1)]],
                            device float* wout [[buffer(2)]],
                            constant long& Tn [[buffer(3)]],
                            constant int& reg [[buffer(4)]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint tg [[threadgroup_position_in_grid]]) {
    long t = long(tg) * 8 + sg;
    if (t >= Tn) return;
    device const float* lg = logits + ulong(t) * (reg * 32);
    float v[16];
    for (int i = 0; i < reg; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
    for (int i = 0; i < reg; ++i) mx = max(mx, v[i]);
    mx = moe_warp_max_xor(mx);
    float sum = 0.0f;
    for (int i = 0; i < reg; ++i) { v[i] = exp(v[i] - mx); sum += v[i]; }
    float rcp = 1.0f / moe_warp_sum_xor(sum);
    for (int i = 0; i < reg; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = int(lane);
        for (int i = 1; i < reg; ++i)
            if (v[i] > best) { best = v[i]; ex = int(lane) + i * 32; }
        for (ushort mm = 16; mm > 0; mm >>= 1) {
            float ob = simd_shuffle_xor(best, mm);
            int oi = simd_shuffle_xor(ex, mm);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == int(lane)) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0u) ids[t * 10 + rank] = ex;
        if (int(lane) == rank) selected = best;
    }
    selected_sum = max(moe_warp_sum_xor(selected_sum), 6.103515625e-5f);
    if (lane < 10u) wout[t * 10 + lane] = selected / selected_sum;
}

kernel void pf_blob_dequant_kernel(device const uchar* blob [[buffer(0)]],
                                   device ushort* gu16 [[buffer(1)]],
                                   device ushort* d16 [[buffer(2)]],
                                   constant int& half_flag [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
    const long O_D_CODES = 1280L * 640, O_GU_SC = O_D_CODES + 2560L * 160,
               O_D_SC = O_GU_SC + 1280L * 40 * 2;
    const long n_gu = 1280L * 640, n_d = 2560L * 160;
    if (long(i) < n_gu) {
        long row = long(i) / 640, byte = long(i) % 640;
        uchar cc = blob[row * 640 + byte];
        float d = dq_ldf16(blob + O_GU_SC + (row * 40 + (byte * 4) / 64) * 2);
        device ushort* o = gu16 + row * 2560 + byte * 4;
        for (int k = 0; k < 4; ++k) {
            float v = float(int((cc >> (2 * k)) & 3) - 1) * d;
            o[k] = half_flag ? dq_f2h(v) : dq_f2bf(v);
        }
    } else if (long(i) < n_gu + n_d) {
        long j = long(i) - n_gu, row = j / 160, byte = j % 160;
        uchar cc = blob[O_D_CODES + row * 160 + byte];
        float d = dq_ldf16(blob + O_D_SC + (row * 10 + (byte * 4) / 64) * 2);
        device ushort* o = d16 + row * 640 + byte * 4;
        for (int k = 0; k < 4; ++k) {
            float v = float(int((cc >> (2 * k)) & 3) - 1) * d;
            o[k] = half_flag ? dq_f2h(v) : dq_f2bf(v);
        }
    }
}

kernel void pf_swiglu_il_kernel(device const float* gu [[buffer(0)]],
                                device ushort* h16 [[buffer(1)]],
                                constant long& n [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (long(i) >= n * 640) return;
    long r = long(i) / 640, k = long(i) % 640;
    float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = dq_f2h(g / (1.0f + exp(-g)) * u);
}

kernel void pf_swiglu_split_kernel(device const float* gu [[buffer(0)]],
                                   device ushort* h16 [[buffer(1)]],
                                   constant long& n [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    if (long(i) >= n * 640) return;
    long r = long(i) / 640, k = long(i) % 640;
    float g = gu[r * 1280 + k], u = gu[r * 1280 + 640 + k];
    h16[i] = dq_f2h(g / (1.0f + exp(-g)) * u);
}

kernel void pf_swiglu_pair_kernel(device const float* g [[buffer(0)]],
                                  device const float* u [[buffer(1)]],
                                  device ushort* h16 [[buffer(2)]],
                                  constant long& n [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
    if (long(i) >= n * 640) return;
    float a = g[i];
    h16[i] = dq_f2h(a / (1.0f + exp(-a)) * u[i]);
}

kernel void pf_copy_i32_kernel(device int* dst [[buffer(0)]],
                               device const int* src [[buffer(1)]],
                               constant long& n [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (long(i) >= n) return;
    dst[i] = src[i];
}

kernel void pf_gather_rows16_kernel(device const ushort* x [[buffer(0)]],
                                    device const int* src [[buffer(1)]],
                                    device ushort* dst [[buffer(2)]],
                                    constant long& n [[buffer(3)]],
                                    constant long& width [[buffer(4)]],
                                    uint i [[thread_position_in_grid]]) {
    if (long(i) >= n * width) return;
    long r = long(i) / width, j = long(i) % width;
    dst[i] = x[ulong(src[r]) * width + j];
}

kernel void pf_moe_combine_kernel(device const float* Dm [[buffer(0)]],
                                  device const int* slot [[buffer(1)]],
                                  device const float* w [[buffer(2)]],
                                  device const float* shared [[buffer(3)]],
                                  device const float* sg [[buffer(4)]],
                                  device float* bo [[buffer(5)]],
                                  constant long& Tn [[buffer(6)]],
                                  uint i [[thread_position_in_grid]]) {
    const int N = 2560;
    if (long(i) >= Tn * N) return;
    int t = int(i) / N, d = int(i) % N;
    float s = 0.0f;
    for (int k = 0; k < 10; ++k)
        s = fma(w[t * 10 + k], Dm[ulong(slot[t * 10 + k]) * N + d], s);
    bo[i] = s + shared[i] * gdn_sigmoid(sg[t]);
}

// ---------------------------------------------------------------- QSA helpers
kernel void pf_rms_rows_kernel(device float* x [[buffer(0)]],
                               device const float* w [[buffer(1)]],
                               constant long& cols [[buffer(2)]],
                               constant long& ld [[buffer(3)]],
                               constant float& eps [[buffer(4)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tcount [[threads_per_threadgroup]]) {
    threadgroup float sh[32];
    device float* r = x + ulong(row) * ld;
    float ss = 0.0f;
    for (long c = tid; c < cols; c += tcount) ss += r[c] * r[c];
    float s = rsqrt(pf_block_sum(ss, sh, tid, tcount) / float(cols) + eps);
    for (long c = tid; c < cols; c += tcount) r[c] = s * r[c] * w[c];
}

// NEOX rotary on the first 64 dims of each head (rope_kernel): pair p
// rotates elements p and p + 32 by pos * theta_scale^p, the position taken
// from the mrope table (cell-indexed) when one is bound.
kernel void pf_rope_kernel(device float* x [[buffer(0)]],
                           constant long& heads [[buffer(1)]],
                           constant long& dim [[buffer(2)]],
                           constant long& ld [[buffer(3)]],
                           constant long& pos0 [[buffer(4)]],
                           constant float& theta_scale [[buffer(5)]],
                           device const int* mtab [[buffer(6)]],
                           constant int& has_mtab [[buffer(7)]],
                           uint pair [[thread_index_in_threadgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    long t = long(row) / heads, h = long(row) % heads;
    device float* p = x + ulong(t) * ld + ulong(h) * dim;
    int pos = has_mtab ? mtab[(pos0 + t) * 3 + int(pair % 3u)] : int(pos0 + t);
    float theta = float(pos) * pow(theta_scale, float(pair));
    float co = cos(theta), si = sin(theta);
    float a = p[pair], b = p[pair + 32];
    p[pair] = a * co - b * si;
    p[pair + 32] = a * si + b * co;
}

kernel void pf_split_q_kernel(device const float* qf [[buffer(0)]],
                              device float* q [[buffer(1)]],
                              constant long& Tn [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (long(i) >= Tn * 24 * 256) return;
    long t = long(i) / (24 * 256), h = (long(i) / 256) % 24, d = long(i) % 256;
    q[i] = qf[ulong(t) * 24 * 512 + ulong(h) * 512 + d];
}

kernel void pf_gate_attn_kernel(device const float* a [[buffer(0)]],
                                device const float* qf [[buffer(1)]],
                                device ushort* o16 [[buffer(2)]],
                                constant long& Tn [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (long(i) >= Tn * 24 * 256) return;
    long t = long(i) / (24 * 256), h = (long(i) / 256) % 24, d = long(i) % 256;
    o16[i] = dq_f2h(a[i] * (1.0f / (1.0f + exp(-qf[ulong(t) * 24 * 512 + ulong(h) * 512 + 256 + d]))));
}

// kv_append_kernel: threadgroup per (token, kv head, 64-value group,
// is_v); flags gate the three destinations (device pool, host mirror,
// staging pool) and the fp16-vs-int8 mode. Host/stage buffers bound to a
// dummy by the launcher when the corresponding KvHostPools pointer is null.
kernel void pf_kv_append_kernel(device const float* K [[buffer(0)]],
                                device const float* V [[buffer(1)]],
                                device const int* table [[buffer(2)]],
                                constant long& pos0 [[buffer(3)]],
                                constant long& page_size [[buffer(4)]],
                                device ushort* k_pool [[buffer(5)]],
                                device ushort* v_pool [[buffer(6)]],
                                device char* k_q [[buffer(7)]],
                                device char* v_q [[buffer(8)]],
                                device ushort* k_scale [[buffer(9)]],
                                device ushort* v_scale [[buffer(10)]],
                                device ushort* h_k_pool [[buffer(11)]],
                                device ushort* h_v_pool [[buffer(12)]],
                                device char* h_k_q [[buffer(13)]],
                                device char* h_v_q [[buffer(14)]],
                                device ushort* h_k_scale [[buffer(15)]],
                                device ushort* h_v_scale [[buffer(16)]],
                                device ushort* s_k_pool [[buffer(17)]],
                                device ushort* s_v_pool [[buffer(18)]],
                                device char* s_k_q [[buffer(19)]],
                                device char* s_v_q [[buffer(20)]],
                                device ushort* s_k_scale [[buffer(21)]],
                                device ushort* s_v_scale [[buffer(22)]],
                                constant int& flags [[buffer(23)]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint3 grp [[threadgroup_position_in_grid]]) {
    // flags: 1 f16 pool present | 2 host f16 | 4 host int8 | 8 stage f16 | 16 stage int8
    long t = long(grp.x);
    int kvh = int(grp.y), g = int(grp.z) >> 1;
    bool is_v = (grp.z & 1u) != 0u;
    int d = g * 64 + int(tid);
    float x = (is_v ? V : K)[ulong(t) * 512 + kvh * 256 + d];
    long pos = pos0 + t;
    long page = table[pos / page_size];
    long row = (page * 2 + kvh) * page_size + pos % page_size;
    long row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    if (flags & 1) {
        ushort hh = dq_f2h(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = hh;
        if (flags & 2) (is_v ? h_v_pool : h_k_pool)[row_id * 256 + d] = hh;
        if (flags & 8) (is_v ? s_v_pool : s_k_pool)[row_id * 256 + d] = hh;
        return;
    }
    // int8 mode: amax over the 64-value group (two simdgroups of 32).
    threadgroup float wm[2];
    float a = fabs(x);
    a = simd_max(a);
    if ((tid & 31u) == 0u) wm[tid >> 5] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = max(wm[0], wm[1]);
    ushort sb = dq_f2h(amax / 127.0f);
    float sf = dq_h2f(sb);
    int q = 0;
    if (sf > 0.0f) { q = int(rint(x / sf)); q = q < -127 ? -127 : (q > 127 ? 127 : q); }  // rint: __float2int_rn
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = char(q);
        if (tid == 0u) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (flags & 4) {
        (is_v ? h_v_q : h_k_q)[row_id * 256 + d] = char(q);
        if (tid == 0u) (is_v ? h_v_scale : h_k_scale)[row_id * 4 + g] = sb;
    }
    if (flags & 16) {
        (is_v ? s_v_q : s_k_q)[row_id * 256 + d] = char(q);
        if (tid == 0u) (is_v ? s_v_scale : s_k_scale)[row_id * 4 + g] = sb;
    }
}

kernel void pf_to_f16_kernel(device const float* x [[buffer(0)]],
                             device ushort* y [[buffer(1)]],
                             constant long& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (long(i) >= n) return;
    y[i] = dq_f2h(x[i]);
}

kernel void pf_to_bf16_kernel(device const float* x [[buffer(0)]],
                              device ushort* y [[buffer(1)]],
                              constant long& n [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (long(i) >= n) return;
    y[i] = dq_f2bf(x[i]);
}

kernel void pf_round_f16_kernel(device const float* x [[buffer(0)]],
                                device float* y [[buffer(1)]],
                                constant long& n [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (long(i) >= n) return;
    y[i] = dq_h2f(dq_f2h(x[i]));
}
