// gr.metal - gated residual / hyper-connection kernels + native GR norm/postops.
// Semantics transcribed from src/kernels/cuda/{gr,fused_gr,native_gr_norm,
// native_gr_postops}.cu. BF16 weights are ushort buffers; bf16_to_f32 in the
// prelude. GEMV reductions: one threadgroup per row, lanes stride the axis,
// threadgroup tree sum (CUDA used warp trees; both are fp32 trees).

// ---- native_gr_norm: weighted RMS norm, tg per row ----
kernel void native_gr_rms_norm_kernel(device const float* in [[buffer(0)]],
                                      device const float* gamma [[buffer(1)]],
                                      device float* out [[buffer(2)]],
                                      constant int& n_cols [[buffer(3)]],
                                      constant float& eps [[buffer(4)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint tcount [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* p = in + ulong(row) * n_cols;
    float ss = 0.0f;
    for (uint i = tid; i < uint(n_cols); i += tcount) ss += p[i] * p[i];
    float tot = tg_sum(ss, scratch, tid, tcount);
    float scale = rsqrt(tot / float(n_cols) + eps);
    for (uint i = tid; i < uint(n_cols); i += tcount)
        out[ulong(row) * n_cols + i] = scale * p[i] * gamma[ulong(row) * n_cols + i];
}

// ---- native_gr_postops ----
kernel void native_gr_down_silu_kernel(device float* lo [[buffer(0)]],
                                       constant int& n [[buffer(1)]],
                                       constant float& scale [[buffer(2)]],
                                       uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= n) return;
    float x = scale * lo[idx] + 0.0f; // SCALE with +0 bias, as pinned
    lo[idx] = gdn_silu(x);
}

// one thread per d; fused flag selects FMA-chain vs ordered-add accumulation
kernel void native_gr_pre_gated_kernel(device const float* xn [[buffer(0)]],
                                       device float* gate [[buffer(1)]],
                                       device float* mixed [[buffer(2)]],
                                       constant int& n_embd [[buffer(3)]],
                                       constant int& hc [[buffer(4)]],
                                       constant float& scale [[buffer(5)]],
                                       constant int& fused [[buffer(6)]],
                                       uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= n_embd) return;
    // Pinned semantics: gate[i] is overwritten with xn*sigmoid(gate); the sum
    // uses the ORIGINAL gate value's sigmoid (w below), FMA-chained when fused,
    // ordered adds starting from the first product when not.
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        ulong i = ulong(c) * n_embd + idx;
        float w = gdn_sigmoid(gate[i]);
        float prod = xn[i] * w;
        gate[i] = prod;
        if (fused) sum = fma(xn[i], w, sum);
        else sum = (c == 0) ? prod : (sum + prod);
    }
    mixed[idx] = scale * sum;
}

kernel void native_gr_post_kernel(device const float* residual [[buffer(0)]],
                                  device const float* block_out [[buffer(1)]],
                                  device const float* inject [[buffer(2)]],
                                  device float* out [[buffer(3)]],
                                  constant int& n_embd [[buffer(4)]],
                                  constant int& hc [[buffer(5)]],
                                  constant float& scale [[buffer(6)]],
                                  uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(n_embd * hc)) return;
    int c = int(idx) / n_embd, d = int(idx) % n_embd;
    float w = 2.0f * gdn_sigmoid(inject[c] * scale);
    out[idx] = fma(block_out[d], w, residual[idx]);
}

// ---- gr.hpp helpers ----
// Per-stream RMS norm: tg per stream; writes xn (f32) and xq (bf16 of xn).
kernel void gr_norm_kernel(device const float* R [[buffer(0)]],
                           device const float* w_norm [[buffer(1)]],
                           device float* xn [[buffer(2)]],
                           device ushort* xq [[buffer(3)]],
                           constant int& n_embd [[buffer(4)]],
                           constant float& eps [[buffer(5)]],
                           constant int& fp32_act [[buffer(6)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint c [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    device const float* p = R + ulong(c) * n_embd;
    float ss = 0.0f;
    for (uint i = tid; i < uint(n_embd); i += 256u) ss += p[i] * p[i];
    float tot = tg_sum(ss, scratch, tid, 256u);
    float rs = rsqrt(tot / float(n_embd) + eps);
    for (uint i = tid; i < uint(n_embd); i += 256u) {
        float xv = p[i] * rs * w_norm[ulong(c) * n_embd + i];
        xn[ulong(c) * n_embd + i] = xv;
        if (!fp32_act) xq[ulong(c) * n_embd + i] = f32_to_bf16(xv);
    }
}

// Generic BF16-weight GEMV, tg per output row.
// act_mode: 0 = float act buffer, 1 = ushort (bf16) act buffer.
// round_act: when act is float, round each act value through bf16 first.
kernel void gr_gemv_kernel(device const float* act_f [[buffer(0)]],
                           device const ushort* act_b [[buffer(1)]],
                           device const ushort* w [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant int& n_in [[buffer(4)]],
                           constant int& act_mode [[buffer(5)]],
                           constant int& round_act [[buffer(6)]],
                           constant int& epilogue [[buffer(7)]], // 0 none, 1 silu(x/hc)
                           constant int& hc [[buffer(8)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    device const ushort* wrow = w + ulong(row) * n_in;
    float acc = 0.0f;
    for (uint i = tid; i < uint(n_in); i += 256u) {
        float a = (act_mode == 0) ? act_f[i] : bf16_to_f32(act_b[i]);
        if (round_act) a = bf16_round(a);
        acc += a * bf16_to_f32(wrow[i]);
    }
    float tot = tg_sum(acc, scratch, tid, 256u);
    if (tid == 0u) {
        if (epilogue == 1) { float x = tot / float(hc); out[row] = gdn_silu(x); }
        else out[row] = tot;
    }
}

// down projection with bf16 output (default gr_read contract): out is lq.
kernel void gr_down_bf16_kernel(device const ushort* xq [[buffer(0)]],
                                device const ushort* w [[buffer(1)]],
                                device ushort* lq [[buffer(2)]],
                                constant int& n_in [[buffer(3)]],
                                constant int& hc [[buffer(4)]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint row [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    device const ushort* wrow = w + ulong(row) * n_in;
    float acc = 0.0f;
    for (uint i = tid; i < uint(n_in); i += 256u) acc += bf16_to_f32(xq[i]) * bf16_to_f32(wrow[i]);
    float tot = tg_sum(acc, scratch, tid, 256u);
    if (tid == 0u) { float x = tot / float(hc); lq[row] = f32_to_bf16(gdn_silu(x)); }
}

// gate: gated[i] = xn[i] * sigmoid(dot(lo_act, w_up[i])); tg per row i.
kernel void gr_gate_kernel(device const float* act_f [[buffer(0)]],
                           device const ushort* act_b [[buffer(1)]],
                           device const ushort* w [[buffer(2)]],
                           device const float* xn [[buffer(3)]],
                           device float* gated [[buffer(4)]],
                           constant int& hc_lr [[buffer(5)]],
                           constant int& act_mode [[buffer(6)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    device const ushort* wrow = w + ulong(row) * hc_lr;
    float acc = 0.0f;
    for (uint i = tid; i < uint(hc_lr); i += 256u) {
        float a = (act_mode == 0) ? act_f[i] : bf16_to_f32(act_b[i]);
        acc += a * bf16_to_f32(wrow[i]);
    }
    float tot = tg_sum(acc, scratch, tid, 256u);
    if (tid == 0u) gated[row] = xn[row] * gdn_sigmoid(tot);
}

kernel void gr_mean_kernel(device const float* gated [[buffer(0)]],
                           device float* mixed [[buffer(1)]],
                           constant int& n_embd [[buffer(2)]],
                           constant int& hc [[buffer(3)]],
                           uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= n_embd) return;
    float m = 0.0f;
    for (int c = 0; c < hc; ++c) m += gated[ulong(c) * n_embd + idx];
    mixed[idx] = m / float(hc);
}

kernel void gr_write_kernel(device const float* R [[buffer(0)]],
                            device const float* block_out [[buffer(1)]],
                            device const float* inject [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant int& n_embd [[buffer(4)]],
                            constant int& hc [[buffer(5)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(n_embd * hc)) return;
    int c = int(idx) / n_embd, d = int(idx) % n_embd;
    float w = 2.0f * gdn_sigmoid(inject[c] / float(hc));
    out[idx] = R[idx] + block_out[d] * w;
}

// ---- fused_gr.hpp ----
// tg per stream: rs[c] = rsqrt(mean(R'[c]^2)+eps), R' folds the pending write.
kernel void fused_gr_norm_kernel(device const float* R [[buffer(0)]],
                                 device const float* bo_prev [[buffer(1)]],
                                 device const float* inj_prev [[buffer(2)]],
                                 device float* rs [[buffer(3)]],
                                 constant int& n_embd [[buffer(4)]],
                                 constant float& eps [[buffer(5)]],
                                 constant int& apply [[buffer(6)]],
                                 constant int& hc [[buffer(7)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint c [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    float gw = apply ? 2.0f * gdn_sigmoid(inj_prev[c] / float(hc)) : 0.0f;
    float ss = 0.0f;
    for (uint i = tid; i < uint(n_embd); i += 256u) {
        float r = R[ulong(c) * n_embd + i];
        if (apply) r = fma(bo_prev[i], gw, r);
        ss += r * r;
    }
    float tot = tg_sum(ss, scratch, tid, 256u);
    if (tid == 0u) rs[c] = rsqrt(tot / float(n_embd) + eps);
}

// GEMV over xn recomputed on the fly: xn[i] = R'[i]*w_norm[i]*rs[i/n_embd].
// epilogue 0: plain (inject), 1: silu(x/hc) (lo).
kernel void fused_gr_gemv_kernel(device const float* R [[buffer(0)]],
                                 device const float* bo_prev [[buffer(1)]],
                                 device const float* inj_prev [[buffer(2)]],
                                 device const float* w_norm [[buffer(3)]],
                                 device const float* rs [[buffer(4)]],
                                 device const ushort* w [[buffer(5)]],
                                 device float* out [[buffer(6)]],
                                 constant int& n_embd [[buffer(7)]],
                                 constant int& hc [[buffer(8)]],
                                 constant int& apply [[buffer(9)]],
                                 constant int& epilogue [[buffer(10)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint row [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    int hc_dim = n_embd * hc;
    device const ushort* wrow = w + ulong(row) * hc_dim;
    float acc = 0.0f;
    for (uint i = tid; i < uint(hc_dim); i += 256u) {
        int c = int(i) / n_embd;
        float r = R[i];
        if (apply) r = fma(bo_prev[int(i) % n_embd], 2.0f * gdn_sigmoid(inj_prev[c] / float(hc)), r);
        float xn = r * w_norm[i] * rs[c];
        acc += xn * bf16_to_f32(wrow[i]);
    }
    float tot = tg_sum(acc, scratch, tid, 256u);
    if (tid == 0u) {
        if (epilogue == 1) { float x = tot / float(hc); out[row] = gdn_silu(x); }
        else out[row] = tot;
    }
}

// One thread per column d: applies the pending write to R_out, gates, means.
kernel void fused_gr_up_kernel(device const float* R [[buffer(0)]],
                               device float* R_out [[buffer(1)]],
                               device const float* bo_prev [[buffer(2)]],
                               device const float* inj_prev [[buffer(3)]],
                               device const float* w_norm [[buffer(4)]],
                               device const float* rs [[buffer(5)]],
                               device const ushort* w_up [[buffer(6)]],
                               device const float* lo [[buffer(7)]],
                               device float* mixed [[buffer(8)]],
                               constant int& n_embd [[buffer(9)]],
                               constant int& hc [[buffer(10)]],
                               constant int& hc_lr [[buffer(11)]],
                               constant int& apply [[buffer(12)]],
                               uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= n_embd) return;
    int d = int(idx);
    float s = 0.0f;
    for (int c = 0; c < hc; ++c) {
        ulong i = ulong(c) * n_embd + d;
        float r = R[i];
        if (apply) {
            r = fma(bo_prev[d], 2.0f * gdn_sigmoid(inj_prev[c] / float(hc)), r);
            R_out[i] = r;
        }
        float xn = r * w_norm[i] * rs[c];
        device const ushort* wrow = w_up + i * hc_lr;
        float acc = 0.0f;
        for (int kk = 0; kk < hc_lr; ++kk) acc += lo[kk] * bf16_to_f32(wrow[kk]);
        s += xn * gdn_sigmoid(acc);
    }
    mixed[d] = s / float(hc);
}
