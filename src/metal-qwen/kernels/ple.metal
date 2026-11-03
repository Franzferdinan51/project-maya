// ple.metal - the PLE block's stages and the native postops kernels.
// Fixed qwen4exp geometry: N = 2560, H = 4 streams, D = 10240, history 9,
// conv kernel 4 / dilation 3. Transcribed from src/kernels/cuda/ple.cu and
// native_ple_postops.cu. DELTA: ple.cu's gnorm/gate (and its private bf16
// GEMV) accumulate in double; Apple GPUs have no FP64 in Metal, so these
// kernels accumulate in float with the same striding - expect ~1e-7 relative
// differences against the CUDA path in those two stages, documented in the
// dense workstream report. The postops TU's kernels are float in CUDA too
// and are transcribed as-is.

constant uint PLE_N = 2560;
constant uint PLE_H = 4;
constant uint PLE_D = 10240;
constant float PLE_EPS = 1e-6f;

// gate (ple.cu form): one threadgroup of 256 per stream.
kernel void ple_gate_k(device const float* key [[buffer(0)]], device const float* query [[buffer(1)]],
                       device float* gate [[buffer(2)]], constant float& inv_sqrt_n [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device const float* kc = key + tg * PLE_N;
    device const float* qc = query + tg * PLE_N;
    float acc = 0.0f;
    for (uint d = tid; d < PLE_N; d += 256) acc += kc[d] * qc[d];
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) sum += part[i];
        float s = sum * inv_sqrt_n;
        float mag = sqrt(fmax(fabs(s), 1e-6f));
        float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
        gate[tg] = 1.0f / (1.0f + exp(-(sgn * mag)));
    }
}

// gate (postops form): one threadgroup of 512 per row, eight strided partial
// lanes per thread summed in order, products materialized (never a dot FMA).
kernel void ple_gate512_k(device const float* key [[buffer(0)]], device const float* query [[buffer(1)]],
                          device float* gate [[buffer(2)]], constant float& scale [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[16];
    float sums[8];
    for (int j = 0; j < 8; ++j) {
        uint d = tid + j * 512;
        float p = (d < PLE_N) ? key[tg * PLE_N + d] * query[tg * PLE_N + d] : 0.0f;
        sums[j] = p;
    }
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += sums[j];
    sum = simd_sum(sum);
    if (lane == 0) partials[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float p = (tid < 16) ? partials[tid] : 0.0f;
        p = simd_sum(p);
        if (tid == 0) {
            float s = scale * p;
            float mag = sqrt(fmax(fabs(s), 1e-6f));
            float sign = float((s > 0.0f) - (s < 0.0f));
            gate[tg] = 1.0f / (1.0f + exp(-(sign * mag)));
        }
    }
}

kernel void ple_bcast_k(device const float* value [[buffer(0)]], device const float* gate [[buffer(1)]],
                        device float* gated [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= PLE_D) return;
    gated[gid] = value[gid % PLE_N] * gate[gid / PLE_N];
}

kernel void ple_bcast_batch_k(device const float* value [[buffer(0)]], device const float* gate [[buffer(1)]],
                              device float* gated [[buffer(2)]], constant ulong& total [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    ulong t = gid / PLE_D, d = gid % PLE_D;
    gated[gid] = value[t * PLE_N + d % PLE_N] * gate[t * PLE_H + d / PLE_N];
}

// The depthwise conv (ple.cu form): taps read history rows 0, 3, 6 and the new
// normalized row; history is ROW-FASTEST (hist[row + 9 * channel]); kW is
// ggml-native kW[k + 4 * c]. out = silu(sum).
kernel void ple_conv_k(device const float* hist [[buffer(0)]], device const float* norm [[buffer(1)]],
                       device const ushort* kW [[buffer(2)]], device float* out [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= PLE_D) return;
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
        int row = 9 - (3 - k) * 3;
        float v = (row == 9) ? norm[gid] : hist[row + 9 * gid];
        acc += dq_h2f(kW[k + 4 * gid]) * v;
    }
    out[gid] = acc / (1.0f + exp(-acc));
}

// Postops conv + residual: conv = silu(sum), result = hidden + gated + silu.
kernel void ple_conv_res_k(device const float* hist [[buffer(0)]], device const float* norm [[buffer(1)]],
                           device const ushort* kW [[buffer(2)]], device const float* hidden [[buffer(3)]],
                           device const float* gated [[buffer(4)]], device float* conv [[buffer(5)]],
                           device float* result [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= PLE_D) return;
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        float x = (k == 3) ? norm[gid] : hist[gid * 9 + 3 * k];
        float term = x * dq_h2f(kW[gid * 4 + k]);
        sum = (k == 0) ? term : sum + term;
    }
    float act = sum / (1.0f + exp(-sum));
    conv[gid] = act;
    result[gid] = hidden[gid] + (gated[gid] + act);
}

kernel void ple_conv_res_batch_k(device const float* hist [[buffer(0)]], device const float* norm [[buffer(1)]],
                                 device const ushort* kW [[buffer(2)]], device float* hidden [[buffer(3)]],
                                 device const float* gated [[buffer(4)]], constant ulong& total [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    int t = int(gid / PLE_D), c = int(gid % PLE_D);
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        int p = t - 9 + 3 * k;
        float x = (p >= 0) ? norm[ulong(p) * PLE_D + c] : hist[c * 9 + (9 + p)];
        float term = x * dq_h2f(kW[c * 4 + k]);
        sum = (k == 0) ? term : sum + term;
    }
    float act = sum / (1.0f + exp(-sum));
    hidden[gid] = hidden[gid] + (gated[gid] + act);
}

kernel void ple_add3_k(device const float* hidden [[buffer(0)]], device const float* gated [[buffer(1)]],
                       device const float* conv [[buffer(2)]], device float* result [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= PLE_D) return;
    result[gid] = hidden[gid] + gated[gid] + conv[gid];
}

// RMS norm, one threadgroup of 1024 per stream-row (the postops rms_rep
// shape): gamma is stream-blocked, gamma[(row % H) * N + col].
kernel void ple_rms_rep_k(device const float* input [[buffer(0)]], device const float* gamma [[buffer(1)]],
                          device float* output [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[32];
    device const float* in = input + tg * PLE_N;
    device float* outp = output + tg * PLE_N;
    device const float* g = gamma + (tg % PLE_H) * PLE_N;
    float partial = 0.0f;
    for (uint col = tid; col < PLE_N; col += 1024) partial += in[col] * in[col];
    partial = simd_sum(partial);
    if (lane == 0) sums[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    if (tid < 32) {
        tot = sums[lane];
        tot = simd_sum(tot);
    }
    // broadcast through threadgroup memory: lane 0 of warp 0 holds the total
    if (tid == 0) sums[0] = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = rsqrt(sums[0] / float(PLE_N) + PLE_EPS);
    for (uint col = tid; col < PLE_N; col += 1024) outp[col] = scale * in[col] * g[col];
}

// Grouped norm (ple.cu gnorm form): one threadgroup of 256 per stream, gamma
// stream-blocked. Same formula as ple_rms_rep_k with eps passed in.
kernel void ple_gnorm_k(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant float& eps [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device const float* xc = x + tg * PLE_N;
    device const float* wc = w + tg * PLE_N;
    device float* yc = y + tg * PLE_N;
    float acc = 0.0f;
    for (uint d = tid; d < PLE_N; d += 256) acc += xc[d] * xc[d];
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += part[i];
        part[0] = 1.0f / sqrt(tot / float(PLE_N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = part[0];
    for (uint d = tid; d < PLE_N; d += 256) yc[d] = xc[d] * scale * wc[d];
}

// History advance: one thread per channel, row-fastest layout.
kernel void ple_hist_k(device float* hist [[buffer(0)]], device const float* norm [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= PLE_D) return;
    device float* h = hist + gid * 9;
    for (int r = 0; r < 8; ++r) h[r] = h[r + 1];
    h[8] = norm[gid];
}

kernel void ple_hist_batch_k(device float* hist [[buffer(0)]], device const float* norm [[buffer(1)]],
                             constant int& T [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= PLE_D) return;
    float h[9];
    for (int r = 0; r < 9; ++r) {
        int p = T - 9 + r;
        h[r] = (p >= 0) ? norm[ulong(p) * PLE_D + gid] : hist[gid * 9 + (T + r)];
    }
    for (int r = 0; r < 9; ++r) hist[gid * 9 + r] = h[r];
}
