// gdn.metal - Gated DeltaNet kernels (decode, single token).
// Semantics transcribed from src/kernels/cuda/{gdn,fused_gdn,native_gdn,
// native_gdn_preprocess}.cu. State layout (S, h_v, S): state[(i*h_v+h)*S+j].
// Accumulation order: every reduction walks i (or the reduction axis) in
// ascending order in one thread, matching the CUDA column-owner order;
// norm sums use a threadgroup tree (order differs, values are sums of
// squares where the CUDA originals themselves used double/warp trees).

// ---- gdn.hpp ----
kernel void gdn_step_kernel(device float* state [[buffer(0)]],
                            device const float* q [[buffer(1)]],
                            device const float* k [[buffer(2)]],
                            device const float* v [[buffer(3)]],
                            device const float* gate [[buffer(4)]],
                            device const float* beta [[buffer(5)]],
                            device float* o [[buffer(6)]],
                            constant int& S [[buffer(7)]],
                            constant int& h_k [[buffer(8)]],
                            constant int& h_v [[buffer(9)]],
                            uint idx [[thread_position_in_grid]]) {
    uint total = uint(h_v * S);
    if (idx >= total) return;
    int h = int(idx) / S, j = int(idx) % S;
    int hk = h % h_k;
    float dec = exp(gate[h]), b = beta[h];
    float sk = 0.0f;
    for (int i = 0; i < S; ++i) {
        device float* p = state + (ulong(i) * h_v + h) * S + j;
        float s = (*p) * dec; *p = s;
        sk += s * k[hk * S + i];
    }
    float d = (v[h * S + j] - sk) * b;
    float dot = 0.0f;
    for (int i = 0; i < S; ++i) {
        device float* p = state + (ulong(i) * h_v + h) * S + j;
        float s = (*p) + k[hk * S + i] * d; *p = s;
        dot += s * q[hk * S + i];
    }
    o[h * S + j] = dot;
}

kernel void gdn_conv_kernel(device float* cs [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device const float* kW [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant int& C [[buffer(4)]],
                            constant int& dc [[buffer(5)]],
                            uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= C) return;
    int c = int(idx);
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += cs[c * (dc - 1) + i] * kW[c * dc + i];
    acc += x[c] * kW[c * dc + dc - 1];
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) cs[c * (dc - 1) + i] = cs[c * (dc - 1) + i + 1];
    if (dc >= 2) cs[c * (dc - 1) + dc - 2] = x[c];
}

// One threadgroup (128 threads) per row; cols <= 1024 handled by striding.
kernel void gdn_l2_kernel(device float* x [[buffer(0)]],
                          constant int& cols [[buffer(1)]],
                          constant float& eps [[buffer(2)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint row [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    device float* p = x + ulong(row) * cols;
    float ss = 0.0f;
    for (uint i = tid; i < uint(cols); i += 128u) ss += p[i] * p[i];
    float tot = tg_sum(ss, scratch, tid, 128u);
    float inv = 1.0f / sqrt(tot + eps);
    for (uint i = tid; i < uint(cols); i += 128u) p[i] *= inv;
}

kernel void gdn_beta_kernel(device float* beta [[buffer(0)]],
                            constant int& n [[buffer(1)]],
                            uint idx [[thread_position_in_grid]]) {
    if (int(idx) < n) beta[idx] = gdn_sigmoid(beta[idx]);
}

kernel void gdn_out_norm_kernel(device const float* o [[buffer(0)]],
                                device const float* z [[buffer(1)]],
                                device const float* gamma [[buffer(2)]],
                                device float* y [[buffer(3)]],
                                constant int& S [[buffer(4)]],
                                constant float& eps [[buffer(5)]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint h [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    device const float* po = o + ulong(h) * S;
    float ss = 0.0f;
    for (uint i = tid; i < uint(S); i += 128u) ss += po[i] * po[i];
    float tot = tg_sum(ss, scratch, tid, 128u);
    float inv = 1.0f / sqrt(tot / float(S) + eps);
    for (uint i = tid; i < uint(S); i += 128u)
        y[ulong(h) * S + i] = po[i] * inv * gamma[i] * gdn_sigmoid(z[ulong(h) * S + i]);
}

// ---- native_gdn.hpp: same recurrence, readout scaled by 1/sqrt(S) ----
kernel void native_gdn_step_kernel(device float* state [[buffer(0)]],
                                   device const float* q [[buffer(1)]],
                                   device const float* k [[buffer(2)]],
                                   device const float* v [[buffer(3)]],
                                   device const float* gate [[buffer(4)]],
                                   device const float* beta [[buffer(5)]],
                                   device float* out [[buffer(6)]],
                                   constant int& h_k [[buffer(7)]],
                                   constant int& h_v [[buffer(8)]],
                                   constant float& scale [[buffer(9)]],
                                   uint idx [[thread_position_in_grid]]) {
    const int S = 128;
    if (idx >= uint(h_v * S)) return;
    int h = int(idx) / S, j = int(idx) % S, hk = h % h_k;
    float g = exp(gate[h]);
    float kv = 0.0f;
    for (int i = 0; i < S; ++i) kv += state[(ulong(i) * h_v + h) * S + j] * k[hk * S + i];
    float delta = (v[h * S + j] - g * kv) * beta[h];
    float att = 0.0f;
    for (int i = 0; i < S; ++i) {
        device float* p = state + (ulong(i) * h_v + h) * S + j;
        float s = g * (*p) + k[hk * S + i] * delta; *p = s;
        att += s * q[hk * S + i];
    }
    out[h * S + j] = att * scale;
}

// ---- native_gdn_preprocess.hpp ----
kernel void native_gdn_conv_silu_kernel(device float* hist [[buffer(0)]],
                                        device const float* input [[buffer(1)]],
                                        device const float* w [[buffer(2)]],
                                        device float* raw [[buffer(3)]],
                                        device float* silu_out [[buffer(4)]],
                                        constant int& channels [[buffer(5)]],
                                        uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= channels) return;
    int c = int(idx);
    float vals[4] = {hist[c * 3], hist[c * 3 + 1], hist[c * 3 + 2], input[c]};
    float sum = 0.0f;
    for (int t = 0; t < 4; ++t) sum += vals[t] * w[c * 4 + t];
    sum = sum + 0.0f;
    raw[c] = sum;
    silu_out[c] = gdn_silu(sum);
    for (int t = 0; t < 3; ++t) hist[c * 3 + t] = vals[t + 1];
}

// [rows,128]: scale = rsqrt(sum/128 + eps/128); y = (scale*x) * (1/sqrt(128))
kernel void native_gdn_l2_kernel(device float* x [[buffer(0)]],
                                 constant float& eps_div [[buffer(1)]],
                                 constant float& scale_after [[buffer(2)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint row [[threadgroup_position_in_grid]]) {
    const int S = 128;
    threadgroup float scratch[32];
    device float* p = x + ulong(row) * S;
    float val = (tid < 128u) ? p[tid] : 0.0f;
    float tot = tg_sum(val * val, scratch, tid, 256u);
    float scale = rsqrt(tot / float(S) + eps_div);
    if (tid < 128u) p[tid] = (scale * val) * scale_after;
}

kernel void native_gdn_gate_kernel(device const float* alpha [[buffer(0)]],
                                   device const float* dt [[buffer(1)]],
                                   device const float* ssm_a [[buffer(2)]],
                                   device float* gate [[buffer(3)]],
                                   constant int& n [[buffer(4)]],
                                   uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= n) return;
    float v = alpha[idx] + dt[idx];
    float sp = v > 20.0f ? v : log(1.0f + exp(v));
    gate[idx] = sp * ssm_a[idx];
}

kernel void native_gdn_out_norm_kernel(device const float* in [[buffer(0)]],
                                       device const float* z [[buffer(1)]],
                                       device const float* gamma [[buffer(2)]],
                                       device float* out [[buffer(3)]],
                                       constant float& eps [[buffer(4)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint row [[threadgroup_position_in_grid]]) {
    const int S = 128;
    threadgroup float scratch[32];
    device const float* p = in + ulong(row) * S;
    float val = (tid < 128u) ? p[tid] : 0.0f;
    float tot = tg_sum(val * val, scratch, tid, 256u);
    float scale = rsqrt(tot / float(S) + eps);
    if (tid < 128u)
        out[ulong(row) * S + tid] = ((scale * val) * gamma[tid]) * gdn_sigmoid(z[ulong(row) * S + tid]);
}

// ---- fused_gdn.hpp ----
// tg per 128-channel head: conv+SiLU, then L2 over the head if head < qk_heads.
kernel void fused_gdn_conv_l2_kernel(device float* hist [[buffer(0)]],
                                     device const float* qkv [[buffer(1)]],
                                     device const float* w [[buffer(2)]],
                                     device float* h [[buffer(3)]],
                                     constant int& qk_heads [[buffer(4)]],
                                     constant float& eps [[buffer(5)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint head [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[32];
    uint c = head * 128u + tid;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2], xv = qkv[c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + xv * w[c * 4 + 3];
    hist[c * 3] = v1; hist[c * 3 + 1] = v2; hist[c * 3 + 2] = xv;
    float y = gdn_silu(sum);
    if (int(head) < qk_heads) {
        float tot = tg_sum(y * y, scratch, tid, 128u);
        y *= rsqrt(tot + eps);
    }
    h[c] = y;
}

// One thread per row (2*h_v rows: alpha then beta). Correctness-first GEMV:
// the row dot walks n in ascending order in one thread.
kernel void fused_gdn_ab_kernel(device const float* x [[buffer(0)]],
                                device const ushort* wa [[buffer(1)]],
                                device const ushort* wb [[buffer(2)]],
                                device const float* dt [[buffer(3)]],
                                device const float* ssm_a [[buffer(4)]],
                                device float* gate [[buffer(5)]],
                                device float* beta [[buffer(6)]],
                                constant int& n [[buffer(7)]],
                                constant int& h_v [[buffer(8)]],
                                uint idx [[thread_position_in_grid]]) {
    if (int(idx) >= 2 * h_v) return;
    bool is_beta = int(idx) >= h_v;
    int r = is_beta ? int(idx) - h_v : int(idx);
    device const ushort* wrow = (is_beta ? wb : wa) + ulong(r) * n;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += bf16_to_f32(wrow[i]) * x[i];
    if (is_beta) beta[r] = gdn_sigmoid(acc);
    else {
        float v = acc + dt[r];
        gate[r] = gdn_softplus_log1p(v) * ssm_a[r];
    }
}

// tg per value head, 128 threads (one per column j), fused step + out norm.
kernel void fused_gdn_step_norm_kernel(device float* state [[buffer(0)]],
                                       device const float* q [[buffer(1)]],
                                       device const float* k [[buffer(2)]],
                                       device const float* v [[buffer(3)]],
                                       device const float* gate [[buffer(4)]],
                                       device const float* beta [[buffer(5)]],
                                       device const float* z [[buffer(6)]],
                                       device const float* gamma [[buffer(7)]],
                                       constant float& eps [[buffer(8)]],
                                       device float* y [[buffer(9)]],
                                       constant int& h_k [[buffer(10)]],
                                       constant int& h_v [[buffer(11)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint head [[threadgroup_position_in_grid]]) {
    const int S = 128;
    threadgroup float scratch[32];
    threadgroup float ocache[128];
    int h = int(head), hk = h % h_k, j = int(tid);
    float g = exp(gate[h]);
    float kv = 0.0f;
    for (int i = 0; i < S; ++i) kv += state[(ulong(i) * h_v + h) * S + j] * k[hk * S + i];
    float delta = (v[h * S + j] - g * kv) * beta[h];
    float o = 0.0f;
    for (int i = 0; i < S; ++i) {
        device float* p = state + (ulong(i) * h_v + h) * S + j;
        float s = g * (*p) + k[hk * S + i] * delta; *p = s;
        o += s * q[hk * S + i];
    }
    o *= rsqrt(float(S));
    ocache[j] = o;
    float tot = tg_sum(o * o, scratch, tid, 128u);
    float scale = rsqrt(tot / float(S) + eps);
    y[h * S + j] = ocache[j] * scale * gamma[j] * gdn_sigmoid(z[h * S + j]);
}
