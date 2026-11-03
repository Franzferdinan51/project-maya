// shared_expert.metal - the shared expert's small elementwise/reduction kernels.
// The three projections themselves go through the dense family's GEMV entries
// (s_gemv / s2_gemv_q8 / native_mmvq host entries), exactly as shared_expert.cu
// composes them; what lives here is SwiGLU, the per-token scalar gate, and the
// output scale. The CUDA legacy SwiGLU and scalar gate accumulate in double;
// MSL has no double, so these use float (the gate's dot is a compensated sum
// over warp partials). The native variants use the same kernels: their CUDA
// forms differ only in using fast exp intrinsics.

// silu(gate) * up, elementwise; out may alias gate.
kernel void shexp_swiglu_k(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                           device float* out [[buffer(2)]], constant int& n [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (int(i) >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

// scalar gate: out[0] = sigmoid(dot(bf16(x), bf16(w))) - one threadgroup.
kernel void shexp_scalar_gate_k(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                                device float* out [[buffer(2)]], constant int& n_embd [[buffer(3)]],
                                ushort tid [[thread_position_in_threadgroup]]) {
    threadgroup float s_part[256];
    float acc = 0.0f;
    for (int i = tid; i < n_embd; i += 256) acc += moe_bf16_to_f32(x[i]) * moe_bf16_to_f32(w[i]);
    s_part[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f, c = 0.0f;
        for (int i = 0; i < 256; ++i) {
            const float y = s_part[i] - c;
            const float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        out[0] = 1.0f / (1.0f + exp(-sum));
    }
}

// in-place sigmoid over the first `n` floats (the native gate's post-step;
// n == 1 single-token, n == n_tok for the verify window).
kernel void shexp_sigmoid_k(device float* gate [[buffer(0)]], constant int& n [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (int(i) >= n) return;
    gate[i] = 1.0f / (1.0f + exp(-gate[i]));
}

// out[i] *= g[0]
kernel void shexp_scale_k(device float* out [[buffer(0)]], device const float* g [[buffer(1)]],
                          constant int& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (int(i) >= n) return;
    out[i] *= g[0];
}

// out[t * n + i] *= g[t]
kernel void shexp_scale_rows_k(device float* out [[buffer(0)]], device const float* g [[buffer(1)]],
                               constant int& n [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
    if (int(pos.x) >= n) return;
    out[ulong(pos.y) * n + pos.x] *= g[pos.y];
}
