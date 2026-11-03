// native_qsa.metal - weighted RMS norm and gate kernels.
// Semantics: src/kernels/cuda/native_qsa.cu.

// One threadgroup per row. Each thread sums the squares of its strided
// columns in float; simdgroup reductions then a threadgroup reduction give
// the row sum (a tree order, like the CUDA warp-shuffle order - neither is
// the sequential order, and the parity harness compares against a double
// reference). out = rsqrt(mean + eps) * x * gamma, in place safe: every
// input read for the reduction precedes the barrier before any write.
kernel void strata_native_qsa_rms_norm_weighted_eps(device const float* input [[buffer(0)]],
                                                    device const float* gamma [[buffer(1)]],
                                                    device float* output [[buffer(2)]],
                                                    constant int& n_cols [[buffer(3)]],
                                                    constant float& epsilon [[buffer(4)]],
                                                    uint tid [[thread_index_in_threadgroup]],
                                                    uint tcount [[threads_per_threadgroup]],
                                                    uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partials[32];
    threadgroup float scale_bcast;
    const size_t base = (size_t)row * (size_t)n_cols;
    float partial = 0.0f;
    for (int col = (int)tid; col < n_cols; col += (int)tcount) {
        const float v = input[base + col];
        partial += v * v;
    }
    partial = simd_sum(partial);
    const uint lane = tid & 31u, sg = tid >> 5;
    if (lane == 0u) partials[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < (tcount >> 5); ++i) total += partials[i];
        scale_bcast = rsqrt(total / (float)n_cols + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = scale_bcast;
    for (int col = (int)tid; col < n_cols; col += (int)tcount)
        output[base + col] = scale * input[base + col] * gamma[col];
}

// out[i] = attn[i] * sigmoid(q_full[head * 2 * head_dim + head_dim + chan])
kernel void strata_native_qsa_gate_apply(device const float* attn [[buffer(0)]],
                                         device const float* q_full [[buffer(1)]],
                                         device float* output [[buffer(2)]],
                                         constant int& n_head [[buffer(3)]],
                                         constant int& head_dim [[buffer(4)]],
                                         uint tid [[thread_position_in_grid]]) {
    if ((long)tid >= (long)n_head * head_dim) return;
    const size_t head = tid / (uint)head_dim, channel = tid % (uint)head_dim;
    const float raw = q_full[head * 2 * (size_t)head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + exp(-raw));
    output[tid] = attn[tid] * sigmoid;
}
