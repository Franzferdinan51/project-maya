// native_flash_attn.metal - short-context flash attention adapter.
// Semantics: src/kernels/cuda/native_flash_attn.cu. Fixed geometry:
// 24 query heads x 256, 2 KV heads x 256, scale 1/16, gathered
// [capacity][2][256] fp16 K/V, optional per-cell fp16 additive mask.
//
// One threadgroup per query head, one thread per dimension. Each thread
// recomputes the row's scores (width <= 256) and runs a sequential
// online softmax for its own dimension - the CUDA kernel tiles the same
// math across warps with a +3*ln2-biased running max whose bias cancels
// in the final division; the sequential order here is the reference
// order the parity harness checks. An invalid step writes the status
// and NaN outputs without reading q/k/v/mask values, as on CUDA.
kernel void strata_native_flash_attn_short_step(device const float* q [[buffer(0)]],
                                                device const ushort* k [[buffer(1)]],
                                                device const ushort* v [[buffer(2)]],
                                                device const int* step [[buffer(3)]],
                                                device const ushort* mask [[buffer(4)]],
                                                device float* out [[buffer(5)]],
                                                device int* status [[buffer(6)]],
                                                constant int& has_mask [[buffer(7)]],
                                                constant int& max_context [[buffer(8)]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint head [[threadgroup_position_in_grid]]) {
    const int width = step[3];  // kStepWidth
    const int nkv = step[1];    // kStepNKv
    const bool valid = width >= 1 && width <= max_context && nkv == width && step[0] == width - 1 &&
                       step[2] == width / 4;
    if (head == 0u && tid == 0u) *status = valid ? 0 : 1;  // Success / UnsupportedStep
    const int d = (int)tid;
    if (!valid) {
        out[head * 256 + d] = as_type<float>(0x7fc00000u);
        return;
    }
    const int kv = (int)head / 12;
    const float scale = 0.0625f;
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int cell = 0; cell < width; ++cell) {
        float dot = 0.0f;
        for (int i = 0; i < 256; ++i)
            dot += (q[head * 256 + i] * scale) * strata_f32_from_f16(k[(cell * 2 + kv) * 256 + i]);
        if (has_mask) dot += strata_f32_from_f16(mask[cell]);
        const float m_new = fmax(m, dot);
        // -inf scores (masked cells) contribute exactly zero, including
        // before the first live cell - the guards keep exp() away from
        // (-inf) - (-inf).
        const float rescale = m == -INFINITY ? 0.0f : exp(m - m_new);
        const float p = dot == -INFINITY ? 0.0f : exp(dot - m_new);
        acc = acc * rescale + p * strata_f32_from_f16(v[(cell * 2 + kv) * 256 + d]);
        l = l * rescale + p;
        m = m_new;
    }
    out[head * 256 + d] = acc / l;
}
