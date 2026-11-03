// qsa_indexer.metal - native QSA indexer key-append kernels.
// Semantics: src/kernels/cuda/native_qsa_indexer.cu. Fixed geometry:
// idx_dim 128, block 4, rotation 64. Incoming keys are rounded through
// fp16 (the CUDA SET_ROWS/GET_ROWS behaviour). The pooled row is the
// 4-cell mean, RMS-normalised (float; the squared-sum reduction is a
// simdgroup tree, not the CUDA double path), weighted, then NEOX-rotated
// at the block's first cell position. Threadgroups are 256 threads wide;
// only the first 128 carry a dimension.

struct IdxNorm {
    float values[128];
    float partials[8];
    float ss_bcast;
};

// mean (already averaged) -> normalised, weighted row in IdxNorm.values.
static inline void idx_apply_norm(threadgroup IdxNorm& N, float mean, device const float* gamma,
                                  float epsilon, uint tid) {
    float sq = tid < 128u ? mean * mean : 0.0f;
    sq = simd_sum(sq);
    const uint lane = tid & 31u, sg = tid >> 5;
    if (lane == 0u) N.partials[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (int i = 0; i < 8; ++i) total += N.partials[i];
        N.ss_bcast = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(N.ss_bcast / 128.0f + epsilon);
    if (tid < 128u) N.values[tid] = scale * mean * gamma[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// NEOX rotation of the normalised row at rope_pos (ident when zero_pos).
static inline float idx_rope_value(threadgroup IdxNorm& N, int d, int rope_pos, float theta_scale,
                                   device const int* mtab, int has_mtab, bool zero_pos) {
    float y = N.values[d];
    if (d < 64) {
        const int pair = d % 32;
        const float theta =
            (zero_pos ? 0.0f : (float)strata_mrope_pos(mtab, has_mtab, rope_pos, pair)) *
            pow(theta_scale, (float)pair);
        const float c = cos(theta), s = sin(theta);
        const float a = N.values[pair], z = N.values[pair + 32];
        y = d < 32 ? a * c - z * s : a * s + z * c;
    }
    return y;
}

// Single append: pos arrives through a device scalar (graph-safe).
kernel void strata_native_indexer_append(device const float* raw [[buffer(0)]],
                                         device const int* pos_dev [[buffer(1)]],
                                         device const float* gamma [[buffer(2)]],
                                         device float* tail [[buffer(3)]], device float* dead [[buffer(4)]],
                                         device float* pooled [[buffer(5)]],
                                         device int* block_pos [[buffer(6)]],
                                         device const int* mtab [[buffer(7)]],
                                         constant int& has_mtab [[buffer(8)]],
                                         constant int& pos_base [[buffer(9)]],
                                         constant float& epsilon [[buffer(10)]],
                                         constant float& theta_scale [[buffer(11)]],
                                         constant int& max_cells [[buffer(12)]],
                                         uint tid [[thread_index_in_threadgroup]]) {
    const int pos = pos_dev[0];
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % 4;
    const int d = (int)tid;
    float incoming = 0.0f;
    if (d < 128) {
        incoming = strata_f32_from_f16(strata_f16_from_f32(raw[d]));
        if (slot < 3) tail[slot * 128 + d] = incoming;
    }
    if (pos != 0 && slot != 3) return;
    threadgroup IdxNorm N;
    float mean = 0.0f;
    if (d < 128) {
        float sum = pos == 0 ? incoming : tail[d];
        for (int j = 1; j < 4; ++j)
            sum += (pos == 0 || j == 3) ? incoming : tail[j * 128 + d];
        mean = fma(0.25f, sum, 0.0f);  // SCALE includes a zero bias
    }
    idx_apply_norm(N, mean, gamma, epsilon, tid);
    if (d >= 128) return;
    const int b = pos / 4;
    const int rope_pos = pos == 0 ? 0 : pos_base + 4 * b;
    const float y = idx_rope_value(N, d, rope_pos, theta_scale, mtab, has_mtab, pos == 0);
    pooled[(size_t)b * 128 + d] = y;
    if (pos == 0) dead[d] = y;
    else pooled[(size_t)(b + 1) * 128 + d] = dead[d];
    if (d == 0 && pos != 0) block_pos[0] = rope_pos;
}

// Batch, first block: pos 0's row, rotation is the identity.
kernel void strata_native_indexer_append_first(device const float* raw [[buffer(0)]],
                                               device const float* gamma [[buffer(1)]],
                                               device float* dead [[buffer(2)]],
                                               device float* pooled [[buffer(3)]],
                                               constant float& epsilon [[buffer(4)]],
                                               constant float& theta_scale [[buffer(5)]],
                                               uint tid [[thread_index_in_threadgroup]]) {
    const int d = (int)tid;
    threadgroup IdxNorm N;
    float mean = 0.0f;
    if (d < 128) {
        const float incoming = strata_f32_from_f16(strata_f16_from_f32(raw[d]));
        float sum = incoming;
        for (int j = 1; j < 4; ++j) sum += incoming;
        mean = fma(0.25f, sum, 0.0f);
    }
    idx_apply_norm(N, mean, gamma, epsilon, tid);
    if (d >= 128) return;
    const float y = N.values[d];  // theta == 0: rotation is the identity
    (void) theta_scale;
    pooled[d] = y;
    dead[d] = y;
}

// Batch, completed blocks: one threadgroup per block b in
// [first_block, last_block]; cells before p0 read the OLD tail rows.
kernel void strata_native_indexer_append_blocks(device const float* raw [[buffer(0)]],
                                                device const float* gamma [[buffer(1)]],
                                                device const float* tail [[buffer(2)]],
                                                device const float* dead [[buffer(3)]],
                                                device float* pooled [[buffer(4)]],
                                                device int* block_pos [[buffer(5)]],
                                                device const int* mtab [[buffer(6)]],
                                                constant int& has_mtab [[buffer(7)]],
                                                constant long& p0 [[buffer(8)]],
                                                constant int& pos_base [[buffer(9)]],
                                                constant float& epsilon [[buffer(10)]],
                                                constant float& theta_scale [[buffer(11)]],
                                                constant long& first_block [[buffer(12)]],
                                                constant long& last_block [[buffer(13)]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint tg [[threadgroup_position_in_grid]]) {
    const long b = first_block + tg;
    const int d = (int)tid;
    threadgroup IdxNorm N;
    float mean = 0.0f;
    if (d < 128) {
        // key(j): this batch's raw row for cells >= p0, the old tail otherwise
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const long cell = b * 4 + j;
            const float v = cell >= p0
                                ? strata_f32_from_f16(strata_f16_from_f32(raw[(cell - p0) * 128 + d]))
                                : tail[j * 128 + d];
            sum = (j == 0) ? v : sum + v;
        }
        mean = fma(0.25f, sum, 0.0f);
    }
    idx_apply_norm(N, mean, gamma, epsilon, tid);
    if (d >= 128) return;
    const int rope_pos = pos_base + 4 * (int)b;
    const float y = idx_rope_value(N, d, rope_pos, theta_scale, mtab, has_mtab, false);
    pooled[(size_t)b * 128 + d] = y;
    if (b == last_block) {
        pooled[(size_t)(b + 1) * 128 + d] = dead[d];
        if (d == 0) block_pos[0] = rope_pos;
    }
}

// Batch, tail refresh: one threadgroup per slot s in 0..2; stores the last
// cell of the batch whose index is congruent to s mod 4 (if any).
kernel void strata_native_indexer_append_tail(device const float* raw [[buffer(0)]],
                                              device float* tail [[buffer(1)]],
                                              constant long& n [[buffer(2)]], constant long& p0 [[buffer(3)]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint s [[threadgroup_position_in_grid]]) {
    const int d = (int)tid;
    if (d >= 128) return;
    const long last = p0 + n - 1;
    const long cell = last - ((last % 4) - (long)s + 4) % 4;
    if (cell < p0) return;
    tail[s * 128 + d] = strata_f32_from_f16(strata_f16_from_f32(raw[(cell - p0) * 128 + d]));
}
