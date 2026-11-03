// qsa_attn.metal - QSA attention over the paged KV pools.
// Semantics: src/kernels/cuda/qsa_decode_attn.cu - 64-cell chunks, per-chunk
// online-softmax partials (max, sum, unnormalised accumulator) in the
// caller's scratch, then a merge across chunks. qsa_prompt_attn_batch runs
// the same two kernels (the CUDA prompt path is a tensor-core variant of
// the same math with FP16 hi/lo splits; this one stays in FP32 throughout,
// so it differs from the CUDA prompt kernel by rounding, not by result).
//
// KV modes (chosen by which pools are set, exactly as on CUDA):
//   0: fp16 pools        1: int8 codes + fp16 scale per 64
//   2: q4_0 pools        3: hybrid - int8 K, q4_0 V
// Fixed geometry: head_dim 256, 12 query heads per KV head.

static inline float attn_q4_elem(device const uchar* pool, long row, int d) {
    // block_q4_0: fp16 scale then 16 bytes; element j in the low nibble of
    // qs[j], j+16 in the high nibble. 144 bytes per row (8 blocks of 32).
    const long base = row * 144 + (d / 32) * 18;
    const float sc = strata_f32_from_f16((ushort)(pool[base] | ((uint)pool[base + 1] << 8)));
    const int rem = d % 32;
    const uchar byte = pool[base + 2 + (rem < 16 ? rem : rem - 16)];
    const int nib = rem < 16 ? (int)(byte & 0x0fu) - 8 : (int)(byte >> 4) - 8;
    return (float)nib * sc;
}
static inline float attn_k_elem(int mode, device const ushort* k_pool, device const char* k_q,
                                device const ushort* k_scale, device const uchar* k_q4, long row, int d) {
    if (mode == 0) return strata_f32_from_f16(k_pool[row * 256 + d]);
    if (mode == 1 || mode == 3) {
        const float sc = strata_f32_from_f16(k_scale[row * 4 + d / 64]);
        return (float)k_q[row * 256 + d] * sc;
    }
    return attn_q4_elem(k_q4, row, d);
}
static inline float attn_v_elem(int mode, device const ushort* v_pool, device const char* v_q,
                                device const ushort* v_scale, device const uchar* v_q4, long row, int d) {
    if (mode == 0) return strata_f32_from_f16(v_pool[row * 256 + d]);
    if (mode == 1) {
        const float sc = strata_f32_from_f16(v_scale[row * 4 + d / 64]);
        return (float)v_q[row * 256 + d] * sc;
    }
    return attn_q4_elem(v_q4, row, d);  // modes 2 and 3: V is q4_0
}

struct AttnPools {
    device const ushort* k_pool;
    device const ushort* v_pool;
    device const char* k_q;
    device const char* v_q;
    device const ushort* k_scale;
    device const ushort* v_scale;
    device const uchar* k_q4;
    device const uchar* v_q4;
    device const int* page_table;
};

// One threadgroup per (chunk, KV head, query); 256 threads.
kernel void strata_qsa_attn_chunk(device const float* q [[buffer(0)]], device const ushort* k_pool [[buffer(1)]],
                                  device const ushort* v_pool [[buffer(2)]], device const char* k_q [[buffer(3)]],
                                  device const char* v_q [[buffer(4)]], device const ushort* k_scale [[buffer(5)]],
                                  device const ushort* v_scale [[buffer(6)]], device const uchar* k_q4 [[buffer(7)]],
                                  device const uchar* v_q4 [[buffer(8)]], device const int* page_table [[buffer(9)]],
                                  device const int* ids [[buffer(10)]], device const int* steps [[buffer(11)]],
                                  device float* part_acc [[buffer(12)]], device float* part_m [[buffer(13)]],
                                  device float* part_l [[buffer(14)]], constant int& mode [[buffer(15)]],
                                  constant int& kv_heads [[buffer(16)]], constant int& page_size [[buffer(17)]],
                                  constant float& scale [[buffer(18)]], constant int& n_chunks [[buffer(19)]],
                                  constant int& cap [[buffer(20)]], constant long& scratch_stride [[buffer(21)]],
                                  uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    const int chunk = (int)tg.x, kvh = (int)tg.y, z = (int)tg.z;
    const int t = (int)tid, lane = t & 31, warp = t >> 5;
    threadgroup float sq[12 * 256];
    threadgroup float sp[12 * 64];
    threadgroup long srow[64];
    const long base = (long)z * scratch_stride;
    device const int* step = steps + z * 4;
    const int n_ids = step[3];  // kStepWidth
    const int c0 = chunk * 64;
    const int n_here = min(64, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < 12) {
            part_m[base + slot * 12 + t] = -FLT_MAX;
            part_l[base + slot * 12 + t] = 0.0f;
        }
        return;
    }
    for (int i = t; i < 12 * 256; i += 256)
        sq[i] = q[((size_t)z * kv_heads * 12 + (size_t)kvh * 12) * 256 + i];
    if (t < 64) {
        long r = -1;
        if (t < n_here) {
            const int cell = ids[(size_t)z * cap + c0 + t];
            const long page = page_table[cell / page_size];
            r = (page * kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const AttnPools P{k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table};
    // scores: each simdgroup takes cells warp, warp+8, ...; each lane holds
    // 8 of the 256 dimensions, summed in expression order then simd_sum.
    for (int c = warp; c < 64; c += 8) {
        if (c >= n_here) {
            if (lane < 12) sp[lane * 64 + c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        for (int j = 0; j < 8; ++j)
            k8[j] = attn_k_elem(mode, P.k_pool, P.k_q, P.k_scale, P.k_q4, srow[c], lane * 8 + j);
        for (int h = 0; h < 12; ++h) {
            float s = 0.0f;
            for (int j = 0; j < 8; ++j) s += k8[j] * sq[h * 256 + lane * 8 + j];
            s = simd_sum(s);
            if (lane == 0) sp[h * 64 + c] = s * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // per-head chunk max and exp-sum: simdgroup w handles heads w and w+8
    for (int h = warp; h < 12; h += 8) {
        const float a = sp[h * 64 + lane], b = sp[h * 64 + lane + 32];
        const float m = simd_max(fmax(a, b));
        const float ea = lane < n_here ? exp(a - m) : 0.0f;
        const float eb = lane + 32 < n_here ? exp(b - m) : 0.0f;
        sp[h * 64 + lane] = ea;
        sp[h * 64 + lane + 32] = eb;
        const float l = simd_sum(ea + eb);
        if (lane == 0) {
            part_m[base + slot * 12 + h] = m;
            part_l[base + slot * 12 + h] = l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // values: thread t owns dimension t for all 12 heads
    float acc[12];
    for (int h = 0; h < 12; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        const float v = attn_v_elem(mode, P.v_pool, P.v_q, P.v_scale, P.v_q4, srow[c], t);
        for (int h = 0; h < 12; ++h) acc[h] = fma(sp[h * 64 + c], v, acc[h]);
    }
    for (int h = 0; h < 12; ++h) part_acc[base + ((size_t)slot * 12 + h) * 256 + t] = acc[h];
}

// One threadgroup per (query head, query); 256 threads, thread = dimension.
kernel void strata_qsa_attn_merge(device const float* part_acc [[buffer(0)]],
                                  device const float* part_m [[buffer(1)]],
                                  device const float* part_l [[buffer(2)]], device float* attn [[buffer(3)]],
                                  constant int& n_chunks [[buffer(4)]], constant int& n_head [[buffer(5)]],
                                  constant long& scratch_stride [[buffer(6)]],
                                  uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
    const int h = (int)tg.x;       // global query head
    const int z = (int)tg.y;       // query
    const int kvh = h / 12, hl = h % 12;
    const int d = (int)tid;
    const long base = (long)z * scratch_stride;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmax(M, part_m[base + (kvh * n_chunks + c) * 12 + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[base + slot * 12 + hl];
        if (m == -FLT_MAX) continue;
        const float w = exp(m - M);
        L = fma(part_l[base + slot * 12 + hl], w, L);
        acc = fma(part_acc[base + ((size_t)slot * 12 + hl) * 256 + d], w, acc);
    }
    attn[((size_t)z * n_head + h) * 256 + d] = L > 0.0f ? acc / L : 0.0f;
}
