// qsa_score.metal - indexer scoring and block selection kernels.
// Semantics: src/kernels/cuda/native_qsa_score.cu (ordered scalar path) and
// src/kernels/cuda/qsa_select.cu. Fixed geometry throughout: idx_dim 128,
// idx heads 4, idx block 4.

// native_qsa_score: one thread per block row r in [0, n_bid] (the last is
// the spare row). Per head a sequential FMA dot in d-ascending order (the
// CUDA scalar fallback's order), ReLU, heads added in order, then bias,
// then the +1e9 incomplete-tail bump on the spare row. An inconsistent
// step writes nothing at all, as on CUDA. The score is fanned out to the
// row's cells [r*4, min(n_kv, (r+1)*4)).
kernel void strata_native_qsa_score(device const float* pooled [[buffer(0)]],
                                    device const float* query [[buffer(1)]],
                                    device const float* bias [[buffer(2)]],
                                    device const int* step [[buffer(3)]],
                                    device float* cells [[buffer(4)]],
                                    constant int& has_bias [[buffer(5)]],
                                    constant int& max_cells [[buffer(6)]],
                                    uint tid [[thread_position_in_grid]]) {
    const int n = step[1];    // kStepNKv
    const int full = step[2]; // kStepNBid
    if (n < 1 || n > max_cells || step[0] != n - 1 || full != n / 4 ||
        step[3] != (n < 2051 ? n : 2051))
        return;
    const int r = (int)tid;
    if (r > full) return;
    float sum = 0.0f;
    for (int h = 0; h < 4; ++h) {
        float acc = 0.0f;
        for (int d = 0; d < 128; ++d) acc = fma(pooled[(size_t)r * 128 + d], query[h * 128 + d], acc);
        sum += acc > 0.0f ? acc : 0.0f;
    }
    if (has_bias) sum += bias[r];
    sum += (r == full && n % 4) ? 1e9f : 0.0f;
    sum += 0.0f;  // the CUDA path's final +0 (keeps -0/+0 behaviour identical)
    for (int i = r * 4; i < n && i < (r + 1) * 4; ++i) cells[i] = sum;
}

// qsa_block_scores: one thread per (query, block). Blocks past the query's
// completed count read the dead (spare) row instead of a pooled row; the
// incomplete-tail bump applies on that row. Dots are sequential float FMAs
// (the CUDA warp kernel reduces float4 partials in tree order; the parity
// harness's double reference bounds both).
kernel void strata_qsa_block_scores(device const float* pooled [[buffer(0)]],
                                    device const float* dead [[buffer(1)]],
                                    device const float* q_idx [[buffer(2)]],
                                    device const int* steps [[buffer(3)]],
                                    device float* out [[buffer(4)]],
                                    constant long& max_blocks [[buffer(5)]],
                                    constant long& reach [[buffer(6)]],
                                    uint tid [[thread_position_in_grid]]) {
    const long qi = (long)tid / reach;
    const long b = (long)tid % reach;
    const int n_kv = steps[qi * 4 + 1];
    const int n_bid = steps[qi * 4 + 2];
    if (b > n_bid || b >= max_blocks) return;
    device const float* key = (b == n_bid) ? dead : pooled + b * 128;
    device const float* q = q_idx + qi * 4 * 128;
    float score = 0.0f;
    for (int h = 0; h < 4; ++h) {
        float d = 0.0f;
        for (int i = 0; i < 128; ++i) d = fma(key[i], q[h * 128 + i], d);
        score += d > 0.0f ? d : 0.0f;
    }
    if (b == n_bid && n_kv % 4 != 0) score += 1e9f;
    out[qi * max_blocks + b] = score;
}

// qsa_block_topk: one threadgroup of 256 threads per query; a direct port
// of the CUDA radix-select kernel. Blocks carry weights (4 cells, the tail
// block its remainder); the threshold is the largest order_key whose
// weighted count of >= reaches the step width; ties at the threshold are
// admitted lowest-block-first until the width is filled; ids are emitted
// ascending by cell. When n_kv <= width the selection is the identity.
kernel void strata_qsa_block_topk(device const float* scores [[buffer(0)]],
                                  device const int* steps [[buffer(1)]], device int* ids [[buffer(2)]],
                                  constant long& max_blocks [[buffer(3)]], constant long& cap [[buffer(4)]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint qi [[threadgroup_position_in_grid]]) {
    threadgroup atomic_int hist[256];
    threadgroup int s_a[256];
    threadgroup int s_b[256];
    threadgroup int s_digit;
    threadgroup int s_above;
    const int t = (int)tid;
    device const int* st = steps + qi * 4;
    const long n_kv = st[1], n_bid = st[2], width = st[3];
    device int* out = ids + qi * cap;
    if (n_kv <= width) {  // everything is selected: the identity, ascending
        for (long j = t; j < n_kv; j += 256) out[j] = (int)j;
        return;
    }
    device const float* sc = scores + qi * max_blocks;
    const long nb = n_bid + 1;  // blocks 0..n_bid, the last possibly empty
    const long per = (nb + 255) / 256;
    const long b0 = (long)t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    // ---- radix select, 8 bits at a time
    uint prefix = 0u;
    int above = 0;  // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += 256) atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (long b = b0; b < b1; ++b) {
            const int w = b < n_bid ? 4 : (int)(n_kv - n_bid * 4);
            if (w == 0) continue;
            const uint k = strata_order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask))
                atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], w, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                const int hd = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (cum + hd >= width) break;
                cum += hd;
            }
            s_digit = d;
            s_above = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= (uint)s_digit << shift;
        above = s_above;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = prefix;
    const long eq_budget = width - above;  // cells equal to thr that fit, lowest index first
    int gt = 0, eq = 0;
    for (long b = b0; b < b1; ++b) {
        const int w = b < n_bid ? 4 : (int)(n_kv - n_bid * 4);
        if (w == 0) continue;
        const uint k = strata_order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < 256; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag;
            s_b[i] = ae;
            ag += g;
            ae += e;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const long eq_before = s_b[t];
    long my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int)my_eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s_a[t] = sel;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < 256; ++i) {
            const int c = s_a[i];
            s_a[i] = a;
            a += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long wpos = s_a[t];
    long eq_left = my_eq;
    for (long b = b0; b < b1; ++b) {
        const int w = b < n_bid ? 4 : (int)(n_kv - n_bid * 4);
        if (w == 0) continue;
        const uint k = strata_order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int)(b * 4 + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int)(b * 4 + c);
        }
    }
}
