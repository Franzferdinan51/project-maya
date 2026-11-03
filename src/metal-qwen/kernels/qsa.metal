// qsa.metal - the QSA decode pipeline kernels (qsa.hpp family).
// Semantics: src/kernels/cuda/qsa.cu.
//
// Where the CUDA kernels accumulate in double (indexer_key_append's norm,
// qsa_index's dots), Metal has no double: this file carries double-single
// (two-float) arithmetic helpers and uses them in exactly those two spots,
// so the values that get rounded to float at the end agree with the CUDA
// double path to far below one float ulp. Everything else is float, as on
// CUDA.

// ---- double-single (Dekker/Knuth) helpers ----------------------------------
struct DS {
    float hi, lo;
};
static inline DS ds_from(float x) { return DS{x, 0.0f}; }
static inline DS ds_add(DS a, DS b) {
    const float s = a.hi + b.hi;
    const float bb = s - a.hi;
    const float err = (a.hi - (s - bb)) + (b.hi - bb);
    const float lo = err + a.lo + b.lo;
    DS r;
    r.hi = s + lo;
    r.lo = lo - (r.hi - s);
    return r;
}
static inline DS ds_mul(DS a, DS b) {
    const float p = a.hi * b.hi;
    const float e = fma(a.hi, b.hi, -p);
    const float lo = e + (a.hi * b.lo + a.lo * b.hi);
    DS r;
    r.hi = p + lo;
    r.lo = lo - (r.hi - p);
    return r;
}
static inline DS ds_mul_f(DS a, float b) { return ds_mul(a, ds_from(b)); }
static inline float ds_to_float(DS a) { return a.hi + a.lo; }
// 1/sqrt(x) to double-single accuracy: float seed + two Newton steps
// y <- y * (1 + (1 - x*y^2) / 2).
static inline DS ds_inv_sqrt(DS x) {
    float y = rsqrt(x.hi);
    DS dy = ds_from(y);
    for (int i = 0; i < 2; ++i) {
        const DS y2 = ds_mul(dy, dy);
        const DS t = ds_add(ds_from(1.0f), ds_mul_f(ds_mul(x, y2), -1.0f));
        dy = ds_mul(dy, ds_add(ds_from(1.0f), ds_mul_f(t, 0.5f)));
    }
    return dy;
}

// ---- kv_append --------------------------------------------------------------
// One thread per (kv head, dim). The page row comes from the step's
// position; a negative page skips the device pool (streaming residency),
// the host pool (identity layout) is written whenever present.
kernel void strata_kv_append(device ushort* k_pool [[buffer(0)]], device ushort* v_pool [[buffer(1)]],
                             device const int* page_table [[buffer(2)]],
                             device const int* step [[buffer(3)]], device const float* kcur [[buffer(4)]],
                             device const float* vcur [[buffer(5)]], device ushort* host_k [[buffer(6)]],
                             device ushort* host_v [[buffer(7)]], constant int& has_host [[buffer(8)]],
                             constant int& kv_heads [[buffer(9)]], constant int& head_dim [[buffer(10)]],
                             constant int& page_size [[buffer(11)]], uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= kv_heads * head_dim) return;
    const long pos = step[0];  // kStepPos
    const int h = (int)tid / head_dim, d = (int)tid - h * head_dim;
    const long page = page_table[pos / page_size];
    if (page >= 0) {
        const long row = (page * kv_heads + h) * page_size + (pos % page_size);
        k_pool[row * head_dim + d] = strata_f16_from_f32(kcur[tid]);
        v_pool[row * head_dim + d] = strata_f16_from_f32(vcur[tid]);
    }
    if (has_host) {
        const long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        host_k[row * head_dim + d] = strata_f16_from_f32(kcur[tid]);
        host_v[row * head_dim + d] = strata_f16_from_f32(vcur[tid]);
    }
}

// ---- indexer_key_append -----------------------------------------------------
// One threadgroup of idx_dim threads. Mirrors the CUDA kernel's structure:
// raw tail store, the pos==0 spare key in double (here double-single), the
// completed block's mean/norm in double-single, spare row rewrite, then
// the in-place NEOX rotation of the completed row with the shared table.
kernel void strata_indexer_key_append(device const float* raw [[buffer(0)]],
                                      device const int* pos_dev [[buffer(1)]],
                                      device const float* w_k_norm [[buffer(2)]],
                                      device float* tail [[buffer(3)]], device float* dead [[buffer(4)]],
                                      device float* pooled [[buffer(5)]],
                                      device int* block_pos [[buffer(6)]],
                                      device const float* cos_tab [[buffer(7)]],
                                      device const float* sin_tab [[buffer(8)]],
                                      device const int* mtab [[buffer(9)]],
                                      constant int& has_mtab [[buffer(10)]],
                                      constant int& pos_base [[buffer(11)]],
                                      constant float& eps [[buffer(12)]],
                                      constant int& idx_dim [[buffer(13)]], constant int& r [[buffer(14)]],
                                      constant int& n_rot [[buffer(15)]],
                                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float mean_hi[1024];
    threadgroup float mean_lo[1024];
    const int d = (int)tid;
    const int pos = pos_dev[0];
    const int slot = pos % r;
    if (slot < r - 1) tail[(size_t)slot * idx_dim + d] = raw[d];
    if (pos == 0) {
        DS ss = ds_from(0.0f);
        for (int i = 0; i < idx_dim; ++i) ss = ds_add(ss, ds_mul_f(ds_from(raw[i]), raw[i]));
        const DS inv = ds_inv_sqrt(ds_add(ds_mul_f(ss, 1.0f / (float)idx_dim), ds_from(eps)));
        // dead = (float)(raw * inv * w), the product carried in DS as the
        // CUDA path carries it in double.
        dead[d] = ds_to_float(ds_mul(ds_mul(ds_from(raw[d]), inv), ds_from(w_k_norm[d])));
        pooled[d] = dead[d];  // rope at position 0 is the identity
    }
    if (slot != r - 1) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        DS m = ds_from(0.0f);
        for (int j = 0; j < r - 1; ++j) m = ds_add(m, ds_from(tail[(size_t)j * idx_dim + d]));
        m = ds_add(m, ds_from(raw[d]));
        m = ds_mul_f(m, 1.0f / (float)r);
        mean_hi[d] = m.hi;
        mean_lo[d] = m.lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    DS ss = ds_from(0.0f);
    for (int i = 0; i < idx_dim; ++i) {
        const DS v = DS{mean_hi[i], mean_lo[i]};
        ss = ds_add(ss, ds_mul(v, v));
    }
    const DS inv = ds_inv_sqrt(ds_add(ds_mul_f(ss, 1.0f / (float)idx_dim), ds_from(eps)));
    const int b = pos / r;
    pooled[(size_t)b * idx_dim + d] =
        ds_to_float(ds_mul(ds_mul(DS{mean_hi[d], mean_lo[d]}, inv), ds_from(w_k_norm[d])));
    pooled[(size_t)(b + 1) * idx_dim + d] = dead[d];
    if (d == 0) block_pos[0] = pos_base + b * r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int nhalf = n_rot / 2;
    if (d < nhalf) {
        device float* row = pooled + (size_t)b * idx_dim;
        const int rp = strata_mrope_pos(mtab, has_mtab, pos_base + b * r, d);
        const size_t toff = (size_t)rp * (size_t)nhalf;
        const float c = cos_tab[toff + d], s = sin_tab[toff + d];
        const float a = row[d], bb = row[nhalf + d];
        row[d] = a * c - bb * s;
        row[nhalf + d] = a * s + bb * c;
    }
}

// ---- qsa_index ---------------------------------------------------------------
// One threadgroup per block, one simdgroup per indexer head. Dots in
// double-single: per-lane strided partials, simd_shuffle_xor tree (the
// CUDA order), then thread 0 adds the ReLU'd heads in order, the bias,
// and fans the score out to the block's cells (the tail block: to n_kv).
kernel void strata_qsa_index(device const float* pooled [[buffer(0)]],
                             device const float* q_idx [[buffer(1)]],
                             device const float* bias [[buffer(2)]], device const int* step [[buffer(3)]],
                             device float* cell_scores [[buffer(4)]], constant int& has_bias [[buffer(5)]],
                             constant int& idx_n_head [[buffer(6)]], constant int& idx_dim [[buffer(7)]],
                             constant long& r [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                             uint b [[threadgroup_position_in_grid]]) {
    threadgroup float dot_hi[32];
    threadgroup float dot_lo[32];
    const long n_bid = step[2];
    const long n_kv = step[1];
    if ((long)b > n_bid) return;
    const int wid = (int)(tid >> 5), lane = (int)(tid & 31u);
    if (wid == 0) {
        dot_hi[lane] = 0.0f;
        dot_lo[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    DS acc = ds_from(0.0f);
    for (int d = lane; d < idx_dim; d += 32)
        acc = ds_add(acc, ds_mul(ds_from(pooled[(size_t)b * idx_dim + d]),
                                 ds_from(q_idx[(size_t)wid * idx_dim + d])));
    for (int o = 16; o > 0; o >>= 1) {
        const DS other = DS{simd_shuffle_xor(acc.hi, o), simd_shuffle_xor(acc.lo, o)};
        acc = ds_add(acc, other);
    }
    if (lane == 0) {
        dot_hi[wid] = acc.hi;
        dot_lo[wid] = acc.lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;
    DS score = ds_from(0.0f);
    for (int h = 0; h < idx_n_head; ++h) {
        const DS v = DS{dot_hi[h], dot_lo[h]};
        const bool pos = v.hi > 0.0f || (v.hi == 0.0f && v.lo > 0.0f);
        if (pos) score = ds_add(score, v);
    }
    if (has_bias) score = ds_add(score, ds_from(bias[b]));
    long lo = (long)b * r, hi = lo + r;
    if ((long)b == n_bid) hi = n_kv;
    if (hi > n_kv) hi = n_kv;
    float sc = ds_to_float(score);
    if ((long)b == n_bid && n_kv % r != 0) sc += 1e9f;
    for (long j = lo; j < hi; ++j) cell_scores[j] = sc;
}

// ---- topk_512 -----------------------------------------------------------------
// One threadgroup of 256 threads; a direct port of the CUDA binary-lifting
// threshold search: the largest order_key whose count of >= reaches the
// step width, then emit greater-than cells ascending, then equals in index
// order until the width is filled.
kernel void strata_topk_512(device const float* scores [[buffer(0)]], device const int* step [[buffer(1)]],
                            device int* out_ids [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int s_a[256];
    threadgroup int s_b[256];
    threadgroup long s_cgt;
    const long n_kv = step[1];
    const long width = step[3];
    const int t = (int)tid;
    const long chunk = (n_kv + 255) / 256;
    const long lo = (long)t * chunk;
    long hi = lo + chunk;
    if (hi > n_kv) hi = n_kv;
    uint v = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        const uint cand = v | (1u << bit);
        int c = 0;
        for (long j = lo; j < hi; ++j)
            if (strata_order_key(scores[j]) >= cand) ++c;
        s_a[t] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 128; s > 0; s >>= 1) {
            if (t < s) s_a[t] += s_a[t + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const int tot = s_a[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tot >= (int)width) v = cand;
    }
    const uint thr = v;
    int gt = 0, eq = 0;
    for (long j = lo; j < hi; ++j) {
        const uint k = strata_order_key(scores[j]);
        if (k > thr) ++gt;
        else if (k == thr) ++eq;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        long ag = 0, ae = 0;
        for (int i = 0; i < 256; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = (int)ag;
            s_b[i] = (int)ae;
            ag += g;
            ae += e;
        }
        s_cgt = ag;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const long off_eq = s_b[t];
    const long eq_budget = width - s_cgt;
    int sel = 0;
    {
        long e = off_eq;
        for (long j = lo; j < hi; ++j) {
            const uint k = strata_order_key(scores[j]);
            if (k > thr) ++sel;
            else if (k == thr && e < eq_budget) {
                ++sel;
                ++e;
            }
        }
    }
    s_a[t] = sel;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        long a = 0;
        for (int i = 0; i < 256; ++i) {
            const int c = s_a[i];
            s_a[i] = (int)a;
            a += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long w = s_a[t];
    long e = off_eq;
    for (long j = lo; j < hi; ++j) {
        const uint k = strata_order_key(scores[j]);
        if (k > thr) out_ids[w++] = (int)j;
        else if (k == thr && e < eq_budget) {
            ++e;
            out_ids[w++] = (int)j;
        }
    }
}

// ---- kv_gather ------------------------------------------------------------------
// One thread per 4-half group, as on CUDA (uint2 copies).
kernel void strata_kv_gather(device const uint2* k_pool [[buffer(0)]], device const uint2* v_pool [[buffer(1)]],
                             device const int* page_table [[buffer(2)]], device const int* ids [[buffer(3)]],
                             device const int* step [[buffer(4)]], device uint2* k_scratch [[buffer(5)]],
                             device uint2* v_scratch [[buffer(6)]], constant int& kv_heads [[buffer(7)]],
                             constant int& head_dim [[buffer(8)]], constant int& page_size [[buffer(9)]],
                             uint tid [[thread_position_in_grid]]) {
    const long n_ids = step[3];  // kStepWidth
    const long per = head_dim / 4;
    const long total = n_ids * kv_heads * per;
    if ((long)tid >= total) return;
    const long id = (long)tid / (kv_heads * per);
    const int rem = (int)((long)tid % (kv_heads * per));
    const int h = (int)(rem / per), q = (int)(rem - h * per);
    const int cell = ids[id];
    const long page = page_table[cell / page_size];
    const long src = ((page * kv_heads + h) * page_size + (cell % page_size)) * per + q;
    const long dst = (id * kv_heads + h) * per + q;
    k_scratch[dst] = k_pool[src];
    v_scratch[dst] = v_pool[src];
}

// ---- qsa_attend -----------------------------------------------------------------
// One threadgroup per query head, head_dim threads. The scores live in
// threadgroup memory sized for the geometry's largest selection width
// (idx_top_k + idx_block - 1 = 2051 for the real shapes); the host wrapper
// refuses a larger capacity. Reduction order mirrors the CUDA kernel:
// strided per-thread scores, simdgroup max/sum, cross-simdgroup via the
// red array, then each thread accumulates its own dimension over cells in
// ascending order with the normalised weights.
kernel void strata_qsa_attend(device const float* q [[buffer(0)]],
                              device const ushort* k_scratch [[buffer(1)]],
                              device const ushort* v_scratch [[buffer(2)]],
                              device const int* step [[buffer(3)]], device float* attn [[buffer(4)]],
                              device float* weights [[buffer(5)]], constant int& has_weights [[buffer(6)]],
                              constant int& n_head [[buffer(7)]], constant int& n_head_kv [[buffer(8)]],
                              constant int& head_dim [[buffer(9)]], uint tid [[thread_index_in_threadgroup]],
                              uint h [[threadgroup_position_in_grid]]) {
    threadgroup float red[32];
    threadgroup float w[2051];
    const long n_ids = step[3];  // kStepWidth
    const int d = (int)tid;
    if (n_ids == 0) {
        for (int i = d; i < head_dim; i += head_dim) attn[(size_t)h * head_dim + i] = 0.0f;
        return;
    }
    const int kv = (int)h / (n_head / n_head_kv);
    const float scale = 1.0f / sqrt((float)head_dim);
    const int lane = d & 31, wid = d >> 5, nwarp = (head_dim + 31) >> 5;
    for (long j = d; j < n_ids; j += head_dim) {
        device const ushort* krow = k_scratch + (j * n_head_kv + kv) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < head_dim; ++i) acc += strata_f32_from_f16(krow[i]) * q[(size_t)h * head_dim + i];
        w[j] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -FLT_MAX;
    for (long j = d; j < n_ids; j += head_dim) mx = fmax(mx, w[j]);
    mx = simd_max(mx);
    if (lane == 0) red[wid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0) {
        mx = lane < nwarp ? red[lane] : -FLT_MAX;
        mx = simd_max(mx);
        if (lane == 0) red[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (long j = d; j < n_ids; j += head_dim) {
        const float e = exp(w[j] - mx);
        w[j] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) red[wid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0) {
        sum = lane < nwarp ? red[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) red[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (long j = 0; j < n_ids; ++j)
        acc += (w[j] * inv) * strata_f32_from_f16(v_scratch[(j * n_head_kv + kv) * head_dim + d]);
    attn[(size_t)h * head_dim + d] = acc;
    if (has_weights) {
        for (long j = d; j < n_ids; j += head_dim) weights[(size_t)h * n_ids + j] = w[j] * inv;
    }
}

// ---- gates ------------------------------------------------------------------------
// out = attn * sigmoid(second half of q_full's per-head row), as f16 / f32.
// The CUDA kernels evaluate the sigmoid in double; float exp here differs
// by ~1e-7 relative, which the parity harness bounds (and which rounds to
// the same f16 in all but pathological ties for the f16 variant).
kernel void strata_qsa_gate_apply(device const float* attn [[buffer(0)]],
                                  device const float* q_full [[buffer(1)]],
                                  device ushort* out [[buffer(2)]], constant int& n_head [[buffer(3)]],
                                  constant int& head_dim [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= n_head * head_dim) return;
    const int h = (int)tid / head_dim, d = (int)tid % head_dim;
    const float g = q_full[(size_t)h * 2 * head_dim + head_dim + d];
    const float sig = 1.0f / (1.0f + exp(-g));
    out[tid] = strata_f16_from_f32(attn[tid] * sig);
}
kernel void strata_qsa_gate_apply_f32(device const float* attn [[buffer(0)]],
                                      device const float* q_full [[buffer(1)]],
                                      device float* out [[buffer(2)]], constant int& n_head [[buffer(3)]],
                                      constant int& head_dim [[buffer(4)]],
                                      uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= n_head * head_dim) return;
    const int h = (int)tid / head_dim, d = (int)tid % head_dim;
    const float g = q_full[(size_t)h * 2 * head_dim + head_dim + d];
    const float sig = 1.0f / (1.0f + exp(-g));
    out[tid] = attn[tid] * sig;
}
