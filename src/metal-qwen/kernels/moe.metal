// moe.metal - MoE routing and combination kernels for the Qwen Metal backend.
// Semantics mirror the CUDA sources in src/kernels/cuda (router_top10.cu,
// router_sigmoid.cu, native_router.cu, native_moe.cu, shared_expert.cu's
// moe_combine_kernel); where a CUDA kernel accumulates in double and MSL has no
// double, the serial sums below use compensated (Kahan) float accumulation in
// the same ascending order. Selection/tie rules are exact.

// ---- router_top10: softmax over ALL experts, stable descending top-k (ties by
// lower index), gather, renormalise with the 2^-14 lower clamp. One threadgroup
// per token; n_expert <= 1024 (host-enforced).
struct RouterArgs {
    int n_expert;
    int k;
};

kernel void router_top10_k(device const float* logits [[buffer(0)]], device int* ids [[buffer(1)]],
                           device float* weights [[buffer(2)]], constant RouterArgs& a [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_position_in_threadgroup]]) {
    threadgroup float s_exp[1024];
    threadgroup float s_p[1024];
    threadgroup uchar s_taken[1024];
    threadgroup float s_v[256];
    threadgroup int s_i[256];
    threadgroup float s_scalar[2];
    const int ne = a.n_expert, k = a.k;
    device const float* l = logits + ulong(tg) * ne;
    // max (exact, order-independent)
    float mx = -INFINITY;
    for (int e = tid; e < ne; e += 256) mx = max(mx, l[e]);
    s_v[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (int i = 0; i < 256; ++i) m = max(m, s_v[i]);
        s_scalar[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = s_scalar[0];
    for (int e = tid; e < ne; e += 256) s_exp[e] = exp(l[e] - mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {  // the sum, ascending, on one thread (as the CUDA kernel keeps it)
        float sum = 0.0f, c = 0.0f;
        for (int e = 0; e < ne; ++e) {
            const float y = s_exp[e] - c;
            const float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        s_scalar[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = s_scalar[0];
    for (int e = tid; e < ne; e += 256) {
        s_p[e] = s_exp[e] * inv;
        s_taken[e] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // k passes of block argmax; strict > on a strided scan plus a (value, index)
    // reduction keeps the LOWEST index on ties.
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = ne;
        for (int e = tid; e < ne; e += 256) {
            if (s_taken[e]) continue;
            if (s_p[e] > bv) {
                bv = s_p[e];
                bi = e;
            }
        }
        s_v[tid] = bv;
        s_i[tid] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = -INFINITY;
            int ix = ne;
            for (int t = 0; t < 256; ++t) {
                if (s_v[t] > v || (s_v[t] == v && s_i[t] < ix)) {
                    v = s_v[t];
                    ix = s_i[t];
                }
            }
            if (ix < ne) {
                ids[ulong(tg) * k + i] = ix;
                weights[ulong(tg) * k + i] = v;
                s_taken[ix] = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {  // renormalise, ascending, with the clamp
        float s = 0.0f, c = 0.0f;
        for (int i = 0; i < k; ++i) {
            const float y = weights[ulong(tg) * k + i] - c;
            const float t = s + y;
            c = (t - s) - y;
            s = t;
        }
        const float sc = max(s, 6.103515625e-5f);  // 2^-14
        for (int i = 0; i < k; ++i) weights[ulong(tg) * k + i] = weights[ulong(tg) * k + i] / sc;
    }
}

// ---- router_sigmoid: p = sigmoid(logits) (unbiased weights), sel = p + bias
// (selection only), stable descending rank over sel, weights gathered from p,
// renormalised (clamp 2^-14) only when norm_w, then scaled by w_scale.
struct RouterSigArgs {
    int n_expert;
    int k;
    float w_scale;
    int norm_w;
    int has_bias;
};

kernel void router_sigmoid_k(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                             device int* ids [[buffer(2)]], device float* weights [[buffer(3)]],
                             constant RouterSigArgs& a [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             ushort tid [[thread_position_in_threadgroup]]) {
    threadgroup float s_p[1024];
    threadgroup float s_sel[1024];
    const int ne = a.n_expert, k = a.k;
    device const float* l = logits + ulong(tg) * ne;
    for (int e = tid; e < ne; e += 256) {
        const float p = 1.0f / (1.0f + exp(-l[e]));
        s_p[e] = p;
        s_sel[e] = a.has_bias ? p + bias[e] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // rank(e) = #{f: sel[f] > sel[e]} + #{f < e: sel[f] == sel[e]}
    for (int e = tid; e < ne; e += 256) {
        const float v = s_sel[e];
        int rank = 0;
        for (int f = 0; f < ne; ++f) {
            const float u = s_sel[f];
            rank += u > v ? 1 : 0;
            rank += (f < e && u == v) ? 1 : 0;
        }
        if (rank < k) ids[ulong(tg) * k + rank] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f, c = 0.0f;
        for (int i = 0; i < k; ++i) {
            const float y = s_p[ids[ulong(tg) * k + i]] - c;
            const float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        const float inv = a.norm_w ? 1.0f / max(sum, 6.103515625e-5f) : 1.0f;
        for (int i = 0; i < k; ++i)
            weights[ulong(tg) * k + i] = s_p[ids[ulong(tg) * k + i]] * inv * a.w_scale;
    }
}

// ---- native_router_top10: the pinned 512-expert / top-10 contract. One
// simdgroup per token; each lane holds experts lane + 32*i (i in 0..15), float
// softmax, selection by repeated warp argmax with the lower index winning ties,
// the selected sum accumulated by the WINNING lane in selection order and
// reduced with the xor butterfly - the CUDA kernel's exact structure.
kernel void native_router_k(device const float* logits [[buffer(0)]], device int* ids [[buffer(1)]],
                            device float* weights [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    device const float* l = logits + ulong(tg) * 512;
    device int* o_ids = ids + ulong(tg) * 10;
    device float* o_w = weights + ulong(tg) * 10;
    float v[16];
    for (int i = 0; i < 16; ++i) v[i] = l[lane + 32 * i];
    float mx = -INFINITY;
    for (int i = 0; i < 16; ++i) mx = max(mx, v[i]);
    mx = moe_warp_max_xor(mx);
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        v[i] = exp(v[i] - mx);
        sum += v[i];
    }
    const float rcp = 1.0f / moe_warp_sum_xor(sum);
    for (int i = 0; i < 16; ++i) {
        v[i] *= rcp;
        if (isnan(v[i])) v[i] = -FLT_MAX;
    }
    float selected = 0.0f, ssum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = int(lane);
        for (int i = 1; i < 16; ++i)
            if (v[i] > best) {
                best = v[i];
                ex = int(lane) + 32 * i;
            }
        for (ushort m = 16; m > 0; m >>= 1) {
            const float ob = simd_shuffle_xor(best, m);
            const int oe = simd_shuffle_xor(ex, m);
            if (ob > best || (ob == best && oe < ex)) {
                best = ob;
                ex = oe;
            }
        }
        if ((ex & 31) == int(lane)) {
            v[ex / 32] = -INFINITY;
            o_ids[rank] = ex;
            ssum += best;
        }
        if (rank == int(lane)) selected = best;
    }
    ssum = max(moe_warp_sum_xor(ssum), 6.103515625e-5f);
    const float inv = 1.0f / ssum;
    if (lane < 10) o_w[lane] = selected * inv;
}

// ---- combine kernels. moe_combine_k: the legacy combination - ascending sum
// of weights[e] * parts[e] over k, shared added PLAIN (never weighted).
// native_combine_k: the pinned contract - first product rounds to float, the
// rest accumulate with FMA in expert order, shared added once afterward.
// Grid: x = n_embd threads, y = token (1 for the single-token forms).
struct CombineArgs {
    int n_embd;
    int k;
    int has_shared;
};

kernel void moe_combine_k(device const float* parts [[buffer(0)]], device const float* weights [[buffer(1)]],
                          device const float* shared [[buffer(2)]], device float* y [[buffer(3)]],
                          constant CombineArgs& a [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    if (int(j) >= a.n_embd) return;
    float acc = 0.0f, c = 0.0f;
    for (int e = 0; e < a.k; ++e) {
        const float term = weights[e] * parts[ulong(e) * a.n_embd + j] - c;
        const float t = acc + term;
        c = (t - acc) - term;
        acc = t;
    }
    if (a.has_shared) acc += shared[j];
    y[j] = acc;
}

kernel void native_combine_k(device const float* parts [[buffer(0)]], device const float* weights [[buffer(1)]],
                             device const float* shared [[buffer(2)]], device float* out [[buffer(3)]],
                             constant CombineArgs& a [[buffer(4)]], uint2 pos [[thread_position_in_grid]]) {
    const int col = int(pos.x);
    if (col >= a.n_embd) return;
    const ulong tk = ulong(pos.y);
    device const float* p = parts + tk * ulong(a.k) * a.n_embd;
    device const float* w = weights + tk * a.k;
    float sum = p[col] * w[0];
    for (int e = 1; e < a.k; ++e) sum = fma(p[ulong(e) * a.n_embd + col], w[e], sum);
    if (a.has_shared) sum += shared[tk * a.n_embd + col];
    out[tk * a.n_embd + col] = sum;
}

// ---- hit selection: which routed entries are resident. Serial on one thread
// (k <= 128 entries; the CUDA version ballots a warp to the same result):
// entries keep routing order, count is the compacted length.
struct SelectArgs {
    int n;
    int n_expert;
};

kernel void moe_hit_select_k(device const int* ids [[buffer(0)]], device const int* res_row [[buffer(1)]],
                             device int* slot [[buffer(2)]], device int* dst [[buffer(3)]],
                             device int* count [[buffer(4)]], constant SelectArgs& a [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    int n_hit = 0;
    for (int i = 0; i < a.n; ++i) {
        const int e = ids[i];
        const int s = (e >= 0 && e < a.n_expert) ? res_row[e] : -1;
        if (s >= 0) {
            slot[n_hit] = s;
            dst[n_hit] = i;
            ++n_hit;
        }
    }
    count[0] = n_hit;
}

// ---- hit add: parts[dst[h]] += hit_out[dst[h]] for h < *count (device count).
struct HitAddArgs {
    int n_embd;
};

kernel void moe_hit_add_k(device float* parts [[buffer(0)]], device const float* hit_out [[buffer(1)]],
                          device const int* dst [[buffer(2)]], device const int* count [[buffer(3)]],
                          constant HitAddArgs& a [[buffer(4)]], uint2 pos [[thread_position_in_grid]]) {
    const int h = int(pos.y);
    if (h >= count[0]) return;
    const int i = int(pos.x);
    if (i >= a.n_embd) return;
    const ulong row = ulong(dst[h]) * a.n_embd;
    parts[row + i] += hit_out[row + i];
}
