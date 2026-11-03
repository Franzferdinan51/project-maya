// verify.metal - the speculative-verify family's GPU kernels, transcribed
// from src/kernels/cuda/verify_kernels.cu (the GDN multi-token trio, the
// selection/copy helpers and the mapped-memory conditionals).
// resident_plan is resident_plan_k below: it manipulates raw pointer VALUES
// (integers it writes, never dereferences), so it runs on the device in
// stream order. fetch_blobs is the one host-semantics entry left in the
// .cpp: on Metal its staging copy is owned by the pool-side DMA path (see
// verify_kernels.cpp), not by a mid-body host sync.
// gpu_stamp's value is sampled on the host (Metal exposes no GPU timestamp
// register to kernels) and written by stamp_k in stream order.

// ---- GRFOLD debug: residual fingerprint (env-gated at the call site) ----
// One threadgroup; the sum is over integers, so the result is independent of
// the thread partition: h = sum_i (bits(x[i]) + phi) * (i+1) * FNV-prime.
kernel void resid_hash_k(device const float* x [[buffer(0)]], device ulong* out [[buffer(1)]],
                         constant int& n [[buffer(2)]], constant int& slot [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong scratch[256];
    ulong h = 0;
    for (int i = int(tid); i < n; i += 256) {
        ulong b = ulong(as_type<uint>(x[i]));
        h += (b + 0x9e3779b97f4a7c15ul) * (ulong(i) + 1ul) * 0x100000001b3ul;
    }
    scratch[tid] = h;
    for (uint step = 128; step > 0; step >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < step) scratch[tid] += scratch[tid + step];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) out[slot] = scratch[0];
}

// GRFOLD debug: strided value sampler (companion to resid_hash_k): copies
// x[k * stride] into out[slot * nsamp + k], in stream order like the hash.
kernel void resid_sample_k(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                           constant int& stride [[buffer(2)]], constant int& slot [[buffer(3)]],
                           constant int& nsamp [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(tid) < nsamp) out[slot * nsamp + int(tid)] = x[int(tid) * stride];
}

// ---- GDN multi-token trio ------------------------------------------------------

kernel void gdn_conv_l2_k(device const float* hist [[buffer(0)]], device const float* qkv [[buffer(1)]],
                          device const float* w [[buffer(2)]], device float* h [[buffer(3)]],
                          constant int& C [[buffer(4)]], constant int& qk_heads [[buffer(5)]],
                          constant float& eps [[buffer(6)]], constant int& t_begin [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint2 tidv [[thread_position_in_threadgroup]]) {
    const uint tid = tidv.x;
    threadgroup float part[4];
    int t = t_begin + int(tg.y);
    int c = int(tg.x) * 128 + int(tid);
    float win[3];
    for (int j = 0; j < 3; ++j) {
        int src = t + j;
        win[j] = (src < 3) ? hist[c * 3 + src] : qkv[long(src - 3) * C + c];
    }
    float x = qkv[long(t) * C + c];
    float sum = win[0] * w[c * 4] + win[1] * w[c * 4 + 1] + win[2] * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + exp(-sum));
    if (int(tg.x) < qk_heads) {
        float sq = y * y;
        for (uint o = 16; o > 0; o >>= 1) sq += simd_shuffle_xor(sq, o);
        if ((tid & 31) == 0) part[tid >> 5] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrt(ss + eps);
    }
    h[long(t) * C + c] = y;
}

kernel void gdn_conv_commit_k(device float* hist [[buffer(0)]], device const float* qkv [[buffer(1)]],
                              constant int& C [[buffer(2)]], device const int* n_keep [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(C)) return;
    int n = n_keep[0];
    if (n <= 0) return;
    int c = int(gid);
    float seq[3];
    for (int j = 0; j < 3; ++j) {
        int src = n + j;
        seq[j] = (src < 3) ? hist[c * 3 + src] : qkv[long(src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

// One warp per output row (2*h_v rows: alpha rows then beta rows), each lane
// accumulating all T tokens over its stride of the row, as in CUDA.
struct GdnAbArgs {
    int n;
    int h_v;
    int T;
};
kernel void gdn_ab_k(device const float* x [[buffer(0)]], device const ushort* wa [[buffer(1)]],
                     device const ushort* wb [[buffer(2)]], device const float* dt [[buffer(3)]],
                     device const float* ssm_a [[buffer(4)]], device float* gate [[buffer(5)]],
                     device float* beta [[buffer(6)]], constant GdnAbArgs& a [[buffer(7)]],
                     uint tg [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    int row = int(tg);
    if (row >= 2 * a.h_v) return;
    bool is_beta = row >= a.h_v;
    int r = is_beta ? row - a.h_v : row;
    device const ushort* wrow = (is_beta ? wb : wa) + long(r) * a.n;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int j = int(lane); j < a.n; j += 32) {
        float wv = dq_bf2f(wrow[j]);
        for (int t = 0; t < a.T; ++t) acc[t] = fma(wv, x[long(t) * a.n + j], acc[t]);
    }
    for (int t = 0; t < a.T; ++t) {
        float v = acc[t];
        for (uint o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
        if (lane == 0) {
            if (is_beta) {
                beta[long(t) * a.h_v + r] = 1.0f / (1.0f + exp(-v));
            } else {
                float u = v + dt[r];
                float sp = (u > 20.0f) ? u : log(1.0f + exp(u));
                gate[long(t) * a.h_v + r] = sp * ssm_a[r];
            }
        }
    }
}

// One threadgroup of 512 (128 cols x 4 row groups) per value head, tokens
// serial - the CUDA block, shared arrays and all.
struct GdnStepArgs {
    int C;
    float eps;
    int h_k;
    int h_v;
    int T;
    int has_nkeep;
    int t_out_begin;
};
kernel void gdn_step_norm_k(device float* state [[buffer(0)]], device const float* hbuf [[buffer(1)]],
                            device const float* gate [[buffer(2)]], device const float* beta [[buffer(3)]],
                            device const float* z [[buffer(4)]], device const float* gamma [[buffer(5)]],
                            device float* y [[buffer(6)]], device const int* n_keep [[buffer(7)]],
                            constant GdnStepArgs& a [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sk[128];
    threadgroup float sq[128];
    threadgroup float red[4][128];
    threadgroup float wsum[16];
    const int S = 128, RG = 4, RPG = 32;
    int head = int(tg);
    int col = int(tid) % S;
    int rg = int(tid) / S;
    int qh = head % a.h_k;
    int qk = S * a.h_k;
    int value_dim = S * a.h_v;
    int n = a.has_nkeep ? n_keep[0] : a.T;
    float s[32];
    long base_off = (long(rg * RPG) * a.h_v + head) * S + col;
    long row_stride = long(a.h_v) * S;
    for (int r = 0; r < RPG; ++r) s[r] = state[base_off + r * row_stride];
    for (int t = 0; t < n; ++t) {
        device const float* ht = hbuf + long(t) * a.C;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < uint(S)) {
            sk[tid] = ht[qk + qh * S + tid];
            sq[tid] = ht[qh * S + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float g = exp(gate[long(t) * a.h_v + head]);
        float kv = 0.0f;
        for (int r = 0; r < RPG; ++r) kv = fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[long(t) * a.h_v + head];
        float o = 0.0f;
        for (int r = 0; r < RPG; ++r) {
            s[r] = fma(g, s[r], sk[rg * RPG + r] * delta);
            o = fma(s[r], sq[rg * RPG + r], o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg][col] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrt(float(S));
            sq_part = oc * oc;
        }
        if (t >= a.t_out_begin) {
            for (uint o2 = 16; o2 > 0; o2 >>= 1) sq_part += simd_shuffle_xor(sq_part, o2);
            if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (rg == 0) {
                float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                float scale = rsqrt(ss / float(S) + a.eps);
                float zz = z[long(t) * value_dim + head * S + col];
                y[long(t) * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + exp(-zz)));
            }
        }
    }
    if (a.has_nkeep && n > 0) {
        for (int r = 0; r < RPG; ++r) state[base_off + r * row_stride] = s[r];
    }
}

// ---- selection / copy helpers ---------------------------------------------------

struct EmbDevArgs {
    long n;
    int code_bits;
    int code_bias;
    int group_elems;
    int has_offsets;
    ulong row_codes;
    ulong row_groups;
};
kernel void embedding_gather_dev_k(device const uchar* codes [[buffer(0)]], device const float* scales [[buffer(1)]],
                                   device const float* offsets [[buffer(2)]], device const int* tokens [[buffer(3)]],
                                   device float* out [[buffer(4)]], constant EmbDevArgs& a [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    long i = gid.x;
    if (i >= a.n) return;
    int t = int(gid.y);
    ulong token = ulong(uint(tokens[t]));
    device const uchar* c = codes + token * a.row_codes;
    device const float* sc = scales + token * a.row_groups;
    int per_byte = 8 / a.code_bits;
    uint mask = (1u << uint(a.code_bits)) - 1u;
    int code = (c[i / per_byte] >> ((i % per_byte) * a.code_bits)) & mask;
    long group = i / a.group_elems;
    float v = float(code + a.code_bias) * sc[group];
    if (a.has_offsets) v += offsets[token * a.row_groups + group];
    out[t * a.n + i] = v;
}

kernel void broadcast_streams_k(device const float* x [[buffer(0)]], device float* R [[buffer(1)]],
                                constant long& n [[buffer(2)]], constant int& hc [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
    long i = gid.x;
    if (i >= n * hc) return;
    R[gid.y * n * hc + i] = x[gid.y * n + i % n];
}

kernel void add_streams_broadcast_k(device const float* h [[buffer(0)]], device const float* e [[buffer(1)]],
                                    device float* R [[buffer(2)]], constant long& n [[buffer(3)]],
                                    constant int& hc [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    long i = gid.x;
    if (i >= n * hc) return;
    R[gid.y * n * hc + i] = h[gid.y * n * hc + i] + e[gid.y * n + i % n];
}

kernel void copy_indexed_k(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
                           constant long& stride [[buffer(2)]], device const int* index [[buffer(3)]],
                           constant long& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    int idx = index[0];
    if (idx < 0 || gid >= ulong(n)) return;
    dst[gid] = src[long(idx) * stride + gid];
}

kernel void rebase_ptrs_k(device ulong* ptr [[buffer(0)]], device const int* n [[buffer(1)]],
                          constant ulong& base [[buffer(2)]], constant long& bytes [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]]) {
    if (int(tid) < n[0]) ptr[tid] = base + ulong(tid) * ulong(bytes);
}

kernel void ident_hits_k(device const int* ids [[buffer(0)]], constant int& n [[buffer(1)]],
                         device int* slot [[buffer(2)]], device int* dst [[buffer(3)]],
                         device int* count [[buffer(4)]], uint tid [[thread_position_in_threadgroup]]) {
    if (int(tid) < n) {
        slot[tid] = ids[tid];
        dst[tid] = int(tid);
    }
    if (tid == 0) count[0] = n;
}

kernel void map_ids_k(device int* ids [[buffer(0)]], device const int* table [[buffer(1)]],
                      constant int& n [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]) {
    if (int(tid) < n) ids[tid] = table[ids[tid]];
}

kernel void dense_steps_k(device const int* cells [[buffer(0)]], constant int& n [[buffer(1)]],
                          device int* steps [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]) {
    if (int(tid) >= n) return;
    int c = cells[tid];
    steps[tid * 4 + 0] = c;
    steps[tid * 4 + 1] = c + 1;
    steps[tid * 4 + 2] = (c + 1) / 4;
    steps[tid * 4 + 3] = c + 1;
}

// Element kernels; the host picks the element width (16/4/1 bytes) by name.
#define GATHER_ROWS_KERNEL(NAME, T)                                                                              \
    kernel void NAME(device const T* src [[buffer(0)]], constant long& row_e [[buffer(1)]],                      \
                     device const int* ids [[buffer(2)]], constant long& n [[buffer(3)]], device T* dst [[buffer(4)]], \
                     uint gid [[thread_position_in_grid]]) {                                                     \
        long total = n * row_e;                                                                                  \
        if (gid >= ulong(total)) return;                                                                         \
        long r = gid / row_e, o = gid % row_e;                                                                   \
        dst[gid] = src[long(ids[r]) * row_e + o];                                                                \
    }
GATHER_ROWS_KERNEL(gather_rows16_k, uint4)
GATHER_ROWS_KERNEL(gather_rows4_k, uint)
GATHER_ROWS_KERNEL(gather_rows1_k, uchar)

kernel void row_top_prob_k(device const float* logits [[buffer(0)]], constant int& n_vocab [[buffer(1)]],
                           device const int* ids [[buffer(2)]], device float* probs [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    device const float* l = logits + tg * n_vocab;
    float m = l[ids[tg]];
    float s = 0.0f;
    for (int i = int(tid); i < n_vocab; i += 1024) s += exp(l[i] - m);
    for (uint o = 16; o > 0; o >>= 1) s += simd_shuffle_xor(s, o);
    if (lane == 0) part[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 32; ++w) tot += part[w];
        probs[tg] = 1.0f / tot;
    }
}

struct MtpArgs {
    long stride;
    int j;
    int has_out;
    int has_probs;
};
kernel void mtp_select_k(device const float* R_src [[buffer(0)]], device const int* ids [[buffer(1)]],
                         device const int* row_dev [[buffer(2)]], device float* R_dst [[buffer(3)]],
                         device int* tok_dst [[buffer(4)]], device int* out [[buffer(5)]],
                         device const float* probs [[buffer(6)]], device float* out_p [[buffer(7)]],
                         constant MtpArgs& a [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    int row = row_dev[0];
    if (gid < ulong(a.stride)) R_dst[gid] = R_src[long(row) * a.stride + gid];
    if (gid == 0) {
        int tok = ids[row];
        tok_dst[0] = tok;
        if (a.has_out) out[a.j] = tok;
        if (a.has_probs) out_p[a.j] = probs[row];
    }
}

kernel void window_ids_k(device int* steps [[buffer(0)]], constant int& window [[buffer(1)]],
                         device int* ids [[buffer(2)]], constant long& stride [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    device int* st = steps + tg * 4;
    int n_kv = st[1];
    int start = (n_kv > window) ? n_kv - window : 0;
    int width = n_kv - start;
    for (int j = int(tid); j < width; j += 256) ids[tg * stride + j] = start + j;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) st[3] = width;
}

// ---- mapped-memory conditionals ---------------------------------------------------

// The flag waits (wait_flag_ge / wait_flag_ge_or) are NOT kernels here: a
// device spin on a host-raised flag can never observe the raise on this
// backend, so the stubs register a held-commit gate with the runtime
// instead (metal_runtime.mm "flag waits", VSTALL-NOTES.md).

// ---- resident_plan (E-6): the host pool's plan layout, built serially on one
// thread - the same integer operations in the same order as the CUDA-side
// host layout (distinct experts in routing order, their entries ascending).
// Only VALUES are written (counts / starts / dst / tok and the blob contents
// addresses as integers); nothing dereferences a pointer value, so this runs
// as a normal in-stream kernel - queue-ordered after the router that produced
// `ids`, with no host sync. (The first Metal port ran this on the host after
// a stream sync, which deadlocked inside the verify body the same way the
// grouped launcher's sync did: the sync waits for flag-spin kernels that
// only the post-body host loop can release.)
struct ResidentPlanArgs {
    int n_entries;
    int k;
    int n_expert;
    int has_off;  // slot_off supplied (else the offset is slot * blob)
    ulong base;   // cache_base contents address (a value, written into the plan)
    long blob;
    long capx;
    uint ring;
};

kernel void resident_plan_k(device const int* ids [[buffer(0)]], device const int* res_layer [[buffer(1)]],
                            device const ulong* slot_off [[buffer(2)]], device int* plan [[buffer(3)]],
                            device uint* skip [[buffer(4)]], constant ResidentPlanArgs& a [[buffer(5)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (int i = 0; i < a.n_entries; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= a.n_expert || res_layer[e] < 0) {
            skip[0] = 0;
            return;
        }
    }
    device int* counts = plan;
    device int* start = plan + 4;
    device int* dst = start + a.capx + 1;
    device int* tok = dst + a.capx;
    const long ptr_off = ((4 + (a.capx + 1) + 2 * a.capx) + 1) & ~1l;
    device ulong* ptr = (device ulong*) (plan + ptr_off);
    device int* start2 = plan + ptr_off + 4 * a.capx;
    int groups = 0, entries = 0;
    for (int i0 = 0; i0 < a.n_entries; ++i0) {
        bool first = true;
        for (int j = 0; j < i0; ++j)
            if (ids[j] == ids[i0]) first = false;
        if (!first) continue;
        const int slot = res_layer[ids[i0]];
        ptr[groups] = a.base + (a.has_off ? slot_off[slot] : ulong(slot) * ulong(a.blob));
        start[groups] = entries;
        for (int i = i0; i < a.n_entries; ++i)
            if (ids[i] == ids[i0]) {
                dst[entries] = i;
                tok[entries] = i / a.k;
                ++entries;
            }
        ++groups;
    }
    start[groups] = entries;
    start2[0] = entries;
    counts[0] = groups;
    counts[1] = entries;
    counts[2] = 0;
    skip[0] = a.ring;
}

kernel void copy_i32_unless_k(device int* dst [[buffer(0)]], device const int* src [[buffer(1)]],
                              constant int& n [[buffer(2)]], device const uint* skip [[buffer(3)]],
                              constant uint& value [[buffer(4)]], uint tid [[thread_position_in_threadgroup]]) {
    if (skip[0] == value) return;
    for (int i = int(tid); i < n; i += 128) dst[i] = src[i];
}

kernel void copy_or_zero_k(device float4* dst [[buffer(0)]], device const float4* src [[buffer(1)]],
                           constant long& n4 [[buffer(2)]], device const uint* skip [[buffer(3)]],
                           constant uint& value [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= ulong(n4)) return;
    dst[gid] = (skip[0] == value) ? float4(0.0f) : src[gid];
}

kernel void stamp_k(device ulong* buf [[buffer(0)]], constant int& i [[buffer(1)]],
                    constant ulong& value [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]) {
    if (tid == 0) buf[i] = value;
}
