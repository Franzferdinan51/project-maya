// kv.metal - the KV storage family: FWHT-256, Q8 and Q4_0 append/gather, and
// the KV-streaming map kernels (reset, ring table, resolve, miss copy).
// Transcribed from src/kernels/cuda/kv_q8.cu, kv_q4.cu and kv_stream.cu.
// Layouts and the step-record indices (kStepPos = 0, kStepWidth = 3) are the
// qsa family's, unchanged.

// ---- FWHT-256: one warp per row, 8 registers per lane, scale 1/16 at load ----
kernel void fwht256_k(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                      constant ulong& n_rows [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_position_in_threadgroup]]) {
    if (tg >= n_rows) return;
    device const float* rs = src + tg * 256;
    device float* rd = dst + tg * 256;
    float reg[8];
    for (int i = 0; i < 8; ++i) reg[i] = rs[i * 32 + lane] * (1.0f / 16.0f);
    for (int h = 1; h < 32; h *= 2) {
        for (int j = 0; j < 8; ++j) {
            float v2 = simd_shuffle_xor(reg[j], h);
            reg[j] = ((lane & h) == 0) ? reg[j] + v2 : v2 - reg[j];
        }
    }
    for (int h = 32; h < 256; h *= 2) {
        int step = h / 32;
        for (int j = 0; j < 8; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                float x = reg[j + k], y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
    for (int i = 0; i < 8; ++i) rd[i * 32 + lane] = reg[i];
}

// ---- Q8 append/gather --------------------------------------------------------

struct KvQ8Args {
    int heads;
    int head_dim;
    int page_size;
    int has_host;
};

// One threadgroup of 64 per (K/V, head, 64-group); tg encodes the triple.
kernel void kv_append_q8_k(device char* k_q [[buffer(0)]], device char* v_q [[buffer(1)]],
                           device ushort* k_scale [[buffer(2)]], device ushort* v_scale [[buffer(3)]],
                           device const int* table [[buffer(4)]], device const int* step [[buffer(5)]],
                           device const float* kcur [[buffer(6)]], device const float* vcur [[buffer(7)]],
                           device char* hk_q [[buffer(8)]], device char* hv_q [[buffer(9)]],
                           device ushort* hk_scale [[buffer(10)]], device ushort* hv_scale [[buffer(11)]],
                           constant KvQ8Args& a [[buffer(12)]], uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float warp_max[2];
    int groups = a.head_dim / 64;
    int per = a.heads * groups;  // tg = ((is_v * heads) + h) * groups + g
    int is_v = int(tg) / per;
    int rem = int(tg) % per;
    int h = rem / groups, g = rem % groups;
    long pos = step[0];
    float x = (is_v ? vcur : kcur)[h * a.head_dim + g * 64 + tid];
    float am = simd_max(fabs(x));
    if (lane == 0) warp_max[sg] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = fmax(warp_max[0], warp_max[1]);
    ushort sbits = dq_f2h(amax / 127.0f);
    float sf = dq_h2f(sbits);
    int q = 0;
    if (sf > 0.0f) q = clamp(int(rint(x / sf)), -127, 127);
    long page = table[pos / a.page_size];
    if (page >= 0) {
        long row = (page * a.heads + h) * a.page_size + (pos % a.page_size);
        (is_v ? v_q : k_q)[row * a.head_dim + g * 64 + tid] = char(q);
        if (tid == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (a.has_host) {
        long row = ((pos / a.page_size) * a.heads + h) * a.page_size + (pos % a.page_size);
        (is_v ? hv_q : hk_q)[row * a.head_dim + g * 64 + tid] = char(q);
        if (tid == 0) (is_v ? hv_scale : hk_scale)[row * groups + g] = sbits;
    }
}

kernel void kv_gather_q8_k(device const char* k_q [[buffer(0)]], device const char* v_q [[buffer(1)]],
                           device const ushort* k_scale [[buffer(2)]], device const ushort* v_scale [[buffer(3)]],
                           device const int* table [[buffer(4)]], device const int* ids [[buffer(5)]],
                           device const int* step [[buffer(6)]], device ushort* k_scratch [[buffer(7)]],
                           device ushort* v_scratch [[buffer(8)]], constant KvQ8Args& a [[buffer(9)]],
                           uint gid [[thread_position_in_grid]]) {
    long n_ids = step[3];
    ulong total = ulong(n_ids) * a.heads * a.head_dim;
    if (gid >= total) return;
    int e = int(gid % uint(a.head_dim));
    int h = int((gid / uint(a.head_dim)) % uint(a.heads));
    long id = long(gid / uint(a.head_dim) / uint(a.heads));
    int cell = ids[id];
    long page = table[cell / a.page_size];
    long row = (page * a.heads + h) * a.page_size + (cell % a.page_size);
    int groups = a.head_dim / 64;
    float ks = dq_h2f(k_scale[row * groups + e / 64]);
    float vs = dq_h2f(v_scale[row * groups + e / 64]);
    k_scratch[gid] = dq_f2h(float(k_q[row * a.head_dim + e]) * ks);
    v_scratch[gid] = dq_f2h(float(v_q[row * a.head_dim + e]) * vs);
}

// ---- Q4_0 append/gather --------------------------------------------------------

// The q4_0 group quantizer shared by both appends: returns the block scale
// bits; `byte` receives this lane's packed nibble pair (lanes 0..15 valid).
inline ushort q4_group(float x, uint lane, thread uint8_t& byte) {
    float amax = fabs(x), mval = x;
    for (uint o = 16; o > 0; o >>= 1) {
        float a2 = simd_shuffle_xor(amax, o);
        float v2 = simd_shuffle_xor(mval, o);
        if (a2 > amax || (a2 == amax && v2 > mval)) {
            amax = a2;
            mval = v2;
        }
    }
    float d = mval / -8.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int q = int(x * id + 8.5f);  // trunc toward zero, as __float2int_rz
    uint qc = uint(clamp(q, 0, 15));
    uint qhi = simd_shuffle_down(qc, 16);
    byte = uint8_t(qc | (qhi << 4));
    (void) lane;
    return dq_f2h(d);
}

inline void q4_store(device uchar* pool, long row, int b, uint lane, ushort d, uint8_t byte) {
    device uchar* blk = pool + row * 144 + b * 18;
    if (lane == 0) {
        blk[0] = uchar(d & 0xFF);
        blk[1] = uchar(d >> 8);
    }
    if (lane < 16) blk[2 + lane] = byte;
}

struct KvQ4Args {
    int heads;
    int head_dim;
    int page_size;
    int has_host;
    int has_stage;
    long pos0;  // batch append only
};

// One warp per (K/V, head, 32-block); tg encodes the triple (batch: token first).
kernel void kv_append_q4_k(device uchar* k_q4 [[buffer(0)]], device uchar* v_q4 [[buffer(1)]],
                           device const int* table [[buffer(2)]], device const int* step [[buffer(3)]],
                           device const float* kcur [[buffer(4)]], device const float* vcur [[buffer(5)]],
                           device uchar* hk_q4 [[buffer(6)]], device uchar* hv_q4 [[buffer(7)]],
                           constant KvQ4Args& a [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                           uint lane [[thread_position_in_threadgroup]]) {
    int blocks = a.head_dim / 32;
    int per = a.heads * blocks;
    int is_v = int(tg) / per % 2;
    int rem = int(tg) % per;
    int h = rem / blocks, b = rem % blocks;
    long pos = step[0];
    float x = (is_v ? vcur : kcur)[h * a.head_dim + b * 32 + lane];
    uint8_t byte;
    ushort d = q4_group(x, lane, byte);
    long page = table[pos / a.page_size];
    if (page >= 0)
        q4_store(is_v ? v_q4 : k_q4, (page * a.heads + h) * a.page_size + (pos % a.page_size), b, lane, d, byte);
    if (a.has_host)
        q4_store(is_v ? hv_q4 : hk_q4, ((pos / a.page_size) * a.heads + h) * a.page_size + (pos % a.page_size),
                 b, lane, d, byte);
}

kernel void kv_append_q4_batch_k(device uchar* k_q4 [[buffer(0)]], device uchar* v_q4 [[buffer(1)]],
                                 device const int* table [[buffer(2)]], device const float* K [[buffer(3)]],
                                 device const float* V [[buffer(4)]], device uchar* hk_q4 [[buffer(5)]],
                                 device uchar* hv_q4 [[buffer(6)]], device uchar* sk_q4 [[buffer(7)]],
                                 device uchar* sv_q4 [[buffer(8)]], constant KvQ4Args& a [[buffer(9)]],
                                 uint tg [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    int blocks = a.head_dim / 32;
    int per = a.heads * blocks;
    int is_v = (int(tg) / per) % 2;
    long t = long(tg) / (2 * per);
    int rem = int(tg) % per;
    int h = rem / blocks, b = rem % blocks;
    long pos = a.pos0 + t;
    float x = (is_v ? V : K)[t * (a.heads * a.head_dim) + h * a.head_dim + b * 32 + lane];
    uint8_t byte;
    ushort d = q4_group(x, lane, byte);
    long page = table[pos / a.page_size];
    long row_id = ((pos / a.page_size) * a.heads + h) * a.page_size + (pos % a.page_size);
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * a.heads + h) * a.page_size + (pos % a.page_size), b, lane, d, byte);
    if (a.has_host) q4_store(is_v ? hv_q4 : hk_q4, row_id, b, lane, d, byte);
    if (a.has_stage) q4_store(is_v ? sv_q4 : sk_q4, row_id, b, lane, d, byte);
}

kernel void kv_gather_q4_k(device const uchar* k_q4 [[buffer(0)]], device const uchar* v_q4 [[buffer(1)]],
                           device const int* table [[buffer(2)]], device const int* ids [[buffer(3)]],
                           device const int* step [[buffer(4)]], device ushort* k_scratch [[buffer(5)]],
                           device ushort* v_scratch [[buffer(6)]], constant KvQ4Args& a [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    long n_ids = step[3];
    ulong total = ulong(n_ids) * a.heads * a.head_dim;
    if (gid >= total) return;
    int e = int(gid % uint(a.head_dim));
    int h = int((gid / uint(a.head_dim)) % uint(a.heads));
    long id = long(gid / uint(a.head_dim) / uint(a.heads));
    int cell = ids[id];
    long page = table[cell / a.page_size];
    long row = (page * a.heads + h) * a.page_size + (cell % a.page_size);
    int b = e / 32, j = e % 32;
    device const uchar* kb = k_q4 + row * (a.head_dim / 32 * 18) + b * 18;
    device const uchar* vb = v_q4 + row * (a.head_dim / 32 * 18) + b * 18;
    uint8_t kbyte = kb[2 + (j < 16 ? j : j - 16)];
    uint8_t vbyte = vb[2 + (j < 16 ? j : j - 16)];
    int kq = ((j < 16) ? (kbyte & 0xF) : (kbyte >> 4)) - 8;
    int vq = ((j < 16) ? (vbyte & 0xF) : (vbyte >> 4)) - 8;
    k_scratch[gid] = dq_f2h(float(kq) * dq_ldf16(kb));
    v_scratch[gid] = dq_f2h(float(vq) * dq_ldf16(vb));
}

// ---- KV streaming -------------------------------------------------------------

kernel void kv_reset_k(device int* page_table [[buffer(0)]], device int* slot_block [[buffer(1)]],
                       device int* slot_stamp [[buffer(2)]], device int* slot_ref [[buffer(3)]],
                       device int* ctl [[buffer(4)]], constant long& n_blocks [[buffer(5)]],
                       constant long& n_slots [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    if (gid < ulong(n_blocks)) page_table[gid] = -1;
    if (gid < ulong(n_slots)) {
        slot_block[gid] = -1;
        slot_stamp[gid] = -1;
        slot_ref[gid] = 0;
    }
    if (gid < 16) ctl[gid] = 0;
}

kernel void kv_ring_k(device int* table [[buffer(0)]], constant long& n_blocks [[buffer(1)]],
                      constant long& n_slots [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= ulong(n_blocks)) return;
    table[gid] = int(gid % ulong(n_slots));
}

// Copy the resolve's missed blocks from the host copy into their slots: up to
// four parallel byte runs (codes/scales planes), 16 bytes per thread step.
struct KvCopyArgs {
    int len[4];
    int n_runs;
    int n_tg;
};
kernel void kv_copy_k(device const uchar* s0 [[buffer(0)]], device uchar* d0 [[buffer(1)]],
                      device const uchar* s1 [[buffer(2)]], device uchar* d1 [[buffer(3)]],
                      device const uchar* s2 [[buffer(4)]], device uchar* d2 [[buffer(5)]],
                      device const uchar* s3 [[buffer(6)]], device uchar* d3 [[buffer(7)]],
                      device const int* miss_block [[buffer(8)]], device const int* miss_slot [[buffer(9)]],
                      device const int* ctl [[buffer(10)]], constant KvCopyArgs& a [[buffer(11)]],
                      uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    device const uchar* srcs[4] = {s0, s1, s2, s3};
    device uchar* dsts[4] = {d0, d1, d2, d3};
    int need = ctl[2];
    for (int k = int(tg); k < need; k += a.n_tg) {
        long b = miss_block[k], sl = miss_slot[k];
        for (int r = 0; r < a.n_runs; ++r) {
            device const uint4* src = (device const uint4*) (srcs[r] + b * a.len[r]);
            device uint4* dst = (device uint4*) (dsts[r] + sl * a.len[r]);
            for (int i = int(tid); i < a.len[r] / 16; i += 128) dst[i] = src[i];
        }
    }
}

// The resolve: ONE threadgroup of 1024, the CUDA kernel's three phases -
// claim misses with a compare-exchange on the page table, pick victims with
// a clock sweep driven by a block scan, re-point the table and the counters.
// Threadgroup atomics stand in for the CUDA shared atomics; the page table
// claim uses a device atomic, as the original's atomicCAS did.
struct KvResolveArgs {
    int n_q;
    int cap;
    int page_size;
    long n_slots;
};

inline int kv_block_scan(int v, threadgroup int* warp_sums, uint tid, uint lane, uint w) {
    int x = v;
    for (uint o = 1; o < 32; o <<= 1) {
        int y = simd_shuffle_up(x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0) {
        int t = warp_sums[lane];
        for (uint o = 1; o < 32; o <<= 1) {
            int y = simd_shuffle_up(t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return excl;  // the block total is warp_sums[31] after the call
}

kernel void kv_resolve_k(device atomic_int* page_table [[buffer(0)]], device int* slot_block [[buffer(1)]],
                         device int* slot_stamp [[buffer(2)]], device int* slot_ref [[buffer(3)]],
                         device int* ctl [[buffer(4)]], device int* miss_block [[buffer(5)]],
                         device int* miss_slot [[buffer(6)]], device const int* ids [[buffer(7)]],
                         device const int* steps [[buffer(8)]], constant KvResolveArgs& a [[buffer(9)]],
                         uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int warp_sums[32];
    threadgroup atomic_int s_nmiss;
    threadgroup atomic_int s_lookups_a;
    threadgroup int s_cut;
    if (tid == 0) {
        atomic_store_explicit(&s_nmiss, 0, memory_order_relaxed);
        atomic_store_explicit(&s_lookups_a, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int epoch = ctl[0] + 1;
    const uint w = tid >> 5;
    // phase 1: hits, claims
    int lookups = 0;
    for (int q = 0; q < a.n_q; ++q) {
        int width = steps[q * 4 + 3];
        device const int* qi = ids + long(q) * a.cap;
        for (int i = int(tid); i < width; i += 1024) {
            int b = qi[i] / a.page_size;
            if (i > 0 && qi[i - 1] / a.page_size == b) continue;
            ++lookups;
            int sl = atomic_load_explicit(&page_table[b], memory_order_relaxed);
            if (sl >= 0) {
                slot_stamp[sl] = epoch;
                slot_ref[sl] = 1;
            } else if (sl == -1) {
                int expected = -1;
                if (atomic_compare_exchange_weak_explicit(&page_table[b], &expected, -2, memory_order_relaxed,
                                                          memory_order_relaxed)) {
                    int idx = atomic_fetch_add_explicit(&s_nmiss, 1, memory_order_relaxed);
                    miss_block[idx] = b;
                }
            }
        }
    }
    if (lookups) atomic_fetch_add_explicit(&s_lookups_a, lookups, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int s_lookups = atomic_load_explicit(&s_lookups_a, memory_order_relaxed);
    const int need = atomic_load_explicit(&s_nmiss, memory_order_relaxed);
    const int n = int(a.n_slots);
    int hand = ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += 1024) {
        int j = int((long(hand) + tid) % n);
        bool mine = slot_stamp[j] == epoch;
        bool cand = !mine && (slot_block[j] < 0 || slot_ref[j] == 0);
        int rank = kv_block_scan(cand ? 1 : 0, warp_sums, tid, lane, w);
        int total = warp_sums[31];
        int want = need - got;
        if (tid == 0) s_cut = 1024;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cand && rank == want - 1) s_cut = int(tid) + 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int cut = s_cut;
        if (cand && rank < want) {
            miss_slot[got + rank] = j;
            slot_stamp[j] = epoch;
        } else if (int(tid) < cut && !mine) {
            slot_ref[j] = 0;
        }
        got += (total < want) ? total : want;
        hand = int((long(hand) + cut) % n);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int placed = (got < need) ? got : need;
    for (int k = int(tid); k < need; k += 1024) {
        int b = miss_block[k];
        if (k >= placed) {
            atomic_store_explicit(&page_table[b], -1, memory_order_relaxed);
            continue;
        }
        int sl = miss_slot[k];
        int old = slot_block[sl];
        if (old >= 0) atomic_store_explicit(&page_table[old], -1, memory_order_relaxed);
        slot_block[sl] = b;
        slot_stamp[sl] = epoch;
        slot_ref[sl] = 1;
        atomic_store_explicit(&page_table[b], sl, memory_order_relaxed);
    }
    if (tid == 0) {
        ctl[0] = epoch;
        ctl[1] = hand;
        ctl[2] = placed;
        if (placed < need) ctl[3] = 1;
        device ulong* c = (device ulong*) (ctl + 4);
        c[0] += ulong(placed);
        c[1] += ulong(s_lookups);
        c[2] += 1;
    }
}
