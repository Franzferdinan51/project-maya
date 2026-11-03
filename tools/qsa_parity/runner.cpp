// runner.cpp - parity checks for the qsa workstream's Metal kernels.
//
// Drives the REAL host launcher TUs (src/metal-qwen/stubs/*.cpp, qsa family)
// through the dispatch contract (launch_shim.mm) and compares against
// scalar references derived from the CUDA sources: double precision where
// the CUDA kernel computes in double (indexer norm, qsa_index dots, rope
// table), sequential float where it computes in float. Prints max abs/rel
// error per entry and exits non-zero on any breach of the stated bound.
//
// Build & run: see build_and_run.sh (Mac only - needs Metal).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/native_rope.hpp"

extern "C" void qsa_shim_init(const char* kernel_dir);
#include "strata/metal_compat/cuda_runtime.h"

using namespace strata::kernels;

static std::mt19937 rng(20261102);
static float frand(float lo = -1.0f, float hi = 1.0f) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}
static void* g_stream = (void*) (uintptr_t) 0x51;  // non-null dummy stream for native_* entries

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) { cudaMalloc((void**) &p, count * sizeof(T)); }
    void fill_random(float lo = -1.0f, float hi = 1.0f) {
        std::vector<T> h(n);
        for (auto& v : h) v = (T) frand(lo, hi);
        std::memcpy(p, h.data(), n * sizeof(T));
    }
    void set(const std::vector<T>& h) { std::memcpy(p, h.data(), n * sizeof(T)); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        std::memcpy(h.data(), p, n * sizeof(T));
        return h;
    }
    void zero() { std::memset(p, 0, n * sizeof(T)); }
};

static int g_failures = 0;
static void report(const char* name, const std::vector<float>& got, const std::vector<double>& ref,
                   double abs_bound) {
    double ma = 0, mr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) got[i] - ref[i]);
        ma = std::max(ma, d);
        mr = std::max(mr, d / std::max(1.0, std::fabs(ref[i])));
    }
    const bool ok = ma <= abs_bound;
    if (!ok) ++g_failures;
    std::printf("%-34s n=%5zu  max_abs=%.3g  max_rel=%.3g  bound=%.1g  %s\n", name, ref.size(), ma, mr,
                abs_bound, ok ? "OK" : "FAIL");
}
static void report_ints(const char* name, const std::vector<int32_t>& got, const std::vector<int32_t>& ref) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        if (got[i] != ref[i]) {
            if (!bad) first = i;
            ++bad;
        }
    if (bad) ++g_failures;
    std::printf("%-34s n=%5zu  mismatches=%zu (first at %zu)  %s\n", name, ref.size(), bad, first,
                bad ? "FAIL" : "OK");
}

// ---------------------------------------------------------------- rope ----
static void test_rope() {
    const int rows = 5, head_dim = 256, n_rot = 64, max_pos = 64;
    Dev<float> x(rows * head_dim), out(rows * head_dim), ct(max_pos * n_rot / 2), st(max_pos * n_rot / 2);
    x.fill_random();
    build_rope_table(n_rot, 1e7, max_pos, ct.p, st.p);
    std::vector<int> pos = {0, 3, 17, 40, 63};
    Dev<int> posd(rows);
    posd.set(pos);
    auto run_ref = [&](const std::vector<int>& pair_pos) {
        auto xh = x.get();
        std::vector<double> ref(rows * head_dim);
        for (int r = 0; r < rows; ++r) {
            for (int d = 0; d < head_dim; ++d) ref[r * head_dim + d] = xh[r * head_dim + d];
            for (int i = 0; i < n_rot / 2; ++i) {
                const double inv = std::pow(1e7, -2.0 * i / n_rot);
                const double ang = (double) pair_pos[r * (n_rot / 2) + i] * inv;
                const double c = std::cos(ang), s = std::sin(ang);
                const double a = xh[r * head_dim + i], b = xh[r * head_dim + n_rot / 2 + i];
                ref[r * head_dim + i] = a * c - b * s;
                ref[r * head_dim + n_rot / 2 + i] = a * s + b * c;
            }
        }
        return ref;
    };
    std::vector<int> pp(rows * (n_rot / 2));
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n_rot / 2; ++i) pp[r * (n_rot / 2) + i] = pos[r];
    rope_neox_apply(x.p, out.p, rows, head_dim, n_rot, ct.p, st.p, posd.p, nullptr);
    cudaDeviceSynchronize();
    report("rope_neox_apply", out.get(), run_ref(pp), 2e-6);

    // M-RoPE table: per-cell (t,h,w), pair i reads stream i%3.
    Dev<int32_t> tab(rows * 3);
    std::vector<int32_t> th(rows * 3);
    for (int r = 0; r < rows; ++r) {
        th[r * 3 + 0] = 5 + r;
        th[r * 3 + 1] = 20 + 2 * r;
        th[r * 3 + 2] = 40 + 3 * r;
    }
    tab.set(th);
    mrope_table_set(tab.p);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n_rot / 2; ++i) pp[r * (n_rot / 2) + i] = th[r * 3 + (i % 3)];
    std::vector<int> cells = {0, 1, 2, 3, 4};
    Dev<int> celld(rows);
    celld.set(cells);
    rope_neox_apply(x.p, out.p, rows, head_dim, n_rot, ct.p, st.p, celld.p, nullptr);
    cudaDeviceSynchronize();
    report("rope_neox_apply (mrope)", out.get(), run_ref(pp), 2e-6);
    mrope_table_set(nullptr);
}

static void test_native_rope() {
    for (int head_dim : {128, 256}) {
        const int rows = 4;
        Dev<float> x(rows * head_dim), out(rows * head_dim);
        x.fill_random();
        std::vector<int> pos = {0, 5, 33, 61};
        Dev<int> posd(rows);
        posd.set(pos);
        native_rope_apply(x.p, out.p, rows, head_dim, 64, 1e7f, posd.p, g_stream);
        cudaDeviceSynchronize();
        auto xh = x.get();
        std::vector<double> ref(rows * head_dim);
        const float ts = powf(1e7f, -2.0f / 64);
        for (int r = 0; r < rows; ++r) {
            for (int d = 0; d < head_dim; ++d) ref[r * head_dim + d] = xh[r * head_dim + d];
            for (int i = 0; i < 32; ++i) {
                const double theta = (double) ((float) pos[r] * powf(ts, (float) i));
                const double c = std::cos(theta), s = std::sin(theta);
                const double a = xh[r * head_dim + i], b = xh[r * head_dim + 32 + i];
                ref[r * head_dim + i] = a * c - b * s;
                ref[r * head_dim + 32 + i] = a * s + b * c;
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "native_rope_apply hd=%d", head_dim);
        report(name, out.get(), ref, 2e-5);
    }
}

// ------------------------------------------------------- native_qsa -------
static void test_native_qsa() {
    {
        const int cols = 128, rows = 3;
        Dev<float> x(rows * cols), g(cols), out(rows * cols);
        x.fill_random();
        g.fill_random(0.5f, 1.5f);
        native_qsa_rms_norm_weighted(x.p, g.p, out.p, cols, rows, 1e-6f, g_stream);
        cudaDeviceSynchronize();
        auto xh = x.get(), gh = g.get();
        std::vector<double> ref(rows * cols);
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int c = 0; c < cols; ++c) ss += (double) xh[r * cols + c] * xh[r * cols + c];
            const double scale = 1.0 / std::sqrt(ss / cols + 1e-6);
            for (int c = 0; c < cols; ++c) ref[r * cols + c] = scale * xh[r * cols + c] * gh[c];
        }
        report("native_qsa_rms_norm_weighted", out.get(), ref, 2e-5);
    }
    {
        const int nh = 24, hd = 256;
        Dev<float> attn(nh * hd), qf(nh * 2 * hd), out(nh * hd);
        attn.fill_random();
        qf.fill_random(-4, 4);
        native_qsa_gate_apply(attn.p, qf.p, out.p, nh, hd, g_stream);
        cudaDeviceSynchronize();
        auto ah = attn.get(), qh = qf.get();
        std::vector<double> ref(nh * hd);
        for (int i = 0; i < nh * hd; ++i) {
            const int h = i / hd, d = i % hd;
            const double g = qh[h * 2 * hd + hd + d];
            ref[i] = ah[i] / (1.0 + std::exp(-g));
        }
        report("native_qsa_gate_apply", out.get(), ref, 2e-6);
    }
}

// -------------------------------------------------- native_qsa_score ------
static void test_native_score() {
    QsaShapes s = qsa_real_shapes();
    const int64_t max_cells = 64, max_blocks = max_cells / 4 + 1;
    Dev<float> pooled(max_blocks * 128), query(4 * 128), bias(max_blocks), cells(max_cells);
    pooled.fill_random(-0.5f, 0.5f);
    query.fill_random(-0.5f, 0.5f);
    bias.fill_random(-0.25f, 0.25f);
    cells.zero();
    int32_t step_h[kStepCount];
    qsa_step_fill(step_h, 36, s);  // n=37, full=9, width=37
    Dev<int32_t> step(kStepCount);
    step.set(std::vector<int32_t>(step_h, step_h + kStepCount));
    native_qsa_score(pooled.p, query.p, bias.p, s, step.p, max_blocks, max_cells, cells.p, g_stream);
    cudaDeviceSynchronize();
    auto ph = pooled.get(), qh = query.get(), bh = bias.get();
    std::vector<double> ref(37);
    for (int r = 0; r <= 9; ++r) {
        float sum = 0.0f;  // the CUDA scalar path is float throughout
        for (int h = 0; h < 4; ++h) {
            float acc = 0.0f;
            for (int d = 0; d < 128; ++d) acc = fmaf(ph[r * 128 + d], qh[h * 128 + d], acc);
            sum += acc > 0 ? acc : 0;
        }
        sum += bh[r];
        if (r == 9) sum += 1e9f;
        for (int i = r * 4; i < 37 && i < (r + 1) * 4; ++i) ref[i] = sum;
    }
    auto got = cells.get();
    got.resize(37);
    report("native_qsa_score", got, ref, 1e-3);
}

// --------------------------------------------------------- qsa_select -----
static uint32_t ref_order_key(float v0) {
    const float v = v0 + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
static std::vector<int32_t> ref_block_topk(const float* sc, int64_t n_kv, int64_t n_bid, int64_t width,
                                           int64_t max_blocks) {
    std::vector<int32_t> out;
    if (n_kv <= width) {
        for (int64_t j = 0; j < n_kv; ++j) out.push_back((int32_t) j);
        return out;
    }
    auto weight = [&](int64_t b) { return b < n_bid ? 4 : (int) (n_kv - n_bid * 4); };
    // largest key whose weighted >= count reaches width (scan distinct keys)
    std::vector<uint32_t> keys;
    for (int64_t b = 0; b <= n_bid; ++b)
        if (weight(b)) keys.push_back(ref_order_key(sc[b]));
    std::sort(keys.begin(), keys.end(), std::greater<uint32_t>());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    uint32_t thr = 0;
    int64_t above = 0;
    for (uint32_t k : keys) {
        int64_t cnt = 0;
        for (int64_t b = 0; b <= n_bid; ++b)
            if (weight(b) && ref_order_key(sc[b]) >= k) cnt += weight(b);
        if (cnt >= width) {
            thr = k;
            above = 0;
            for (int64_t b = 0; b <= n_bid; ++b)
                if (weight(b) && ref_order_key(sc[b]) > k) above += weight(b);
            break;
        }
    }
    const int64_t budget = width - above;
    int64_t used = 0;
    for (int64_t b = 0; b <= n_bid && (int64_t) out.size() < width; ++b) {
        const int w = weight(b);
        if (!w) continue;
        const uint32_t k = ref_order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out.push_back((int32_t) (b * 4 + c));
        } else if (k == thr) {
            for (int c = 0; c < w && used < budget; ++c, ++used) out.push_back((int32_t) (b * 4 + c));
        }
    }
    return out;
}

static void test_select() {
    QsaShapes s = qsa_real_shapes();
    const int64_t nq = 2, max_blocks = 17;
    Dev<float> pooled(max_blocks * 128), dead(128), qidx(nq * 4 * 128), scores(nq * max_blocks);
    pooled.fill_random(-0.5f, 0.5f);
    dead.fill_random(-0.5f, 0.5f);
    qidx.fill_random(-0.5f, 0.5f);
    Dev<int32_t> steps(nq * kStepCount);
    std::vector<int32_t> sh(nq * kStepCount);
    qsa_step_fill(sh.data(), 36, s);
    qsa_step_fill(sh.data() + kStepCount, 200, s);  // n=201, n_bid=50 > max_blocks on purpose? no: cap reach
    // keep the second query inside max_blocks: pos 60 -> n_bid 15
    qsa_step_fill(sh.data() + kStepCount, 60, s);
    steps.set(sh);
    qsa_block_scores(pooled.p, dead.p, qidx.p, steps.p, nq, max_blocks, s, scores.p, nullptr);
    cudaDeviceSynchronize();
    auto ph = pooled.get(), dh = dead.get(), qh = qidx.get();
    std::vector<double> ref(nq * max_blocks, 0.0);
    const int poss[2] = {36, 60};
    for (int qi = 0; qi < nq; ++qi) {
        const int n_kv = poss[qi] + 1, n_bid = n_kv / 4;
        for (int b = 0; b <= n_bid; ++b) {
            const float* key = b == n_bid ? dh.data() : &ph[b * 128];
            float score = 0.0f;  // float throughout, as in the CUDA kernel
            for (int h = 0; h < 4; ++h) {
                float acc = 0.0f;
                for (int d = 0; d < 128; ++d) acc = fmaf(key[d], qh[qi * 512 + h * 128 + d], acc);
                score += acc > 0 ? acc : 0;
            }
            if (b == n_bid && n_kv % 4) score += 1e9f;
            ref[qi * max_blocks + b] = score;
        }
    }
    report("qsa_block_scores", scores.get(), ref, 1e-3);

    // topk over those block scores with a small width: use steps whose
    // width forces real selection (pos 200 -> width 201 over 51 blocks)
    const int64_t max_blocks2 = 52, cap = qsa_selection_width(kTopkMaxCells, s);
    Dev<float> scores2(max_blocks2);
    Dev<int32_t> ids(nq * cap);
    std::vector<float> sc(max_blocks2);
    for (auto& v : sc) v = frand(-2, 2);
    sc[3] = sc[7] = sc[11] = 0.5f;  // exact ties at plausible thresholds
    scores2.set(sc);
    Dev<int32_t> steps2(kStepCount);
    int32_t st2[kStepCount];
    qsa_step_fill(st2, 200, s);
    steps2.set(std::vector<int32_t>(st2, st2 + kStepCount));
    // single-query launches through both entries
    qsa_block_topk_ref(scores2.p, steps2.p, 1, max_blocks2, cap, s, ids.p, nullptr);
    cudaDeviceSynchronize();
    auto want = ref_block_topk(sc.data(), 201, 50, 201 - 0 /* width below */, max_blocks2);
    // width for pos 200: min(201, 2051) = 201 == n_kv -> identity; use pos 3000 for a real cut
    (void) want;
    qsa_step_fill(st2, 3000, s);  // n=3001, n_bid=750 - too many for max_blocks2; use scores sized 751
    Dev<float> scores3(751);
    std::vector<float> sc3(751);
    for (auto& v : sc3) v = frand(-2, 2);
    sc3[5] = sc3[9] = 1.25f;
    scores3.set(sc3);
    steps2.set(std::vector<int32_t>(st2, st2 + kStepCount));
    Dev<int32_t> ids3(cap);
    qsa_block_topk(scores3.p, steps2.p, 1, 751, cap, s, ids3.p, nullptr);
    cudaDeviceSynchronize();
    auto want3 = ref_block_topk(sc3.data(), 3001, 750, 2051, 751);
    auto got3 = ids3.get();
    got3.resize(want3.size());
    report_ints("qsa_block_topk", got3, want3);
}

// ------------------------------------------------------ indexer ---------
// Double-precision simulation of the CUDA append state machine.
struct IndexerSim {
    std::vector<double> tail = std::vector<double>(3 * 128, 0.0);
    std::vector<double> dead = std::vector<double>(128, 0.0);
    std::vector<double> pooled = std::vector<double>(64 * 128, 0.0);
    std::vector<float> gamma;
    int32_t block_pos = -1;
    double eps = 1e-6;
    void append(const float* raw_f, int pos, int pos_base) {
        std::vector<double> in(128);
        for (int d = 0; d < 128; ++d) in[d] = (double) f32_from_f16(f16_from_f32(raw_f[d]));
        const int slot = pos % 4;
        if (slot < 3)
            for (int d = 0; d < 128; ++d) tail[slot * 128 + d] = in[d];
        if (pos != 0 && slot != 3) return;
        std::vector<double> mean(128);
        for (int d = 0; d < 128; ++d) {
            double sum = pos == 0 ? in[d] : tail[d];
            for (int j = 1; j < 4; ++j) sum += (pos == 0 || j == 3) ? in[d] : tail[j * 128 + d];
            mean[d] = 0.25 * sum;
        }
        double ss = 0;
        for (int d = 0; d < 128; ++d) ss += mean[d] * mean[d];
        const double scale = 1.0 / std::sqrt(ss / 128 + eps);
        const int b = pos / 4;
        const int rope_pos = pos == 0 ? 0 : pos_base + 4 * b;
        std::vector<double> val(128);
        for (int d = 0; d < 128; ++d) val[d] = scale * mean[d] * gamma[d];
        const float ts = powf(1e7f, -2.0f / 64);
        for (int d = 0; d < 128; ++d) {
            double y = val[d];
            if (d < 64) {
                const int pair = d % 32;
                const double theta = pos == 0 ? 0.0 : (double) ((float) rope_pos * powf(ts, (float) pair));
                const double c = std::cos(theta), sn = std::sin(theta);
                y = d < 32 ? val[pair] * c - val[pair + 32] * sn : val[pair] * sn + val[pair + 32] * c;
            }
            pooled[b * 128 + d] = y;
        }
        if (pos == 0) {
            for (int d = 0; d < 128; ++d) dead[d] = pooled[d];
        } else {
            for (int d = 0; d < 128; ++d) pooled[(b + 1) * 128 + d] = dead[d];
            block_pos = rope_pos;
        }
    }
};

static void test_indexer() {
    QsaShapes s = qsa_real_shapes();
    const int cells = 9;
    std::vector<float> raw(cells * 128), gamma(128);
    for (auto& v : raw) v = frand(-0.7f, 0.7f);
    for (auto& v : gamma) v = frand(0.5f, 1.5f);
    IndexerSim sim;
    sim.gamma.assign(gamma.begin(), gamma.end());
    for (int pos = 0; pos < cells; ++pos) sim.append(&raw[pos * 128], pos, 0);

    // sequential appends through the device entry
    Dev<float> tail(3 * 128), dead(128), pooled(64 * 128), gammad(128);
    Dev<int32_t> block_pos(1), relpos(1);
    gammad.set(gamma);
    tail.zero();
    dead.zero();
    pooled.zero();
    QsaIndexerBuffers b{tail.p, dead.p, pooled.p, block_pos.p};
    Dev<float> rawd(cells * 128);
    rawd.set(raw);
    for (int pos = 0; pos < cells; ++pos) {
        relpos.set({pos});
        native_qsa_indexer_append(rawd.p + pos * 128, relpos.p, 0, gammad.p, 1e-6f, b, s, 64, 1e7f, g_stream);
        cudaDeviceSynchronize();  // the next host write to relpos must not race this launch
    }
    std::vector<double> refp(pooled.n);
    for (size_t i = 0; i < pooled.n; ++i) refp[i] = sim.pooled[i];
    report("native_indexer_append pooled", pooled.get(), refp, 1e-4);
    report("native_indexer_append dead", dead.get(),
           std::vector<double>(sim.dead.begin(), sim.dead.end()), 1e-4);
    report("native_indexer_append tail", tail.get(),
           std::vector<double>(sim.tail.begin(), sim.tail.end()), 1e-6);
    if (block_pos.get()[0] != sim.block_pos) {
        ++g_failures;
        std::printf("%-34s got %d want %d  FAIL\n", "native_indexer block_pos", block_pos.get()[0],
                    sim.block_pos);
    } else {
        std::printf("%-34s value %d  OK\n", "native_indexer block_pos", sim.block_pos);
    }

    // batch append over the same cells from a fresh state
    Dev<float> tail2(3 * 128), dead2(128), pooled2(64 * 128);
    Dev<int32_t> block_pos2(1);
    tail2.zero();
    dead2.zero();
    pooled2.zero();
    QsaIndexerBuffers b2{tail2.p, dead2.p, pooled2.p, block_pos2.p};
    native_qsa_indexer_append_batch(rawd.p, cells, 0, 0, gammad.p, 1e-6f, b2, s, 64, 1e7f, g_stream);
    cudaDeviceSynchronize();
    report("native_indexer_batch pooled", pooled2.get(), refp, 1e-4);
    report("native_indexer_batch tail", tail2.get(), std::vector<double>(sim.tail.begin(), sim.tail.end()),
           1e-6);
}

// ------------------------------------------- full qsa decode chain ------
static void test_chain() {
    QsaShapes s = qsa_real_shapes();
    const int64_t cap = 2051, max_blocks = 17, cells = 64;
    const int pages = 16;  // page_size 4
    Dev<int32_t> page_table(pages);
    std::vector<int32_t> pt(pages);
    for (int i = 0; i < pages; ++i) pt[i] = i;
    page_table.set(pt);
    const int rows = pages * 2 * 4;  // [page][kv_head][page_size] rows of head_dim
    Dev<uint16_t> kpool(rows * 256), vpool(rows * 256);
    kpool.zero();
    vpool.zero();
    Dev<float> ct(64 * 32), st(64 * 32);
    build_rope_table(64, 1e7, 64, ct.p, st.p);
    Dev<float> tail(3 * 128), dead(128), pooled(64 * 128), wnorm(128);
    tail.zero();
    dead.zero();
    pooled.zero();
    wnorm.fill_random(0.5f, 1.5f);
    Dev<int32_t> block_pos(1), stepd(kStepCount);
    std::vector<float> kcur(7 * 2 * 256), vcur(7 * 2 * 256), rawk(7 * 128), qfull(24 * 256), qidx(4 * 128);
    for (auto& v : kcur) v = frand(-0.6f, 0.6f);
    for (auto& v : vcur) v = frand(-0.6f, 0.6f);
    for (auto& v : rawk) v = frand(-0.6f, 0.6f);
    for (auto& v : qfull) v = frand(-0.6f, 0.6f);
    for (auto& v : qidx) v = frand(-0.6f, 0.6f);
    Dev<float> kcurd(7 * 2 * 256), vcurd(7 * 2 * 256), rawkd(7 * 128), qfulld(24 * 256), qidxd(4 * 128);
    kcurd.set(kcur);
    vcurd.set(vcur);
    rawkd.set(rawk);
    qfulld.set(qfull);
    qidxd.set(qidx);

    // reference state
    std::vector<uint16_t> rk(rows * 256, 0), rv(rows * 256, 0);
    auto wnorm_h = wnorm.get();
    // scalar model of indexer_key_append (double throughout, table rope)
    std::vector<double> mtail(3 * 128, 0.0), mdead(128, 0.0), mpooled(64 * 128, 0.0);
    auto ct_h = ct.get(), st_h = st.get();
    auto model_append = [&](const float* raw, int p) {
        const int slot = p % 4;
        if (slot < 3)
            for (int d = 0; d < 128; ++d) mtail[slot * 128 + d] = raw[d];
        if (p == 0) {
            double ss = 0;
            for (int i = 0; i < 128; ++i) ss += (double) raw[i] * raw[i];
            const double inv = 1.0 / std::sqrt(ss / 128 + (double) qsa_rms_eps());
            for (int d = 0; d < 128; ++d) mdead[d] = (double) raw[d] * inv * wnorm_h[d];
            for (int d = 0; d < 128; ++d) mpooled[d] = mdead[d];
        }
        if (slot != 3) return;
        std::vector<double> mean(128);
        for (int d = 0; d < 128; ++d) {
            double m = 0;
            for (int j = 0; j < 3; ++j) m += mtail[j * 128 + d];
            mean[d] = (m + raw[d]) / 4;
        }
        double ss = 0;
        for (int i = 0; i < 128; ++i) ss += mean[i] * mean[i];
        const double inv = 1.0 / std::sqrt(ss / 128 + (double) qsa_rms_eps());
        const int b = p / 4;
        for (int d = 0; d < 128; ++d) mpooled[b * 128 + d] = mean[d] * inv * wnorm_h[d];
        for (int d = 0; d < 128; ++d) mpooled[(b + 1) * 128 + d] = mdead[d];
        for (int i = 0; i < 32; ++i) {
            const double c = ct_h[(size_t) (b * 4) * 32 + i], sn = st_h[(size_t) (b * 4) * 32 + i];
            const double a = mpooled[b * 128 + i], bb = mpooled[b * 128 + 32 + i];
            mpooled[b * 128 + i] = a * c - bb * sn;
            mpooled[b * 128 + 32 + i] = a * sn + bb * c;
        }
    };
    const int pos = 6;
    for (int p = 0; p <= pos; ++p) {
        int32_t sh[kStepCount];
        qsa_step_fill(sh, p, s);
        stepd.set(std::vector<int32_t>(sh, sh + kStepCount));
        kv_append_step(kpool.p, vpool.p, page_table.p, stepd.p, kcurd.p + p * 2 * 256, vcurd.p + p * 2 * 256, s,
                       nullptr);
        for (int h = 0; h < 2; ++h)
            for (int d = 0; d < 256; ++d) {
                const int row = ((p / 4) * 2 + h) * 4 + (p % 4);
                rk[row * 256 + d] = f16_from_f32(kcur[p * 512 + h * 256 + d]);
                rv[row * 256 + d] = f16_from_f32(vcur[p * 512 + h * 256 + d]);
            }
        Dev<int32_t> posd(1);
        posd.set({p});
        QsaIndexerBuffers ib{tail.p, dead.p, pooled.p, block_pos.p};
        indexer_key_append(rawkd.p + p * 128, posd.p, 0, wnorm.p, qsa_rms_eps(), ib, s, ct.p, st.p, nullptr);
        model_append(&rawk[p * 128], p);
    }
    cudaDeviceSynchronize();
    report("indexer_key_append pooled", pooled.get(), mpooled, 1e-4);
    report("indexer_key_append dead", dead.get(), mdead, 1e-4);
    report("indexer_key_append tail", tail.get(), mtail, 1e-6);
    {
        auto gotk = kpool.get(), gotv = vpool.get();
        size_t bad = 0;
        for (size_t i = 0; i < rk.size(); ++i) bad += (gotk[i] != rk[i]) + (gotv[i] != rv[i]);
        if (bad) ++g_failures;
        std::printf("%-34s mismatches=%zu  %s\n", "kv_append_step pools (bit-exact)", bad, bad ? "FAIL" : "OK");
    }
    // index + topk + gather + attend at pos 6 (n_kv 7, n_bid 1, width 7)
    Dev<float> cell_scores(cells), attn(24 * 256), weights(24 * cap);
    Dev<int32_t> ids(cap);
    qsa_index_step(pooled.p, qidxd.p, nullptr, s, stepd.p, max_blocks, cell_scores.p, nullptr);
    topk_512_step(cell_scores.p, s, cap, stepd.p, ids.p, nullptr);
    Dev<uint16_t> kscr(cap * 2 * 256), vscr(cap * 2 * 256);
    kv_gather_step(kpool.p, vpool.p, page_table.p, ids.p, stepd.p, cap, s, kscr.p, vscr.p, nullptr);
    qsa_attend_step(qfulld.p, kscr.p, vscr.p, stepd.p, cap, s, attn.p, weights.p, nullptr);
    cudaDeviceSynchronize();
    // reference index scores from the DEVICE pooled rows (the append model
    // is checked separately in test_indexer), double dots as on CUDA
    auto pooled_h = pooled.get();
    std::vector<double> ref_scores(7);
    for (int b = 0; b <= 1; ++b) {
        double score = 0;
        for (int h = 0; h < 4; ++h) {
            double acc = 0;
            for (int d = 0; d < 128; ++d) acc += (double) pooled_h[b * 128 + d] * qidx[h * 128 + d];
            score += acc > 0 ? acc : 0;
        }
        float scf = (float) score;   // the CUDA kernel's double -> float store
        if (b == 1) scf += 1e9f;     // incomplete tail (7 % 4 != 0), added in float
        for (int j = b * 4; j < 7 && j < (b + 1) * 4; ++j) ref_scores[j] = scf;
    }
    auto got_scores = cell_scores.get();
    got_scores.resize(7);
    report("qsa_index_step cell_scores", got_scores, ref_scores, 1e-3);
    auto got_ids = ids.get();
    got_ids.resize(7);
    report_ints("topk_512_step ids (identity)", got_ids, {0, 1, 2, 3, 4, 5, 6});
    // reference attention over the gathered cells 0..6, float-sequential
    std::vector<double> ref_attn(24 * 256);
    {
        std::vector<double> p(7);
        for (int h = 0; h < 24; ++h) {
            const int kv = h / 12;
            double mx = -1e30;
            std::vector<double> sc(7);
            for (int j = 0; j < 7; ++j) {
                const int row = ((j / 4) * 2 + kv) * 4 + (j % 4);
                double acc = 0;
                for (int d = 0; d < 256; ++d) acc += (double) f32_from_f16(rk[row * 256 + d]) * qfull[h * 256 + d];
                sc[j] = acc / 16.0;
                mx = std::max(mx, sc[j]);
            }
            double sum = 0;
            for (int j = 0; j < 7; ++j) {
                p[j] = std::exp(sc[j] - mx);
                sum += p[j];
            }
            for (int d = 0; d < 256; ++d) {
                double acc = 0;
                for (int j = 0; j < 7; ++j) {
                    const int row = ((j / 4) * 2 + kv) * 4 + (j % 4);
                    acc += (p[j] / sum) * (double) f32_from_f16(rv[row * 256 + d]);
                }
                ref_attn[h * 256 + d] = acc;
            }
        }
    }
    report("qsa_attend_step attn", attn.get(), ref_attn, 2e-5);
}

// --------------------------------------------- decode/prompt attention --
struct AttnFixture {
    QsaShapes s = qsa_real_shapes();
    int64_t cap = 100;
    int pages = 25;
    Dev<int32_t> page_table{25};
    Dev<uint16_t> kpool{25 * 2 * 4 * 256}, vpool{25 * 2 * 4 * 256};
    Dev<int8_t> kq{25 * 2 * 4 * 256}, vq{25 * 2 * 4 * 256};
    Dev<uint16_t> kscale{25 * 2 * 4 * 4}, vscale{25 * 2 * 4 * 4};
    Dev<float> q{2 * 24 * 256};
    Dev<int32_t> ids{2 * 100}, steps{2 * kStepCount};
    std::vector<int32_t> ids_h{std::vector<int32_t>(200)};
    AttnFixture() {
        std::vector<int32_t> pt(pages);
        for (int i = 0; i < pages; ++i) pt[i] = i;
        page_table.set(pt);
        kpool.fill_random();
        vpool.fill_random();
        std::vector<int8_t> codes(25 * 2 * 4 * 256);
        for (auto& c : codes) c = (int8_t) (frand(-1, 1) * 100);
        kq.set(codes);
        for (auto& c : codes) c = (int8_t) (frand(-1, 1) * 100);
        vq.set(codes);
        kscale.fill_random(0.005f, 0.02f);
        vscale.fill_random(0.005f, 0.02f);
        q.fill_random(-0.5f, 0.5f);
        for (auto& id : ids_h) id = (int32_t) (frand(0, 1) * 99.99f);
        ids.set(ids_h);
        std::vector<int32_t> sh(2 * kStepCount);
        qsa_step_fill(sh.data(), 39, s);           // width 40
        qsa_step_fill(sh.data() + kStepCount, 20, s);  // width 21
        steps.set(sh);
    }
    // reference attention for query qi in the given mode, double math
    std::vector<double> ref(int qi, int mode) const {
        const int widths[2] = {40, 21};
        const int width = widths[qi];
        auto kh = kpool.get(), vh = vpool.get();
        auto kqh = kq.get(), vqh = vq.get();
        auto ksh = kscale.get(), vsh = vscale.get();
        auto qh = q.get();
        auto kval = [&](int row, int d) -> double {
            if (mode == 0) return f32_from_f16(kh[row * 256 + d]);
            return (double) kqh[row * 256 + d] * f32_from_f16(ksh[row * 4 + d / 64]);
        };
        auto vval = [&](int row, int d) -> double {
            if (mode == 0) return f32_from_f16(vh[row * 256 + d]);
            return (double) vqh[row * 256 + d] * f32_from_f16(vsh[row * 4 + d / 64]);
        };
        std::vector<double> out(24 * 256);
        for (int h = 0; h < 24; ++h) {
            const int kv = h / 12;
            std::vector<double> sc(width);
            double mx = -1e30;
            for (int j = 0; j < width; ++j) {
                const int cell = ids_h[qi * 100 + j];
                const int row = ((cell / 4) * 2 + kv) * 4 + (cell % 4);
                double acc = 0;
                for (int d = 0; d < 256; ++d) acc += kval(row, d) * qh[qi * 24 * 256 + h * 256 + d];
                sc[j] = acc / 16.0;
                mx = std::max(mx, sc[j]);
            }
            double sum = 0;
            std::vector<double> p(width);
            for (int j = 0; j < width; ++j) {
                p[j] = std::exp(sc[j] - mx);
                sum += p[j];
            }
            for (int d = 0; d < 256; ++d) {
                double acc = 0;
                for (int j = 0; j < width; ++j) {
                    const int cell = ids_h[qi * 100 + j];
                    const int row = ((cell / 4) * 2 + kv) * 4 + (cell % 4);
                    acc += (p[j] / sum) * vval(row, d);
                }
                out[h * 256 + d] = acc;
            }
        }
        return out;
    }
};

static void test_attn() {
    AttnFixture f;
    QsaShapes& s = f.s;
    const int64_t stride = (int64_t) qsa_decode_attn_scratch_floats(f.cap, s);
    Dev<float> scratch(2 * stride), attn(2 * 24 * 256);
    {
        QsaAttnPools pools;
        pools.k_pool = f.kpool.p;
        pools.v_pool = f.vpool.p;
        pools.page_table = f.page_table.p;
        qsa_decode_attn_batch(f.q.p, pools, f.ids.p, f.steps.p, f.cap, s, scratch.p, attn.p, 2, nullptr);
        cudaDeviceSynchronize();
        auto got = attn.get();
        std::vector<float> g0(got.begin(), got.begin() + 24 * 256), g1(got.begin() + 24 * 256, got.end());
        report("decode_attn_batch f16 q0", g0, f.ref(0, 0), 2e-4);
        report("decode_attn_batch f16 q1", g1, f.ref(1, 0), 2e-4);
        // prompt entry over the same fixture
        Dev<float> attn2(2 * 24 * 256);
        const bool ok = qsa_prompt_attn_batch(f.q.p, pools, f.ids.p, f.steps.p, f.cap, s, attn2.p, 2, nullptr);
        cudaDeviceSynchronize();
        if (!ok) {
            ++g_failures;
            std::printf("%-34s returned false  FAIL\n", "qsa_prompt_attn_batch f16");
        } else {
            auto got2 = attn2.get();
            std::vector<float> h0(got2.begin(), got2.begin() + 24 * 256);
            report("prompt_attn_batch f16 q0", h0, f.ref(0, 0), 2e-4);
        }
    }
    {
        QsaAttnPools pools;
        pools.k_q = f.kq.p;
        pools.v_q = f.vq.p;
        pools.k_scale = f.kscale.p;
        pools.v_scale = f.vscale.p;
        pools.page_table = f.page_table.p;
        Dev<float> attn3(24 * 256);
        qsa_decode_attn_step(f.q.p, pools, f.ids.p, f.steps.p, f.cap, s, scratch.p, attn3.p, nullptr);
        cudaDeviceSynchronize();
        report("decode_attn_step int8 q0", attn3.get(), f.ref(0, 1), 2e-4);
    }
}

// ---------------------------------------------------------- flash -------
static void test_flash() {
    QsaShapes s = qsa_real_shapes();
    const int64_t cap = 256;
    Dev<float> q(24 * 256), out(24 * 256);
    Dev<uint16_t> k(cap * 2 * 256), v(cap * 2 * 256), mask(256);
    Dev<int32_t> step(kStepCount), status(1);
    q.fill_random(-0.5f, 0.5f);
    k.fill_random();
    v.fill_random();
    std::vector<uint16_t> mh(256);
    for (auto& m : mh) m = f16_from_f32(frand(-0.5f, 0.5f));
    mh[3] = 0xfc00;  // -inf
    mask.set(mh);
    int32_t sh[kStepCount];
    qsa_step_fill(sh, 36, s);  // width 37
    step.set(std::vector<int32_t>(sh, sh + kStepCount));
    native_flash_attn_short_step(q.p, k.p, v.p, step.p, cap, 256, s, out.p, status.p, mask.p, g_stream);
    cudaDeviceSynchronize();
    if (status.get()[0] != 0) {
        ++g_failures;
        std::printf("%-34s status %d want 0  FAIL\n", "flash status (valid step)", status.get()[0]);
    }
    auto qh = q.get();
    auto kh = k.get(), vh = v.get();
    std::vector<double> ref(24 * 256);
    for (int h = 0; h < 24; ++h) {
        const int kv = h / 12;
        std::vector<double> sc(37);
        double mx = -1e30;
        for (int c = 0; c < 37; ++c) {
            double acc = 0;
            for (int d = 0; d < 256; ++d)
                acc += (double) qh[h * 256 + d] * 0.0625 * f32_from_f16(kh[(c * 2 + kv) * 256 + d]);
            sc[c] = acc + (double) f32_from_f16(mh[c]);
            if (sc[c] > mx) mx = sc[c];
        }
        double sum = 0;
        std::vector<double> p(37);
        for (int c = 0; c < 37; ++c) {
            p[c] = std::exp(sc[c] - mx);
            sum += p[c];
        }
        for (int d = 0; d < 256; ++d) {
            double acc = 0;
            for (int c = 0; c < 37; ++c) acc += (p[c] / sum) * f32_from_f16(vh[(c * 2 + kv) * 256 + d]);
            ref[h * 256 + d] = acc;
        }
    }
    report("native_flash_attn_short_step", out.get(), ref, 2e-4);
    // invalid step: width 300 > max_context
    int32_t bad[kStepCount] = {299, 300, 75, 300};
    step.set(std::vector<int32_t>(bad, bad + kStepCount));
    native_flash_attn_short_step(q.p, k.p, v.p, step.p, cap, 256, s, out.p, status.p, mask.p, g_stream);
    cudaDeviceSynchronize();
    auto og = out.get();
    const bool nan_ok = std::isnan(og[0]) && std::isnan(og[24 * 256 - 1]);
    if (status.get()[0] != 1 || !nan_ok) {
        ++g_failures;
        std::printf("%-34s status %d nan %d  FAIL\n", "flash invalid step", status.get()[0], (int) nan_ok);
    } else {
        std::printf("%-34s status 1 + NaN outputs  OK\n", "flash invalid step");
    }
}

// ---------------- KV pool family ----------------
// Scalar refs transcribed from kv_q8.cu / kv_q4.cu / kv_stream.cu.

static void report_bad(const char* name, long bad) {
    if (bad) ++g_failures;
    std::printf("%-34s mismatches=%ld  %s\n", name, bad, bad ? "FAIL" : "OK");
}

static void test_fwht() {
    const int rows = 3;
    Dev<float> in(rows * 256), out(rows * 256);
    in.fill_random();
    fwht256_cuda(in.p, out.p, rows, nullptr);
    cudaDeviceSynchronize();
    auto ih = in.get();
    std::vector<double> ref(rows * 256);
    for (int r = 0; r < rows; ++r) {
        double a[256];
        for (int i = 0; i < 256; ++i) a[i] = ih[r * 256 + i] * (1.0 / 16.0);
        for (int len = 1; len < 256; len *= 2)
            for (int i = 0; i < 256; i += 2 * len)
                for (int j = 0; j < len; ++j) {
                    double u = a[i + j], v = a[i + j + len];
                    a[i + j] = u + v; a[i + j + len] = u - v;
                }
        for (int i = 0; i < 256; ++i) ref[r * 256 + i] = a[i];
    }
    report("fwht256_cuda", out.get(), ref, 2e-4);
}

static void test_kv_q8() {
    QsaShapes s = qsa_real_shapes();  // kv heads 2, head_dim 256, page 4
    const int64_t pages = 4, rows = pages * s.n_head_kv * s.page_size;
    const int64_t codes = rows * s.head_dim, scales = rows * (s.head_dim / 64);
    Dev<int8_t> kq(codes), vq(codes), hkq(codes), hvq(codes);
    Dev<uint16_t> ks(scales), vs(scales), hks(scales), hvs(scales);
    kq.zero(); vq.zero(); hkq.zero(); hvq.zero(); ks.zero(); vs.zero(); hks.zero(); hvs.zero();
    const std::vector<int32_t> table = {2, 0, 3, 1};
    Dev<int32_t> dtab(4), dstep(kStepCount);
    dtab.set(table);
    std::vector<int32_t> stepv(kStepCount, 0);
    stepv[kStepPos] = 13;
    dstep.set(stepv);
    Dev<float> dkc(s.n_head_kv * s.head_dim), dvc(s.n_head_kv * s.head_dim);
    dkc.fill_random(); dvc.fill_random();
    auto kc = dkc.get(), vc = dvc.get();
    KvHostPools host{}; host.k_q = hkq.p; host.v_q = hvq.p; host.k_scale = hks.p; host.v_scale = hvs.p;
    kv_append_q8_step(kq.p, vq.p, ks.p, vs.p, dtab.p, dstep.p, dkc.p, dvc.p, s, nullptr, &host);
    cudaDeviceSynchronize();
    std::vector<int8_t> ek(codes, 0), ev(codes, 0), ehk(codes, 0), ehv(codes, 0);
    std::vector<uint16_t> eks(scales, 0), evs(scales, 0), ehks(scales, 0), ehvs(scales, 0);
    auto ref_append = [&](const std::vector<float>& cur, int8_t* pool, uint16_t* sc, int8_t* hpool, uint16_t* hsc) {
        const long pos = 13, page = table[pos / s.page_size];
        for (int64_t h = 0; h < s.n_head_kv; ++h)
            for (int g = 0; g < s.head_dim / 64; ++g) {
                float amax = 0;
                for (int t = 0; t < 64; ++t) amax = std::max(amax, std::fabs(cur[h * s.head_dim + g * 64 + t]));
                const uint16_t sb = f16_from_f32(amax / 127.0f);
                const float sf = f32_from_f16(sb);
                const long row = (page * s.n_head_kv + h) * s.page_size + pos % s.page_size;
                const long hrow = ((pos / s.page_size) * s.n_head_kv + h) * s.page_size + pos % s.page_size;
                for (int t = 0; t < 64; ++t) {
                    int q = 0;
                    if (sf > 0) { q = (int) std::rint(cur[h * s.head_dim + g * 64 + t] / sf); q = std::max(-127, std::min(127, q)); }
                    pool[row * s.head_dim + g * 64 + t] = (int8_t) q;
                    hpool[hrow * s.head_dim + g * 64 + t] = (int8_t) q;
                }
                sc[row * (s.head_dim / 64) + g] = sb;
                hsc[hrow * (s.head_dim / 64) + g] = sb;
            }
    };
    ref_append(kc, ek.data(), eks.data(), ehk.data(), ehks.data());
    ref_append(vc, ev.data(), evs.data(), ehv.data(), ehvs.data());
    auto gk = kq.get(), gv = vq.get(), ghk = hkq.get(), ghv = hvq.get();
    auto gks = ks.get(), gvs = vs.get(), ghks = hks.get(), ghvs = hvs.get();
    long bad = 0;
    for (size_t i = 0; i < (size_t) codes; ++i) bad += (gk[i] != ek[i]) + (gv[i] != ev[i]) + (ghk[i] != ehk[i]) + (ghv[i] != ehv[i]);
    for (size_t i = 0; i < (size_t) scales; ++i) bad += (gks[i] != eks[i]) + (gvs[i] != evs[i]) + (ghks[i] != ehks[i]) + (ghvs[i] != ehvs[i]);
    report_bad("kv_append_q8_step", bad);
    Dev<int32_t> dids(3), dst2(kStepCount);
    dids.set({13, 0, 15});
    std::vector<int32_t> st2(kStepCount, 0);
    st2[kStepWidth] = 3;
    dst2.set(st2);
    const int64_t scr = 3 * s.n_head_kv * s.head_dim;
    Dev<uint16_t> dks(scr), dvs(scr);
    kv_gather_q8_step(kq.p, vq.p, ks.p, vs.p, dtab.p, dids.p, dst2.p, 3, s, dks.p, dvs.p, nullptr);
    cudaDeviceSynchronize();
    auto gkscr = dks.get(), gvscr = dvs.get();
    const std::vector<int32_t> idsv = {13, 0, 15};
    bad = 0;
    for (int id = 0; id < 3; ++id) {
        const long cell = idsv[id], page = table[cell / s.page_size];
        for (int64_t h = 0; h < s.n_head_kv; ++h) {
            const long row = (page * s.n_head_kv + h) * s.page_size + cell % s.page_size;
            for (int64_t e = 0; e < s.head_dim; ++e) {
                const float ksf = f32_from_f16(eks[row * (s.head_dim / 64) + e / 64]);
                const float vsf = f32_from_f16(evs[row * (s.head_dim / 64) + e / 64]);
                const size_t o = (id * s.n_head_kv + h) * s.head_dim + e;
                bad += gkscr[o] != f16_from_f32((float) ek[row * s.head_dim + e] * ksf);
                bad += gvscr[o] != f16_from_f32((float) ev[row * s.head_dim + e] * vsf);
            }
        }
    }
    report_bad("kv_gather_q8_step", bad);
}

static void ref_q4_block(const float* x32, uint8_t* blk) {
    float amax = std::fabs(x32[0]), mval = x32[0];
    for (int i = 1; i < 32; ++i) {
        const float a2 = std::fabs(x32[i]), v2 = x32[i];
        if (a2 > amax || (a2 == amax && v2 > mval)) { amax = a2; mval = v2; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0 ? 1.0f / d : 0.0f;
    const uint16_t db = f16_from_f32(d);
    blk[0] = (uint8_t) (db & 0xFF); blk[1] = (uint8_t) (db >> 8);
    for (int j = 0; j < 16; ++j) {
        auto q = [&](int i) { int v = (int) (x32[i] * id + 8.5f); return (uint8_t) std::max(0, std::min(15, v)); };
        blk[2 + j] = (uint8_t) (q(j) | (q(j + 16) << 4));
    }
}

static void test_kv_q4() {
    QsaShapes s = qsa_real_shapes();
    const int64_t pages = 4, rows = pages * s.n_head_kv * s.page_size;
    const int64_t bytes = rows * 144;
    Dev<uint8_t> k4(bytes), v4(bytes), hk4(bytes), hv4(bytes), sk4(bytes), sv4(bytes);
    k4.zero(); v4.zero(); hk4.zero(); hv4.zero(); sk4.zero(); sv4.zero();
    const std::vector<int32_t> table = {2, 0, 3, 1};
    Dev<int32_t> dtab(4), dstep(kStepCount);
    dtab.set(table);
    std::vector<int32_t> stepv(kStepCount, 0);
    stepv[kStepPos] = 13;
    dstep.set(stepv);
    Dev<float> dkc(s.n_head_kv * s.head_dim), dvc(s.n_head_kv * s.head_dim);
    dkc.fill_random(); dvc.fill_random();
    auto kc = dkc.get(), vc = dvc.get();
    KvHostPools host{}; host.k_q4 = hk4.p; host.v_q4 = hv4.p;
    kv_append_q4_step(k4.p, v4.p, dtab.p, dstep.p, dkc.p, dvc.p, s, nullptr, &host);
    cudaDeviceSynchronize();
    std::vector<uint8_t> ek(bytes, 0), ev(bytes, 0), ehk(bytes, 0), ehv(bytes, 0);
    auto ref_app = [&](const std::vector<float>& cur, uint8_t* pool, uint8_t* hpool, long pos) {
        const long page = table[pos / s.page_size];
        for (int64_t h = 0; h < s.n_head_kv; ++h)
            for (int b = 0; b < 8; ++b) {
                const long row = (page * s.n_head_kv + h) * s.page_size + pos % s.page_size;
                const long hrow = ((pos / s.page_size) * s.n_head_kv + h) * s.page_size + pos % s.page_size;
                ref_q4_block(&cur[h * s.head_dim + b * 32], pool + row * 144 + b * 18);
                ref_q4_block(&cur[h * s.head_dim + b * 32], hpool + hrow * 144 + b * 18);
            }
    };
    ref_app(kc, ek.data(), ehk.data(), 13); ref_app(vc, ev.data(), ehv.data(), 13);
    auto gk = k4.get(), gv = v4.get(), ghk = hk4.get(), ghv = hv4.get();
    long bad = 0;
    for (size_t i = 0; i < (size_t) bytes; ++i) bad += (gk[i] != ek[i]) + (gv[i] != ev[i]) + (ghk[i] != ehk[i]) + (ghv[i] != ehv[i]);
    report_bad("kv_append_q4_step", bad);
    Dev<float> dK(2 * kc.size()), dV(2 * vc.size());
    dK.fill_random(); dV.fill_random();
    auto K = dK.get(), V = dV.get();
    KvHostPools stage{}; stage.k_q4 = sk4.p; stage.v_q4 = sv4.p;
    kv_append_q4(k4.p, v4.p, dtab.p, 8, 2, dK.p, dV.p, s, nullptr, &host, &stage);
    cudaDeviceSynchronize();
    auto gsk = sk4.get(), gsv = sv4.get();
    bad = 0;
    for (int t = 0; t < 2; ++t) {
        const long pos = 8 + t;
        for (int64_t h = 0; h < s.n_head_kv; ++h)
            for (int b = 0; b < 8; ++b) {
                uint8_t blk[18];
                const long irow = ((pos / s.page_size) * s.n_head_kv + h) * s.page_size + pos % s.page_size;
                ref_q4_block(&K[t * kc.size() + h * s.head_dim + b * 32], blk);
                for (int i = 0; i < 18; ++i) bad += gsk[irow * 144 + b * 18 + i] != blk[i];
                ref_q4_block(&V[t * vc.size() + h * s.head_dim + b * 32], blk);
                for (int i = 0; i < 18; ++i) bad += gsv[irow * 144 + b * 18 + i] != blk[i];
            }
    }
    report_bad("kv_append_q4 batch+stage", bad);
    Dev<int32_t> dids(2), dst2(kStepCount);
    dids.set({13, 9});
    std::vector<int32_t> st2(kStepCount, 0);
    st2[kStepWidth] = 2;
    dst2.set(st2);
    const int64_t scr = 2 * s.n_head_kv * s.head_dim;
    Dev<uint16_t> dks(scr), dvs(scr);
    kv_gather_q4_step(k4.p, v4.p, dtab.p, dids.p, dst2.p, 2, s, dks.p, dvs.p, nullptr);
    cudaDeviceSynchronize();
    auto gkscr = dks.get(), gvscr = dvs.get();
    gk = k4.get(); gv = v4.get();  // pools after both appends
    const std::vector<int32_t> idsv = {13, 9};
    bad = 0;
    for (int id = 0; id < 2; ++id) {
        const long cell = idsv[id], page = table[cell / s.page_size];
        for (int64_t h = 0; h < s.n_head_kv; ++h) {
            const long row = (page * s.n_head_kv + h) * s.page_size + cell % s.page_size;
            for (int64_t e = 0; e < s.head_dim; ++e) {
                auto deq = [&](const std::vector<uint8_t>& pool) {
                    const uint8_t* blk = &pool[row * 144 + (e / 32) * 18];
                    const float d = f32_from_f16((uint16_t) (blk[0] | (blk[1] << 8)));
                    const int j = (int) (e % 32);
                    const uint8_t byte = blk[2 + (j < 16 ? j : j - 16)];
                    const int q = ((j < 16) ? (byte & 0xF) : (byte >> 4)) - 8;
                    return f16_from_f32((float) q * d);
                };
                const size_t o = (id * s.n_head_kv + h) * s.head_dim + e;
                bad += gkscr[o] != deq(gk);
                bad += gvscr[o] != deq(gv);
            }
        }
    }
    report_bad("kv_gather_q4_step", bad);
}

static void test_kv_stream() {
    QsaShapes s = qsa_real_shapes();  // page 4, kv heads 2, head_dim 256
    const int64_t n_blocks = 8, n_slots = 4;
    KvStreamMap m{};
    Dev<int32_t> pt(n_blocks), sb(n_slots), ss(n_slots), sr(n_slots), ctl(kKvCtlInts), mb(n_slots), ms(n_slots);
    m.page_table = pt.p; m.slot_block = sb.p; m.slot_stamp = ss.p; m.slot_ref = sr.p;
    m.ctl = ctl.p; m.miss_block = mb.p; m.miss_slot = ms.p;
    m.n_blocks = n_blocks; m.n_slots = n_slots;
    kv_stream_reset(m, nullptr);
    cudaDeviceSynchronize();
    long bad = 0;
    for (auto v : pt.get()) bad += v != -1;
    for (auto v : sb.get()) bad += v != -1;
    for (auto v : ss.get()) bad += v != -1;
    for (auto v : sr.get()) bad += v != 0;
    for (auto v : ctl.get()) bad += v != 0;
    report_bad("kv_stream_reset", bad);
    bad = kv_block_bytes(s, kKvF16) != (uint64_t) (s.n_head_kv * s.page_size * s.head_dim * 2 * 2) ? 1 : 0;
    bad += kv_block_bytes(s, kKvQ4) != (uint64_t) (s.n_head_kv * s.page_size * 144 * 2) ? 1 : 0;
    report_bad("kv_block_bytes", bad);
    const int64_t rows = s.n_head_kv * s.page_size, bb = rows * s.head_dim * 2;  // block bytes per plane
    Dev<uint16_t> dhk(n_blocks * bb / 2), dhv(n_blocks * bb / 2), dsk(n_slots * bb / 2), dsv(n_slots * bb / 2);
    {
        std::vector<uint16_t> hk(n_blocks * bb / 2), hv(n_blocks * bb / 2);
        for (size_t i = 0; i < hk.size(); ++i) { hk[i] = (uint16_t) (i * 7 + 1); hv[i] = (uint16_t) (i * 13 + 5); }
        dhk.set(hk); dhv.set(hv);
    }
    dsk.zero(); dsv.zero();
    QsaAttnPools slots{}; slots.k_pool = dsk.p; slots.v_pool = dsv.p;
    KvHostPools host{}; host.k_pool = dhk.p; host.v_pool = dhv.p;
    std::vector<int32_t> ptv(n_blocks, -1); ptv[2] = 1;
    std::vector<int32_t> sbv(n_slots, -1); sbv[1] = 2;
    pt.set(ptv); sb.set(sbv);
    Dev<int32_t> dids(8), dsteps(kStepCount);
    std::vector<int32_t> ids(8);
    for (int i = 0; i < 4; ++i) { ids[i] = 8 + i; ids[4 + i] = 20 + i; }  // block 2 hit, block 5 miss
    dids.set(ids);
    std::vector<int32_t> steps(kStepCount, 0);
    steps[kStepWidth] = 8;
    dsteps.set(steps);
    kv_stream_resolve(m, slots, host, kKvF16, dids.p, dsteps.p, 1, 8, s, nullptr);
    cudaDeviceSynchronize();
    auto ptg = pt.get(), sbg = sb.get(), ssg = ss.get(), srg = sr.get(), ctlg = ctl.get();
    bad = 0;
    bad += ptg[5] != 0;
    bad += sbg[0] != 5;
    bad += ssg[1] != 1;
    bad += srg[1] != 1;
    bad += ctlg[0] != 1 || ctlg[2] != 1;
    report_bad("kv_stream_resolve state", bad);
    const KvStreamCounters c = kv_stream_counters(m);
    bad = (c.misses != 1) + (c.lookups != 2) + (c.calls != 1) + (c.overflow ? 1 : 0);
    report_bad("kv_stream_counters", bad);
    auto gsk = dsk.get(), hk = dhk.get();
    bad = 0;
    for (int64_t i = 0; i < bb / 2; ++i) bad += gsk[i] != hk[5 * (bb / 2) + i];
    report_bad("kv_stream_resolve copy", bad);
    Dev<uint16_t> dstg(n_blocks * bb / 2), dstgv(n_blocks * bb / 2);
    QsaAttnPools stage{}; stage.k_pool = dstg.p; stage.v_pool = dstgv.p;
    kv_stage_from_host(stage, host, kKvF16, 3, s, nullptr);
    cudaDeviceSynchronize();
    auto gst = dstg.get();
    bad = 0;
    for (int64_t i = 0; i < 3 * (bb / 2); ++i) bad += gst[i] != hk[i];
    report_bad("kv_stage_from_host", bad);
    kv_ring_table(pt.p, n_blocks, n_slots, nullptr);
    cudaDeviceSynchronize();
    ptg = pt.get();
    bad = 0;
    for (int64_t i = 0; i < n_blocks; ++i) bad += ptg[i] != (int32_t) (i % n_slots);
    report_bad("kv_ring_table", bad);
    Dev<uint16_t> drk(n_slots * bb / 2), drv(n_slots * bb / 2);
    drk.zero(); drv.zero();
    QsaAttnPools rslots{}; rslots.k_pool = drk.p; rslots.v_pool = drv.p;
    kv_ring_restore(rslots, host, kKvF16, 5, 8, n_slots, s, nullptr);
    cudaDeviceSynchronize();
    auto grk = drk.get();
    bad = 0;
    for (int b = 5; b < 8; ++b)
        for (int64_t i = 0; i < bb / 2; ++i) bad += grk[(b % n_slots) * (bb / 2) + i] != hk[b * (bb / 2) + i];
    report_bad("kv_ring_restore", bad);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qsa_parity_runner <kernels-dir>\n");
        return 2;
    }
    qsa_shim_init(argv[1]);
    test_rope();
    test_native_rope();
    test_native_qsa();
    test_native_score();
    test_select();
    test_indexer();
    test_chain();
    test_attn();
    test_flash();
    test_fwht();
    test_kv_q8();
    test_kv_q4();
    test_kv_stream();
    std::printf("%s (%d failures)\n", g_failures ? "PARITY FAIL" : "PARITY PASS", g_failures);
    return g_failures ? 1 : 0;
}
