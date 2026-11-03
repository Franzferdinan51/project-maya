// tools/moe_parity/moe_ref.cpp - VM-side parity harness for the moe workstream.
//
// Scalar references transcribed from the CUDA sources (themselves the spec),
// plus float simulations of the MSL kernels' arithmetic (MSL has no double;
// serial sums in the kernels are compensated float in the same order). This
// measures the error the MSL rounding introduces BEFORE any Mac run, and
// dumps seeded fixtures (inputs + reference outputs) a Mac mini-runner can
// check the real kernels against (see README.md in this directory).
//
// Build: g++ -std=c++17 -O2 -o /tmp/moe_ref moe_ref.cpp && /tmp/moe_ref [fixture_dir]
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(20261102);
static float frand(float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng);
}

// ---------------- router_top10: CUDA semantics (double) ----------------
static void ref_router_top10(const float* l, int ne, int k, int* ids, float* w) {
    double mx = -INFINITY;
    for (int e = 0; e < ne; ++e) mx = std::max(mx, (double) l[e]);
    std::vector<double> ex(ne);
    double sum = 0;
    for (int e = 0; e < ne; ++e) {
        ex[e] = std::exp((double) l[e] - mx);
        sum += ex[e];
    }
    const float inv = (float) (1.0 / sum);
    std::vector<float> p(ne);
    for (int e = 0; e < ne; ++e) p[e] = (float) (ex[e] * inv);
    std::vector<char> taken(ne, 0);
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = ne;
        for (int e = 0; e < ne; ++e) {
            if (taken[e]) continue;
            if (p[e] > bv) {
                bv = p[e];
                bi = e;
            }
        }
        ids[i] = bi;
        w[i] = bv;
        taken[bi] = 1;
    }
    double s = 0;
    for (int i = 0; i < k; ++i) s += (double) w[i];
    const double sc = std::max(s, 6.103515625e-05);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) w[i] / sc);
}

// Float simulation of the MSL kernel (compensated serial sums, float exp).
static float kahan_add(float sum, float v, float& c) {
    const float y = v - c;
    const float t = sum + y;
    c = (t - sum) - y;
    return t;
}
static void sim_router_top10(const float* l, int ne, int k, int* ids, float* w) {
    float mx = -INFINITY;
    for (int e = 0; e < ne; ++e) mx = std::max(mx, l[e]);
    std::vector<float> ex(ne), p(ne);
    for (int e = 0; e < ne; ++e) ex[e] = std::exp(l[e] - mx);
    float sum = 0, c = 0;
    for (int e = 0; e < ne; ++e) sum = kahan_add(sum, ex[e], c);
    const float inv = 1.0f / sum;
    for (int e = 0; e < ne; ++e) p[e] = ex[e] * inv;
    std::vector<char> taken(ne, 0);
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = ne;
        for (int e = 0; e < ne; ++e) {
            if (taken[e]) continue;
            if (p[e] > bv) {
                bv = p[e];
                bi = e;
            }
        }
        ids[i] = bi;
        w[i] = bv;
        taken[bi] = 1;
    }
    float s = 0;
    c = 0;
    for (int i = 0; i < k; ++i) s = kahan_add(s, w[i], c);
    const float sc = std::max(s, 6.103515625e-5f);
    for (int i = 0; i < k; ++i) w[i] = w[i] / sc;
}

// ---------------- native_router: lane-exact float simulation ----------------
// Mirrors the CUDA/MSL structure operation for operation: lane holds
// lane+32*i, xor-butterfly max/sum, selection by warp argmax, selected sum
// accumulated by the winning lane.
static void sim_native_router(const float* l, int32_t* ids, float* w) {
    float v[32][16];
    for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 16; ++i) v[lane][i] = l[lane + 32 * i];
    auto butterfly = [](float x[32], bool is_max) {
        for (int m = 16; m > 0; m >>= 1) {
            float prev[32];
            std::memcpy(prev, x, sizeof prev);  // a shuffle reads every lane's pre-step value
            for (int lane = 0; lane < 32; ++lane) {
                const float o = prev[lane ^ m];
                x[lane] = is_max ? std::max(x[lane], o) : x[lane] + o;
            }
        }
    };
    float mx[32];
    for (int lane = 0; lane < 32; ++lane) {
        mx[lane] = -INFINITY;
        for (int i = 0; i < 16; ++i) mx[lane] = std::max(mx[lane], v[lane][i]);
    }
    butterfly(mx, true);
    float sum[32];
    for (int lane = 0; lane < 32; ++lane) {
        sum[lane] = 0;
        for (int i = 0; i < 16; ++i) {
            v[lane][i] = std::exp(v[lane][i] - mx[lane]);
            sum[lane] += v[lane][i];
        }
    }
    butterfly(sum, false);
    for (int lane = 0; lane < 32; ++lane) {
        const float rcp = 1.0f / sum[lane];
        for (int i = 0; i < 16; ++i) {
            v[lane][i] *= rcp;
            if (std::isnan(v[lane][i])) v[lane][i] = -FLT_MAX;
        }
    }
    float selected[32] = {}, ssum_lane[32] = {};
    for (int rank = 0; rank < 10; ++rank) {
        float best[32];
        int ex[32];
        for (int lane = 0; lane < 32; ++lane) {
            best[lane] = v[lane][0];
            ex[lane] = lane;
            for (int i = 1; i < 16; ++i)
                if (v[lane][i] > best[lane]) {
                    best[lane] = v[lane][i];
                    ex[lane] = lane + 32 * i;
                }
        }
        for (int m = 16; m > 0; m >>= 1) {
            float pb[32];
            int pe[32];
            std::memcpy(pb, best, sizeof pb);
            std::memcpy(pe, ex, sizeof pe);
            for (int lane = 0; lane < 32; ++lane) {
                const float ob = pb[lane ^ m];
                const int oe = pe[lane ^ m];
                if (ob > best[lane] || (ob == best[lane] && oe < ex[lane])) {
                    best[lane] = ob;
                    ex[lane] = oe;
                }
            }
        }
        const int winner = ex[0];
        ids[rank] = winner;
        ssum_lane[winner & 31] += best[0];
        v[winner & 31][winner / 32] = -INFINITY;
        selected[rank] = best[0];
    }
    butterfly(ssum_lane, false);
    const float inv = 1.0f / std::max(ssum_lane[0], 6.103515625e-5f);
    for (int lane = 0; lane < 10; ++lane) w[lane] = selected[lane] * inv;
}

// ---------------- router_sigmoid ----------------
static void ref_router_sigmoid(const float* l, const float* b, int ne, int k, float w_scale, bool norm_w,
                               int* ids, float* w) {
    std::vector<float> p(ne), sel(ne);
    for (int e = 0; e < ne; ++e) {
        p[e] = 1.0f / (1.0f + std::exp(-l[e]));
        sel[e] = b ? p[e] + b[e] : p[e];
    }
    for (int e = 0; e < ne; ++e) {
        int rank = 0;
        for (int f = 0; f < ne; ++f) {
            rank += sel[f] > sel[e] ? 1 : 0;
            rank += (f < e && sel[f] == sel[e]) ? 1 : 0;
        }
        if (rank < k) ids[rank] = e;
    }
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += (double) p[ids[i]];
    const float inv = (float) (norm_w ? 1.0 / std::max(sum, 6.103515625e-5) : 1.0);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) p[ids[i]] * inv * w_scale);
}

// Float simulation of the MSL sigmoid kernel: identical selection, compensated
// float renorm instead of the reference's double sum.
static void sim_router_sigmoid(const float* l, const float* b, int ne, int k, float w_scale, bool norm_w,
                               int* ids, float* w) {
    std::vector<float> p(ne), sel(ne);
    for (int e = 0; e < ne; ++e) {
        p[e] = 1.0f / (1.0f + std::exp(-l[e]));
        sel[e] = b ? p[e] + b[e] : p[e];
    }
    for (int e = 0; e < ne; ++e) {
        int rank = 0;
        for (int f = 0; f < ne; ++f) {
            rank += sel[f] > sel[e] ? 1 : 0;
            rank += (f < e && sel[f] == sel[e]) ? 1 : 0;
        }
        if (rank < k) ids[rank] = e;
    }
    float sum = 0, c = 0;
    for (int i = 0; i < k; ++i) sum = kahan_add(sum, p[ids[i]], c);
    const float inv = norm_w ? 1.0f / std::max(sum, 6.103515625e-5f) : 1.0f;
    for (int i = 0; i < k; ++i) w[i] = p[ids[i]] * inv * w_scale;
}

// ---------------- combines ----------------
static void ref_moe_combine(const float* parts, const float* w, const float* shared, float* y, int n, int k) {
    for (int j = 0; j < n; ++j) {
        double acc = 0;
        for (int e = 0; e < k; ++e) acc += (double) w[e] * (double) parts[(size_t) e * n + j];
        if (shared) acc += (double) shared[j];
        y[j] = (float) acc;
    }
}
static void sim_native_combine(const float* parts, const float* w, const float* shared, float* y, int n, int k) {
    for (int j = 0; j < n; ++j) {
        float sum = parts[j] * w[0];
        for (int e = 1; e < k; ++e) sum = std::fma(parts[(size_t) e * n + j], w[e], sum);
        if (shared) sum += shared[j];
        y[j] = sum;
    }
}

// ---------------- s2 expert (scalar, chunk-serial) ----------------
static float f16_at(const uint8_t* p) {
    const uint16_t bits = p[0] | (p[1] << 8);
    // fp16 -> fp32
    const int sign = (bits >> 15) & 1, exp = (bits >> 10) & 31, man = bits & 1023;
    float v;
    if (exp == 0) v = std::ldexp((float) man / 1024.0f, -14);
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = std::ldexp(1.0f + (float) man / 1024.0f, exp - 15);
    return sign ? -v : v;
}
static void f16_put(uint8_t* p, float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const int sign = (x >> 16) & 0x8000;
    int exp = (int) ((x >> 23) & 255) - 112;
    int man = x & 0x7fffff;
    uint16_t h;
    if (exp <= 0) h = (uint16_t) sign;
    else if (exp >= 31) h = (uint16_t) (sign | 0x7c00);
    else h = (uint16_t) (sign | (exp << 10) | (man >> 13));
    p[0] = h & 255;
    p[1] = h >> 8;
}
constexpr int S2H = 2560, S2FF = 640;
constexpr size_t O_D_CODES = 819200, O_GU_SCALES = 1228800, O_D_SCALES = 1331200, BLOB = O_D_SCALES + 2560 * 10 * 2;
static float s2_row(const uint8_t* blob, const uint8_t* codes, const uint8_t* scales, const uint8_t* x,
                    const float* xs, int n_chunks) {
    float acc = 0;
    for (int c = 0; c < n_chunks; ++c) {
        const uint8_t* cb = codes + c * 8;
        const uint8_t* xb = x + c * 34;
        const float dx = xs ? xs[c] : f16_at(xb);
        int s = 0, hx = 0;
        for (int j = 0; j < 8; ++j) {
            const unsigned byte = cb[j];
            const int cw[4] = {int(byte & 3), int((byte >> 2) & 3), int((byte >> 4) & 3), int((byte >> 6) & 3)};
            for (int t = 0; t < 4; ++t) {
                const int xv = (int8_t) xb[2 + 4 * j + t];
                s += cw[t] * xv;
                hx += xv;
            }
        }
        const float dw = f16_at(scales + (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

static void dump(const std::string& dir, const char* name, const void* data, size_t bytes) {
    if (dir.empty()) return;
    FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
    std::fwrite(data, 1, bytes, f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    double worst = 0;
    // ---- routers over seeded tokens, incl. an all-equal tie case and a wide-range case
    {
        const int T = 64, NE = 512, K = 10;
        std::vector<float> logits(T * NE);
        for (auto& v : logits) v = frand(-8, 8);
        for (int e = 0; e < NE; ++e) logits[60 * NE + e] = 1.25f;             // exact ties everywhere
        for (int e = 0; e < NE; ++e) logits[61 * NE + e] = frand(-30, 30);    // wide range
        int exact_t = 0, exact_n = 0, exact_s = 0;
        double werr_t = 0, werr_n = 0, werr_s = 0;
        for (int t = 0; t < T; ++t) {
            const float* l = logits.data() + t * NE;
            int ir[10], is[10];
            float wr[10], ws[10];
            ref_router_top10(l, NE, K, ir, wr);
            sim_router_top10(l, NE, K, is, ws);
            bool same = true;
            for (int i = 0; i < K; ++i) {
                same &= ir[i] == is[i];
                werr_t = std::max(werr_t, (double) std::fabs(wr[i] - ws[i]));
            }
            exact_t += same;
            int32_t in[10];
            float wn[10];
            sim_native_router(l, in, wn);
            same = true;
            for (int i = 0; i < K; ++i) {
                same &= ir[i] == (int) in[i];
                // native weights are float-softmax weights: compare against the sim generic float path
                werr_n = std::max(werr_n, (double) std::fabs(ws[i] - wn[i]));
            }
            exact_n += same;
            std::vector<float> bias(NE);
            for (auto& v : bias) v = frand(-0.05f, 0.05f);
            if (t == 60)
                for (auto& v : bias) v = 0.0f;
            int ig[10];
            float wg[10];
            ref_router_sigmoid(l, bias.data(), NE, K, 2.5f, true, ig, wg);
            // sigmoid sim = same float math; only the renorm sum differs (double vs compensated)
            int ig2[10];
            float wg2[10];
            sim_router_sigmoid(l, bias.data(), NE, K, 2.5f, true, ig2, wg2);
            same = true;
            for (int i = 0; i < K; ++i) {
                same &= ig[i] == ig2[i];
                werr_s = std::max(werr_s, (double) std::fabs(wg[i] - wg2[i]));
            }
            exact_s += same;
            if (t == 0) {
                dump(dir, "router_logits.bin", l, NE * 4);
                dump(dir, "router_top10_ids.bin", ir, K * 4);
                dump(dir, "router_top10_w.bin", wr, K * 4);
                dump(dir, "router_bias.bin", bias.data(), NE * 4);
                dump(dir, "router_sigmoid_ids.bin", ig, K * 4);
                dump(dir, "router_sigmoid_w.bin", wg, K * 4);
            }
        }
        std::printf("router_top10   ids exact %d/%d tokens, max |w_ref - w_sim| %.3e\n", exact_t, T, werr_t);
        std::printf("native_router  ids exact %d/%d tokens (vs double ref), max |w_sim_generic - w_native| %.3e\n",
                    exact_n, T, werr_n);
        std::printf("router_sigmoid ids exact %d/%d tokens, max |w_ref - w_sim| %.3e (incl. zero-bias tie token)\n",
                    exact_s, T, werr_s);
        worst = std::max(worst, werr_t);
    }
    // ---- combines
    {
        const int N = 2560, K = 10;
        std::vector<float> parts(K * N), w(K), shared(N), yr(N), ys(N);
        for (auto& v : parts) v = frand(-2, 2);
        for (auto& v : w) v = frand(0, 0.2f);
        for (auto& v : shared) v = frand(-1, 1);
        ref_moe_combine(parts.data(), w.data(), shared.data(), yr.data(), N, K);
        sim_native_combine(parts.data(), w.data(), shared.data(), ys.data(), N, K);
        double e = 0;
        for (int j = 0; j < N; ++j) e = std::max(e, (double) std::fabs(yr[j] - ys[j]));
        std::printf("combine        max |double_ref - fma_sim| %.3e over %d elements\n", e, N);
        dump(dir, "combine_parts.bin", parts.data(), parts.size() * 4);
        dump(dir, "combine_w.bin", w.data(), K * 4);
        dump(dir, "combine_shared.bin", shared.data(), N * 4);
        dump(dir, "combine_ref.bin", yr.data(), N * 4);
        worst = std::max(worst, e);
    }
    // ---- s2 single expert, full gu -> swiglu -> quantize -> down, scalar
    {
        std::vector<uint8_t> blob(BLOB);
        for (auto& v : blob) v = (uint8_t) (rng() & 255);
        // scales regions get small positive fp16 scales instead of random bits
        for (size_t o = O_GU_SCALES; o < O_GU_SCALES + (size_t) 2 * S2FF * 40 * 2; o += 2) f16_put(&blob[o], frand(0.005f, 0.05f));
        for (size_t o = O_D_SCALES; o < BLOB; o += 2) f16_put(&blob[o], frand(0.005f, 0.05f));
        std::vector<uint8_t> x((S2H / 32) * 34);
        for (int c = 0; c < S2H / 32; ++c) {
            f16_put(&x[c * 34], frand(0.01f, 0.1f));
            for (int j = 0; j < 32; ++j) x[c * 34 + 2 + j] = (uint8_t) (int8_t) frand(-64, 64);
        }
        std::vector<float> gu(2 * S2FF);
        for (int i = 0; i < 2 * S2FF; ++i)
            gu[(i & 1) ? (S2FF + (i >> 1)) : (i >> 1)] =
                s2_row(blob.data(), blob.data() + (size_t) i * 640, blob.data() + O_GU_SCALES + (size_t) i * 40 * 2,
                       x.data(), nullptr, S2H / 32);
        std::vector<float> h(S2FF);
        for (int r = 0; r < S2FF; ++r) {
            const float g = gu[r];
            h[r] = (float) ((g / (1.0 + std::exp(-(double) g))) * gu[S2FF + r]);
        }
        // quantize_q8_0 of h (ggml form: d = amax/127 in fp16, round-to-nearest-even-ish via std::nearbyint)
        std::vector<uint8_t> hq((S2FF / 32) * 34);
        for (int c = 0; c < S2FF / 32; ++c) {
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(h[c * 32 + j]));
            const float d = amax / 127.0f;
            uint8_t db[2];
            f16_put(db, d);
            const float dd = f16_at(db);
            hq[c * 34] = db[0];
            hq[c * 34 + 1] = db[1];
            for (int j = 0; j < 32; ++j) {
                int v = (int) std::nearbyint(h[c * 32 + j] / dd);
                v = std::max(-127, std::min(127, v));
                hq[c * 34 + 2 + j] = (uint8_t) (int8_t) v;
            }
        }
        std::vector<float> out(S2H);
        for (int r = 0; r < S2H; ++r)
            out[r] = s2_row(blob.data(), blob.data() + O_D_CODES + (size_t) r * 160,
                            blob.data() + O_D_SCALES + (size_t) r * 10 * 2, hq.data(), nullptr, S2FF / 32);
        double e = 0;
        for (float v : out) e = std::max(e, (double) std::fabs(v));
        std::printf("s2 expert      scalar pipeline ran, max |out| %.3f (fixture dumped for the Mac runner)\n", e);
        dump(dir, "s2_blob.bin", blob.data(), blob.size());
        dump(dir, "s2_x_q8_0.bin", x.data(), x.size());
        dump(dir, "s2_gate_up_ref.bin", gu.data(), gu.size() * 4);
        dump(dir, "s2_out_ref.bin", out.data(), out.size() * 4);
    }
    std::printf("worst float-vs-double delta seen: %.3e\n", worst);
    return 0;
}
