// tools/gdn_gr_scalar_ref.cpp - VM-side parity harness for the GDN/GR Metal
// workstream. Implements every entry twice from the CUDA sources:
//   ref: double precision, straightforward order (ground truth)
//   emu: float, in the exact accumulation order the MSL kernels use
// Seeded inputs; prints max abs/rel error of emu vs ref per entry.
// Build: g++ -O2 -std=c++17 tools/gdn_gr_scalar_ref.cpp -o /tmp/gdn_ref && /tmp/gdn_ref
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using std::vector;
static std::mt19937 rng(12345);
static float frand(float s = 1.0f) { return std::uniform_real_distribution<float>(-s, s)(rng); }
static uint16_t to_bf16(float f) { uint32_t u; __builtin_memcpy(&u, &f, 4); uint32_t l = (u >> 16) & 1; return uint16_t((u + 0x7FFF + l) >> 16); }
static float from_bf16(uint16_t h) { uint32_t u = uint32_t(h) << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
struct Err { double a = 0, r = 0; void add(double e, double ref) { a = std::max(a, std::fabs(e)); r = std::max(r, std::fabs(e) / (std::fabs(ref) + 1e-6)); } void print(const char* n) { std::printf("%-28s max_abs=%.3e max_rel=%.3e\n", n, a, r); } };

int main() {
    const int S = 128, HK = 16, HV = 48;
    // ---- gdn_step / native_gdn_step (native adds 1/sqrt(S) readout scale) ----
    {
        vector<double> st(HV * S * S), stf;
        vector<float> q(HK * S), k(HK * S), v(HV * S), gate(HV), beta(HV);
        for (auto& x : st) x = frand(0.3f);
        for (auto& x : q) x = frand(0.2f); for (auto& x : k) x = frand(0.2f);
        for (auto& x : v) x = frand(); for (auto& x : gate) x = frand(0.5f); for (auto& x : beta) x = 1.0f / (1.0f + std::exp(-frand()));
        vector<double> o_ref(HV * S); vector<float> st_e(st.begin(), st.end()), o_e(HV * S);
        for (int h = 0; h < HV; ++h) { int hk = h % HK; double dec = std::exp((double)gate[h]);
            for (int j = 0; j < S; ++j) { double sk = 0; for (int i = 0; i < S; ++i) { double& s = st[(i * HV + h) * S + j]; s *= dec; sk += s * k[hk * S + i]; }
                double d = (v[h * S + j] - sk) * beta[h]; double dot = 0;
                for (int i = 0; i < S; ++i) { double& s = st[(i * HV + h) * S + j]; s += k[hk * S + i] * d; dot += s * q[hk * S + i]; }
                o_ref[h * S + j] = dot; } }
        for (int h = 0; h < HV; ++h) { int hk = h % HK; float dec = std::exp(gate[h]);
            for (int j = 0; j < S; ++j) { float sk = 0; for (int i = 0; i < S; ++i) { float& s = st_e[(i * HV + h) * S + j]; s *= dec; sk += s * k[hk * S + i]; }
                float d = (v[h * S + j] - sk) * beta[h]; float dot = 0;
                for (int i = 0; i < S; ++i) { float& s = st_e[(i * HV + h) * S + j]; s += k[hk * S + i] * d; dot += s * q[hk * S + i]; }
                o_e[h * S + j] = dot; } }
        Err e, es; for (int i = 0; i < HV * S; ++i) e.add(o_e[i] - o_ref[i], o_ref[i]);
        for (size_t i = 0; i < st_e.size(); ++i) es.add(st_e[i] - st[i], st[i]);
        e.print("gdn_step out"); es.print("gdn_step state");
    }
    // ---- conv + silu (native & gdn_conv share taps; gdn has no silu) ----
    {
        const int C = 10240; vector<float> hist(C * 3), x(C), w(C * 4);
        for (auto& a : hist) a = frand(0.5f); for (auto& a : x) a = frand(); for (auto& a : w) a = frand(0.5f);
        Err e; for (int c = 0; c < C; ++c) { double acc = 0; for (int t = 0; t < 3; ++t) acc += (double)hist[c * 3 + t] * w[c * 4 + t]; acc += (double)x[c] * w[c * 4 + 3];
            float accf = hist[c*3]*w[c*4] + hist[c*3+1]*w[c*4+1] + hist[c*3+2]*w[c*4+2] + x[c]*w[c*4+3];
            e.add(accf - acc, acc); }
        e.print("conv_step/native_conv raw");
    }
    // ---- l2 norms ----
    {
        Err e1, e2;
        for (int row = 0; row < 34; ++row) { vector<float> x(S); for (auto& a : x) a = frand();
            double ss = 0; for (auto a : x) ss += (double)a * a;
            double inv = 1.0 / std::sqrt(ss + 1e-6); float ssf = 0; for (auto a : x) ssf += a * a; float invf = 1.0f / std::sqrt(ssf + 1e-6f);
            for (int i = 0; i < S; ++i) e1.add(x[i] * invf - x[i] * inv, x[i] * inv);
            // native: rsqrt(sum/128 + eps/128) * x * (1/sqrt(128))
            double inv2 = 1.0 / std::sqrt(ss / S + 1e-6 / S) / std::sqrt((double)S);
            float inv2f = (1.0f / std::sqrt(ssf / S + 1e-6f / S)) * (1.0f / std::sqrt((float)S));
            for (int i = 0; i < S; ++i) e2.add(x[i] * inv2f - x[i] * inv2, x[i] * inv2); }
        e1.print("gdn_l2_norm"); e2.print("native_gdn_l2_norm");
    }
    // ---- gate / beta / out_norm (elementwise+norm) ----
    {
        Err eg, eb, eo;
        for (int h = 0; h < HV; ++h) { float a = frand(2.0f), dt = frand(0.5f), sa = frand();
            double v = (double)a + dt; double sp = v > 20 ? v : std::log1p(std::exp(v));
            float vf = a + dt; float spf = vf > 20 ? vf : std::log(1.0 + std::exp((double)vf));
            eg.add(spf * sa - sp * sa, sp * sa);
            double sg = 1.0 / (1.0 + std::exp(-(double)a)); float sgf = 1.0f / (1.0f + std::exp(-a));
            eb.add(sgf - sg, sg); }
        eg.print("native_gdn_gate"); eb.print("beta_gate (both)");
        for (int h = 0; h < HV; ++h) { vector<float> o(S), z(S), g(S); for (int i = 0; i < S; ++i) { o[i] = frand(); z[i] = frand(); g[i] = frand(); }
            double ss = 0; for (auto a : o) ss += (double)a * a; double inv = 1.0 / std::sqrt(ss / S + 1e-6);
            float ssf = 0; for (auto a : o) ssf += a * a; float invf = 1.0f / std::sqrt(ssf / S + 1e-6f);
            for (int i = 0; i < S; ++i) { double r = o[i] * inv * g[i] / (1.0 + std::exp(-(double)z[i]));
                float rf = o[i] * invf * g[i] * (1.0f / (1.0f + std::exp(-z[i]))); eo.add(rf - r, r); } }
        eo.print("out_norm (gdn/native/fused)");
    }
    // ---- fused_gdn_ab (bf16 GEMV rows) ----
    {
        const int N = 2560; vector<float> x(N); vector<uint16_t> w(N);
        for (auto& a : x) a = frand(0.3f); for (auto& a : w) a = to_bf16(frand(0.1f));
        double acc = 0; for (int i = 0; i < N; ++i) acc += (double)x[i] * from_bf16(w[i]);
        float accf = 0; for (int i = 0; i < N; ++i) accf += x[i] * from_bf16(w[i]);
        Err e; e.add(accf - acc, acc); e.print("fused_gdn_ab row dot");
    }
    // ---- GR: norm, down/up/inject GEMVs, mean, write, postops ----
    {
        const int N = 2560, HC = 4, LR = 320, D = N * HC;
        vector<float> R(D), wn(D); for (auto& a : R) a = frand(0.5f); for (auto& a : wn) a = frand();
        vector<double> xn_ref(D); vector<float> xn_e(D);
        for (int c = 0; c < HC; ++c) { double ss = 0; for (int i = 0; i < N; ++i) ss += (double)R[c * N + i] * R[c * N + i];
            double rs = 1.0 / std::sqrt(ss / N + 1e-6); float ssf = 0; for (int i = 0; i < N; ++i) ssf += R[c * N + i] * R[c * N + i];
            float rsf = 1.0f / std::sqrt(ssf / N + 1e-6f);
            for (int i = 0; i < N; ++i) { xn_ref[c * N + i] = R[c * N + i] * rs * wn[c * N + i]; xn_e[c * N + i] = R[c * N + i] * rsf * wn[c * N + i]; } }
        Err e; for (int i = 0; i < D; ++i) e.add(xn_e[i] - xn_ref[i], xn_ref[i]); e.print("gr rms norm (xn)");
        // down dot for one row
        vector<uint16_t> wd(D); for (auto& a : wd) a = to_bf16(frand(0.05f));
        double acc = 0; for (int i = 0; i < D; ++i) acc += xn_ref[i] * from_bf16(wd[i]);
        float accf = 0; for (int i = 0; i < D; ++i) accf += xn_e[i] * from_bf16(wd[i]);
        Err ed; ed.add(accf - acc, acc); ed.print("gr down/up/inject dot");
        // post: w=2*sigmoid(inj/hc)
        Err ep; for (int c = 0; c < HC; ++c) { float inj = frand(); double wr = 2.0 / (1.0 + std::exp(-(double)inj / HC));
            float wf = 2.0f / (1.0f + std::exp(-inj / HC)); ep.add(wf - wr, wr); }
        ep.print("gr_write/native_post weight");
    }
    std::printf("done\n");
    return 0;
}
