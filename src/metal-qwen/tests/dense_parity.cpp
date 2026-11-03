// src/metal-qwen/tests/dense_parity.cpp - VM parity harness for the dense
// workstream. The oracle is the tree's own validated scalar reference
// (include/strata/artifact/dequant.hpp) plus a naive fp32 matvec. The "model"
// side is a CPU transliteration of the exact expressions the MSL kernels in
// src/metal-qwen/kernels/dense.metal + common.metalh carry (dq_elem, the
// sequential GEMVs, quantize_q8_1, mmvq), so a pass here validates the
// algorithms the Metal kernels implement; executing the MSL itself is the Mac
// parity runner's job (see the workstream report's MAC-QUEUE).
//
// Seeded inputs (splitmix64), sizes shaped like the Qwen set's dense tensors.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/artifact/dequant.hpp"

namespace {

uint64_t rng_state = 0x123456789abcdef0ull;
uint64_t rnd() {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
float rndf(float lo, float hi) { return lo + (hi - lo) * float((rnd() >> 40) & 0xFFFFFF) / float(0x1000000); }
uint16_t f2h(float f) {
    // round-to-nearest-even fp32 -> fp16 bits (inputs here are tame)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000, exp = ((u >> 23) & 0xFF) - 112, man = u & 0x7FFFFF;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7BFF);
    uint32_t h = sign | (exp << 10) | (man >> 13);
    if ((man >> 12) & 1) h += 1;  // approximate RNE for test data generation only
    return (uint16_t) h;
}

struct Err {
    double max_abs = 0, max_rel = 0;
    void add(float got, float want) {
        double d = std::fabs(double(got) - double(want));
        if (d > max_abs) max_abs = d;
        double r = d / (std::fabs(double(want)) + 1e-6);
        if (r > max_rel) max_rel = r;
    }
};

// ---- model side: transliteration of common.metalh's dq_elem ------------------
float model_h2f(uint16_t b) { return strata::fp16_to_fp32(b); }
uint16_t ld16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }

float dq_elem_model(const uint8_t* row, int type, uint32_t e) {
    static const float kIq4NlF[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    if (type == 42) {
        const uint8_t* b = row + (e / 64) * 18;
        uint32_t j = e % 64;
        int code = (b[2 + j / 4] >> ((j % 4) * 2)) & 3;
        return float(code - 1) * model_h2f(ld16(b));
    }
    if (type == 2) {
        const uint8_t* b = row + (e / 32) * 18;
        uint32_t j = e % 32;
        int nib = (j < 16) ? (b[2 + j] & 0xF) : (b[2 + j - 16] >> 4);
        return float(nib - 8) * model_h2f(ld16(b));
    }
    if (type == 6) {
        const uint8_t* b = row + (e / 32) * 22;
        uint32_t j = e % 32;
        uint32_t qh = uint32_t(b[2]) | (uint32_t(b[3]) << 8) | (uint32_t(b[4]) << 16) | (uint32_t(b[5]) << 24);
        int x = (j < 16) ? int((b[6 + j] & 0xF) | (((qh >> j) << 4) & 0x10))
                         : int((b[6 + j - 16] >> 4) | (((qh >> (j - 4)) & 0x10)));
        return float(x - 16) * model_h2f(ld16(b));
    }
    if (type == 8) {
        const uint8_t* b = row + (e / 32) * 34;
        return float(int(int8_t(b[2 + (e % 32)]))) * model_h2f(ld16(b));
    }
    if (type == 20) {
        const uint8_t* b = row + (e / 32) * 18;
        uint32_t j = e % 32;
        int nib = (j < 16) ? (b[2 + j] & 0xF) : (b[2 + j - 16] >> 4);
        return model_h2f(ld16(b)) * kIq4NlF[nib];
    }
    if (type == 11) {
        const uint8_t* b = row + (e / 256) * 110;
        uint32_t w = e % 256, gi = w / 32, t = w % 32, n = gi / 4, jj = gi % 4;
        uint32_t aux[4];
        std::memcpy(aux, b + 96, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*) aux;
        uint32_t is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
        float dl = model_h2f(ld16(b + 108)) * float(scales[is] - 32);
        uint32_t shift = 2 * jj;
        uint32_t m = 1u << (n * 4 + jj);
        int q = int((b[32 + n * 32 + t] >> shift) & 3) - ((b[t] & m) ? 0 : 4);
        return dl * float(q);
    }
    if (type == 12 || type == 13) {
        bool k5 = (type == 13);
        const uint8_t* b = row + (e / 256) * (k5 ? 176 : 144);
        uint32_t w = e % 256, gi = w / 32, j64 = gi / 2, hi = gi % 2, l = w % 32;
        int sc, mn;
        if (gi < 4) {
            sc = b[4 + gi] & 63;
            mn = b[4 + gi + 4] & 63;
        } else {
            sc = (b[4 + gi + 4] & 0xF) | ((b[4 + gi - 4] >> 6) << 4);
            mn = (b[4 + gi + 4] >> 4) | ((b[4 + gi] >> 6) << 4);
        }
        float d1 = model_h2f(ld16(b)) * float(sc), m1 = model_h2f(ld16(b + 2)) * float(mn);
        if (!k5) {
            const uint8_t* q = b + 16 + 32 * j64;
            int nib = hi ? (q[l] >> 4) : (q[l] & 0xF);
            return d1 * float(nib) - m1;
        }
        const uint8_t* ql = b + 48 + 32 * j64;
        uint32_t u = 1u << (2 * j64 + hi);
        int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
        return d1 * float(nib + ((b[16 + l] & u) ? 16 : 0)) - m1;
    }
    if (type == 14) {
        const uint8_t* b = row + (e / 256) * 210;
        uint32_t w = e % 256, gi = w / 32, n = gi / 4, qu = gi % 4, l = w % 32;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        int q;
        if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
        else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
        else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
        else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
        int scale = int(int8_t(b[192 + 8 * n + (l / 16) + 2 * qu]));
        return model_h2f(ld16(b + 208)) * float(scale) * float(q - 32);
    }
    if (type == 23) {
        const uint8_t* b = row + (e / 256) * 136;
        uint32_t ib = (e % 256) / 32, j = e % 32;
        int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((ld16(b + 2) >> (2 * ib)) & 3) << 4);
        float dl = model_h2f(ld16(b)) * float(ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        int nib = (j < 16) ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
        return dl * kIq4NlF[nib];
    }
    return 0.0f;
}

void oracle_dequant(int type, const uint8_t* block, float* out) {
    switch (type) {
    case 42: strata::dequantize_q2_0(block, out); break;
    case 2: strata::dequantize_q4_0(block, out); break;
    case 6: strata::dequantize_q5_0(block, out); break;
    case 8: strata::dequantize_q8_0(block, out); break;
    case 20: strata::dequantize_iq4_nl(block, out); break;
    case 11: strata::dequantize_q3_K(block, out); break;
    case 12: strata::dequantize_q4_K(block, out); break;
    case 13: strata::dequantize_q5_K(block, out); break;
    case 14: strata::dequantize_q6_K(block, out); break;
    case 23: strata::dequantize_iq4_xs(block, out); break;
    }
}
int block_elems(int t) { return (t == 42) ? 64 : (t == 2 || t == 6 || t == 8 || t == 20) ? 32 : 256; }
int block_bytes(int t) {
    switch (t) {
    case 42: case 2: case 20: return 18;
    case 6: return 22;
    case 8: return 34;
    case 11: return 110;
    case 12: return 144;
    case 13: return 176;
    case 14: return 210;
    case 23: return 136;
    }
    return 0;
}

const char* type_name(int t) {
    switch (t) {
    case 42: return "Q2_0";
    case 2: return "Q4_0";
    case 6: return "Q5_0";
    case 8: return "Q8_0";
    case 20: return "IQ4_NL";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 23: return "IQ4_XS";
    }
    return "?";
}

}  // namespace

int main() {
    int failures = 0;
    // ---- dequant element parity, every format --------------------------------
    for (int type : {42, 2, 6, 8, 20, 11, 12, 13, 14, 23}) {
        const int be = block_elems(type), bb = block_bytes(type);
        std::vector<uint8_t> row(bb * 4);
        for (auto& b : row) b = uint8_t(rnd());
        // sane scales: overwrite the f16 scale fields with ~0.05
        for (int blk = 0; blk < 4; ++blk) {
            uint16_t d = f2h(0.05f);
            uint8_t* b = row.data() + blk * bb;
            auto put = [&](int off) { b[off] = uint8_t(d & 0xFF); b[off + 1] = uint8_t(d >> 8); };
            if (type == 14) put(208);
            else if (type == 11) put(108);
            else {
                put(0);
                if (type == 12 || type == 13) put(2);
            }
        }
        Err err;
        std::vector<float> want(be);
        for (int blk = 0; blk < 4; ++blk) {
            oracle_dequant(type, row.data() + blk * bb, want.data());
            for (int e = 0; e < be; ++e) err.add(dq_elem_model(row.data(), type, blk * be + e), want[e]);
        }
        std::printf("dequant %-7s max_abs=%.3g max_rel=%.3g %s\n", type_name(type), err.max_abs, err.max_rel,
                    err.max_abs == 0 ? "BITWISE" : "DIFF");
        if (err.max_abs != 0) ++failures;
    }

    // ---- S2 GEMV: model (s2_gemv_k order) vs oracle dequant + naive matvec --
    {
        const int n_in = 2560, n_out = 64, nb = n_in / 64;
        std::vector<uint8_t> codes(n_out * nb * 16);
        std::vector<float> scales(n_out * nb), y_model(n_out), y_oracle(n_out);
        std::vector<uint16_t> x(n_in);
        for (auto& c : codes) c = uint8_t(rnd());
        for (auto& s : scales) s = rndf(0.01f, 0.1f);
        for (auto& v : x) v = f2h(rndf(-1.5f, 1.5f));
        // model: dequant_s2 planes, sequential order as in s2_gemv_k
        for (int o = 0; o < n_out; ++o) {
            float acc = 0;
            for (int b = 0; b < nb; ++b)
                for (int j = 0; j < 64; ++j) {
                    int code = (codes[o * nb * 16 + b * 16 + j / 4] >> ((j % 4) * 2)) & 3;
                    acc += float(code - 1) * scales[o * nb + b] * strata::fp16_to_fp32(x[b * 64 + j]);
                }
            y_model[o] = acc;
        }
        // oracle: artifact q2_0 dequant of the equivalent GGUF blocks + naive matvec.
        // The S2 plane form and the GGUF block form carry the same values: GGUF block
        // = {f16 d, qs[16]}; here d comes from the plane scale rounded to f16.
        for (int o = 0; o < n_out; ++o) {
            float acc = 0;
            for (int b = 0; b < nb; ++b) {
                uint8_t blk[18];
                uint16_t d = f2h(scales[o * nb + b]);
                blk[0] = uint8_t(d & 0xFF);
                blk[1] = uint8_t(d >> 8);
                std::memcpy(blk + 2, codes.data() + o * nb * 16 + b * 16, 16);
                float wrow[64];
                strata::dequantize_q2_0(blk, wrow);
                for (int j = 0; j < 64; ++j) acc += wrow[j] * strata::fp16_to_fp32(x[b * 64 + j]);
            }
            y_oracle[o] = acc;
        }
        Err err;
        for (int o = 0; o < n_out; ++o) err.add(y_model[o], y_oracle[o]);
        std::printf("s2_gemv (vs f16-rounded-scale oracle) max_abs=%.3g max_rel=%.3g\n", err.max_abs, err.max_rel);
        // the scale rounding to f16 dominates; the plane form itself is exact when
        // the scale is f16-representable:
        Err err2;
        for (int o = 0; o < n_out; ++o) {
            float acc = 0;
            for (int b = 0; b < nb; ++b)
                for (int j = 0; j < 64; ++j) {
                    int code = (codes[o * nb * 16 + b * 16 + j / 4] >> ((j % 4) * 2)) & 3;
                    acc += float(code - 1) * strata::fp16_to_fp32(f2h(scales[o * nb + b])) *
                           strata::fp16_to_fp32(x[b * 64 + j]);
                }
            err2.add(acc, y_oracle[o]);
        }
        std::printf("s2_gemv (f16-scale model vs oracle) max_abs=%.3g max_rel=%.3g %s\n", err2.max_abs,
                    err2.max_rel, err2.max_abs == 0 ? "BITWISE" : "DIFF");
        if (err2.max_abs != 0) ++failures;
    }

    // ---- quantize_q8_1 model vs the ggml formula ------------------------------
    {
        const int n = 32 * 40;
        std::vector<float> x(n);
        for (auto& v : x) v = rndf(-2.f, 2.f);
        Err ed, eq;
        for (int b = 0; b < n / 32; ++b) {
            float amax = 0, sum = 0;
            for (int j = 0; j < 32; ++j) {
                amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
                sum += x[b * 32 + j];
            }
            float d = std::fmin(amax / 127.0f, 65504.0f);
            ed.add(strata::fp16_to_fp32(f2h(d)), d);
            for (int j = 0; j < 32; ++j) {
                int q = (amax == 0) ? 0 : int(std::clamp(std::round(x[b * 32 + j] / d), -127.0f, 127.0f));
                eq.add(float(q) * d, x[b * 32 + j]);
            }
        }
        std::printf("quantize_q8_1 scale f16 rounding max_rel=%.3g; dequant roundtrip max_abs=%.3g\n", ed.max_rel,
                    eq.max_abs);
    }

    // ---- mmvq model (Q2_0 + Q4_K) vs oracle dequant + q8_1-dequant matvec -----
    for (int type : {42, 12}) {
        const int n_in = 2560, n_out = 32, be = block_elems(type), bb = block_bytes(type);
        std::vector<uint8_t> w(size_t(n_out) * (n_in / be) * bb);
        for (auto& c : w) c = uint8_t(rnd());
        for (int r = 0; r < n_out * (n_in / be); ++r) {
            uint16_t d = f2h(0.03f);
            uint8_t* b = w.data() + size_t(r) * bb;
            b[0] = uint8_t(d & 0xFF);
            b[1] = uint8_t(d >> 8);
            if (type == 12) {
                b[2] = uint8_t(d & 0xFF);
                b[3] = uint8_t(d >> 8);
            }
            if (type == 14) {}
        }
        std::vector<float> xf(n_in);
        for (auto& v : xf) v = rndf(-1.f, 1.f);
        // q8_1 quantize (model formula), then dequantized activations
        std::vector<float> act(n_in);
        for (int b = 0; b < n_in / 32; ++b) {
            float amax = 0, sum = 0;
            for (int j = 0; j < 32; ++j) {
                amax = std::fmax(amax, std::fabs(xf[b * 32 + j]));
                sum += xf[b * 32 + j];
            }
            (void) sum;
            float d = strata::fp16_to_fp32(f2h(std::fmin(amax / 127.0f, 65504.0f)));
            for (int j = 0; j < 32; ++j) {
                int q = (amax == 0) ? 0 : int(std::clamp(std::round(xf[b * 32 + j] / (amax / 127.0f)), -127.0f, 127.0f));
                act[b * 32 + j] = float(q) * d;
            }
        }
        Err err;
        for (int o = 0; o < n_out; ++o) {
            float acc_model = 0, acc_oracle = 0;
            std::vector<float> wrow(n_in);
            for (int blk = 0; blk < n_in / be; ++blk)
                oracle_dequant(type, w.data() + (size_t(o) * (n_in / be) + blk) * bb, wrow.data() + blk * be);
            for (int i = 0; i < n_in; ++i) {
                acc_model += dq_elem_model(w.data() + size_t(o) * (n_in / be) * bb, type, i) * act[i];
                acc_oracle += wrow[i] * act[i];
            }
            err.add(acc_model, acc_oracle);
        }
        std::printf("mmvq %s model vs oracle-dequant matvec max_abs=%.3g max_rel=%.3g %s\n", type_name(type),
                    err.max_abs, err.max_rel, err.max_abs == 0 ? "BITWISE" : "DIFF");
        if (err.max_abs != 0) ++failures;
    }

    std::printf(failures ? "dense_parity: %d FAILURES\n" : "dense_parity: all bitwise checks passed\n", failures);
    return failures ? 1 : 0;
}
