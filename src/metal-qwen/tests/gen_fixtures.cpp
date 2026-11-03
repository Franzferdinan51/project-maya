// Fixture generator for the Metal Qwen kernel parity harness. Host-only: it
// links no GPU code. Every expected output comes from one of:
//   * the tree's validated scalar dequantizers (include/strata/artifact/dequant.hpp),
//     the same reference the upstream CUDA parity programs chain to ggml;
//   * a double-precision scalar transcription of the math in the kernel's own
//     header contract / upstream CUDA source, written here independently.
// Deterministic: all randomness is splitmix64 over the fixture's fixed seed.
// Usage: gen_fixtures <out_dir>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "fixture_format.hpp"
#include "strata/artifact/dequant.hpp"

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uni(float lo, float hi) {  // uniform in [lo, hi)
        return lo + (hi - lo) * (float)((next() >> 40) * (1.0 / 16777216.0));
    }
    uint8_t byte() { return (uint8_t)(next() >> 56); }
};

uint16_t f32_to_f16(float v) {
    // Round-to-nearest-EVEN, matching the device conversions the fixtures are
    // checked against: __float2half_rn in the CUDA TU (dequant_bf16.cu's
    // put(H16)) and MSL's half(float). The previous implementation rounded
    // exact ties UP (it tested only the top dropped bit and always added the
    // increment), which put expected f16 bits 1 ulp high on every tie - the
    // dequant_f16 failures on iq4_xs/q4_k/q5_k/q6_k, whose values are integer
    // multiples of a power of two and land on f16 ties constantly.
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = (int)((x >> 23) & 0xFF) - 112;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {  // inf / NaN input
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    if (exp >= 31) {  // finite overflow: saturates to inf
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {  // subnormal half, or zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const int shift = 14 - exp;  // dropped bits below the kept subnormal mantissa
        uint32_t m = man >> shift;
        const uint32_t dropped = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (dropped > half || (dropped == half && (m & 1u))) ++m;  // may carry into smallest normal
        return (uint16_t)(sign | m);
    }
    const uint32_t dropped = man & 0x1FFFu;
    if (dropped > 0x1000u || (dropped == 0x1000u && ((man >> 13) & 1u))) {
        man += 0x2000u;
        if (man & 0x800000u) {  // mantissa overflow: bump the exponent (may reach inf)
            if (exp + 1 >= 31) return (uint16_t)(sign | 0x7C00u);
            return (uint16_t)(sign | ((uint32_t)(exp + 1) << 10));
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}
uint16_t f32_to_bf16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)(x >> 16);
    x += 0x7FFFu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}

mqfx::Tensor mk(const char* name, mqfx::DType dt, std::vector<int64_t> shape, std::vector<uint8_t> bytes) {
    mqfx::Tensor t;
    t.name = name; t.dtype = dt; t.shape = std::move(shape); t.bytes = std::move(bytes);
    return t;
}
mqfx::Tensor mk_f32(const char* name, std::vector<int64_t> shape, const std::vector<float>& v) {
    std::vector<uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return mk(name, mqfx::DType::F32, std::move(shape), std::move(b));
}
mqfx::Tensor mk_i32(const char* name, std::vector<int64_t> shape, const std::vector<int32_t>& v) {
    std::vector<uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return mk(name, mqfx::DType::I32, std::move(shape), std::move(b));
}
mqfx::Tensor mk_u16(const char* name, std::vector<int64_t> shape, const std::vector<uint16_t>& v) {
    std::vector<uint8_t> b(v.size() * 2);
    std::memcpy(b.data(), v.data(), b.size());
    return mk(name, mqfx::DType::U16, std::move(shape), std::move(b));
}
mqfx::Tensor mk_u8(const char* name, std::vector<int64_t> shape, const std::vector<uint8_t>& v) {
    return mk(name, mqfx::DType::U8, std::move(shape), v);
}

std::string g_out;
int g_written = 0;
void emit(mqfx::Fixture& fx) {
    const std::string path = g_out + "/" + fx.entry + "." + fx.case_name + ".mqfx";
    if (!mqfx::write_fixture(path, fx)) { std::fprintf(stderr, "FAILED to write %s\n", path.c_str()); std::exit(1); }
    ++g_written;
    std::printf("wrote %s\n", path.c_str());
}

// ---- GGUF block geometry -------------------------------------------------
struct BlockInfo { int type; const char* name; int elems; int bytes; int d_off; };
const BlockInfo kBlocks[] = {
    {42, "q2_0", 64, 18, 0},   {2, "q4_0", 32, 18, 0},  {6, "q5_0", 32, 22, 0},
    {8, "q8_0", 32, 34, 0},    {20, "iq4_nl", 32, 18, 0}, {23, "iq4_xs", 256, 136, 0},
    {11, "q3_k", 256, 110, 108}, {12, "q4_k", 256, 144, 0}, {13, "q5_k", 256, 176, 0},
    {14, "q6_k", 256, 210, 208},
};
void dequant_block(int type, const uint8_t* blk, float* out) {
    switch (type) {
        case 42: strata::dequantize_q2_0(blk, out); break;
        case 2: strata::dequantize_q4_0(blk, out); break;
        case 6: strata::dequantize_q5_0(blk, out); break;
        case 8: strata::dequantize_q8_0(blk, out); break;
        case 20: strata::dequantize_iq4_nl(blk, out); break;
        case 23: strata::dequantize_iq4_xs(blk, out); break;
        case 11: strata::dequantize_q3_K(blk, out); break;
        case 12: strata::dequantize_q4_K(blk, out); break;
        case 13: strata::dequantize_q5_K(blk, out); break;
        case 14: strata::dequantize_q6_K(blk, out); break;
        default: std::fprintf(stderr, "no dequantizer for type %d\n", type); std::exit(1);
    }
}
// Fabricate raw blocks: random code bytes, but every fp16 scale field pinned to
// a finite normal value so outputs stay finite (random scale bits could be
// inf/NaN, which would make the fixture about NaN handling, not dequant math).
std::vector<uint8_t> fake_blocks(const BlockInfo& bi, int n_blocks, Rng& rng) {
    static const uint16_t kScales[] = {0x3C00, 0x3800, 0x3400, 0x2C00, 0xBC00, 0xB400, 0x1800, 0x4000};
    std::vector<uint8_t> raw((size_t)n_blocks * bi.bytes);
    for (auto& b : raw) b = rng.byte();
    for (int i = 0; i < n_blocks; ++i) {
        uint8_t* blk = raw.data() + (size_t)i * bi.bytes;
        const uint16_t d = kScales[rng.next() % 8];
        blk[bi.d_off] = (uint8_t)(d & 0xFF);
        blk[bi.d_off + 1] = (uint8_t)(d >> 8);
        if (bi.type == 12 || bi.type == 13) {  // dmin next to d in K-quants
            const uint16_t dm = kScales[rng.next() % 8];
            blk[2] = (uint8_t)(dm & 0xFF);
            blk[3] = (uint8_t)(dm >> 8);
        }
    }
    return raw;
}
std::vector<float> dequant_all(int type, const BlockInfo& bi, const std::vector<uint8_t>& raw) {
    const int n_blocks = (int)raw.size() / bi.bytes;
    std::vector<float> out((size_t)n_blocks * bi.elems);
    for (int i = 0; i < n_blocks; ++i) dequant_block(type, raw.data() + (size_t)i * bi.bytes, out.data() + (size_t)i * bi.elems);
    return out;
}

// ---- 1. dequant family ----------------------------------------------------
void gen_dequant() {
    for (const auto& bi : kBlocks) {
        const int rows = 2, cols = bi.elems * 2;
        Rng rng(0xD000 + (uint64_t)bi.type);
        auto raw = fake_blocks(bi, rows * (cols / bi.elems), rng);
        auto want = dequant_all(bi.type, bi, raw);
        for (const char* entry : {"dequant_f32", "dequant_bf16", "dequant_f16"}) {
            mqfx::Fixture fx;
            fx.family = "dequant"; fx.entry = entry;
            fx.case_name = std::string(bi.name) + "_2rows";
            fx.seed = 0xD000 + (uint64_t)bi.type;
            fx.params_i = {{"ggml_type", bi.type}, {"rows", rows}, {"cols", cols}, {"row0", 0}};
            fx.inputs = {mk_u8("blocks", {rows * (cols / bi.elems) * bi.bytes}, raw)};
            if (std::strcmp(entry, "dequant_f32") == 0) {
                fx.tol_kind = mqfx::TolKind::Bitwise;
                fx.outputs = {mk_f32("out", {rows, cols}, want)};
            } else {
                // Expected bits are the reference values rounded once, RNE,
                // exactly as a correct device conversion must produce them.
                std::vector<uint16_t> bits(want.size());
                for (size_t i = 0; i < want.size(); ++i)
                    bits[i] = std::strcmp(entry, "dequant_bf16") == 0 ? f32_to_bf16(want[i]) : f32_to_f16(want[i]);
                fx.tol_kind = mqfx::TolKind::Bitwise;
                fx.outputs = {mk_u16("out", {rows, cols}, bits)};
            }
            emit(fx);
        }
    }
    // dequant_s2: the canonical S2 planes (codes + f32 scales), not raw GGUF.
    {
        Rng rng(0xD200);
        const int64_t n_blocks = 8;
        std::vector<uint8_t> codes((size_t)n_blocks * 16);
        for (auto& b : codes) b = rng.byte();
        std::vector<float> scales((size_t)n_blocks);
        for (auto& s : scales) s = rng.uni(0.05f, 0.4f);
        std::vector<float> want((size_t)n_blocks * 64);
        for (int64_t blk = 0; blk < n_blocks; ++blk)
            for (int j = 0; j < 64; ++j) {
                const int code = (codes[(size_t)blk * 16 + j / 4] >> ((j % 4) * 2)) & 3;
                want[(size_t)blk * 64 + j] = (float)(code - 1) * scales[(size_t)blk];
            }
        mqfx::Fixture fx;
        fx.family = "dequant"; fx.entry = "dequant_s2"; fx.case_name = "canonical_8blocks"; fx.seed = 0xD200;
        fx.tol_kind = mqfx::TolKind::Bitwise;
        fx.params_i = {{"n_blocks", n_blocks}};
        fx.inputs = {mk_u8("codes", {n_blocks * 16}, codes), mk_f32("scales", {n_blocks}, scales)};
        fx.outputs = {mk_f32("out", {n_blocks * 64}, want)};
        emit(fx);
    }
}

// ---- canonical S-form decode + GEMV reference -----------------------------
const int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
double s_decode(const uint8_t* codes_row, int64_t i, int code_bits, int code_bias, bool iq4nl) {
    int code;
    if (code_bits == 2) code = (codes_row[i / 4] >> ((i % 4) * 2)) & 3;
    else if (code_bits == 4) code = (i % 2 == 0) ? (codes_row[i / 2] & 0xF) : (codes_row[i / 2] >> 4);
    else code = codes_row[i];
    return iq4nl ? (double)kIq4Nl[code] : (double)(code + code_bias);
}
struct SCase {
    const char* entry; const char* name; uint64_t seed;
    int64_t n_in, n_out; int code_bits, code_bias, group; bool iq4nl;
};
void gen_s_gemv_for(const SCase& c, bool split) {
    Rng rng(c.seed);
    const int64_t bytes_per_row = c.n_in * c.code_bits / 8;
    const int64_t groups_per_row = c.n_in / c.group;
    std::vector<uint16_t> x((size_t)c.n_in);
    std::vector<double> xd((size_t)c.n_in);
    for (int64_t i = 0; i < c.n_in; ++i) {
        x[(size_t)i] = f32_to_f16(rng.uni(-1.0f, 1.0f));
        xd[(size_t)i] = strata::fp16_to_fp32(x[(size_t)i]);
    }
    std::vector<uint8_t> codes((size_t)c.n_out * bytes_per_row);
    for (auto& b : codes) b = rng.byte();
    std::vector<float> scales((size_t)c.n_out * groups_per_row);
    for (auto& s : scales) s = rng.uni(0.02f, 0.2f);
    std::vector<float> want((size_t)c.n_out);
    for (int64_t o = 0; o < c.n_out; ++o) {
        double acc = 0;
        const uint8_t* row = codes.data() + (size_t)o * bytes_per_row;
        for (int64_t i = 0; i < c.n_in; ++i)
            acc += xd[(size_t)i] * s_decode(row, i, c.code_bits, c.code_bias, c.iq4nl) *
                   (double)scales[(size_t)o * groups_per_row + i / c.group];
        want[(size_t)o] = (float)acc;
    }
    mqfx::Fixture fx;
    fx.family = "gemv"; fx.entry = split ? "s_gemv_split" : c.entry; fx.case_name = c.name; fx.seed = c.seed;
    fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 1e-4; fx.tol_rel = 2e-3;
    fx.params_i = {{"n_in", c.n_in}, {"n_out", c.n_out}, {"code_bits", c.code_bits},
                   {"code_bias", c.code_bias}, {"group_elems", c.group}, {"codebook", c.iq4nl ? 1 : 0},
                   {"has_offset", 0}, {"threads_per_row", 32}};
    fx.inputs = {mk_u16("x", {c.n_in}, x), mk_u8("codes", {c.n_out * bytes_per_row}, codes),
                 mk_f32("scales", {c.n_out * groups_per_row}, scales)};
    fx.outputs = {mk_f32("y", {c.n_out}, want)};
    emit(fx);
}
void gen_gemv() {
    // Real Qwen shapes, row-subsetted on n_out so fixtures stay small: the
    // expert gate/up projection is [2560 in, 640 out], the down projection
    // [640 in, 2560 out], the shared-expert down is IQ4_NL [640 in, 2560 out].
    gen_s_gemv_for({"s_gemv", "s2_gate_2560x64", 0x6E01, 2560, 64, 2, -1, 64, false}, false);
    gen_s_gemv_for({"s_gemv", "s2_gate_2560x64", 0x6E01, 2560, 64, 2, -1, 64, false}, true);
    gen_s_gemv_for({"s_gemv", "s2_down_640x64", 0x6E02, 640, 64, 2, -1, 64, false}, false);
    gen_s_gemv_for({"s_gemv", "iq4nl_shexp_down_640x64", 0x6E03, 640, 64, 4, 0, 32, true}, false);
    // s2_gemv: the expert S2 entry, same planes as the gate case.
    gen_s_gemv_for({"s2_gemv", "s2_gate_2560x64", 0x6E01, 2560, 64, 2, -1, 64, false}, false);
    // s2_gemv_q8: activations pre-quantized to Q8_0 blocks (the vec_dot_type
    // of Q2_0 weights); the reference dots the DEQUANTIZED activation.
    {
        Rng rng(0x6E04);
        const int64_t n_in = 2560, n_out = 64;
        std::vector<float> xf((size_t)n_in);
        for (auto& v : xf) v = rng.uni(-1.0f, 1.0f);
        std::vector<uint8_t> act((size_t)(n_in / 32) * 34);
        std::vector<double> xd((size_t)n_in);
        for (int64_t b = 0; b < n_in / 32; ++b) {
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xf[(size_t)b * 32 + j]));
            const float d = amax / 127.0f;
            const uint16_t db = f32_to_f16(d);
            uint8_t* blk = act.data() + (size_t)b * 34;
            blk[0] = (uint8_t)(db & 0xFF); blk[1] = (uint8_t)(db >> 8);
            const double dd = strata::fp16_to_fp32(db);
            for (int j = 0; j < 32; ++j) {
                const long q = std::lrint((double)xf[(size_t)b * 32 + j] / dd);
                blk[2 + j] = (uint8_t)(int8_t)std::max(-127L, std::min(127L, q));
                xd[(size_t)b * 32 + j] = dd * (double)(int8_t)blk[2 + j];
            }
        }
        const int64_t bytes_per_row = n_in / 4, groups_per_row = n_in / 64;
        std::vector<uint8_t> codes((size_t)n_out * bytes_per_row);
        for (auto& b : codes) b = rng.byte();
        std::vector<float> scales((size_t)n_out * groups_per_row);
        for (auto& s : scales) s = rng.uni(0.02f, 0.2f);
        std::vector<float> want((size_t)n_out);
        for (int64_t o = 0; o < n_out; ++o) {
            double acc = 0;
            const uint8_t* row = codes.data() + (size_t)o * bytes_per_row;
            for (int64_t i = 0; i < n_in; ++i)
                acc += xd[(size_t)i] * s_decode(row, i, 2, -1, false) * (double)scales[(size_t)o * groups_per_row + i / 64];
            want[(size_t)o] = (float)acc;
        }
        mqfx::Fixture fx;
        fx.family = "gemv"; fx.entry = "s2_gemv_q8"; fx.case_name = "q8act_gate_2560x64"; fx.seed = 0x6E04;
        fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 1e-4; fx.tol_rel = 2e-3;
        fx.params_i = {{"n_in", n_in}, {"n_out", n_out}, {"threads_per_row", 32}};
        fx.inputs = {mk_u8("act", {(n_in / 32) * 34}, act), mk_u8("codes", {n_out * bytes_per_row}, codes),
                     mk_f32("scales", {n_out * groups_per_row}, scales)};
        fx.outputs = {mk_f32("y", {n_out}, want)};
        emit(fx);
    }
}

// ---- native_mmvq family ----------------------------------------------------
// The Q8_1 activation image the mmvq entries consume: per 32 elements
// {fp16 d, fp16 sum_of_original_floats, int8 qs[32]} = 36 bytes.
std::vector<uint8_t> quantize_q8_1(const std::vector<float>& x, std::vector<double>& x_deq) {
    const size_t nb = x.size() / 32;
    std::vector<uint8_t> out(nb * 36);
    x_deq.assign(x.size(), 0.0);
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0;
        double sum = 0;
        for (int j = 0; j < 32; ++j) { amax = std::max(amax, std::fabs(x[b * 32 + j])); sum += x[b * 32 + j]; }
        // The CUDA quantizer's exact contract (native_mmvq.cu,
        // native_quantize_q8_1_kernel): the quotient uses the UNROUNDED float
        // scale d = fminf(amax/127, 65504), rounded by roundf (half away from
        // zero); only the STORED scale is the f16 rounding of d, and the
        // dequantized activation is stored_d * q. The previous version divided
        // by the rounded scale and used lrint (ties-to-even), which flips q by
        // one on ~1% of elements and was the whole native_*_f32 failure set -
        // the *_mmvq cases passed because they consume this block as input.
        const float d = std::min(amax / 127.0f, 65504.0f);
        const uint16_t db = f32_to_f16(d);
        const float sum_clamped = std::max(-65504.0f, std::min(65504.0f, (float)sum));
        const uint16_t sb = f32_to_f16(sum_clamped);
        uint8_t* blk = out.data() + b * 36;
        blk[0] = (uint8_t)(db & 0xFF); blk[1] = (uint8_t)(db >> 8);
        blk[2] = (uint8_t)(sb & 0xFF); blk[3] = (uint8_t)(sb >> 8);
        const double dd = strata::fp16_to_fp32(db);
        for (int j = 0; j < 32; ++j) {
            long q = amax == 0.0f ? 0 : (long)std::round(x[b * 32 + j] / d);
            q = std::max(-127L, std::min(127L, q));
            blk[4 + j] = (uint8_t)(int8_t)q;
            x_deq[b * 32 + j] = dd * (double)q;
        }
    }
    return out;
}
void gen_mmvq() {
    const int types[] = {42, 20, 23, 8, 12};  // Q2_0, IQ4_NL, IQ4_XS, Q8_0, Q4_K
    for (int type : types) {
        const BlockInfo* bi = nullptr;
        for (const auto& b : kBlocks) if (b.type == type) bi = &b;
        const int64_t n_in = (bi->elems == 256) ? 256 : (type == 42 ? 2560 : 640);
        const int64_t n_out = 16, ncols = 1;
        Rng rng(0xAA00 + (uint64_t)type);
        auto raw = fake_blocks(*bi, (int)(n_out * (n_in / bi->elems)), rng);
        auto w = dequant_all(type, *bi, raw);
        std::vector<float> xf((size_t)n_in);
        for (auto& v : xf) v = rng.uni(-1.0f, 1.0f);
        std::vector<double> xdq;
        auto xq = quantize_q8_1(xf, xdq);
        std::vector<float> want((size_t)n_out);
        for (int64_t o = 0; o < n_out; ++o) {
            double acc = 0;
            for (int64_t i = 0; i < n_in; ++i) acc += (double)w[(size_t)o * n_in + i] * xdq[(size_t)i];
            want[(size_t)o] = (float)acc;
        }
        const char* suffix = bi->name;
        for (const char* entry_kind : {"generic", "typed", "f32"}) {
            mqfx::Fixture fx;
            fx.family = "gemv";
            fx.entry = entry_kind[0] == 'g' ? "native_mmvq"
                       : entry_kind[0] == 't' ? std::string("native_") + suffix + "_mmvq"
                                              : std::string("native_") + suffix + "_f32";
            fx.case_name = std::string(suffix) + "_native_dims";
            fx.seed = 0xAA00 + (uint64_t)type;
            // The reference dots dequantized weights with the dequantized
            // Q8_1 activation in double; the device accumulates in f32 over
            // quantized integers, so the bound is relative, not bitwise.
            fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 1e-3; fx.tol_rel = 5e-3;
            fx.params_i = {{"ggml_type", type}, {"n_in", n_in}, {"n_out", n_out}, {"ncols", ncols}};
            fx.inputs = {mk_u8("weights", {(int64_t)raw.size()}, raw)};
            if (entry_kind[0] == 'f') fx.inputs.push_back(mk_f32("x", {n_in}, xf));
            else fx.inputs.push_back(mk_u8("x_q8_1", {(int64_t)xq.size()}, xq));
            fx.outputs = {mk_f32("y", {n_out}, want)};
            emit(fx);
        }
    }
}

// ---- native_expert_grouped ----------------------------------------------------
// The grouped native experts end to end: per group, a fabricated blob in the
// NativeExpertLayout ([gate rows | up rows | down rows] of raw GGUF blocks);
// the reference is the double-precision chain the CUDA TU computes - dequant
// dots against the dequantized q8_1 activation, swiglu (plain or clamped),
// q8_1 quantization of the intermediate (quantize_q8_1's exact contract),
// down dots. grp_ptr itself is not a fixture input: the runner fills it with
// the device addresses of the per-group blob inputs, as the engine does.
void gen_native_expert_grouped() {
    struct Case {
        const char* name;
        int gu, d;
        int64_t n_embd, n_ff;
        std::vector<int32_t> grp_start;  // groups + 1
        std::vector<int32_t> ent_tok, ent_dst;
        double limit;  // < 0 means +infinity (the plain arm)
        uint64_t seed;
    };
    const Case cases[] = {
        {"q2_0_two_groups_perm", 42, 42, 128, 64, {0, 2, 3}, {0, 1, 2}, {1, 0, 2}, -1.0, 0xE001},
        {"q5k_iq4nl_one_group", 13, 20, 256, 256, {0, 2}, {1, 0}, {0, 1}, -1.0, 0xE002},
        {"iq4xs_q2_0_clamped", 23, 42, 256, 64, {0, 1, 3}, {2, 0, 1}, {2, 1, 0}, 5.0, 0xE003},
        {"q2_0_iq4xs_down", 42, 23, 128, 256, {0, 2}, {0, 1}, {1, 0}, -1.0, 0xE004},
    };
    for (const Case& cs : cases) {
        const BlockInfo* gu_bi = nullptr;
        const BlockInfo* d_bi = nullptr;
        for (const auto& b : kBlocks) {
            if (b.type == cs.gu) gu_bi = &b;
            if (b.type == cs.d) d_bi = &b;
        }
        const int64_t cap_groups = (int64_t)cs.grp_start.size() - 1;
        const int64_t cap_entries = (int64_t)cs.ent_tok.size();
        int n_tok = 0;
        for (int32_t t : cs.ent_tok) n_tok = std::max(n_tok, (int)t + 1);
        Rng rng(cs.seed);
        // Blobs: gate and up are n_ff rows of n_embd; down is n_embd rows of n_ff.
        std::vector<std::vector<uint8_t>> blobs;
        std::vector<std::vector<float>> gate_w, up_w, down_w;
        for (int64_t g = 0; g < cap_groups; ++g) {
            auto gate_raw = fake_blocks(*gu_bi, (int)(cs.n_ff * (cs.n_embd / gu_bi->elems)), rng);
            auto up_raw = fake_blocks(*gu_bi, (int)(cs.n_ff * (cs.n_embd / gu_bi->elems)), rng);
            auto down_raw = fake_blocks(*d_bi, (int)(cs.n_embd * (cs.n_ff / d_bi->elems)), rng);
            gate_w.push_back(dequant_all(cs.gu, *gu_bi, gate_raw));
            up_w.push_back(dequant_all(cs.gu, *gu_bi, up_raw));
            down_w.push_back(dequant_all(cs.d, *d_bi, down_raw));
            std::vector<uint8_t> blob;
            blob.insert(blob.end(), gate_raw.begin(), gate_raw.end());
            blob.insert(blob.end(), up_raw.begin(), up_raw.end());
            blob.insert(blob.end(), down_raw.begin(), down_raw.end());
            blobs.push_back(std::move(blob));
        }
        // Activations: one q8_1 image per token.
        std::vector<float> xf((size_t)n_tok * cs.n_embd);
        for (auto& v : xf) v = rng.uni(-1.5f, 1.5f);
        std::vector<double> xdq;
        auto xq = quantize_q8_1(xf, xdq);
        // Reference chain per entry.
        std::vector<float> want((size_t)cap_entries * cs.n_embd, 0.0f);
        for (int64_t g = 0; g < cap_groups; ++g) {
            for (int32_t e = cs.grp_start[(size_t)g]; e < cs.grp_start[(size_t)g + 1]; ++e) {
                const double* xa = xdq.data() + (size_t)cs.ent_tok[(size_t)e] * cs.n_embd;
                std::vector<float> h((size_t)cs.n_ff);
                for (int64_t r = 0; r < cs.n_ff; ++r) {
                    double gv = 0, uv = 0;
                    for (int64_t i = 0; i < cs.n_embd; ++i) {
                        gv += (double)gate_w[(size_t)g][(size_t)r * cs.n_embd + i] * xa[i];
                        uv += (double)up_w[(size_t)g][(size_t)r * cs.n_embd + i] * xa[i];
                    }
                    double gg = gv, uu = uv;
                    if (cs.limit >= 0) {
                        gg = std::min(gv, cs.limit);
                        uu = std::max(-cs.limit, std::min(uv, cs.limit));
                    }
                    h[(size_t)r] = (float)((gg / (1.0 + std::exp(-gg))) * uu);
                }
                std::vector<double> hdq;
                quantize_q8_1(h, hdq);
                float* out_row = want.data() + (size_t)cs.ent_dst[(size_t)e] * cs.n_embd;
                for (int64_t r = 0; r < cs.n_embd; ++r) {
                    double acc = 0;
                    for (int64_t j = 0; j < cs.n_ff; ++j)
                        acc += (double)down_w[(size_t)g][(size_t)r * cs.n_ff + j] * hdq[(size_t)j];
                    out_row[r] = (float)acc;
                }
            }
        }
        mqfx::Fixture fx;
        fx.family = "moe";
        fx.entry = "native_expert_grouped";
        fx.case_name = cs.name;
        fx.seed = cs.seed;
        fx.tol_kind = mqfx::TolKind::MaxRel;
        fx.tol_abs = 1e-2;
        fx.tol_rel = 2e-2;
        fx.params_i = {{"gu_type", cs.gu}, {"d_type", cs.d}, {"n_embd", cs.n_embd}, {"n_ff", cs.n_ff},
                       {"cap_groups", cap_groups}, {"cap_entries", cap_entries}, {"has_limit", cs.limit >= 0 ? 1 : 0}};
        fx.params_f = {{"swiglu_limit", cs.limit >= 0 ? cs.limit : 0.0}};
        for (int64_t g = 0; g < cap_groups; ++g)
            fx.inputs.push_back(mk_u8(("blob" + std::to_string(g)).c_str(), {(int64_t)blobs[(size_t)g].size()},
                                      blobs[(size_t)g]));
        fx.inputs.push_back(mk_i32("grp_start", {cap_groups + 1}, cs.grp_start));
        fx.inputs.push_back(mk_i32("n_groups", {1}, {(int32_t)cap_groups}));
        fx.inputs.push_back(mk_i32("ent_tok", {cap_entries}, cs.ent_tok));
        fx.inputs.push_back(mk_i32("ent_dst", {cap_entries}, cs.ent_dst));
        fx.inputs.push_back(mk_u8("x_q8_1", {(int64_t)xq.size()}, xq));
        fx.outputs = {mk_f32("out", {cap_entries, cs.n_embd}, want)};
        emit(fx);
        // The based form (native_expert_grouped_based), exercised the way
        // record_window calls it: every blob a slice of ONE arena allocation
        // (256-aligned stride), grp_ptr filled by the runner with the slices'
        // addresses, the plan read on the device. Same reference output.
        mqfx::Fixture fx2;
        fx2.family = "moe";
        fx2.entry = "native_expert_grouped_based";
        fx2.case_name = cs.name;
        fx2.seed = cs.seed;
        fx2.tol_kind = mqfx::TolKind::MaxRel;
        fx2.tol_abs = 1e-2;
        fx2.tol_rel = 2e-2;
        const int64_t blob_bytes = (int64_t)blobs[0].size();
        const int64_t stride = (blob_bytes + 255) & ~255ll;
        std::vector<uint8_t> arena((size_t)stride * (size_t)cap_groups, 0);
        for (int64_t g = 0; g < cap_groups; ++g)
            std::memcpy(arena.data() + (size_t)g * (size_t)stride, blobs[(size_t)g].data(),
                        blobs[(size_t)g].size());
        fx2.params_i = {{"gu_type", cs.gu}, {"d_type", cs.d}, {"n_embd", cs.n_embd}, {"n_ff", cs.n_ff},
                        {"cap_groups", cap_groups}, {"cap_entries", cap_entries},
                        {"has_limit", cs.limit >= 0 ? 1 : 0}, {"blob_stride", stride}};
        fx2.params_f = {{"swiglu_limit", cs.limit >= 0 ? cs.limit : 0.0}};
        fx2.inputs.push_back(mk_u8("arena", {(int64_t)arena.size()}, arena));
        fx2.inputs.push_back(mk_i32("grp_start", {cap_groups + 1}, cs.grp_start));
        fx2.inputs.push_back(mk_i32("n_groups", {1}, {(int32_t)cap_groups}));
        fx2.inputs.push_back(mk_i32("ent_tok", {cap_entries}, cs.ent_tok));
        fx2.inputs.push_back(mk_i32("ent_dst", {cap_entries}, cs.ent_dst));
        fx2.inputs.push_back(mk_u8("x_q8_1", {(int64_t)xq.size()}, xq));
        fx2.outputs = {mk_f32("out", {cap_entries, cs.n_embd}, want)};
        emit(fx2);
    }
}

// ---- verify_moe_assembly ------------------------------------------------------
// The verify window's post() expert assembly as ONE composition (the fixture
// DIVERGE-NOTES.md prescribes): a window whose routed entries are split
// across a VRAM grouped_based call (zero-based grp_start), a PCIe
// grouped_based call over the SAME dst/tok arrays and the SAME scratch and
// hit_out, whose grp_start CONTINUES the entry numbering (start2), and CPU
// rows that arrive through the mapped ymiss rows - finished by
// copy_rows_from_mapped (GPU rows zeroed, CPU rows copied) and moe_hit_add.
// Per-entry fixtures cannot see this composition: the grouped entries are
// exercised there with cap == the real counts and a single call, while
// record_window passes cap_groups == cap_entries == T*K (oversized) for BOTH
// calls and reuses one scratch/hit_out pair. Experts are SHARED across the
// window's tokens (one blob, entries from several rows), so a row/entry
// indexing slip perturbs every row. Reference: the grouped fixtures' double
// chain per entry; CPU rows are their ymiss values; hit_out rows no grouped
// call owns must keep the runner's stale sentinel. The plan arrays are sized
// capx = max_t*K (the window's T may be below the verifier's max_t), with
// stale zeros past the live entries, exactly as the mapped plan carries them.
void gen_verify_moe_assembly() {
    struct Case {
        const char* name;
        uint64_t seed;
        int64_t T, K, n_embd, n_ff;
        int64_t capx;          // plan array sizing (max_t*K); 0 -> T*K
        std::vector<int32_t> ids;   // routed expert per entry i (T*K, i = t*K + slot)
        std::vector<int> tier;      // per distinct expert in routing order: 0 VRAM, 1 PCIe, 2 CPU
        bool full = false;     // also run shared_expert_multi + combine -> bo
    };
    const Case cases[] = {
        {"t2_mixed_vram_pcie_cpu", 0xE201, 2, 3, 128, 64, 0,
         {10, 11, 12, 11, 13, 10}, {0, 1, 2, 0}},
        {"t2_vram_only_shared", 0xE202, 2, 3, 128, 64, 0,
         {20, 21, 20, 22, 20, 21}, {0, 0, 2, 0}},
        // The capstone's real geometry: n_embd 2560, n_ff 640, K = 10, and a
        // T=2 window inside a max_t = 4 verifier (capx = 40 plan arrays).
        {"t2_real_dims_capx40", 0xE203, 2, 10, 2560, 640, 40,
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 11, 12, 13, 14, 15, 2, 4},
         {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2}},
        // A full T=4 window at real geometry, experts shared across all rows.
        {"t4_real_dims", 0xE204, 4, 10, 2560, 640, 0,
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 12, 2, 13, 3, 14, 4, 15, 5,
          16, 6, 17, 7, 18, 8, 19, 9, 20, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15},
         {0, 0, 1, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1}},
        // Full post() FFN: the grouped assembly above, plus the shared expert
        // and the batched combine, at real geometry.
        {"t2_full_ffn", 0xE205, 2, 10, 2560, 640, 40,
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 11, 12, 13, 14, 15, 2, 4},
         {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2}, true},
        {"t4_full_ffn", 0xE206, 4, 10, 2560, 640, 0,
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 12, 2, 13, 3, 14, 4, 15, 5,
          16, 6, 17, 7, 18, 8, 19, 9, 20, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15},
         {0, 0, 1, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1}, true},
    };
    const int gu = 42, d = 42;
    const BlockInfo* bi = nullptr;
    for (const auto& b : kBlocks) if (b.type == gu) bi = &b;
    for (const Case& cs : cases) {
        const int64_t T = cs.T, K = cs.K, cap = T * K;
        const int64_t n_embd = cs.n_embd, n_ff = cs.n_ff;
        const int64_t capx = cs.capx ? cs.capx : cap;
        Rng rng(cs.seed);
        // Distinct experts in routing order.
        int32_t distinct[64];
        int nd = 0;
        for (int i = 0; i < T * K; ++i) {
            bool seen = false;
            for (int j = 0; j < nd; ++j) seen = seen || distinct[j] == cs.ids[(size_t)i];
            if (!seen) distinct[nd++] = cs.ids[(size_t)i];
        }
        // One fabricated blob per GPU distinct expert.
        std::map<int32_t, std::vector<uint8_t>> blob_of;
        std::map<int32_t, std::vector<float>> gate_w, up_w, down_w;
        for (int j = 0; j < nd; ++j) {
            if (cs.tier[(size_t)j] == 2) continue;
            const int32_t ex = distinct[j];
            auto gate_raw = fake_blocks(*bi, (int)(n_ff * (n_embd / bi->elems)), rng);
            auto up_raw = fake_blocks(*bi, (int)(n_ff * (n_embd / bi->elems)), rng);
            auto down_raw = fake_blocks(*bi, (int)(n_embd * (n_ff / bi->elems)), rng);
            gate_w[ex] = dequant_all(gu, *bi, gate_raw);
            up_w[ex] = dequant_all(gu, *bi, up_raw);
            down_w[ex] = dequant_all(d, *bi, down_raw);
            std::vector<uint8_t> blob;
            blob.insert(blob.end(), gate_raw.begin(), gate_raw.end());
            blob.insert(blob.end(), up_raw.begin(), up_raw.end());
            blob.insert(blob.end(), down_raw.begin(), down_raw.end());
            blob_of[ex] = std::move(blob);
        }
        const int64_t blob_bytes = (int64_t)blob_of.begin()->second.size();
        const int64_t stride = (blob_bytes + 255) & ~255ll;
        // Plan, built exactly as expert_pool_dispatch_multi builds it: VRAM
        // groups first (distinct order), then PCIe groups, entries numbered
        // continuously across both; dst/tok shared by both calls.
        std::vector<int32_t> dst, tok, start, start2;
        std::vector<int32_t> vram_experts, pcie_experts;
        int entries = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < nd; ++j) {
                if (cs.tier[(size_t)j] != pass) continue;  // pass 0: VRAM, pass 1: PCIe
                (pass == 0 ? start : start2).push_back(entries);
                (pass == 0 ? vram_experts : pcie_experts).push_back(distinct[j]);
                for (int i = 0; i < T * K; ++i)
                    if (cs.ids[(size_t)i] == distinct[j]) {
                        dst.push_back(i);
                        tok.push_back(i / (int32_t)K);
                        ++entries;
                    }
            }
            (pass == 0 ? start : start2).push_back(entries);
        }
        const int32_t counts[4] = {(int32_t)vram_experts.size(), entries, (int32_t)pcie_experts.size(), 0};
        // Activations: one q8_1 image per token.
        std::vector<float> xf((size_t)T * n_embd);
        for (auto& v : xf) v = rng.uni(-1.5f, 1.5f);
        std::vector<double> xdq;
        auto xq = quantize_q8_1(xf, xdq);
        // Grouped reference per entry (double chain, as gen_native_expert_grouped).
        std::vector<float> grouped_row((size_t)T * K * n_embd, 0.0f);
        for (int e = 0; e < entries; ++e) {
            const int32_t ex = cs.ids[(size_t)dst[(size_t)e]];
            const double* xa = xdq.data() + (size_t)tok[(size_t)e] * n_embd;
            std::vector<float> h((size_t)n_ff);
            for (int64_t r = 0; r < n_ff; ++r) {
                double gv = 0, uv = 0;
                for (int64_t i = 0; i < n_embd; ++i) {
                    gv += (double)gate_w[ex][(size_t)r * n_embd + i] * xa[i];
                    uv += (double)up_w[ex][(size_t)r * n_embd + i] * xa[i];
                }
                h[(size_t)r] = (float)((gv / (1.0 + std::exp(-gv))) * uv);
            }
            std::vector<double> hdq;
            quantize_q8_1(h, hdq);
            float* out_row = grouped_row.data() + (size_t)dst[(size_t)e] * n_embd;
            for (int64_t r = 0; r < n_embd; ++r) {
                double acc = 0;
                for (int64_t j = 0; j < n_ff; ++j)
                    acc += (double)down_w[ex][(size_t)r * n_ff + j] * hdq[(size_t)j];
                out_row[r] = (float)acc;
            }
        }
        // ymiss: CPU rows carry real values; GPU rows carry junk the assembly
        // must discard (copy_rows zeroes them), never add.
        std::vector<char> is_gpu((size_t)T * K, 0);
        for (int32_t r : dst) is_gpu[(size_t)r] = 1;
        std::vector<float> ymiss((size_t)T * K * n_embd);
        for (int64_t i = 0; i < T * K; ++i)
            for (int64_t j = 0; j < n_embd; ++j)
                ymiss[(size_t)i * n_embd + j] = is_gpu[(size_t)i] ? 999.5f : rng.uni(-1.0f, 1.0f);
        // Expected: hit_out keeps the stale sentinel off the GPU rows; parts
        // is the grouped result on GPU rows and the ymiss value on CPU rows.
        const float kStale = 777.25f;
        std::vector<float> want_hit((size_t)T * K * n_embd, kStale), want_parts((size_t)T * K * n_embd);
        for (int64_t i = 0; i < T * K; ++i)
            for (int64_t j = 0; j < n_embd; ++j) {
                if (is_gpu[(size_t)i]) want_hit[(size_t)i * n_embd + j] = grouped_row[(size_t)i * n_embd + j];
                want_parts[(size_t)i * n_embd + j] =
                    is_gpu[(size_t)i] ? grouped_row[(size_t)i * n_embd + j] : ymiss[(size_t)i * n_embd + j];
            }
        // Arenas: VRAM blobs at the aligned stride; staging blobs packed at
        // exactly blob_bytes (the rebase_ptrs stride record_window uses).
        std::vector<uint8_t> arena_vram((size_t)stride * vram_experts.size(), 0),
            arena_stage((size_t)blob_bytes * std::max<size_t>(pcie_experts.size(), 1), 0);
        for (size_t g = 0; g < vram_experts.size(); ++g)
            std::memcpy(arena_vram.data() + g * (size_t)stride, blob_of[vram_experts[g]].data(), (size_t)blob_bytes);
        for (size_t g = 0; g < pcie_experts.size(); ++g)
            std::memcpy(arena_stage.data() + g * (size_t)blob_bytes, blob_of[pcie_experts[g]].data(),
                        (size_t)blob_bytes);
        // Full-FFN extras: shared expert (same double chain; the sigmoid gate
        // scalar over the f32 x and the exact bf16 gate_inp values) and the
        // weighted combine of the parts reference.
        std::vector<uint8_t> sh_gate_raw, sh_up_raw, sh_down_raw;
        std::vector<uint16_t> gate_inp_bf16;
        std::vector<float> wts, want_shared, want_bo;
        if (cs.full) {
            sh_gate_raw = fake_blocks(*bi, (int)(n_ff * (n_embd / bi->elems)), rng);
            sh_up_raw = fake_blocks(*bi, (int)(n_ff * (n_embd / bi->elems)), rng);
            sh_down_raw = fake_blocks(*bi, (int)(n_embd * (n_ff / bi->elems)), rng);
            const auto sh_gate_w = dequant_all(gu, *bi, sh_gate_raw);
            const auto sh_up_w = dequant_all(gu, *bi, sh_up_raw);
            const auto sh_down_w = dequant_all(d, *bi, sh_down_raw);
            std::vector<float> gate_inp((size_t)n_embd);
            for (auto& v : gate_inp) v = rng.uni(-0.5f, 0.5f);
            gate_inp_bf16.resize((size_t)n_embd);
            std::vector<float> gate_inp_exact((size_t)n_embd);
            for (int64_t i = 0; i < n_embd; ++i) {
                uint32_t u;
                std::memcpy(&u, &gate_inp[(size_t)i], 4);
                const uint32_t lsb = (u >> 16) & 1u;
                u += 0x7FFFu + lsb;  // round to nearest even
                const uint16_t b = (uint16_t)(u >> 16);
                gate_inp_bf16[(size_t)i] = b;
                const uint32_t f = (uint32_t)b << 16;
                std::memcpy(&gate_inp_exact[(size_t)i], &f, 4);
            }
            wts.resize((size_t)T * K);
            for (auto& v : wts) v = rng.uni(0.05f, 0.3f);
            want_shared.assign((size_t)T * n_embd, 0.0f);
            for (int64_t t = 0; t < T; ++t) {
                const double* xa = xdq.data() + (size_t)t * n_embd;
                std::vector<float> h((size_t)n_ff);
                for (int64_t r = 0; r < n_ff; ++r) {
                    double gv = 0, uv = 0;
                    for (int64_t i = 0; i < n_embd; ++i) {
                        gv += (double)sh_gate_w[(size_t)r * n_embd + i] * xa[i];
                        uv += (double)sh_up_w[(size_t)r * n_embd + i] * xa[i];
                    }
                    h[(size_t)r] = (float)((gv / (1.0 + std::exp(-gv))) * uv);
                }
                std::vector<double> hdq;
                quantize_q8_1(h, hdq);
                double gd = 0;
                for (int64_t i = 0; i < n_embd; ++i) gd += (double)xf[(size_t)t * n_embd + i] * gate_inp_exact[(size_t)i];
                const double gs = 1.0 / (1.0 + std::exp(-gd));
                for (int64_t r = 0; r < n_embd; ++r) {
                    double acc = 0;
                    for (int64_t j = 0; j < n_ff; ++j)
                        acc += (double)sh_down_w[(size_t)r * n_ff + j] * hdq[(size_t)j];
                    want_shared[(size_t)t * n_embd + r] = (float)(acc * gs);
                }
            }
            want_bo.assign((size_t)T * n_embd, 0.0f);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t j = 0; j < n_embd; ++j) {
                    double acc = want_shared[(size_t)t * n_embd + j];
                    for (int64_t e = 0; e < K; ++e)
                        acc += (double)wts[(size_t)t * K + e] * want_parts[(size_t)(t * K + e) * n_embd + j];
                    want_bo[(size_t)t * n_embd + j] = (float)acc;
                }
        }
        mqfx::Fixture fx;
        fx.family = "moe";
        fx.entry = "verify_moe_assembly";
        fx.case_name = cs.name;
        fx.seed = cs.seed;
        fx.tol_kind = mqfx::TolKind::MaxRel;
        fx.tol_abs = 1e-2;
        fx.tol_rel = 2e-2;
        fx.params_i = {{"gu_type", gu}, {"d_type", d}, {"n_embd", n_embd}, {"n_ff", n_ff},
                       {"T", T}, {"K", K}, {"cap", cap}, {"capx", capx},
                       {"n_vram_groups", (int64_t)vram_experts.size()},
                       {"n_pcie_groups", (int64_t)pcie_experts.size()},
                       {"blob_stride", stride}, {"blob_bytes", blob_bytes}};
        fx.inputs.push_back(mk_u8("arena_vram", {(int64_t)arena_vram.size()}, arena_vram));
        fx.inputs.push_back(mk_u8("arena_stage", {(int64_t)arena_stage.size()}, arena_stage));
        fx.inputs.push_back(mk_i32("counts", {4}, std::vector<int32_t>(counts, counts + 4)));
        fx.inputs.push_back(mk_i32("grp_start", {(int64_t)start.size()}, start));
        fx.inputs.push_back(mk_i32("grp_start2", {(int64_t)start2.size()}, start2));
        fx.inputs.push_back(mk_i32("ent_dst", {(int64_t)dst.size()}, dst));
        fx.inputs.push_back(mk_i32("ent_tok", {(int64_t)tok.size()}, tok));
        fx.inputs.push_back(mk_u8("x_q8_1", {(int64_t)xq.size()}, xq));
        fx.inputs.push_back(mk_f32("ymiss", {T * K, n_embd}, ymiss));
        fx.outputs = {mk_f32("hit_out", {T * K, n_embd}, want_hit), mk_f32("parts", {T * K, n_embd}, want_parts)};
        if (cs.full) {
            fx.params_i.push_back({"with_shared", 1});
            fx.inputs.push_back(mk_f32("x", {T, n_embd}, xf));
            fx.inputs.push_back(mk_f32("w", {T * K}, wts));
            fx.inputs.push_back(mk_u8("sh_gate", {(int64_t)sh_gate_raw.size()}, sh_gate_raw));
            fx.inputs.push_back(mk_u8("sh_up", {(int64_t)sh_up_raw.size()}, sh_up_raw));
            fx.inputs.push_back(mk_u8("sh_down", {(int64_t)sh_down_raw.size()}, sh_down_raw));
            fx.inputs.push_back(mk_u16("sh_gate_inp", {n_embd}, gate_inp_bf16));
            fx.outputs.push_back(mk_f32("shared", {T, n_embd}, want_shared));
            fx.outputs.push_back(mk_f32("bo", {T, n_embd}, want_bo));
        }
        emit(fx);
    }
}

// ---- fused_gr_read (GRFOLD) ---------------------------------------------------
// The hyper-connection fold at its APPLY boundary: fused_gr_read as the
// window body drives it (verify.cpp gr_read_group / layer.cpp fused path),
// single token and fused_gr_read_multi, with a NONTRIVIAL bo_prev carried
// from a previous block. Upstream's gr_parity covers the CUDA multi against
// the CUDA single (same code, no independent reference) and the metal
// fixture table mislabels this entry GLM-family/not-fixtureable, so no
// metal fixture has ever checked the fold against a scalar reference.
// Reference: double-precision transcription of src/kernels/cuda/fused_gr.cu:
//   gw[c] = apply ? 2*sigmoid(inj_prev[c]/hc) : 0
//   R'[c,d] = R[c,d] + bo_prev[d]*gw[c]        (only folded when apply)
//   rs[c] = rsqrt(mean_d R'[c,d]^2 + eps);  xn = R' * w_norm * rs[c]
//   lo[k] = silu((w_down[k].xn)/hc);  inject[c] = w_inject[c].xn
//   mixed[d] = mean_c xn[c,d] * sigmoid(w_up[c*n_embd+d].lo)
//   R_out written ONLY when apply (the up kernel's contract); the runner
//   prefills R_out/inject_out with sentinels so an unwanted write - or a
//   missing one - is itself the failure. Cases include the verify shape
//   R_out == R (in-place) at T=2, which no upstream parity case exercises
//   (gr_parity's multi uses a separate R_out).
void gen_fused_gr_read() {
    const int64_t N = 2560, HC = 4, LR = 320, D = N * HC;
    const double eps = 1e-6;
    struct Case { const char* name; uint64_t seed; int64_t T; bool apply; bool inject; bool inplace; };
    const Case cases[] = {
        {"apply1_inject_t1", 0xF101, 1, true, true, false},
        {"apply0_inject_t1", 0xF102, 1, false, true, false},
        {"apply1_noinject_t1", 0xF103, 1, true, false, false},
        {"apply1_inject_t1_inplace", 0xF104, 1, true, true, true},
        {"apply1_inject_t2_multi", 0xF105, 2, true, true, false},
        {"apply1_inject_t2_inplace", 0xF106, 2, true, true, true},
        {"apply1_inject_t4_multi", 0xF107, 4, true, true, false},
    };
    auto bf2f = [](uint16_t h) { const uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; };
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    for (const Case& cs : cases) {
        const int64_t T = cs.T;
        Rng rng(cs.seed);
        std::vector<float> R((size_t)T * D), bo((size_t)T * N), inj((size_t)T * HC), w_norm((size_t)D);
        for (auto& v : R) v = rng.uni(-1.5f, 1.5f);
        for (auto& v : bo) v = rng.uni(-1.2f, 1.2f);          // a previous block's real output scale
        for (auto& v : inj) v = rng.uni(-3.0f, 3.0f);         // gates 2*sigmoid(inj/4) span ~0.64..1.36
        for (auto& v : w_norm) v = rng.uni(-0.9f, 0.9f);
        auto mk_w = [&](int64_t rows, int64_t cols) {
            std::vector<uint16_t> w((size_t)rows * cols);
            for (auto& v : w) v = f32_to_bf16(rng.uni(-0.08f, 0.08f));
            return w;
        };
        const std::vector<uint16_t> w_down = mk_w(LR, D), w_up = mk_w(D, LR), w_inj = mk_w(HC, D);
        // Double reference per token.
        const float kSentinel = 555.25f;
        std::vector<float> want_rout((size_t)T * D, kSentinel), want_lo((size_t)T * LR), want_rs((size_t)T * HC),
            want_inj((size_t)T * HC, kSentinel), want_mixed((size_t)T * N);
        for (int64_t t = 0; t < T; ++t) {
            const float* Rt = R.data() + (size_t)t * D;
            const float* bot = bo.data() + (size_t)t * N;
            const float* injt = inj.data() + (size_t)t * HC;
            std::vector<double> Rp((size_t)D), xn((size_t)D), rs((size_t)HC);
            for (int64_t c = 0; c < HC; ++c) {
                const double gw = cs.apply ? 2.0 * sigmoid((double)injt[c] / (double)HC) : 0.0;
                double ss = 0;
                for (int64_t d = 0; d < N; ++d) {
                    const double r = cs.apply ? (double)Rt[c * N + d] + (double)bot[d] * gw : (double)Rt[c * N + d];
                    Rp[(size_t)c * N + d] = r;
                    ss += r * r;
                }
                rs[(size_t)c] = 1.0 / std::sqrt(ss / (double)N + eps);
            }
            for (int64_t i = 0; i < D; ++i) xn[(size_t)i] = Rp[(size_t)i] * (double)w_norm[(size_t)i] * rs[i / N];
            std::vector<double> lo((size_t)LR);
            for (int64_t k = 0; k < LR; ++k) {
                double acc = 0;
                for (int64_t i = 0; i < D; ++i) acc += (double)bf2f(w_down[(size_t)k * D + i]) * xn[(size_t)i];
                const double x = acc / (double)HC;
                lo[(size_t)k] = x * sigmoid(x);
                want_lo[(size_t)t * LR + k] = (float)lo[(size_t)k];
            }
            if (cs.inject)
                for (int64_t c = 0; c < HC; ++c) {
                    double acc = 0;
                    for (int64_t i = 0; i < D; ++i) acc += (double)bf2f(w_inj[(size_t)c * D + i]) * xn[(size_t)i];
                    want_inj[(size_t)t * HC + c] = (float)acc;
                }
            for (int64_t d = 0; d < N; ++d) {
                double s = 0;
                for (int64_t c = 0; c < HC; ++c) {
                    double acc = 0;
                    const uint16_t* wr = w_up.data() + (size_t)(c * N + d) * LR;
                    for (int64_t k = 0; k < LR; ++k) acc += lo[(size_t)k] * (double)bf2f(wr[k]);
                    s += xn[(size_t)c * N + d] * sigmoid(acc);
                }
                want_mixed[(size_t)t * N + d] = (float)(s / (double)HC);
            }
            if (cs.apply)
                for (int64_t i = 0; i < D; ++i) want_rout[(size_t)t * D + i] = (float)Rp[(size_t)i];
            for (int64_t c = 0; c < HC; ++c) want_rs[(size_t)t * HC + c] = (float)rs[(size_t)c];
        }
        mqfx::Fixture fx;
        fx.family = "gr";
        fx.entry = "fused_gr_read";
        fx.case_name = cs.name;
        fx.seed = cs.seed;
        fx.tol_kind = mqfx::TolKind::MaxRel;
        fx.tol_abs = 1e-2;
        fx.tol_rel = 2e-2;
        fx.params_i = {{"T", T}, {"apply", cs.apply ? 1 : 0}, {"with_inject", cs.inject ? 1 : 0},
                       {"inplace", cs.inplace ? 1 : 0}, {"n_embd", N}, {"hc", HC}, {"hc_lr", LR}};
        fx.params_f = {{"eps", eps}};
        fx.inputs.push_back(mk_f32("R", {T, D}, R));
        fx.inputs.push_back(mk_f32("bo_prev", {T, N}, bo));
        fx.inputs.push_back(mk_f32("inj_prev", {T, HC}, inj));
        fx.inputs.push_back(mk_f32("w_norm", {D}, w_norm));
        fx.inputs.push_back(mk_u16("w_down", {LR, D}, w_down));
        fx.inputs.push_back(mk_u16("w_up", {D, LR}, w_up));
        if (cs.inject) fx.inputs.push_back(mk_u16("w_inject", {HC, D}, w_inj));
        fx.outputs = {mk_f32("R_out", {T, D}, want_rout), mk_f32("lo", {T, LR}, want_lo),
                      mk_f32("rs", {T, HC}, want_rs), mk_f32("inject_out", {T, HC}, want_inj),
                      mk_f32("mixed", {T, N}, want_mixed)};
        emit(fx);
    }
}

// ---- resident_plan ------------------------------------------------------------
// The E-6 device plan builder (verify_kernels.hpp): distinct experts in
// routing order, entries ascending, blob pointers as cache_base + offset
// VALUES in the plan. Expected plan computed with the layout's own integer
// arithmetic; bitwise comparison (the plan is integers, not floats). The
// plan buffer starts filled with a sentinel so untouched words are checked
// too; the non-resident case must write nothing but *skip = 0.
void gen_resident_plan() {
    struct Case {
        const char* name;
        std::vector<int32_t> ids;       // n_entries routing ids
        int k;
        std::vector<int32_t> res;       // per-expert slot or -1
        bool with_off;                  // slot_off supplied (else slot * blob)
        uint32_t ring;
        std::vector<int64_t> off;       // per-SLOT byte offsets (with_off)
    };
    const int64_t blob = 4096, capx = 8;
    const Case cases[] = {
        {"two_groups_slot_stride", {5, 2, 5, 2, 2, 5}, 2, {-1, -1, 3, -1, -1, 7}, false, 9, {}},
        {"three_groups_slot_off", {0, 4, 4, 0, 3, 3, 3}, 1, {2, -1, -1, 5, 1}, true, 4,
         {0, 4608, 9216, 13824, 18432, 23040}},
        {"one_not_resident", {1, 6, 1}, 1, {0, -1, 2, -1, -1, -1, -1}, false, 7, {}},
    };
    const int64_t plan_words = ((4 + (capx + 1) + 2 * capx + 1) & ~1ll) + 4 * capx + (capx + 1) + 1;
    for (const Case& cs : cases) {
        const int64_t n_entries = (int64_t)cs.ids.size();
        const std::vector<int64_t>& slot_off = cs.off;  // per-slot, like hits.slot_off
        std::vector<int32_t> want((size_t)plan_words, (int32_t)0x5A5A5A5A);
        uint32_t want_skip = 0;
        bool all_res = true;
        for (int32_t e : cs.ids)
            if (e < 0 || e >= (int32_t)cs.res.size() || cs.res[(size_t)e] < 0) all_res = false;
        if (all_res) {
            const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
            int32_t* start = want.data() + 4;
            int32_t* dst = start + capx + 1;
            int32_t* tok = dst + capx;
            auto* ptr = (uint64_t*) (want.data() + ptr_off);
            int32_t* start2 = want.data() + ptr_off + 4 * capx;
            int groups = 0, entries = 0;
            for (int64_t i0 = 0; i0 < n_entries; ++i0) {
                bool first = true;
                for (int64_t j = 0; j < i0; ++j)
                    if (cs.ids[(size_t)j] == cs.ids[(size_t)i0]) first = false;
                if (!first) continue;
                const int32_t slot = cs.res[(size_t)cs.ids[(size_t)i0]];
                // cache_base is passed as 0 by the runner (the entry only
                // writes the value), so the expected ptr words are the raw
                // slot offsets.
                ptr[groups] = (uint64_t) (cs.with_off ? slot_off[(size_t)slot] : (int64_t)slot * blob);
                start[groups] = entries;
                for (int64_t i = i0; i < n_entries; ++i)
                    if (cs.ids[(size_t)i] == cs.ids[(size_t)i0]) {
                        dst[entries] = (int32_t)i;
                        tok[entries] = (int32_t)(i / cs.k);
                        ++entries;
                    }
                ++groups;
            }
            start[groups] = entries;
            start2[0] = entries;
            want[0] = groups;
            want[1] = entries;
            want[2] = 0;
            want_skip = cs.ring;
        }
        mqfx::Fixture fx;
        fx.family = "verify";
        fx.entry = "resident_plan";
        fx.case_name = cs.name;
        fx.seed = 0xE100;
        fx.tol_kind = mqfx::TolKind::Bitwise;
        fx.params_i = {{"n_entries", n_entries}, {"k", cs.k}, {"n_expert", (int64_t)cs.res.size()},
                       {"blob", blob}, {"capx", capx}, {"ring", (int64_t)cs.ring},
                       {"with_off", cs.with_off ? 1 : 0}, {"plan_words", plan_words}};
        fx.inputs.push_back(mk_i32("ids", {n_entries}, cs.ids));
        fx.inputs.push_back(mk_i32("res_layer", {(int64_t)cs.res.size()}, cs.res));
        if (cs.with_off) {
            std::vector<int64_t> off64 = slot_off;
            mqfx::Tensor t;
            t.name = "slot_off";
            t.dtype = mqfx::DType::I64;
            t.shape = {(int64_t)off64.size()};
            t.bytes.assign((const uint8_t*)off64.data(), (const uint8_t*)off64.data() + off64.size() * 8);
            fx.inputs.push_back(t);
        }
        fx.outputs.push_back(mk_i32("plan", {plan_words}, want));
        std::vector<int32_t> skip_want = {(int32_t)want_skip};
        fx.outputs.push_back(mk_i32("skip", {1}, skip_want));
        emit(fx);
    }
}

// ---- routers ----------------------------------------------------------------
void router_ref(const float* logits, int n_expert, int k, int32_t* ids, float* w) {
    std::vector<double> p(n_expert);
    double mx = -1e300;
    for (int i = 0; i < n_expert; ++i) mx = std::max(mx, (double)logits[i]);
    double sum = 0;
    for (int i = 0; i < n_expert; ++i) { p[i] = std::exp((double)logits[i] - mx); sum += p[i]; }
    for (auto& v : p) v /= sum;
    std::vector<int> order(n_expert);
    for (int i = 0; i < n_expert; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
    double s = 0;
    for (int j = 0; j < k; ++j) { ids[j] = order[j]; s += p[order[j]]; }
    s = std::max(s, 6.103515625e-05);  // the 2**-14 renormalisation clamp
    for (int j = 0; j < k; ++j) w[j] = (float)(p[order[j]] / s);
}
void gen_routers() {
    {  // router_top10 at the Qwen geometry: 512 experts, k = 10
        Rng rng(0x9001);
        const int n_tok = 3, n_exp = 512, k = 10;
        std::vector<float> logits((size_t)n_tok * n_exp);
        for (auto& v : logits) v = rng.uni(-6.0f, 6.0f);
        // Row 2 is the tie case: three exactly equal top logits, the lowest
        // indices must win in order.
        for (int i = 0; i < n_exp; ++i) logits[(size_t)2 * n_exp + i] = -3.0f;
        logits[(size_t)2 * n_exp + 7] = 5.0f;
        logits[(size_t)2 * n_exp + 100] = 5.0f;
        logits[(size_t)2 * n_exp + 300] = 5.0f;
        std::vector<int32_t> ids((size_t)n_tok * k);
        std::vector<float> w((size_t)n_tok * k);
        for (int t = 0; t < n_tok; ++t) router_ref(logits.data() + (size_t)t * n_exp, n_exp, k, ids.data() + (size_t)t * k, w.data() + (size_t)t * k);
        mqfx::Fixture fx;
        fx.family = "router"; fx.entry = "router_top10"; fx.case_name = "qwen_512_top10_ties"; fx.seed = 0x9001;
        fx.tol_kind = mqfx::TolKind::MaxAbs; fx.tol_abs = 1e-6;
        fx.params_i = {{"n_tokens", n_tok}, {"n_expert", n_exp}, {"k", k}};
        fx.inputs = {mk_f32("logits", {n_tok, n_exp}, logits)};
        fx.outputs = {mk_i32("ids", {n_tok, k}, ids), mk_f32("weights", {n_tok, k}, w)};
        emit(fx);
        // native_router_top10: the same contract pinned to one token / multi rows.
        for (const char* entry : {"native_router_top10", "native_router_top10_multi"}) {
            mqfx::Fixture f2 = fx;
            f2.entry = entry;
            const bool multi = std::string(entry).find("multi") != std::string::npos;
            f2.case_name = std::string("qwen_512_top10_ties_") + (multi ? "multi" : "single");
            if (!multi) {  // single: keep row 0 only
                f2.inputs = {mk_f32("logits", {1, n_exp}, std::vector<float>(logits.begin(), logits.begin() + n_exp))};
                f2.outputs = {mk_i32("ids", {1, k}, std::vector<int32_t>(ids.begin(), ids.begin() + k)),
                              mk_f32("weights", {1, k}, std::vector<float>(w.begin(), w.begin() + k))};
                f2.params_i = {{"n_tokens", 1}, {"n_expert", n_exp}, {"k", k}};
            }
            emit(f2);
        }
    }
    {  // router_sigmoid: sigmoid + selection-only bias, renorm, scale 2.5
        Rng rng(0x9002);
        const int n_tok = 2, n_exp = 256, k = 10;
        std::vector<float> logits((size_t)n_tok * n_exp), bias(n_exp);
        for (auto& v : logits) v = rng.uni(-4.0f, 4.0f);
        for (auto& v : bias) v = rng.uni(-0.5f, 0.5f);
        std::vector<int32_t> ids((size_t)n_tok * k);
        std::vector<float> w((size_t)n_tok * k);
        for (int t = 0; t < n_tok; ++t) {
            const float* row = logits.data() + (size_t)t * n_exp;
            std::vector<double> p(n_exp), sel(n_exp);
            for (int i = 0; i < n_exp; ++i) { p[i] = 1.0 / (1.0 + std::exp(-(double)row[i])); sel[i] = p[i] + bias[i]; }
            std::vector<int> order(n_exp);
            for (int i = 0; i < n_exp; ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a] > sel[b]; });
            double s = 0;
            for (int j = 0; j < k; ++j) s += p[order[j]];
            s = std::max(s, 6.103515625e-05);
            for (int j = 0; j < k; ++j) { ids[(size_t)t * k + j] = order[j]; w[(size_t)t * k + j] = (float)(p[order[j]] / s * 2.5); }
        }
        mqfx::Fixture fx;
        fx.family = "router"; fx.entry = "router_sigmoid"; fx.case_name = "bias_select_only"; fx.seed = 0x9002;
        fx.tol_kind = mqfx::TolKind::MaxAbs; fx.tol_abs = 1e-6;
        fx.params_i = {{"n_tokens", n_tok}, {"n_expert", n_exp}, {"k", k}, {"norm_w", 1}};
        fx.params_f = {{"w_scale", 2.5}};
        fx.inputs = {mk_f32("logits", {n_tok, n_exp}, logits), mk_f32("bias", {n_exp}, bias)};
        fx.outputs = {mk_i32("ids", {n_tok, k}, ids), mk_f32("weights", {n_tok, k}, w)};
        emit(fx);
    }
}

// ---- sampler (greedy path of sample_tokens) ----------------------------------
void gen_sampler() {
    Rng rng(0x5A01);
    const int n_tok = 2, n_vocab = 1024;
    std::vector<float> logits((size_t)n_tok * n_vocab);
    for (auto& v : logits) v = rng.uni(-5.0f, 5.0f);
    logits[17] = 9.0f;  // row 0 argmax
    for (int i = 0; i < n_vocab; ++i) logits[(size_t)n_vocab + i] = -2.0f;
    logits[(size_t)n_vocab + 5] = 7.0f;
    logits[(size_t)n_vocab + 900] = 7.0f;  // exact tie: the lower index wins
    mqfx::Fixture fx;
    fx.family = "sampler"; fx.entry = "sample_tokens"; fx.case_name = "greedy_tie"; fx.seed = 0x5A01;
    fx.tol_kind = mqfx::TolKind::Bitwise;
    fx.params_i = {{"n_tokens", n_tok}, {"n_vocab", n_vocab}, {"greedy", 1}, {"top_k", 0},
                   {"penalty_last_n", 0}, {"history_len", 0}};
    fx.params_f = {{"temperature", 0.0}, {"top_p", 1.0}, {"min_p", 0.0}};
    fx.inputs = {mk_f32("logits", {n_tok, n_vocab}, logits)};
    fx.outputs = {mk_i32("out", {n_tok}, {17, 5})};
    emit(fx);
}

// ---- PLE postops --------------------------------------------------------------
void grouped_norm(const double* in, const float* gamma, double* out) {
    // Per-stream (4 streams of 2560) RMS norm with eps 1e-6 on the mean.
    for (int c = 0; c < 4; ++c) {
        double acc = 0;
        for (int d = 0; d < 2560; ++d) acc += in[c * 2560 + d] * in[c * 2560 + d];
        const double inv = 1.0 / std::sqrt(acc / 2560.0 + 1e-6);
        for (int d = 0; d < 2560; ++d) out[c * 2560 + d] = in[c * 2560 + d] * inv * (double)gamma[c * 2560 + d];
    }
}
void gen_ple() {
    {
        Rng rng(0x9E01);
        const int HC = 10240, E = 2560, HIST = 9;
        std::vector<float> pkey(HC), hidden(HC), value(E), hist((size_t)HIST * HC);
        std::vector<float> nk(HC), nq(HC), nc(HC);
        std::vector<uint16_t> convw((size_t)4 * HC);
        for (auto& v : pkey) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : hidden) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : value) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : hist) v = rng.uni(-0.8f, 0.8f);
        for (auto& v : nk) v = rng.uni(0.5f, 1.5f);
        for (auto& v : nq) v = rng.uni(0.5f, 1.5f);
        for (auto& v : nc) v = rng.uni(0.5f, 1.5f);
        for (auto& v : convw) v = f32_to_f16(rng.uni(-0.3f, 0.3f));
        std::vector<double> key(HC), query(HC), gated(HC), norm(HC), conv(HC);
        std::vector<double> pkeyd(pkey.begin(), pkey.end()), hiddend(hidden.begin(), hidden.end()), gated_in(HC);
        grouped_norm(pkeyd.data(), nk.data(), key.data());
        grouped_norm(hiddend.data(), nq.data(), query.data());
        double gate[4];
        for (int c = 0; c < 4; ++c) {
            double s = 0;
            for (int d = 0; d < 2560; ++d) s += key[c * 2560 + d] * query[c * 2560 + d];
            s /= std::sqrt(2560.0);
            const double sr = std::copysign(std::sqrt(std::max(std::fabs(s), 1e-6)), s);
            gate[c] = 1.0 / (1.0 + std::exp(-sr));
            for (int d = 0; d < 2560; ++d) gated_in[c * 2560 + d] = (double)value[d] * gate[c];
        }
        grouped_norm(gated_in.data(), nc.data(), norm.data());
        for (int ch = 0; ch < HC; ++ch) {
            double acc = 0;
            for (int k = 0; k < 4; ++k) {
                const int pos = 9 - (3 - k) * 3;  // dilation 3 over hist rows 0..8 + current at 9
                const double tap = (pos == 9) ? norm[ch] : (double)hist[(size_t)pos + (size_t)9 * ch];
                acc += (double)strata::fp16_to_fp32(convw[(size_t)k + 4 * ch]) * tap;
            }
            conv[ch] = acc;
        }
        auto tof = [](const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); };
        std::vector<float> result(HC);
        // The conv OUTPUT buffer holds the silu activation, not the raw sum:
        // native_ple_postops.cu's conv_residual_kernel writes
        // conv[c] = activation where activation = sum / (1 + expf(-sum)), and
        // the residual adds that same activation. The previous fixture stored
        // the raw sum in "conv" (result was already correct), which was the
        // whole native_ple_postops.real_geometry failure.
        std::vector<double> conv_act(HC);
        for (int i = 0; i < HC; ++i) {
            const double silu = conv[i] / (1.0 + std::exp(-conv[i]));
            conv_act[i] = silu;
            result[i] = (float)((double)hidden[i] + gated_in[i] + silu);
        }
        std::vector<float> gatef = {(float)gate[0], (float)gate[1], (float)gate[2], (float)gate[3]};
        mqfx::Fixture fx;
        fx.family = "ple"; fx.entry = "native_ple_postops"; fx.case_name = "real_geometry"; fx.seed = 0x9E01;
        fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 2e-4; fx.tol_rel = 2e-3;
        fx.inputs = {mk_f32("projected_key", {HC}, pkey), mk_f32("hidden", {HC}, hidden),
                     mk_f32("value", {E}, value), mk_f32("history", {HIST, HC}, hist),
                     mk_f32("norm_key", {HC}, nk), mk_f32("norm_query", {HC}, nq), mk_f32("norm_conv", {HC}, nc),
                     mk_u16("conv1d_f16", {4, HC}, convw)};
        fx.outputs = {mk_f32("key", {HC}, tof(key)), mk_f32("gate", {4}, gatef), mk_f32("gated", {HC}, tof(gated_in)),
                      mk_f32("normalized", {HC}, tof(norm)), mk_f32("conv", {HC}, tof(conv_act)),
                      mk_f32("result", {HC}, result)};
        emit(fx);
    }
    {  // ple_history_advance: shift rows up by one, append at the last row.
        Rng rng(0x9E02);
        const int HC = 10240, HIST = 9;
        std::vector<float> hist((size_t)HIST * HC), norm(HC);
        for (auto& v : hist) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : norm) v = rng.uni(-1.0f, 1.0f);
        std::vector<float> want((size_t)HIST * HC);
        for (int r = 0; r < HIST - 1; ++r)
            for (int ch = 0; ch < HC; ++ch) want[(size_t)r + (size_t)HIST * ch] = hist[(size_t)(r + 1) + (size_t)HIST * ch];
        for (int ch = 0; ch < HC; ++ch) want[(size_t)(HIST - 1) + (size_t)HIST * ch] = norm[ch];
        mqfx::Fixture fx;
        fx.family = "ple"; fx.entry = "ple_history_advance"; fx.case_name = "shift_append"; fx.seed = 0x9E02;
        fx.tol_kind = mqfx::TolKind::Bitwise;
        fx.inputs = {mk_f32("hist", {HIST, HC}, hist), mk_f32("normalized", {HC}, norm)};
        fx.outputs = {mk_f32("hist_out", {HIST, HC}, want)};
        emit(fx);
    }
}

// ---- GDN -----------------------------------------------------------------------
void gdn_ref(std::vector<double>& state, const std::vector<float>& q, const std::vector<float>& k,
             const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta,
             int S, int h_k, int h_v, std::vector<float>& o, bool scale_readout) {
    auto at = [&](int i, int h, int j) -> double& { return state[((size_t)i * h_v + h) * S + j]; };
    o.assign((size_t)h_v * S, 0.0f);
    for (int h = 0; h < h_v; ++h) {
        const int src = h % h_k;
        const double dec = std::exp((double)gate[h]);
        const double b = (double)beta[h];
        for (int j = 0; j < S; ++j) {
            for (int i = 0; i < S; ++i) at(i, h, j) *= dec;
            double sk = 0;
            for (int i = 0; i < S; ++i) sk += at(i, h, j) * (double)k[(size_t)src * S + i];
            const double d = ((double)v[(size_t)h * S + j] - sk) * b;
            for (int i = 0; i < S; ++i) at(i, h, j) += (double)k[(size_t)src * S + i] * d;
            double dot = 0;
            for (int i = 0; i < S; ++i) dot += at(i, h, j) * (double)q[(size_t)src * S + i];
            o[(size_t)h * S + j] = (float)(scale_readout ? dot / std::sqrt((double)S) : dot);
        }
    }
}
void gen_gdn_one(const char* entry, uint64_t seed, int S, int h_k, int h_v, bool native) {
    Rng rng(seed);
    std::vector<float> state0((size_t)S * h_v * S), q((size_t)h_k * S), k((size_t)h_k * S), v((size_t)h_v * S);
    std::vector<float> gate(h_v), beta(h_v);
    for (auto& x : state0) x = rng.uni(-0.3f, 0.3f);
    for (auto& x : q) x = rng.uni(-0.5f, 0.5f);
    for (auto& x : k) x = rng.uni(-0.5f, 0.5f);
    for (auto& x : v) x = rng.uni(-1.0f, 1.0f);
    for (auto& x : gate) x = rng.uni(-4.0f, -1.0f);  // deep decay: the regime where order traps are visible
    for (auto& x : beta) x = rng.uni(0.2f, 1.0f);
    std::vector<float> q_in = q;
    if (!native) {  // gdn_step's contract: q arrives L2-normalised and pre-scaled by 1/sqrt(S)
        for (auto& x : q_in) x = (float)((double)x / std::sqrt((double)S));
    }
    std::vector<double> st(state0.begin(), state0.end());
    std::vector<float> o;
    gdn_ref(st, q_in, k, v, gate, beta, S, h_k, h_v, o, native);
    std::vector<float> st_out(st.begin(), st.end());
    mqfx::Fixture fx;
    fx.family = "gdn"; fx.entry = entry;
    fx.case_name = native ? "s128_small_heads" : "s16_small_heads";
    fx.seed = seed;
    fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 1e-5; fx.tol_rel = 1e-4;
    fx.params_i = {{"S", S}, {"h_k", h_k}, {"h_v", h_v}};
    fx.inputs = {mk_f32("state", {S, h_v, S}, state0), mk_f32("q", {h_k, S}, q_in), mk_f32("k", {h_k, S}, k),
                 mk_f32("v", {h_v, S}, v), mk_f32("gate", {h_v}, gate), mk_f32("beta", {h_v}, beta)};
    fx.outputs = {mk_f32("state_out", {S, h_v, S}, st_out), mk_f32("o", {h_v, S}, o)};
    emit(fx);
}
void gen_gdn() {
    gen_gdn_one("gdn_step", 0x6D01, 16, 2, 4, false);
    gen_gdn_one("native_gdn_step", 0x6D02, 128, 2, 4, true);
    {  // gdn_l2_norm: eps is a floor on the SQUARED norm, not the mean
        Rng rng(0x6D03);
        const int rows = 4, cols = 128;
        std::vector<float> x((size_t)rows * cols);
        for (auto& v : x) v = rng.uni(-3.0f, 3.0f);
        std::vector<float> want = x;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double)x[(size_t)r * cols + i] * x[(size_t)r * cols + i];
            const double inv = 1.0 / std::sqrt(acc + 1e-6);
            for (int i = 0; i < cols; ++i) want[(size_t)r * cols + i] = (float)((double)x[(size_t)r * cols + i] * inv);
        }
        mqfx::Fixture fx;
        fx.family = "gdn"; fx.entry = "gdn_l2_norm"; fx.case_name = "eps_on_sumsq"; fx.seed = 0x6D03;
        fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 1e-6; fx.tol_rel = 1e-6;
        fx.params_i = {{"rows", rows}, {"cols", cols}};
        fx.params_f = {{"eps", 1e-6}};
        fx.inputs = {mk_f32("x", {rows, cols}, x)};
        fx.outputs = {mk_f32("out", {rows, cols}, want)};
        emit(fx);
    }
}

// ---- prefill (prompt-chunk kernels, include/strata/prefill/kernels.hpp) --------
// References: scalar transcriptions of src/prefill/kernels.cu, computed in
// double except where the kernel's own rounding is the contract (the int8
// KV append, whose scale/quantize steps are float by construction). The
// geometry constants are the kernel TU's own (N 2560, HC 4, LR 320, S 128,
// HK 16, HV 48, C 10240). 16-bit outputs are stored as F16/BF16 tensors so
// the runner compares by value; their tolerances cover one ulp of the
// format (bf16 2^-7, f16 2^-10 relative) plus the reference's slack.
mqfx::Tensor mk_f16t(const char* name, std::vector<int64_t> shape, const std::vector<uint16_t>& v) {
    std::vector<uint8_t> b(v.size() * 2);
    std::memcpy(b.data(), v.data(), b.size());
    return mk(name, mqfx::DType::F16, std::move(shape), std::move(b));
}
mqfx::Tensor mk_bf16t(const char* name, std::vector<int64_t> shape, const std::vector<uint16_t>& v) {
    std::vector<uint8_t> b(v.size() * 2);
    std::memcpy(b.data(), v.data(), b.size());
    return mk(name, mqfx::DType::BF16, std::move(shape), std::move(b));
}
double sigm_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }
float pf_f16_val(uint16_t h) {  // IEEE half bits -> float (exact)
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const int exp = (int)((h >> 10) & 0x1Fu);
    const uint32_t man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0x1F) u = sign | 0x7F800000u | (man << 13);
    else if (exp == 0) {
        if (man == 0) u = sign;
        else {
            int e = -1;
            uint32_t mm = man;
            while ((mm & 0x400u) == 0) { mm <<= 1; ++e; }
            u = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mm & 0x3FFu) << 13);
        }
    } else u = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void pf_base(mqfx::Fixture& fx, const char* entry, const char* case_name, uint64_t seed, mqfx::TolKind kind,
             double ta, double tr) {
    fx.family = "prefill"; fx.entry = entry; fx.case_name = case_name; fx.seed = seed;
    fx.tol_kind = kind; fx.tol_abs = ta; fx.tol_rel = tr;
}

void gen_prefill() {
    const int N = 2560, HC = 4, D = N * HC, LR = 320, S = 128, HK = 16, HV = 48, C = 10240;
    const double EPS = 1e-6;
    // ---- hyper-connection ----
    {
        const int T = 2;
        Rng rng(0x9F00);
        std::vector<float> R((size_t)T * D), w((size_t)HC * N);
        for (auto& v : R) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : w) v = rng.uni(0.5f, 1.5f);
        std::vector<float> xn((size_t)T * D);
        std::vector<uint16_t> xn16((size_t)T * D);
        std::vector<float> rs((size_t)T * HC);
        for (int row = 0; row < T * HC; ++row) {
            double ss = 0;
            for (int d = 0; d < N; ++d) ss += (double)R[(size_t)row * N + d] * R[(size_t)row * N + d];
            const double r = 1.0 / std::sqrt(ss / N + EPS);
            rs[row] = (float)r;
            const int c = row % HC;
            for (int d = 0; d < N; ++d) {
                const double v = (double)R[(size_t)row * N + d] * r * w[(size_t)c * N + d];
                xn[(size_t)row * N + d] = (float)v;
                xn16[(size_t)row * N + d] = f32_to_bf16((float)v);
            }
        }
        // One fixture per tolerance class: the f32 outputs check tight, the
        // bf16 image checks at one-ulp-of-format slack (a fixture's
        // tolerance is shared by all its outputs).
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_norm", "t2", 0x9F00, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
            fx.params_i = {{"T", T}};
            fx.params_f = {{"eps", EPS}};
            fx.inputs = {mk_f32("R", {T, D}, R), mk_f32("w_norm", {HC, N}, w)};
            fx.outputs = {mk_f32("xn", {T, D}, xn)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_norm", "t2_bf16", 0x9F00, mqfx::TolKind::MaxRel, 1e-3, 2e-2);
            fx.params_i = {{"T", T}};
            fx.params_f = {{"eps", EPS}};
            fx.inputs = {mk_f32("R", {T, D}, R), mk_f32("w_norm", {HC, N}, w)};
            fx.outputs = {mk_bf16t("xn16", {T, D}, xn16)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_norm_rs", "t2", 0x9F00, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
            fx.params_i = {{"T", T}};
            fx.params_f = {{"eps", EPS}};
            fx.inputs = {mk_f32("R", {T, D}, R), mk_f32("w_norm", {HC, N}, w)};
            fx.outputs = {mk_f32("rs", {T, HC}, rs)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_norm_rs", "t2_bf16", 0x9F00, mqfx::TolKind::MaxRel, 1e-3, 2e-2);
            fx.params_i = {{"T", T}};
            fx.params_f = {{"eps", EPS}};
            fx.inputs = {mk_f32("R", {T, D}, R), mk_f32("w_norm", {HC, N}, w)};
            fx.outputs = {mk_bf16t("xn16", {T, D}, xn16)};
            emit(fx);
        }
        // gr_mix_r / gr_mix share the mix math; gr_mix consumes xn, gr_mix_r
        // recomputes it from R/rs/w.
        std::vector<float> gated((size_t)T * D);
        for (auto& v : gated) v = rng.uni(-2.0f, 2.0f);
        std::vector<float> mixed((size_t)T * N);
        std::vector<uint16_t> mixed16((size_t)T * N), mixed_h((size_t)T * N);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < N; ++d) {
                double s = 0;
                for (int c = 0; c < HC; ++c)
                    s += (double)xn[(size_t)(t * HC + c) * N + d] * sigm_d(gated[(size_t)(t * HC + c) * N + d]);
                s /= HC;
                mixed[(size_t)t * N + d] = (float)s;
                mixed16[(size_t)t * N + d] = f32_to_bf16((float)s);
                mixed_h[(size_t)t * N + d] = f32_to_f16((float)s);
            }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_mix_r", "t2", 0x9F01, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
            fx.params_i = {{"T", T}};
            fx.inputs = {mk_f32("R", {T, D}, R), mk_f32("rs", {T, HC}, rs), mk_f32("w_norm", {HC, N}, w),
                         mk_f32("gated", {T, D}, gated)};
            fx.outputs = {mk_f32("mixed", {T, N}, mixed)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_mix_r", "t2_img", 0x9F01, mqfx::TolKind::MaxRel, 1e-3, 2e-2);
            fx.params_i = {{"T", T}};
            fx.inputs = {mk_f32("R", {T, D}, R), mk_f32("rs", {T, HC}, rs), mk_f32("w_norm", {HC, N}, w),
                         mk_f32("gated", {T, D}, gated)};
            fx.outputs = {mk_bf16t("mixed16", {T, N}, mixed16), mk_f16t("mixed_h", {T, N}, mixed_h)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_mix", "t2", 0x9F01, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
            fx.params_i = {{"T", T}};
            fx.inputs = {mk_f32("xn", {T, D}, xn), mk_f32("gated", {T, D}, gated)};
            fx.outputs = {mk_f32("mixed", {T, N}, mixed)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_mix", "t2_img", 0x9F01, mqfx::TolKind::MaxRel, 1e-3, 2e-2);
            fx.params_i = {{"T", T}};
            fx.inputs = {mk_f32("xn", {T, D}, xn), mk_f32("gated", {T, D}, gated)};
            fx.outputs = {mk_bf16t("mixed16", {T, N}, mixed16), mk_f16t("mixed_h", {T, N}, mixed_h)};
            emit(fx);
        }
    }
    {
        const int T = 2;
        Rng rng(0x9F02);
        std::vector<float> lo((size_t)T * LR);
        for (auto& v : lo) v = rng.uni(-6.0f, 6.0f);
        std::vector<uint16_t> lo16(lo.size());
        for (size_t i = 0; i < lo.size(); ++i) {
            const double x = (double)lo[i] / HC;
            lo16[i] = f32_to_bf16((float)(x / (1.0 + std::exp(-x))));
        }
        mqfx::Fixture fx;
        pf_base(fx, "gr_silu", "t2", 0x9F02, mqfx::TolKind::MaxRel, 1e-3, 2e-2);
        fx.params_i = {{"T", T}};
        fx.inputs = {mk_f32("lo", {T, LR}, lo)};
        fx.outputs = {mk_bf16t("lo16", {T, LR}, lo16)};
        emit(fx);
    }
    {
        const int T = 2;
        Rng rng(0x9F03);
        std::vector<float> R0((size_t)T * D), bo((size_t)T * N), inj((size_t)T * HC), wn((size_t)HC * N);
        for (auto& v : R0) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : bo) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : inj) v = rng.uni(-2.0f, 2.0f);
        for (auto& v : wn) v = rng.uni(0.5f, 1.5f);
        std::vector<float> Rw = R0;
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < HC; ++c) {
                const double sc = 2.0 * sigm_d((double)inj[(size_t)t * HC + c] / HC);
                for (int d = 0; d < N; ++d)
                    Rw[(size_t)(t * HC + c) * N + d] =
                        (float)((double)Rw[(size_t)(t * HC + c) * N + d] + (double)bo[(size_t)t * N + d] * sc);
            }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_write", "t2", 0x9F03, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
            fx.params_i = {{"T", T}, {"inj_ld", HC}};
            fx.inputs = {mk_f32("R", {T, D}, R0), mk_f32("bo", {T, N}, bo), mk_f32("inj", {T, HC}, inj)};
            fx.outputs = {mk_f32("R_out", {T, D}, Rw)};
            emit(fx);
        }
        // gr_write_norm_rs over the same written rows, with wn as next norm
        std::vector<float> rs((size_t)T * HC);
        std::vector<uint16_t> xn16((size_t)T * D);
        for (int row = 0; row < T * HC; ++row) {
            double ss = 0;
            for (int d = 0; d < N; ++d) ss += (double)Rw[(size_t)row * N + d] * Rw[(size_t)row * N + d];
            const double r = 1.0 / std::sqrt(ss / N + EPS);
            rs[row] = (float)r;
            const int c = row % HC;
            for (int d = 0; d < N; ++d)
                xn16[(size_t)row * N + d] = f32_to_bf16((float)((double)Rw[(size_t)row * N + d] * r * wn[(size_t)c * N + d]));
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_write_norm_rs", "t2", 0x9F03, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
            fx.params_i = {{"T", T}, {"inj_ld", HC}};
            fx.params_f = {{"eps", EPS}};
            fx.inputs = {mk_f32("R", {T, D}, R0), mk_f32("bo", {T, N}, bo), mk_f32("inj", {T, HC}, inj),
                         mk_f32("w_norm_next", {HC, N}, wn)};
            fx.outputs = {mk_f32("R_out", {T, D}, Rw), mk_f32("rs", {T, HC}, rs)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gr_write_norm_rs", "t2_bf16", 0x9F03, mqfx::TolKind::MaxRel, 1e-3, 2e-2);
            fx.params_i = {{"T", T}, {"inj_ld", HC}};
            fx.params_f = {{"eps", EPS}};
            fx.inputs = {mk_f32("R", {T, D}, R0), mk_f32("bo", {T, N}, bo), mk_f32("inj", {T, HC}, inj),
                         mk_f32("w_norm_next", {HC, N}, wn)};
            fx.outputs = {mk_bf16t("xn16", {T, D}, xn16)};
            emit(fx);
        }
        std::vector<float> e((size_t)T * N), Rb((size_t)T * D);
        for (auto& v : e) v = rng.uni(-1.0f, 1.0f);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < HC; ++c)
                for (int d = 0; d < N; ++d) Rb[(size_t)(t * HC + c) * N + d] = e[(size_t)t * N + d];
        mqfx::Fixture fx;
        pf_base(fx, "gr_broadcast", "t2", 0x9F03, mqfx::TolKind::Bitwise, 0, 0);
        fx.params_i = {{"T", T}};
        fx.inputs = {mk_f32("e", {T, N}, e)};
        fx.outputs = {mk_f32("R_out", {T, D}, Rb)};
        emit(fx);
    }
    // ---- GDN ----
    {
        const int T = 3;
        Rng rng(0x9F10);
        std::vector<float> ab((size_t)T * 2 * HV), dt(HV), sa(HV);
        for (auto& v : ab) v = rng.uni(-3.0f, 3.0f);
        for (auto& v : dt) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : sa) v = rng.uni(-2.0f, -0.5f);
        std::vector<float> gate((size_t)T * HV), beta((size_t)T * HV);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < HV; ++h) {
                const double v = (double)ab[(size_t)t * 2 * HV + h] + dt[h];
                gate[(size_t)t * HV + h] = (float)((v > 20.0 ? v : std::log1p(std::exp(v))) * sa[h]);
                beta[(size_t)t * HV + h] = (float)sigm_d(ab[(size_t)t * 2 * HV + HV + h]);
            }
        mqfx::Fixture fx;
        pf_base(fx, "gdn_gates", "t3", 0x9F10, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
        fx.params_i = {{"T", T}};
        fx.inputs = {mk_f32("ab", {T, 2 * HV}, ab), mk_f32("dt", {HV}, dt), mk_f32("ssm_a", {HV}, sa)};
        fx.outputs = {mk_f32("gate", {T, HV}, gate), mk_f32("beta", {T, HV}, beta)};
        emit(fx);
    }
    for (int T : {3, 65}) {  // 3: the serial walk; 65: the tiled walk + hist fixup
        Rng rng(0x9F11 + (uint64_t)T);
        std::vector<float> hist0((size_t)C * 3), qkv((size_t)T * C), cw((size_t)C * 4);
        for (auto& v : hist0) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : qkv) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : cw) v = rng.uni(-0.5f, 0.5f);
        std::vector<double> h((size_t)T * C);
        std::vector<float> hist = hist0;
        for (int c = 0; c < C; ++c) {
            double v0 = hist0[(size_t)c * 3], v1 = hist0[(size_t)c * 3 + 1], v2 = hist0[(size_t)c * 3 + 2];
            for (int t = 0; t < T; ++t) {
                const double x = qkv[(size_t)t * C + c];
                const double s = v0 * cw[(size_t)c * 4] + v1 * cw[(size_t)c * 4 + 1] + v2 * cw[(size_t)c * 4 + 2] +
                                 x * cw[(size_t)c * 4 + 3];
                h[(size_t)t * C + c] = s / (1.0 + std::exp(-s));
                v0 = v1; v1 = v2; v2 = x;
            }
            hist[(size_t)c * 3] = (float)v0; hist[(size_t)c * 3 + 1] = (float)v1; hist[(size_t)c * 3 + 2] = (float)v2;
        }
        for (int t = 0; t < T; ++t)  // the L2 norm of the q and k head rows
            for (int head = 0; head < 2 * HK; ++head) {
                double ss = 0;
                for (int i = 0; i < S; ++i) ss += h[(size_t)t * C + head * S + i] * h[(size_t)t * C + head * S + i];
                const double inv = 1.0 / std::sqrt(ss + EPS);
                for (int i = 0; i < S; ++i) h[(size_t)t * C + head * S + i] *= inv;
            }
        std::vector<float> hf(h.begin(), h.end());
        mqfx::Fixture fx;
        pf_base(fx, "gdn_conv", T == 3 ? "t3_serial" : "t65_tiled", 0x9F11 + (uint64_t)T, mqfx::TolKind::MaxRel,
                1e-5, 1e-4);
        fx.params_i = {{"T", T}};
        fx.params_f = {{"eps", EPS}};
        fx.inputs = {mk_f32("history", {C, 3}, hist0), mk_f32("qkv", {T, C}, qkv), mk_f32("conv_w", {C, 4}, cw)};
        fx.outputs = {mk_f32("history_out", {C, 3}, hist), mk_f32("h", {T, C}, hf)};
        emit(fx);
    }
    {
        const int T = 2;
        Rng rng(0x9F12);
        std::vector<float> state0((size_t)S * HV * S), h((size_t)T * C), gate((size_t)T * HV), beta((size_t)T * HV),
            z((size_t)T * HV * S), gamma(S);
        for (auto& v : state0) v = rng.uni(-0.2f, 0.2f);
        for (auto& v : h) v = rng.uni(-0.7f, 0.7f);
        for (auto& v : gate) v = rng.uni(-2.0f, -0.2f);  // the kernel exponentiates the gate
        for (auto& v : beta) v = rng.uni(0.1f, 0.9f);
        for (auto& v : z) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : gamma) v = rng.uni(0.5f, 1.5f);
        // double state, indexed [row][head][col] with row stride HV*S
        std::vector<double> st(state0.begin(), state0.end());
        std::vector<float> y((size_t)T * HV * S);
        std::vector<uint16_t> y16(y.size());
        for (int head = 0; head < HV; ++head) {
            const int qh = head % HK;
            std::vector<double> oc(S);
            for (int t = 0; t < T; ++t) {
                const double g = std::exp((double)gate[(size_t)t * HV + head]);
                const double b = beta[(size_t)t * HV + head];
                for (int col = 0; col < S; ++col) {
                    double kv = 0;
                    for (int r = 0; r < S; ++r)
                        kv += st[((size_t)r * HV + head) * S + col] * h[(size_t)t * C + HK * S + qh * S + r];
                    const double delta = ((double)h[(size_t)t * C + 2 * HK * S + head * S + col] - g * kv) * b;
                    double o = 0;
                    for (int r = 0; r < S; ++r) {
                        double& sref = st[((size_t)r * HV + head) * S + col];
                        sref = g * sref + (double)h[(size_t)t * C + HK * S + qh * S + r] * delta;
                        o += sref * h[(size_t)t * C + qh * S + r];
                    }
                    oc[col] = o / std::sqrt((double)S);
                }
                double ss = 0;
                for (int col = 0; col < S; ++col) ss += oc[col] * oc[col];
                const double inv = 1.0 / std::sqrt(ss / S + EPS);
                for (int col = 0; col < S; ++col) {
                    const double v = oc[col] * inv * gamma[col] *
                                     sigm_d(z[((size_t)t * HV + head) * S + col]);
                    y[((size_t)t * HV + head) * S + col] = (float)v;
                    y16[((size_t)t * HV + head) * S + col] = f32_to_f16((float)v);
                }
            }
        }
        std::vector<float> state_out(st.begin(), st.end());
        mqfx::Fixture fx;
        pf_base(fx, "gdn_recurrence", "t2", 0x9F12, mqfx::TolKind::MaxRel, 2e-4, 2e-3);
        fx.params_i = {{"T", T}};
        fx.params_f = {{"eps", EPS}};
        fx.inputs = {mk_f32("state", {S, HV, S}, state0), mk_f32("h", {T, C}, h), mk_f32("gate", {T, HV}, gate),
                     mk_f32("beta", {T, HV}, beta), mk_f32("z", {T, HV, S}, z), mk_f32("gamma", {S}, gamma)};
        fx.outputs = {mk_f32("state_out", {S, HV, S}, state_out), mk_f32("y", {T, HV, S}, y),
                      mk_f16t("y16", {T, HV, S}, y16)};
        emit(fx);
    }
    // ---- MoE ----
    for (int NE : {512, 256}) {
        const int T = 3;
        Rng rng(0x9F20 + (uint64_t)NE);
        std::vector<float> logits((size_t)T * NE);
        for (auto& v : logits) v = rng.uni(-4.0f, 4.0f);
        std::vector<int32_t> ids((size_t)T * 10);
        std::vector<float> wts((size_t)T * 10);
        for (int t = 0; t < T; ++t) {
            std::vector<double> p(NE);
            double mx = -1e300;
            for (int i = 0; i < NE; ++i) mx = std::max(mx, (double)logits[(size_t)t * NE + i]);
            double sum = 0;
            for (int i = 0; i < NE; ++i) { p[i] = std::exp((double)logits[(size_t)t * NE + i] - mx); sum += p[i]; }
            for (int i = 0; i < NE; ++i) p[i] /= sum;
            double sel_sum = 0;
            std::vector<double> sel(10);
            for (int rank = 0; rank < 10; ++rank) {
                int best = 0;
                for (int i = 1; i < NE; ++i)
                    if (p[i] > p[best]) best = i;
                ids[(size_t)t * 10 + rank] = best;
                sel[rank] = p[best];
                sel_sum += p[best];
                p[best] = -std::numeric_limits<double>::infinity();
            }
            sel_sum = std::max(sel_sum, 6.103515625e-5);
            for (int rank = 0; rank < 10; ++rank) wts[(size_t)t * 10 + rank] = (float)(sel[rank] / sel_sum);
        }
        mqfx::Fixture fx;
        pf_base(fx, "route", NE == 512 ? "ne512" : "ne256", 0x9F20 + (uint64_t)NE, mqfx::TolKind::MaxRel, 1e-6,
                1e-4);
        fx.params_i = {{"T", T}, {"n_expert", NE}};
        fx.inputs = {mk_f32("logits", {T, NE}, logits)};
        fx.outputs = {mk_i32("ids", {T, 10}, ids), mk_f32("weights", {T, 10}, wts)};
        emit(fx);
    }
    {
        Rng rng(0x9F21);
        const int n = 3;
        std::vector<float> gu((size_t)n * 1280), g((size_t)n * 640), u((size_t)n * 640);
        for (auto& v : gu) v = rng.uni(-3.0f, 3.0f);
        for (auto& v : g) v = rng.uni(-3.0f, 3.0f);
        for (auto& v : u) v = rng.uni(-3.0f, 3.0f);
        auto silu_mul = [](double a, double b) { return a / (1.0 + std::exp(-a)) * b; };
        std::vector<uint16_t> hil((size_t)n * 640), hsplit((size_t)n * 640), hpair((size_t)n * 640);
        for (int r = 0; r < n; ++r)
            for (int k = 0; k < 640; ++k) {
                hil[(size_t)r * 640 + k] = f32_to_f16((float)silu_mul(gu[(size_t)r * 1280 + 2 * k], gu[(size_t)r * 1280 + 2 * k + 1]));
                hsplit[(size_t)r * 640 + k] = f32_to_f16((float)silu_mul(gu[(size_t)r * 1280 + k], gu[(size_t)r * 1280 + 640 + k]));
                hpair[(size_t)r * 640 + k] = f32_to_f16((float)silu_mul(g[(size_t)r * 640 + k], u[(size_t)r * 640 + k]));
            }
        {
            mqfx::Fixture fx;
            pf_base(fx, "swiglu_interleaved", "n3", 0x9F21, mqfx::TolKind::MaxRel, 1e-4, 3e-3);
            fx.params_i = {{"n", n}};
            fx.inputs = {mk_f32("gu", {n, 1280}, gu)};
            fx.outputs = {mk_f16t("h16", {n, 640}, hil)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "swiglu_split", "n3", 0x9F21, mqfx::TolKind::MaxRel, 1e-4, 3e-3);
            fx.params_i = {{"n", n}};
            fx.inputs = {mk_f32("gu", {n, 1280}, gu)};
            fx.outputs = {mk_f16t("h16", {n, 640}, hsplit)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "swiglu_pair", "n3", 0x9F21, mqfx::TolKind::MaxRel, 1e-4, 3e-3);
            fx.params_i = {{"n", n}};
            fx.inputs = {mk_f32("g", {n, 640}, g), mk_f32("u", {n, 640}, u)};
            fx.outputs = {mk_f16t("h16", {n, 640}, hpair)};
            emit(fx);
        }
    }
    {
        Rng rng(0x9F22);
        std::vector<int32_t> src(1000);
        for (auto& v : src) v = (int32_t)(rng.next() % 200000) - 100000;
        mqfx::Fixture fx;
        pf_base(fx, "copy_i32", "n1000", 0x9F22, mqfx::TolKind::Bitwise, 0, 0);
        fx.params_i = {{"n", 1000}};
        fx.inputs = {mk_i32("src", {1000}, src)};
        fx.outputs = {mk_i32("dst", {1000}, src)};
        emit(fx);
    }
    {
        Rng rng(0x9F23);
        const int n = 5, width = 2560, rows = 8;
        std::vector<uint16_t> x((size_t)rows * width);
        for (auto& v : x) v = f32_to_bf16(rng.uni(-2.0f, 2.0f));
        std::vector<int32_t> srcv = {3, 0, 7, 7, 1};
        std::vector<uint16_t> dst((size_t)n * width);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < width; ++j) dst[(size_t)i * width + j] = x[(size_t)srcv[i] * width + j];
        mqfx::Fixture fx;
        pf_base(fx, "gather_rows16", "n5_w2560", 0x9F23, mqfx::TolKind::Bitwise, 0, 0);
        fx.params_i = {{"n", n}, {"width", width}};
        fx.inputs = {mk("x16", mqfx::DType::BF16, {rows, width},
                        [&] { std::vector<uint8_t> b(x.size() * 2); std::memcpy(b.data(), x.data(), b.size()); return b; }()),
                     mk_i32("src", {n}, srcv)};
        fx.outputs = {mk("dst16", mqfx::DType::BF16, {n, width},
                         [&] { std::vector<uint8_t> b(dst.size() * 2); std::memcpy(b.data(), dst.data(), b.size()); return b; }())};
        emit(fx);
    }
    {
        const int T = 2, rows = T * 10;
        Rng rng(0x9F24);
        std::vector<float> Dm((size_t)rows * N), w((size_t)T * 10), shared((size_t)T * N), sg(T);
        std::vector<int32_t> slot((size_t)T * 10);
        for (auto& v : Dm) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : w) v = rng.uni(0.0f, 1.0f);
        for (auto& v : shared) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : sg) v = rng.uni(-2.0f, 2.0f);
        for (auto& v : slot) v = (int32_t)(rng.next() % rows);
        std::vector<float> bo((size_t)T * N);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < N; ++d) {
                double s = 0;
                for (int k = 0; k < 10; ++k)
                    s += (double)w[(size_t)t * 10 + k] * Dm[(size_t)slot[(size_t)t * 10 + k] * N + d];
                bo[(size_t)t * N + d] = (float)(s + (double)shared[(size_t)t * N + d] * sigm_d(sg[t]));
            }
        mqfx::Fixture fx;
        pf_base(fx, "moe_combine", "t2", 0x9F24, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
        fx.params_i = {{"T", T}};
        fx.inputs = {mk_f32("D", {rows, N}, Dm), mk_i32("slot", {T, 10}, slot), mk_f32("w", {T, 10}, w),
                     mk_f32("shared", {T, N}, shared), mk_f32("sg", {T}, sg)};
        fx.outputs = {mk_f32("bo", {T, N}, bo)};
        emit(fx);
    }
    // ---- QSA helpers ----
    {
        Rng rng(0x9F30);
        struct Case { int rows, cols, ld; const char* name; };
        for (Case cs : {Case{3, 2560, 2560, "rows3_full"}, Case{2, 128, 192, "rows2_strided"}}) {
            std::vector<float> x0((size_t)cs.rows * cs.ld), w(cs.cols);
            for (auto& v : x0) v = rng.uni(-2.0f, 2.0f);
            for (auto& v : w) v = rng.uni(0.5f, 1.5f);
            std::vector<float> x = x0;
            for (int r = 0; r < cs.rows; ++r) {
                double ss = 0;
                for (int c = 0; c < cs.cols; ++c) ss += (double)x0[(size_t)r * cs.ld + c] * x0[(size_t)r * cs.ld + c];
                const double s = 1.0 / std::sqrt(ss / cs.cols + EPS);
                for (int c = 0; c < cs.cols; ++c)
                    x[(size_t)r * cs.ld + c] = (float)((double)x0[(size_t)r * cs.ld + c] * s * w[c]);
            }
            mqfx::Fixture fx;
            pf_base(fx, "rms_rows", cs.name, 0x9F30, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
            fx.params_i = {{"rows", cs.rows}, {"cols", cs.cols}, {"ld", cs.ld}};
            fx.params_f = {{"eps", EPS}};
            fx.inputs = {mk_f32("x", {cs.rows, cs.ld}, x0), mk_f32("w", {cs.cols}, w)};
            fx.outputs = {mk_f32("x_out", {cs.rows, cs.ld}, x)};
            emit(fx);
        }
    }
    {
        const int T = 2, heads = 24, dim = 256, ld = 6144, pos0 = 5;
        Rng rng(0x9F31);
        std::vector<float> x0((size_t)T * ld);
        for (auto& v : x0) v = rng.uni(-1.0f, 1.0f);
        std::vector<float> x = x0;
        const double theta_scale = std::pow(1e7, -2.0 / 64.0);
        for (int t = 0; t < T; ++t)
            for (int hd = 0; hd < heads; ++hd)
                for (int pair = 0; pair < 32; ++pair) {
                    const double th = (double)(pos0 + t) * std::pow(theta_scale, (double)pair);
                    const double co = std::cos(th), si = std::sin(th);
                    float& a = x[(size_t)t * ld + hd * dim + pair];
                    float& b = x[(size_t)t * ld + hd * dim + pair + 32];
                    const double av = a, bv = b;
                    a = (float)(av * co - bv * si);
                    b = (float)(av * si + bv * co);
                }
        mqfx::Fixture fx;
        pf_base(fx, "rope", "t2_q", 0x9F31, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
        fx.params_i = {{"T", T}, {"heads", heads}, {"dim", dim}, {"ld", ld}, {"pos0", pos0}};
        fx.params_f = {{"freq_base", 1e7}};
        fx.inputs = {mk_f32("x", {T, ld}, x0)};
        fx.outputs = {mk_f32("x_out", {T, ld}, x)};
        emit(fx);
    }
    {
        const int T = 2;
        Rng rng(0x9F32);
        std::vector<float> qf((size_t)T * 24 * 512), attn((size_t)T * 24 * 256);
        for (auto& v : qf) v = rng.uni(-2.0f, 2.0f);
        for (auto& v : attn) v = rng.uni(-1.0f, 1.0f);
        std::vector<float> q((size_t)T * 24 * 256);
        std::vector<uint16_t> o16(q.size());
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < 24; ++h)
                for (int d = 0; d < 256; ++d) {
                    q[((size_t)t * 24 + h) * 256 + d] = qf[((size_t)t * 24 + h) * 512 + d];
                    o16[((size_t)t * 24 + h) * 256 + d] =
                        f32_to_f16((float)((double)attn[((size_t)t * 24 + h) * 256 + d] *
                                           sigm_d(qf[((size_t)t * 24 + h) * 512 + 256 + d])));
                }
        {
            mqfx::Fixture fx;
            pf_base(fx, "split_q", "t2", 0x9F32, mqfx::TolKind::Bitwise, 0, 0);
            fx.params_i = {{"T", T}};
            fx.inputs = {mk_f32("q_full", {T, 24, 512}, qf)};
            fx.outputs = {mk_f32("q", {T, 24, 256}, q)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "gate_attn", "t2", 0x9F32, mqfx::TolKind::MaxRel, 1e-4, 3e-3);
            fx.params_i = {{"T", T}};
            fx.inputs = {mk_f32("attn", {T, 24, 256}, attn), mk_f32("q_full", {T, 24, 512}, qf)};
            fx.outputs = {mk_f16t("out16", {T, 24, 256}, o16)};
            emit(fx);
        }
    }
    {
        Rng rng(0x9F33);
        const int n = 1000;
        std::vector<float> x(n);
        for (auto& v : x) v = rng.uni(-30.0f, 30.0f);
        std::vector<uint16_t> y16(n), yb(n);
        std::vector<float> yr(n);
        for (int i = 0; i < n; ++i) {
            y16[i] = f32_to_f16(x[i]);
            yb[i] = f32_to_bf16(x[i]);
            yr[i] = pf_f16_val(y16[i]);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "to_f16", "n1000", 0x9F33, mqfx::TolKind::MaxRel, 0, 2e-3);
            fx.params_i = {{"n", n}};
            fx.inputs = {mk_f32("x", {n}, x)};
            fx.outputs = {mk_f16t("y", {n}, y16)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "to_bf16", "n1000", 0x9F33, mqfx::TolKind::MaxRel, 0, 2e-2);
            fx.params_i = {{"n", n}};
            fx.inputs = {mk_f32("x", {n}, x)};
            fx.outputs = {mk_bf16t("y", {n}, yb)};
            emit(fx);
        }
        {
            mqfx::Fixture fx;
            pf_base(fx, "round_f16", "n1000", 0x9F33, mqfx::TolKind::MaxRel, 0, 2e-3);
            fx.params_i = {{"n", n}};
            fx.inputs = {mk_f32("x", {n}, x)};
            fx.outputs = {mk_f32("y", {n}, yr)};
            emit(fx);
        }
    }
    // kv_append: pools sized [64][256]; page_size 4, pos0 3 (the two tokens
    // straddle a page boundary; the second page is non-resident).
    {
        const int T = 2, page_size = 4, pos0 = 3, pool_rows = 64;
        Rng rng(0x9F34);
        std::vector<float> K((size_t)T * 512), V((size_t)T * 512);
        for (auto& v : K) v = rng.uni(-2.0f, 2.0f);
        for (auto& v : V) v = rng.uni(-2.0f, 2.0f);
        std::vector<int32_t> table = {1, -1, 0, 2, 0, 0, 0, 0};
        {
            std::vector<uint16_t> kp((size_t)pool_rows * 256, 0), vp = kp, hkp = kp, hvp = kp;
            auto put = [&](std::vector<uint16_t>& pool, bool id_layout, int kvh, int d, long pos, uint16_t hv) {
                const long page = table[pos / page_size];
                const long row = id_layout ? ((pos / page_size) * 2 + kvh) * page_size + pos % page_size
                                           : (page * 2 + kvh) * page_size + pos % page_size;
                if (id_layout || page >= 0) pool[(size_t)row * 256 + d] = hv;
            };
            for (int t = 0; t < T; ++t)
                for (int kvh = 0; kvh < 2; ++kvh)
                    for (int d = 0; d < 256; ++d) {
                        const long pos = pos0 + t;
                        const uint16_t hk = f32_to_f16(K[(size_t)t * 512 + kvh * 256 + d]);
                        const uint16_t hv = f32_to_f16(V[(size_t)t * 512 + kvh * 256 + d]);
                        put(kp, false, kvh, d, pos, hk);
                        put(vp, false, kvh, d, pos, hv);
                        put(hkp, true, kvh, d, pos, hk);
                        put(hvp, true, kvh, d, pos, hv);
                    }
            mqfx::Fixture fx;
            // tol_abs covers one f16 ulp over the |v| <= 2 input range: the
            // generator's f32_to_f16 rounds half-up, the MSL half()
            // conversion rounds to-nearest-even, and the two differ by one
            // ulp on exact ties. Placement exactness is guarded by the
            // bitwise int8 sibling case below.
            pf_base(fx, "kv_append", "f16_host", 0x9F34, mqfx::TolKind::MaxAbs, 0.0015, 0);
            fx.params_i = {{"T", T}, {"pos0", pos0}, {"page_size", page_size}, {"pool_rows", pool_rows}, {"mode", 16}};
            fx.inputs = {mk_f32("K", {T, 512}, K), mk_f32("V", {T, 512}, V), mk_i32("page_table", {8}, table)};
            fx.outputs = {mk_f16t("k_pool", {pool_rows, 256}, kp), mk_f16t("v_pool", {pool_rows, 256}, vp),
                          mk_f16t("host_k_pool", {pool_rows, 256}, hkp), mk_f16t("host_v_pool", {pool_rows, 256}, hvp)};
            emit(fx);
        }
        {
            std::vector<int8_t> kq((size_t)pool_rows * 256, 0), vq = kq;
            std::vector<uint16_t> ks((size_t)pool_rows * 4, 0), vs = ks;
            auto quant = [&](const std::vector<float>& X, std::vector<int8_t>& q, std::vector<uint16_t>& sc) {
                for (int t = 0; t < T; ++t) {
                    const long pos = pos0 + t;
                    const long page = table[pos / page_size];
                    if (page < 0) continue;
                    for (int kvh = 0; kvh < 2; ++kvh)
                        for (int g = 0; g < 4; ++g) {
                            float amax = 0;
                            for (int d = 0; d < 64; ++d)
                                amax = std::max(amax, std::fabs(X[(size_t)t * 512 + kvh * 256 + g * 64 + d]));
                            const uint16_t sb = f32_to_f16(amax / 127.0f);
                            const float sf = pf_f16_val(sb);
                            const long row = (page * 2 + kvh) * page_size + pos % page_size;
                            sc[(size_t)row * 4 + g] = sb;
                            for (int d = 0; d < 64; ++d) {
                                int qq = 0;
                                if (sf > 0.0f) {
                                    qq = (int)std::nearbyint(X[(size_t)t * 512 + kvh * 256 + g * 64 + d] / sf);
                                    qq = std::max(-127, std::min(127, qq));
                                }
                                q[(size_t)row * 256 + g * 64 + d] = (int8_t)qq;
                            }
                        }
                }
            };
            quant(K, kq, ks);
            quant(V, vq, vs);
            mqfx::Fixture fx;
            pf_base(fx, "kv_append", "int8", 0x9F34, mqfx::TolKind::Bitwise, 0, 0);
            fx.params_i = {{"T", T}, {"pos0", pos0}, {"page_size", page_size}, {"pool_rows", pool_rows}, {"mode", 8}};
            fx.inputs = {mk_f32("K", {T, 512}, K), mk_f32("V", {T, 512}, V), mk_i32("page_table", {8}, table)};
            fx.outputs = {mk("k_q", mqfx::DType::I8, {pool_rows, 256},
                             std::vector<uint8_t>((const uint8_t*)kq.data(), (const uint8_t*)kq.data() + kq.size())),
                          mk("v_q", mqfx::DType::I8, {pool_rows, 256},
                             std::vector<uint8_t>((const uint8_t*)vq.data(), (const uint8_t*)vq.data() + vq.size())),
                          mk_f16t("k_scale", {pool_rows, 4}, ks), mk_f16t("v_scale", {pool_rows, 4}, vs)};
            emit(fx);
        }
    }
}

// ---- QSA score, following the in-tree precedent tests/hip/native_qsa_score.cpp --
// Same geometry (idx_dim 128, 4 heads, 4-cell blocks, 7 live cells of 8),
// same formula-generated (not random) inputs, same float FMA-order reference,
// same bitwise comparison, same observability guard: both a positive and a
// negative per-head dot must occur or the fixture proves nothing.
void gen_qsa_score() {
    const int kDim = 128, kHeads = 4, kBlock = 4, kMaxCells = 8, kMaxBlocks = 3, kCells = 7;
    std::vector<float> pooled((size_t)kMaxBlocks * kDim), query((size_t)kHeads * kDim), bias(kMaxBlocks);
    const float head_sign[kHeads] = {1.0f, -1.0f, 1.0f, -1.0f};
    for (int row = 0; row < kMaxBlocks; ++row)
        for (int d = 0; d < kDim; ++d)
            pooled[(size_t)row * kDim + d] = (float)(row + 1) * (1.0f + (float)(d % 7) / 8.0f) / 128.0f;
    for (int head = 0; head < kHeads; ++head)
        for (int d = 0; d < kDim; ++d)
            query[(size_t)head * kDim + d] = head_sign[head] * (1.0f + (float)((head + d) % 5) / 16.0f) / 16.0f;
    for (int row = 0; row < kMaxBlocks; ++row) bias[row] = ((float)row - 1.0f) / 8.0f;
    const int32_t step[4] = {kCells - 1, kCells, kCells / kBlock, kCells};  // pos, n_kv, n_bid, width
    std::vector<float> expected(kMaxCells, -123.0f);
    bool pos = false, neg = false;
    for (int row = 0; row <= step[2]; ++row) {
        float sum = 0.0f;
        for (int head = 0; head < kHeads; ++head) {
            float dot = 0.0f;
            for (int d = 0; d < kDim; ++d) dot = std::fmaf(pooled[(size_t)row * kDim + d], query[(size_t)head * kDim + d], dot);
            pos = pos || dot > 0.0f;
            neg = neg || dot < 0.0f;
            sum = sum + (dot > 0.0f ? dot : 0.0f);
        }
        sum = sum + bias[row];
        if (row == step[2] && kCells % kBlock != 0) sum = sum + 1.0e9f;
        const int lo = row * kBlock;
        const int hi = std::min((row + 1) * kBlock, kCells);
        for (int cell = lo; cell < hi; ++cell) expected[cell] = sum;
    }
    if (!pos || !neg) { std::fprintf(stderr, "qsa_score fixture lost its observability guard\n"); std::exit(1); }
    mqfx::Fixture fx;
    fx.family = "qsa"; fx.entry = "native_qsa_score"; fx.case_name = "hip_precedent_7cells"; fx.seed = 0;
    fx.tol_kind = mqfx::TolKind::Bitwise;
    fx.params_i = {{"max_blocks", kMaxBlocks}, {"max_cells", kMaxCells}, {"real_shapes", 1}};
    fx.params_f = {{"out_fill", -123.0}};
    fx.inputs = {mk_f32("pooled", {kMaxBlocks, kDim}, pooled), mk_f32("query", {kHeads, kDim}, query),
                 mk_f32("block_bias", {kMaxBlocks}, bias),
                 mk_i32("step", {4}, std::vector<int32_t>(step, step + 4))};
    fx.outputs = {mk_f32("cell_scores", {kMaxCells}, expected)};
    emit(fx);
}

// ---- QSA (smallest meaningful cases) -------------------------------------------
void gen_qsa() {
    const int n_head = 2, head_dim = 8;
    {  // gate apply: out[h][d] = attn[h][d] * sigmoid(q_full[h][head_dim + d])
        Rng rng(0x9A01);
        std::vector<float> attn((size_t)n_head * head_dim), qf((size_t)n_head * 2 * head_dim);
        for (auto& v : attn) v = rng.uni(-1.0f, 1.0f);
        for (auto& v : qf) v = rng.uni(-2.0f, 2.0f);
        std::vector<float> want(attn.size());
        for (int h = 0; h < n_head; ++h)
            for (int d = 0; d < head_dim; ++d) {
                const double g = qf[(size_t)h * 2 * head_dim + head_dim + d];
                want[(size_t)h * head_dim + d] = (float)((double)attn[(size_t)h * head_dim + d] / (1.0 + std::exp(-g)));
            }
        for (const char* entry : {"qsa_gate_apply_f32", "native_qsa_gate_apply"}) {
            mqfx::Fixture fx;
            fx.family = "qsa"; fx.entry = entry; fx.case_name = "small_heads"; fx.seed = 0x9A01;
            fx.tol_kind = mqfx::TolKind::MaxAbs; fx.tol_abs = 1e-6;
            fx.params_i = {{"n_head", n_head}, {"head_dim", head_dim}, {"n_head_kv", 1}, {"n_rot", 4},
                           {"idx_n_head", 1}, {"idx_dim", 8}, {"idx_block", 4}, {"idx_top_k", 16}, {"page_size", 4}};
            fx.inputs = {mk_f32("attn", {n_head, head_dim}, attn), mk_f32("q_full", {n_head, 2 * head_dim}, qf)};
            fx.outputs = {mk_f32("out", {n_head, head_dim}, want)};
            emit(fx);
        }
    }
    {  // native_qsa_rms_norm_weighted: plain gamma RMS norm, eps on the mean
        Rng rng(0x9A02);
        const int rows = 2, cols = 64;
        std::vector<float> x((size_t)rows * cols), gamma(cols);
        for (auto& v : x) v = rng.uni(-2.0f, 2.0f);
        for (auto& v : gamma) v = rng.uni(0.5f, 1.5f);
        std::vector<float> want(x.size());
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double)x[(size_t)r * cols + i] * x[(size_t)r * cols + i];
            const double inv = 1.0 / std::sqrt(acc / cols + 1e-6);
            for (int i = 0; i < cols; ++i) want[(size_t)r * cols + i] = (float)((double)x[(size_t)r * cols + i] * inv * gamma[i]);
        }
        mqfx::Fixture fx;
        fx.family = "qsa"; fx.entry = "native_qsa_rms_norm_weighted"; fx.case_name = "small_rows"; fx.seed = 0x9A02;
        fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 1e-6; fx.tol_rel = 1e-6;
        fx.params_i = {{"n_rows", rows}, {"n_cols", cols}};
        fx.params_f = {{"eps", 1e-6}};
        fx.inputs = {mk_f32("input", {rows, cols}, x), mk_f32("gamma", {cols}, gamma)};
        fx.outputs = {mk_f32("output", {rows, cols}, want)};
        emit(fx);
    }
    {  // GRFOLD: the qb batched shape the window body actually launches -
       // a whole window's Q head rows at real head_dim.
        Rng rng(0x9A03);
        const int rows = 64, cols = 256;
        std::vector<float> x((size_t)rows * cols), gamma(cols);
        for (auto& v : x) v = rng.uni(-2.0f, 2.0f);
        for (auto& v : gamma) v = rng.uni(0.5f, 1.5f);
        std::vector<float> want(x.size());
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < cols; ++i) acc += (double)x[(size_t)r * cols + i] * x[(size_t)r * cols + i];
            const double inv = 1.0 / std::sqrt(acc / cols + 1e-6);
            for (int i = 0; i < cols; ++i) want[(size_t)r * cols + i] = (float)((double)x[(size_t)r * cols + i] * inv * gamma[i]);
        }
        mqfx::Fixture fx;
        fx.family = "qsa"; fx.entry = "native_qsa_rms_norm_weighted"; fx.case_name = "rows64_hd256"; fx.seed = 0x9A03;
        fx.tol_kind = mqfx::TolKind::MaxRel; fx.tol_abs = 1e-6; fx.tol_rel = 1e-6;
        fx.params_i = {{"n_rows", rows}, {"n_cols", cols}};
        fx.params_f = {{"eps", 1e-6}};
        fx.inputs = {mk_f32("input", {rows, cols}, x), mk_f32("gamma", {cols}, gamma)};
        fx.outputs = {mk_f32("output", {rows, cols}, want)};
        emit(fx);
    }
}

}  // namespace

// native_rope_apply (the QSA layers' pinned rotary): reference transcribed from
// src/kernels/cuda/native_rope.cu - theta = pos * theta_scale^pair in float on
// the host scale, NEOX half-split over the first n_rot channels, tail copied.
// The engine always calls it in place, batched over a whole window's head rows
// (rows = n_tokens * n_head) with per-row positions - the shape no other
// fixture exercised (native_rope was "planned" in FIXTURE-FORMAT.md).
void gen_native_rope() {
    const int DIM = 256, NROT = 64;
    const double FREQ = 1e7;
    const float theta_scale = std::pow((float) FREQ, -2.0f / (float) NROT);
    struct Case { const char* name; std::vector<int32_t> pos; };
    std::vector<Case> cases;
    cases.push_back({"rows32_pos27", std::vector<int32_t>(32, 27)});
    { std::vector<int32_t> p(64, 27); std::fill(p.begin() + 32, p.end(), 28); cases.push_back({"rows64_pos27_28", p}); }
    { std::vector<int32_t> p(64, 5); std::fill(p.begin() + 32, p.end(), 900); cases.push_back({"rows64_pos5_900", p}); }
    { std::vector<int32_t> p(128); for (int i = 0; i < 128; ++i) p[i] = 100 + i / 32; cases.push_back({"rows128_blocks", p}); }
    int ci = 0;
    for (const Case& cs : cases) {
        Rng rng(0x9F40 + ci++);
        const int rows = (int) cs.pos.size();
        std::vector<float> x0((size_t) rows * DIM);
        for (auto& v : x0) v = rng.uni(-1.5f, 1.5f);
        std::vector<float> x = x0;
        for (int r = 0; r < rows; ++r)
            for (int pair = 0; pair < NROT / 2; ++pair) {
                const double th = (double) cs.pos[r] * std::pow((double) theta_scale, (double) pair);
                const double co = std::cos(th), si = std::sin(th);
                float& a = x[(size_t) r * DIM + pair];
                float& b = x[(size_t) r * DIM + pair + NROT / 2];
                const double av = a, bv = b;
                a = (float) (av * co - bv * si);
                b = (float) (av * si + bv * co);
            }
        mqfx::Fixture fx;
        pf_base(fx, "native_rope", cs.name, 0x9F40, mqfx::TolKind::MaxRel, 1e-5, 1e-4);
        fx.params_i = {{"rows", rows}, {"dim", DIM}, {"n_rot", NROT}};
        fx.params_f = {{"freq_base", FREQ}};
        fx.inputs = {mk_f32("x", {rows, DIM}, x0), mk_i32("pos", {rows}, cs.pos)};
        fx.outputs = {mk_f32("x_out", {rows, DIM}, x)};
        emit(fx);
    }
}

// qsa_attn_batch: qsa_decode_attn_batch with n_q = 2 against a scalar
// reference - the GRFOLD regression pin for multi-row QSA attention (the
// live capstone fault sat in a T=2 window's second row while every staged
// input verified identical). Geometry mirrors the live layer: NKV 2,
// NH 24, HD 256, page_size 4, 28 resident cells, identity selection.
void gen_qsa_attn_batch() {
    const int NKV = 2, NH = 24, HD = 256, PS = 4, NCELL = 28, CAP = 128, NQ = 2;
    const int PAGES = (NCELL + PS - 1) / PS;
    auto f16v = [](uint16_t x) -> double {
        const int e = (x >> 10) & 0x1f, f = x & 0x3ff;
        const double v = e == 0 ? (double) f * 5.960464477539063e-08 : std::ldexp(1024.0 + f, e - 25);
        return (x >> 15) ? -v : v;
    };
    Rng rng(0x9F41);
    std::vector<uint16_t> kp((size_t) PAGES * NKV * PS * HD), vp = kp;
    std::vector<float> q((size_t) NQ * NH * HD);
    for (auto& b : kp) b = f32_to_f16(rng.uni(-2.0f, 2.0f));
    for (auto& b : vp) b = f32_to_f16(rng.uni(-2.0f, 2.0f));
    for (auto& v : q) v = rng.uni(-1.0f, 1.0f);
    std::vector<int32_t> table(PAGES), ids((size_t) NQ * CAP, 0), steps((size_t) NQ * 4);
    for (int p = 0; p < PAGES; ++p) table[p] = p;
    for (int z = 0; z < NQ; ++z)
        for (int i = 0; i < NCELL; ++i) ids[(size_t) z * CAP + i] = i;
    steps[0] = 26; steps[1] = 27; steps[2] = 6; steps[3] = 27;
    steps[4] = 27; steps[5] = 28; steps[6] = 7; steps[7] = 28;
    auto pool = [&](const std::vector<uint16_t>& P, int cell, int kvh, int d) -> double {
        const int page = table[cell / PS];
        return f16v(P[(((size_t) page * NKV + kvh) * PS + (cell % PS)) * HD + d]);
    };
    std::vector<float> attn((size_t) NQ * NH * HD, 0.0f);
    const double scale = 1.0 / 16.0;
    for (int z = 0; z < NQ; ++z) {
        const int width = steps[(size_t) z * 4 + 3];
        for (int kvh = 0; kvh < NKV; ++kvh)
            for (int hl = 0; hl < 12; ++hl) {
                const int h = kvh * 12 + hl;
                double sc[NCELL], mx = -1e30;
                for (int c = 0; c < width; ++c) {
                    double s = 0;
                    for (int d = 0; d < HD; ++d) s += (double) q[((size_t) z * NH + h) * HD + d] * pool(kp, c, kvh, d);
                    sc[c] = s * scale;
                    if (sc[c] > mx) mx = sc[c];
                }
                double sum = 0;
                for (int c = 0; c < width; ++c) { sc[c] = std::exp(sc[c] - mx); sum += sc[c]; }
                for (int d = 0; d < HD; ++d) {
                    double a = 0;
                    for (int c = 0; c < width; ++c) a += sc[c] * pool(vp, c, kvh, d);
                    attn[((size_t) z * NH + h) * HD + d] = (float) (a / sum);
                }
            }
    }
    mqfx::Fixture fx;
    pf_base(fx, "qsa_attn_batch", "nq2", 0x9F41, mqfx::TolKind::MaxAbs, 5e-4, 0);
    fx.params_i = {{"cap", CAP}, {"n_cells", NCELL}};
    fx.inputs = {mk_f16t("k_pool", {PAGES * NKV * PS, HD}, kp), mk_f16t("v_pool", {PAGES * NKV * PS, HD}, vp),
                 mk_i32("page_table", {PAGES}, table), mk_f32("q", {NQ, NH * HD}, q),
                 mk_i32("ids", {NQ, CAP}, ids), mk_i32("steps", {NQ, 4}, steps)};
    fx.outputs = {mk_f32("attn", {NQ, NH * HD}, attn)};
    emit(fx);
    // dup case: both query slots carry the SAME query (row 1's data and
    // step) - the two output rows must come out identical; separates a
    // slot-side defect (scratch/output indexing) from an input-side one.
    {
        std::vector<float> q2 = q;
        std::copy(q.begin() + (size_t) NH * HD, q.end(), q2.begin());
        std::vector<int32_t> steps2 = steps;
        steps2[0] = steps[4]; steps2[1] = steps[5]; steps2[2] = steps[6]; steps2[3] = steps[7];
        std::vector<float> attn2 = attn;
        std::copy(attn.begin() + (size_t) NH * HD, attn.end(), attn2.begin());
        mqfx::Fixture fx2;
        pf_base(fx2, "qsa_attn_batch", "nq2_dup", 0x9F41, mqfx::TolKind::MaxAbs, 5e-4, 0);
        fx2.params_i = {{"cap", CAP}, {"n_cells", NCELL}};
        fx2.inputs = {mk_f16t("k_pool", {PAGES * NKV * PS, HD}, kp), mk_f16t("v_pool", {PAGES * NKV * PS, HD}, vp),
                      mk_i32("page_table", {PAGES}, table), mk_f32("q", {NQ, NH * HD}, q2),
                      mk_i32("ids", {NQ, CAP}, ids), mk_i32("steps", {NQ, 4}, steps2)};
        fx2.outputs = {mk_f32("attn", {NQ, NH * HD}, attn2)};
        emit(fx2);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gen_fixtures <out_dir>\n"); return 2; }
    g_out = argv[1];
    gen_dequant();
    gen_gemv();
    gen_mmvq();
    gen_native_expert_grouped();
    gen_verify_moe_assembly();
    gen_fused_gr_read();
    gen_native_rope();
    gen_qsa_attn_batch();
    gen_resident_plan();
    gen_routers();
    gen_sampler();
    gen_ple();
    gen_gdn();
    gen_prefill();
    gen_qsa();
    gen_qsa_score();
    std::printf("gen_fixtures: %d fixtures written to %s\n", g_written, g_out.c_str());
    return 0;
}
