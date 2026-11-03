// metal_kernel_parity: the Mac-side runner of the Qwen Metal parity oracle.
//
// Loads every .mqfx fixture in a directory, calls the REAL host entry points
// (the exact signatures the engine calls, via strata_kernels), and compares
// against the fixture's expected outputs under the fixture's own tolerance.
//
// Each case runs in a forked child: the Stage-1 stub entries abort() by
// design, and the harness must report that as a per-case FAIL and keep going
// rather than dying with the first stub. Against real kernels the same
// isolation contains a faulting kernel to its own case.
//
// Exit code: number of failed cases (capped at 125), 0 = all pass.
// Usage: metal_kernel_parity <fixture_dir> [--entry <name>] [--list]

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/wait.h>
#include <unistd.h>

#include "fixture_format.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/dequant_s2.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/s2_gemv.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/fused_gr.hpp"

namespace k = strata::kernels;

namespace {

[[noreturn]] void die(const char* what, cudaError_t e) {
    std::fprintf(stderr, "runtime: %s: %s\n", what, cudaGetErrorString(e));
    _exit(2);
}
#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) die(#call, _e); } while (0)

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    Dev() = default;
    Dev(const void* host, size_t n) { alloc(n); if (n) CK(cudaMemcpy(p, host, n, cudaMemcpyHostToDevice)); }
    explicit Dev(size_t n) { alloc(n); }
    void alloc(size_t n) { bytes = n; if (n) CK(cudaMalloc(&p, n)); }
    void fill_f32(float v, size_t count) {
        std::vector<float> h(count, v);
        if (count) CK(cudaMemcpy(p, h.data(), count * 4, cudaMemcpyHostToDevice));
    }
    std::vector<uint8_t> get() const {
        std::vector<uint8_t> h(bytes);
        if (bytes) CK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
        return h;
    }
    template <typename T> T* as() const { return reinterpret_cast<T*>(p); }
};

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
float bf16_to_f32(uint16_t h) {
    const uint32_t f = (uint32_t)h << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
double elem_as_double(const mqfx::Tensor& t, size_t i) {
    switch (t.dtype) {
        case mqfx::DType::F32: { float v; std::memcpy(&v, t.bytes.data() + i * 4, 4); return v; }
        case mqfx::DType::F16: { uint16_t v; std::memcpy(&v, t.bytes.data() + i * 2, 2); return f16_to_f32(v); }
        case mqfx::DType::BF16: { uint16_t v; std::memcpy(&v, t.bytes.data() + i * 2, 2); return bf16_to_f32(v); }
        case mqfx::DType::I32: { int32_t v; std::memcpy(&v, t.bytes.data() + i * 4, 4); return v; }
        case mqfx::DType::I64: { int64_t v; std::memcpy(&v, t.bytes.data() + i * 8, 8); return (double)v; }
        case mqfx::DType::I8: return (int8_t)t.bytes[i];
        case mqfx::DType::U8: return t.bytes[i];
        case mqfx::DType::U16: { uint16_t v; std::memcpy(&v, t.bytes.data() + i * 2, 2); return v; }
    }
    return 0;
}
bool elem_bits_equal(const mqfx::Tensor& a, const uint8_t* got, size_t i) {
    const size_t sz = mqfx::dtype_size(a.dtype);
    return std::memcmp(a.bytes.data() + i * sz, got + i * sz, sz) == 0;
}

// Compare one expected tensor against device result bytes. Returns max error.
bool compare(const mqfx::Fixture& fx, const mqfx::Tensor& want, const std::vector<uint8_t>& got, double& max_err) {
    const size_t sz = mqfx::dtype_size(want.dtype);
    if (got.size() != want.bytes.size()) {
        std::fprintf(stderr, "  size mismatch for %s: got %zu want %zu\n", want.name.c_str(), got.size(), want.bytes.size());
        return false;
    }
    const int64_t n = want.numel();
    max_err = 0;
    const bool intlike = want.dtype == mqfx::DType::I32 || want.dtype == mqfx::DType::I64 ||
                         want.dtype == mqfx::DType::U8 || want.dtype == mqfx::DType::I8;
    for (int64_t i = 0; i < n; ++i) {
        if (fx.tol_kind == mqfx::TolKind::Bitwise || intlike) {
            if (!elem_bits_equal(want, got.data(), (size_t)i)) {
                max_err = 1;
                std::fprintf(stderr, "  %s[%lld] bits differ: want %g\n", want.name.c_str(), (long long)i,
                             elem_as_double(want, (size_t)i));
                return false;
            }
        } else {
            mqfx::Tensor gt;
            gt.dtype = want.dtype;
            gt.bytes.assign(got.begin(), got.end());
            const double g = elem_as_double(gt, (size_t)i);
            const double w = elem_as_double(want, (size_t)i);
            const double err = std::fabs(g - w);
            const double lim = fx.tol_kind == mqfx::TolKind::MaxAbs ? fx.tol_abs : fx.tol_abs + fx.tol_rel * std::fabs(w);
            if (err > max_err) max_err = err;
            if (!(err <= lim)) {
                std::fprintf(stderr, "  %s[%lld] got %.9g want %.9g (err %.3g > %.3g)\n", want.name.c_str(),
                             (long long)i, g, w, err, lim);
                return false;
            }
        }
    }
    (void)sz;
    return true;
}

cudaStream_t g_stream = nullptr;
void* stream() { return (void*)g_stream; }

const mqfx::Tensor& need(const mqfx::Fixture& fx, const char* name) {
    const mqfx::Tensor* t = fx.in(name);
    if (!t) { std::fprintf(stderr, "fixture missing input %s\n", name); _exit(2); }
    return *t;
}
Dev up(const mqfx::Fixture& fx, const char* name) {
    const mqfx::Tensor& t = need(fx, name);
    return Dev(t.bytes.data(), t.bytes.size());
}

// Run one fixture's entry point. Fills `results` (expected-output name -> host
// bytes). Returns false only for harness errors; kernel aborts never return.
bool dispatch(const mqfx::Fixture& fx, std::map<std::string, std::vector<uint8_t>>& results) {
    const std::string& e = fx.entry;
    auto out_bytes = [&](const char* name, Dev& d) { results[name] = d.get(); };
    if (e == "dequant_f32" || e == "dequant_bf16" || e == "dequant_f16") {
        Dev blocks = up(fx, "blocks");
        const int64_t rows = fx.param_i("rows"), cols = fx.param_i("cols");
        const int type = (int)fx.param_i("ggml_type");
        if (e == "dequant_f32") {
            Dev out((size_t)rows * cols * 4);
            k::dequant_f32(type, blocks.p, fx.param_i("row0"), rows, cols, out.as<float>(), stream());
            CK(cudaStreamSynchronize(g_stream));
            out_bytes("out", out);
        } else {
            Dev out((size_t)rows * cols * 2);
            if (e == "dequant_bf16") k::dequant_bf16(type, blocks.p, fx.param_i("row0"), rows, cols, out.as<uint16_t>(), stream());
            else k::dequant_f16(type, blocks.p, fx.param_i("row0"), rows, cols, out.as<uint16_t>(), stream());
            CK(cudaStreamSynchronize(g_stream));
            out_bytes("out", out);
        }
        return true;
    }
    if (e == "dequant_s2") {
        Dev codes = up(fx, "codes"), scales = up(fx, "scales");
        Dev out((size_t)fx.param_i("n_blocks") * 64 * 4);
        k::dequant_s2(codes.as<uint8_t>(), scales.as<float>(), out.as<float>(), fx.param_i("n_blocks"));
        out_bytes("out", out);
        return true;
    }
    if (e == "s_gemv" || e == "s_gemv_split" || e == "s2_gemv" || e == "s2_gemv_q8") {
        Dev scales = up(fx, "scales");
        Dev y((size_t)fx.param_i("n_out") * 4);
        const int64_t n_in = fx.param_i("n_in"), n_out = fx.param_i("n_out");
        if (e == "s2_gemv") {
            Dev x = up(fx, "x"), codes = up(fx, "codes");
            k::s2_gemv(x.as<uint16_t>(), codes.as<uint8_t>(), scales.as<float>(), y.as<float>(), n_in, n_out);
        } else if (e == "s2_gemv_q8") {
            Dev act = up(fx, "act"), codes = up(fx, "codes");
            k::s2_gemv_q8(act.as<uint8_t>(), codes.as<uint8_t>(), scales.as<float>(), y.as<float>(), n_in, n_out,
                          (int)fx.param_i("threads_per_row"), stream());
            CK(cudaStreamSynchronize(g_stream));
        } else {
            Dev x = up(fx, "x"), codes = up(fx, "codes");
            k::SForm form;
            form.code_bits = (int)fx.param_i("code_bits");
            form.code_bias = (int)fx.param_i("code_bias");
            form.group_elems = (int)fx.param_i("group_elems");
            form.codebook = fx.param_i("codebook") == 1 ? k::Codebook::Iq4Nl : k::Codebook::Affine;
            form.has_offset = fx.param_i("has_offset") != 0;
            if (e == "s_gemv") k::s_gemv(x.as<uint16_t>(), codes.as<uint8_t>(), scales.as<float>(), nullptr, y.as<float>(), n_in, n_out, form);
            else k::s_gemv_split(x.as<uint16_t>(), codes.as<uint8_t>(), scales.as<float>(), nullptr, y.as<float>(), n_in, n_out, form, (int)fx.param_i("threads_per_row"));
        }
        out_bytes("y", y);
        return true;
    }
    if (e == "native_mmvq" || (e.rfind("native_", 0) == 0 && e.find("_mmvq") == e.size() - 5) ||
        (e.rfind("native_", 0) == 0 && e.find("_f32") == e.size() - 4 && e.find("qsa") == std::string::npos)) {
        static const std::map<std::string, int> kTypeBySuffix = {
            {"q2_0", 42}, {"q4_0", 2}, {"q5_0", 6}, {"q8_0", 8}, {"iq4_nl", 20},
            {"iq4_xs", 23}, {"q3_k", 11}, {"q4_k", 12}, {"q5_k", 13}, {"q6_k", 14}};
        const int type = (int)fx.param_i("ggml_type");
        const int64_t n_in = fx.param_i("n_in"), n_out = fx.param_i("n_out");
        const int ncols = (int)fx.param_i("ncols", 1);
        Dev w = up(fx, "weights");
        Dev y((size_t)n_out * ncols * 4);
        if (e == "native_mmvq") {
            Dev xq = up(fx, "x_q8_1");
            k::native_mmvq(type, w.p, xq.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream());
        } else if (e.find("_mmvq") != std::string::npos) {
            Dev xq = up(fx, "x_q8_1");
            const std::string suffix = e.substr(7, e.size() - 7 - 5);
            auto it = kTypeBySuffix.find(suffix);
            if (it == kTypeBySuffix.end()) return false;
            switch (it->second) {
                case 42: k::native_q2_0_mmvq(w.p, xq.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 20: k::native_iq4_nl_mmvq(w.p, xq.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 23: k::native_iq4_xs_mmvq(w.p, xq.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 8: k::native_q8_0_mmvq(w.p, xq.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 12: k::native_q4_k_mmvq(w.p, xq.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                default: return false;
            }
        } else {  // _f32 composition: quantize into caller scratch, then mmvq
            Dev x = up(fx, "x");
            Dev scratch(k::native_q8_1_bytes((int)n_in, ncols));
            const std::string suffix = e.substr(7, e.size() - 7 - 4);
            auto it = kTypeBySuffix.find(suffix);
            if (it == kTypeBySuffix.end()) return false;
            switch (it->second) {
                case 42: k::native_q2_0_f32(w.p, x.as<float>(), scratch.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 20: k::native_iq4_nl_f32(w.p, x.as<float>(), scratch.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 23: k::native_iq4_xs_f32(w.p, x.as<float>(), scratch.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 8: k::native_q8_0_f32(w.p, x.as<float>(), scratch.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                case 12: k::native_q4_k_f32(w.p, x.as<float>(), scratch.p, y.as<float>(), (int)n_in, (int)n_out, ncols, stream()); break;
                default: return false;
            }
        }
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("y", y);
        return true;
    }
    if (e == "router_top10") {
        Dev logits = up(fx, "logits");
        const int64_t nt = fx.param_i("n_tokens"), ne = fx.param_i("n_expert"), kk = fx.param_i("k");
        Dev ids((size_t)nt * kk * 4), w((size_t)nt * kk * 4);
        k::router_top10(logits.as<float>(), (int)nt, (int)ne, (int)kk, ids.as<int>(), w.as<float>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("ids", ids);
        out_bytes("weights", w);
        return true;
    }
    if (e == "router_sigmoid") {
        Dev logits = up(fx, "logits"), bias = up(fx, "bias");
        const int64_t nt = fx.param_i("n_tokens"), ne = fx.param_i("n_expert"), kk = fx.param_i("k");
        Dev ids((size_t)nt * kk * 4), w((size_t)nt * kk * 4);
        k::router_sigmoid(logits.as<float>(), bias.as<float>(), (int)nt, (int)ne, (int)kk,
                          (float)fx.param_f("w_scale"), fx.param_i("norm_w") != 0, ids.as<int>(), w.as<float>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("ids", ids);
        out_bytes("weights", w);
        return true;
    }
    if (e == "native_router_top10" || e == "native_router_top10_multi") {
        Dev logits = up(fx, "logits");
        const int64_t nt = fx.param_i("n_tokens"), kk = fx.param_i("k");
        Dev ids((size_t)nt * kk * 4), w((size_t)nt * kk * 4);
        if (e == "native_router_top10") k::native_router_top10(logits.as<float>(), ids.as<int32_t>(), w.as<float>(), stream());
        else k::native_router_top10_multi(logits.as<float>(), ids.as<int32_t>(), w.as<float>(), (int)nt, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("ids", ids);
        out_bytes("weights", w);
        return true;
    }
    if (e == "sample_tokens") {
        Dev logits = up(fx, "logits");
        const int64_t nt = fx.param_i("n_tokens"), nv = fx.param_i("n_vocab");
        Dev out((size_t)nt * 4);
        k::SamplerParams p;
        p.greedy = fx.param_i("greedy") != 0;
        p.temperature = (float)fx.param_f("temperature");
        p.top_k = (int)fx.param_i("top_k");
        p.top_p = (float)fx.param_f("top_p", 1.0);
        p.min_p = (float)fx.param_f("min_p", 0.0);
        p.penalty_last_n = (int)fx.param_i("penalty_last_n");
        k::sample_tokens(logits.as<float>(), (int)nt, (int)nv, nullptr, 0, p, out.as<int>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("out", out);
        return true;
    }
    if (e == "native_ple_postops") {
        Dev pk = up(fx, "projected_key"), hidden = up(fx, "hidden"), value = up(fx, "value"), hist = up(fx, "history");
        Dev nk = up(fx, "norm_key"), nq = up(fx, "norm_query"), nc = up(fx, "norm_conv"), conv = up(fx, "conv1d_f16");
        const size_t HC = 10240;
        Dev key(HC * 4), query(HC * 4), gate(4 * 4), gated(HC * 4), norm(HC * 4), conv_o(HC * 4), result(HC * 4);
        k::PleWeights w{};
        w.norm_key = nk.as<float>();
        w.norm_query = nq.as<float>();
        w.norm_conv = nc.as<float>();
        w.conv1d_f16 = conv.as<uint16_t>();
        k::NativePlePostopsBuffers b{};
        b.key = key.as<float>(); b.query = query.as<float>(); b.gate = gate.as<float>();
        b.gated = gated.as<float>(); b.normalized = norm.as<float>(); b.conv = conv_o.as<float>();
        b.result = result.as<float>();
        k::native_ple_postops(pk.as<float>(), hidden.as<float>(), value.as<float>(), hist.as<float>(), w, b, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("key", key);
        out_bytes("gate", gate);
        out_bytes("gated", gated);
        out_bytes("normalized", norm);
        out_bytes("conv", conv_o);
        out_bytes("result", result);
        return true;
    }
    if (e == "ple_history_advance") {
        Dev hist = up(fx, "hist"), norm = up(fx, "normalized");
        k::ple_history_advance(hist.as<float>(), norm.as<float>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("hist_out", hist);
        return true;
    }
    if (e == "gdn_step" || e == "native_gdn_step") {
        Dev state = up(fx, "state"), q = up(fx, "q"), kk = up(fx, "k"), v = up(fx, "v"), gate = up(fx, "gate"), beta = up(fx, "beta");
        const int64_t S = fx.param_i("S"), hv = fx.param_i("h_v");
        Dev o((size_t)hv * S * 4);
        k::GdnShapes s;
        s.S = S; s.h_k = fx.param_i("h_k"); s.h_v = hv;
        if (e == "gdn_step") k::gdn_step(state.as<float>(), q.as<float>(), kk.as<float>(), v.as<float>(), gate.as<float>(), beta.as<float>(), o.as<float>(), s, stream());
        else k::native_gdn_step(state.as<float>(), q.as<float>(), kk.as<float>(), v.as<float>(), gate.as<float>(), beta.as<float>(), o.as<float>(), s, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("state_out", state);
        out_bytes("o", o);
        return true;
    }
    if (e == "gdn_l2_norm") {
        Dev x = up(fx, "x");
        k::gdn_l2_norm(x.as<float>(), fx.param_i("rows"), fx.param_i("cols"), (float)fx.param_f("eps"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("out", x);
        return true;
    }
    if (e == "qsa_gate_apply_f32") {
        Dev attn = up(fx, "attn"), qf = up(fx, "q_full");
        Dev out(need(fx, "attn").bytes.size());
        k::QsaShapes s;
        s.n_head = fx.param_i("n_head"); s.n_head_kv = fx.param_i("n_head_kv"); s.head_dim = fx.param_i("head_dim");
        s.n_rot = fx.param_i("n_rot"); s.idx_n_head = fx.param_i("idx_n_head"); s.idx_dim = fx.param_i("idx_dim");
        s.idx_block = fx.param_i("idx_block"); s.idx_top_k = fx.param_i("idx_top_k"); s.page_size = fx.param_i("page_size");
        k::qsa_gate_apply_f32(attn.as<float>(), qf.as<float>(), s, out.as<float>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("out", out);
        return true;
    }
    if (e == "native_qsa_gate_apply") {
        Dev attn = up(fx, "attn"), qf = up(fx, "q_full");
        Dev out(need(fx, "attn").bytes.size());
        k::native_qsa_gate_apply(attn.as<float>(), qf.as<float>(), out.as<float>(), (int)fx.param_i("n_head"), (int)fx.param_i("head_dim"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("out", out);
        return true;
    }
    if (e == "native_qsa_rms_norm_weighted") {
        Dev in = up(fx, "input"), gamma = up(fx, "gamma");
        Dev out(need(fx, "input").bytes.size());
        k::native_qsa_rms_norm_weighted(in.as<float>(), gamma.as<float>(), out.as<float>(), (int)fx.param_i("n_cols"),
                                        (int)fx.param_i("n_rows"), (float)fx.param_f("eps"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("output", out);
        return true;
    }
    if (e == "native_qsa_score") {
        Dev pooled = up(fx, "pooled"), query = up(fx, "query"), bias = up(fx, "block_bias"), step = up(fx, "step");
        Dev scores((size_t)fx.param_i("max_cells") * 4);
        if (fx.params_f.size()) scores.fill_f32((float)fx.param_f("out_fill", 0.0), (size_t)fx.param_i("max_cells"));
        k::native_qsa_score(pooled.as<float>(), query.as<float>(), bias.as<float>(), k::qsa_real_shapes(),
                            step.as<int32_t>(), fx.param_i("max_blocks"), fx.param_i("max_cells"), scores.as<float>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("cell_scores", scores);
        return true;
    }
    if (e == "native_expert_grouped") {
        const int64_t cap_groups = fx.param_i("cap_groups"), cap_entries = fx.param_i("cap_entries");
        const int64_t n_embd = fx.param_i("n_embd"), n_ff = fx.param_i("n_ff");
        std::vector<Dev> blobs;
        std::vector<unsigned long long> ptrs;
        for (int64_t g = 0; g < cap_groups; ++g) {
            blobs.push_back(up(fx, ("blob" + std::to_string(g)).c_str()));
            ptrs.push_back((unsigned long long)(uintptr_t)blobs.back().p);
        }
        Dev grp_ptr(ptrs.data(), ptrs.size() * 8);
        Dev grp_start = up(fx, "grp_start"), n_groups = up(fx, "n_groups");
        Dev ent_tok = up(fx, "ent_tok"), ent_dst = up(fx, "ent_dst"), xq = up(fx, "x_q8_1");
        Dev scratch(k::native_expert_scratch_bytes(cap_entries, n_ff));
        Dev out((size_t)cap_entries * n_embd * 4);
        CK(cudaMemset(out.p, 0, out.bytes));
        const k::NativeExpertLayout L =
            k::native_expert_layout((int)fx.param_i("gu_type"), (int)fx.param_i("d_type"), n_embd, n_ff);
        const float limit = fx.param_i("has_limit") ? (float)fx.param_f("swiglu_limit")
                                                    : std::numeric_limits<float>::infinity();
        k::native_expert_grouped(L, grp_ptr.as<unsigned long long>(), grp_start.as<int32_t>(),
                                 n_groups.as<int32_t>(), ent_dst.as<int32_t>(), ent_tok.as<int32_t>(), cap_groups,
                                 cap_entries, xq.p, scratch.p, out.as<float>(), stream(), limit);
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("out", out);
        return true;
    }
    if (e == "native_expert_grouped_based") {
        // The record_window form: all blobs are slices of the one arena
        // allocation, grp_ptr holds their contents addresses, and the entry
        // must not sync or read the plan back - the runner deliberately does
        // NOT sync before the call, so the entry runs with the stream's
        // earlier work (the input uploads' copies are runtime-immediate;
        // the entry's own launches are what the final sync waits for).
        const int64_t cap_groups = fx.param_i("cap_groups"), cap_entries = fx.param_i("cap_entries");
        const int64_t n_embd = fx.param_i("n_embd"), n_ff = fx.param_i("n_ff");
        const int64_t stride = fx.param_i("blob_stride");
        Dev arena = up(fx, "arena");
        std::vector<unsigned long long> ptrs;
        for (int64_t g = 0; g < cap_groups; ++g)
            ptrs.push_back((unsigned long long)(uintptr_t)arena.p + (unsigned long long)(g * stride));
        Dev grp_ptr(ptrs.data(), ptrs.size() * 8);
        Dev grp_start = up(fx, "grp_start"), n_groups = up(fx, "n_groups");
        Dev ent_tok = up(fx, "ent_tok"), ent_dst = up(fx, "ent_dst"), xq = up(fx, "x_q8_1");
        Dev scratch(k::native_expert_scratch_bytes(cap_entries, n_ff));
        Dev out((size_t)cap_entries * n_embd * 4);
        CK(cudaMemset(out.p, 0, out.bytes));
        const k::NativeExpertLayout L =
            k::native_expert_layout((int)fx.param_i("gu_type"), (int)fx.param_i("d_type"), n_embd, n_ff);
        const float limit = fx.param_i("has_limit") ? (float)fx.param_f("swiglu_limit")
                                                    : std::numeric_limits<float>::infinity();
        k::native_expert_grouped_based(L, arena.p, (uint64_t)arena.bytes, grp_ptr.as<unsigned long long>(),
                                       grp_start.as<int32_t>(), n_groups.as<int32_t>(), ent_dst.as<int32_t>(),
                                       ent_tok.as<int32_t>(), cap_groups, cap_entries, xq.p, scratch.p,
                                       out.as<float>(), stream(), limit);
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("out", out);
        return true;
    }
    if (e == "verify_moe_assembly") {
        // post()'s grouped + parts composition, replayed on the real entries:
        // the device plan in verify.cpp's exact layout (capx == cap, as
        // record_window sizes it), TWO grouped_based calls over the shared
        // dst/tok arrays, one scratch and one hit_out - the VRAM call with a
        // zero-based grp_start, the PCIe call with grp_start2 CONTINUING the
        // entry numbering, ptrs rebased at blob_bytes stride into staging -
        // then copy_rows_from_mapped + moe_hit_add into parts.
        const int64_t n_embd = fx.param_i("n_embd"), n_ff = fx.param_i("n_ff");
        const int64_t T = fx.param_i("T"), K = fx.param_i("K"), cap = fx.param_i("cap");
        const int64_t nv = fx.param_i("n_vram_groups"), np = fx.param_i("n_pcie_groups");
        const int64_t stride = fx.param_i("blob_stride"), blob_bytes = fx.param_i("blob_bytes");
        Dev arena_vram = up(fx, "arena_vram"), arena_stage = up(fx, "arena_stage");
        Dev xq = up(fx, "x_q8_1"), ymiss = up(fx, "ymiss");
        const auto& counts_t = need(fx, "counts");
        const auto& start_t = need(fx, "grp_start");
        const auto& start2_t = need(fx, "grp_start2");
        const auto& dst_t = need(fx, "ent_dst");
        const auto& tok_t = need(fx, "ent_tok");
        auto rd_i32 = [](const mqfx::Tensor& t, int64_t i) {
            int32_t v; std::memcpy(&v, t.bytes.data() + (size_t)i * 4, 4); return v;
        };
        const int64_t entries = rd_i32(counts_t, 1);
        // Plan layout (int32 indices), mirroring verify.cpp post(). The plan
        // arrays are sized capx = max_t*K, which may exceed this window's cap.
        int64_t capx = fx.param_i("capx");
        if (capx < cap) capx = cap;
        const int64_t i_counts = 0, i_start = 4, i_dst = i_start + capx + 1, i_tok = i_dst + capx;
        const int64_t pad = (4 + (capx + 1) + 3 * capx) & 1;
        const int64_t i_ptr = i_tok + capx + pad;         // as int32 units; u64 array
        const int64_t i_ptr2 = i_ptr + 2 * capx;
        const int64_t i_start2 = i_ptr2 + 2 * capx;
        const size_t plan_bytes = (size_t)(i_start2 + capx + 1) * 4;
        std::vector<uint8_t> plan(plan_bytes, 0);
        auto put_i = [&](int64_t idx, int32_t v) { std::memcpy(plan.data() + (size_t)idx * 4, &v, 4); };
        auto put_u = [&](int64_t idx, uint64_t v) { std::memcpy(plan.data() + (size_t)idx * 4, &v, 8); };
        for (int i = 0; i < 4; ++i) put_i(i_counts + i, rd_i32(counts_t, i));
        for (int64_t i = 0; i <= nv; ++i) put_i(i_start + i, rd_i32(start_t, i));
        for (int64_t i = 0; i < entries; ++i) {
            put_i(i_dst + i, rd_i32(dst_t, i));
            put_i(i_tok + i, rd_i32(tok_t, i));
        }
        for (int64_t g = 0; g < nv; ++g)
            put_u(i_ptr + 2 * g, (uint64_t)(uintptr_t)arena_vram.p + (uint64_t)g * (uint64_t)stride);
        for (int64_t q = 0; q < np; ++q)
            put_u(i_ptr2 + 2 * q, (uint64_t)(uintptr_t)arena_stage.p + (uint64_t)q * (uint64_t)blob_bytes);
        for (int64_t i = 0; i <= np; ++i) put_i(i_start2 + i, rd_i32(start2_t, i));
        Dev plan_d(plan.data(), plan_bytes);
        const int32_t* p_counts = plan_d.as<int32_t>() + i_counts;
        const int32_t* p_start = plan_d.as<int32_t>() + i_start;
        const int32_t* p_dst = plan_d.as<int32_t>() + i_dst;
        const int32_t* p_tok = plan_d.as<int32_t>() + i_tok;
        const uint64_t* p_ptr = reinterpret_cast<const uint64_t*>(plan_d.as<int32_t>() + i_ptr);
        const uint64_t* p_ptr2 = reinterpret_cast<const uint64_t*>(plan_d.as<int32_t>() + i_ptr2);
        const int32_t* p_start2 = plan_d.as<int32_t>() + i_start2;
        const k::NativeExpertLayout L =
            k::native_expert_layout((int)fx.param_i("gu_type"), (int)fx.param_i("d_type"), n_embd, n_ff);
        Dev scratch((size_t)k::native_expert_scratch_bytes(cap, n_ff));
        CK(cudaMemset(scratch.p, 0, scratch.bytes));
        Dev hit_out((size_t)cap * n_embd * 4);
        hit_out.fill_f32(777.25f, (size_t)cap * n_embd);
        Dev parts((size_t)cap * n_embd * 4);
        parts.fill_f32(-555.5f, (size_t)cap * n_embd);
        // VRAM share (identical caps and scratch/hit_out as record_window).
        k::native_expert_grouped_based(L, arena_vram.p, (uint64_t)arena_vram.bytes, p_ptr, p_start, p_counts,
                                       p_dst, p_tok, cap, cap, xq.p, scratch.p, hit_out.as<float>(), stream(),
                                       std::numeric_limits<float>::infinity());
        // PCIe share through the staging arena.
        k::native_expert_grouped_based(L, arena_stage.p, (uint64_t)arena_stage.bytes, p_ptr2, p_start2,
                                       p_counts + 2, p_dst, p_tok, cap, cap, xq.p, scratch.p,
                                       hit_out.as<float>(), stream(), std::numeric_limits<float>::infinity());
        k::copy_rows_from_mapped(parts.as<float>(), ymiss.as<float>(), T * K, n_embd, p_dst, p_counts + 1,
                                 stream());
        k::moe_hit_add(parts.as<float>(), hit_out.as<float>(), p_dst, p_counts + 1, cap, n_embd, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("hit_out", hit_out);
        out_bytes("parts", parts);
        if (fx.param_i("with_shared")) {
            // The rest of post()'s FFN: the shared expert over the window's
            // rows, then the batched combine into bo.
            Dev x = up(fx, "x"), wgt = up(fx, "w");
            Dev sh_gate = up(fx, "sh_gate"), sh_up = up(fx, "sh_up"), sh_down = up(fx, "sh_down");
            Dev gate_inp = up(fx, "sh_gate_inp");
            Dev q81((size_t)(n_embd / 32) * 36 * T);
            CK(cudaMemset(q81.p, 0, q81.bytes));
            Dev gate_s((size_t)T * n_ff * 4), up_s((size_t)T * n_ff * 4), g_s((size_t)T * 4);
            Dev shared((size_t)T * n_embd * 4);
            shared.fill_f32(888.25f, (size_t)T * n_embd);
            Dev bo((size_t)T * n_embd * 4);
            bo.fill_f32(-999.5f, (size_t)T * n_embd);
            k::NativeSharedWeights nw;
            nw.gate_type = (int)fx.param_i("gu_type");
            nw.up_type = (int)fx.param_i("gu_type");
            nw.down_type = (int)fx.param_i("d_type");
            nw.gate_data = sh_gate.p;
            nw.up_data = sh_up.p;
            nw.down_data = sh_down.p;
            nw.q8_1 = q81.p;
            k::shared_expert_set_native_bf16(true);
            k::shared_expert_multi((int)T, x.as<float>(), nullptr, nw, gate_inp.as<uint16_t>(),
                                   gate_s.as<float>(), up_s.as<float>(), g_s.as<float>(), shared.as<float>(),
                                   n_embd, n_ff, stream());
            k::native_moe_combine_multi(parts.as<float>(), wgt.as<float>(), shared.as<float>(), bo.as<float>(),
                                        n_embd, K, (int)T, stream());
            CK(cudaStreamSynchronize(g_stream));
            out_bytes("shared", shared);
            out_bytes("bo", bo);
        }
        return true;
    }
    if (e == "fused_gr_read") {
        // GRFOLD: the hyper-connection fold at the apply boundary, driven
        // exactly as verify.cpp's gr_read_group drives it (per-token slices,
        // R_out == R when the case says inplace). R_out and inject_out are
        // prefilled with the generator's sentinel: apply=0 must leave R_out
        // untouched and a missing w_inject must leave inject_out untouched.
        const int64_t T = fx.param_i("T"), N = fx.param_i("n_embd"), HC = fx.param_i("hc"), LR = fx.param_i("hc_lr");
        const int64_t D = N * HC;
        const bool apply = fx.param_i("apply") != 0, withinj = fx.param_i("with_inject") != 0,
                   inplace = fx.param_i("inplace") != 0;
        Dev R = up(fx, "R"), bo = up(fx, "bo_prev"), injp = up(fx, "inj_prev"), wn = up(fx, "w_norm");
        Dev wd = up(fx, "w_down"), wu = up(fx, "w_up");
        Dev wi;
        if (withinj) wi = up(fx, "w_inject");
        Dev rout;
        if (!inplace) { rout.alloc((size_t)T * D * 4); rout.fill_f32(555.25f, (size_t)T * D); }
        Dev lo((size_t)T * LR * 4), rs((size_t)T * HC * 4), mixed((size_t)T * N * 4);
        Dev injout((size_t)T * HC * 4);
        injout.fill_f32(555.25f, (size_t)T * HC);
        Dev xn((size_t)T * D * 4);
        std::vector<k::FusedGrArgs> fa((size_t)T);
        for (int64_t t = 0; t < T; ++t) {
            k::FusedGrArgs& a = fa[(size_t)t];
            a.R = R.as<float>() + (size_t)t * D;
            a.R_out = inplace ? R.as<float>() + (size_t)t * D : rout.as<float>() + (size_t)t * D;
            a.apply = apply;
            a.bo_prev = bo.as<float>() + (size_t)t * N;
            a.inj_prev = injp.as<float>() + (size_t)t * HC;
            a.w_norm = wn.as<float>();
            a.w_down = wd.as<uint16_t>();
            a.w_up = wu.as<uint16_t>();
            a.w_inject = withinj ? wi.as<uint16_t>() : nullptr;
            a.eps = (float)fx.param_f("eps");
            a.lo = lo.as<float>() + (size_t)t * LR;
            a.rs = rs.as<float>() + (size_t)t * HC;
            a.inject_out = injout.as<float>() + (size_t)t * HC;
            a.mixed = mixed.as<float>() + (size_t)t * N;
        }
        if (T == 1) k::fused_gr_read(fa[0], stream());
        else k::fused_gr_read_multi(fa.data(), (int)T, xn.as<float>(), stream());
        CK(cudaStreamSynchronize(g_stream));  // this backend's memcpy does not order against compute
        if (inplace) out_bytes("R_out", R);
        else out_bytes("R_out", rout);
        out_bytes("lo", lo);
        out_bytes("rs", rs);
        out_bytes("inject_out", injout);
        out_bytes("mixed", mixed);
        return true;
    }
    if (e == "resident_plan") {
        const int64_t plan_words = fx.param_i("plan_words"), capx = fx.param_i("capx");
        Dev ids = up(fx, "ids"), res = up(fx, "res_layer");
        Dev plan((size_t)plan_words * 4);
        {
            std::vector<int32_t> sentinel((size_t)plan_words, (int32_t)0x5A5A5A5A);
            CK(cudaMemcpy(plan.p, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice));
        }
        Dev skip(4);
        CK(cudaMemset(skip.p, 0xA5, 4));
        const unsigned long long* slot_off = nullptr;
        Dev off;
        if (fx.param_i("with_off")) {
            off = up(fx, "slot_off");
            slot_off = off.as<unsigned long long>();
        }
        // cache_base passed as 0: the entry only writes the value into the
        // plan, so the expected ptr words are the raw slot offsets.
        k::resident_plan(ids.as<int32_t>(), (int)fx.param_i("n_entries"), (int)fx.param_i("k"),
                         res.as<int32_t>(), (int)fx.param_i("n_expert"), nullptr, slot_off,
                         fx.param_i("blob"), plan.as<int32_t>(), capx, skip.as<uint32_t>(),
                         (uint32_t)fx.param_i("ring"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("plan", plan);
        out_bytes("skip", skip);
        return true;
    }
    // ---- prefill family (strata::prefill, include/strata/prefill/kernels.hpp) ----
    namespace pf = strata::prefill;
    const int64_t pT = fx.param_i("T", 0);
    if (e == "gr_norm" || e == "gr_norm_rs") {
        Dev R = up(fx, "R"), w = up(fx, "w_norm");
        Dev xn((size_t)pT * 10240 * 4), xn16((size_t)pT * 10240 * 2), rs((size_t)pT * 4 * 4);
        if (e == "gr_norm") pf::gr_norm(R.as<float>(), w.as<float>(), (float)fx.param_f("eps"), xn.as<float>(), xn16.as<uint16_t>(), pT, stream());
        else pf::gr_norm_rs(R.as<float>(), w.as<float>(), (float)fx.param_f("eps"), rs.as<float>(), xn16.as<uint16_t>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        if (e == "gr_norm") out_bytes("xn", xn);
        else out_bytes("rs", rs);
        out_bytes("xn16", xn16);
        return true;
    }
    if (e == "gr_mix_r" || e == "gr_mix") {
        Dev gated = up(fx, "gated");
        Dev mixed((size_t)pT * 2560 * 4), mixed16((size_t)pT * 2560 * 2), mixed_h((size_t)pT * 2560 * 2);
        if (e == "gr_mix_r") {
            Dev R = up(fx, "R"), rs = up(fx, "rs"), w = up(fx, "w_norm");
            pf::gr_mix_r(R.as<float>(), rs.as<float>(), w.as<float>(), gated.as<float>(), mixed.as<float>(),
                         mixed16.as<uint16_t>(), pT, stream(), mixed_h.as<uint16_t>());
        } else {
            Dev xn = up(fx, "xn");
            pf::gr_mix(xn.as<float>(), gated.as<float>(), mixed.as<float>(), mixed16.as<uint16_t>(), pT, stream(),
                       mixed_h.as<uint16_t>());
        }
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("mixed", mixed);
        out_bytes("mixed16", mixed16);
        out_bytes("mixed_h", mixed_h);
        return true;
    }
    if (e == "gr_silu") {
        Dev lo = up(fx, "lo");
        Dev lo16(lo.bytes / 2);
        pf::gr_silu(lo.as<float>(), lo16.as<uint16_t>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("lo16", lo16);
        return true;
    }
    if (e == "gr_write" || e == "gr_write_norm_rs") {
        Dev R = up(fx, "R"), bo = up(fx, "bo"), inj = up(fx, "inj");
        const int64_t inj_ld = fx.param_i("inj_ld");
        if (e == "gr_write") {
            pf::gr_write(R.as<float>(), bo.as<float>(), inj.as<float>(), inj_ld, pT, stream());
        } else {
            Dev wn = up(fx, "w_norm_next");
            Dev rs((size_t)pT * 4 * 4), xn16((size_t)pT * 10240 * 2);
            pf::gr_write_norm_rs(R.as<float>(), bo.as<float>(), inj.as<float>(), inj_ld, wn.as<float>(),
                                 (float)fx.param_f("eps"), rs.as<float>(), xn16.as<uint16_t>(), pT, stream());
            CK(cudaStreamSynchronize(g_stream));
            out_bytes("rs", rs);
            out_bytes("xn16", xn16);
        }
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("R_out", R);
        return true;
    }
    if (e == "gr_broadcast") {
        Dev ev = up(fx, "e");
        Dev R((size_t)pT * 10240 * 4);
        pf::gr_broadcast(ev.as<float>(), R.as<float>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("R_out", R);
        return true;
    }
    if (e == "gdn_gates") {
        Dev ab = up(fx, "ab"), dt = up(fx, "dt"), sa = up(fx, "ssm_a");
        Dev gate((size_t)pT * 48 * 4), beta((size_t)pT * 48 * 4);
        pf::gdn_gates(ab.as<float>(), dt.as<float>(), sa.as<float>(), gate.as<float>(), beta.as<float>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("gate", gate);
        out_bytes("beta", beta);
        return true;
    }
    if (e == "gdn_conv") {
        Dev hist = up(fx, "history"), qkv = up(fx, "qkv"), cw = up(fx, "conv_w");
        Dev h(qkv.bytes);
        pf::gdn_conv(hist.as<float>(), qkv.as<float>(), cw.as<float>(), h.as<float>(), pT, (float)fx.param_f("eps"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("history_out", hist);
        out_bytes("h", h);
        return true;
    }
    if (e == "gdn_recurrence") {
        Dev state = up(fx, "state"), h = up(fx, "h"), gate = up(fx, "gate"), beta = up(fx, "beta"), z = up(fx, "z"),
            gamma = up(fx, "gamma");
        Dev y((size_t)pT * 48 * 128 * 4), y16((size_t)pT * 48 * 128 * 2);
        pf::gdn_recurrence(state.as<float>(), h.as<float>(), gate.as<float>(), beta.as<float>(), z.as<float>(),
                           gamma.as<float>(), (float)fx.param_f("eps"), y.as<float>(), y16.as<uint16_t>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("state_out", state);
        out_bytes("y", y);
        out_bytes("y16", y16);
        return true;
    }
    if (e == "route") {
        Dev logits = up(fx, "logits");
        Dev ids((size_t)pT * 10 * 4), w((size_t)pT * 10 * 4);
        pf::route(logits.as<float>(), ids.as<int32_t>(), w.as<float>(), pT, fx.param_i("n_expert"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("ids", ids);
        out_bytes("weights", w);
        return true;
    }
    if (e == "swiglu_interleaved" || e == "swiglu_split") {
        Dev gu = up(fx, "gu");
        Dev h16((size_t)fx.param_i("n") * 640 * 2);
        if (e == "swiglu_interleaved") pf::swiglu_interleaved(gu.as<float>(), h16.as<uint16_t>(), fx.param_i("n"), stream());
        else pf::swiglu_split(gu.as<float>(), h16.as<uint16_t>(), fx.param_i("n"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("h16", h16);
        return true;
    }
    if (e == "swiglu_pair") {
        Dev g = up(fx, "g"), u = up(fx, "u");
        Dev h16((size_t)fx.param_i("n") * 640 * 2);
        pf::swiglu_pair(g.as<float>(), u.as<float>(), h16.as<uint16_t>(), fx.param_i("n"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("h16", h16);
        return true;
    }
    if (e == "copy_i32") {
        Dev src = up(fx, "src");
        Dev dst(src.bytes);
        pf::copy_i32(dst.as<int32_t>(), src.as<int32_t>(), fx.param_i("n"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("dst", dst);
        return true;
    }
    if (e == "gather_rows16") {
        Dev x = up(fx, "x16"), src = up(fx, "src");
        Dev dst((size_t)fx.param_i("n") * fx.param_i("width") * 2);
        pf::gather_rows16(x.as<uint16_t>(), src.as<int32_t>(), dst.as<uint16_t>(), fx.param_i("n"),
                          fx.param_i("width"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("dst16", dst);
        return true;
    }
    if (e == "moe_combine") {
        Dev Dm = up(fx, "D"), slot = up(fx, "slot"), w = up(fx, "w"), shared = up(fx, "shared"), sg = up(fx, "sg");
        Dev bo((size_t)pT * 2560 * 4);
        pf::moe_combine(Dm.as<float>(), slot.as<int32_t>(), w.as<float>(), shared.as<float>(), sg.as<float>(),
                        bo.as<float>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("bo", bo);
        return true;
    }
    if (e == "rms_rows") {
        Dev x = up(fx, "x"), w = up(fx, "w");
        pf::rms_rows(x.as<float>(), w.as<float>(), fx.param_i("rows"), fx.param_i("cols"), fx.param_i("ld"),
                     (float)fx.param_f("eps"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("x_out", x);
        return true;
    }
    if (e == "rope") {
        Dev x = up(fx, "x");
        pf::rope(x.as<float>(), pT, fx.param_i("heads"), fx.param_i("dim"), fx.param_i("ld"), fx.param_i("pos0"),
                 (float)fx.param_f("freq_base"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("x_out", x);
        return true;
    }
    if (e == "native_rope") {
        Dev x = up(fx, "x"), pos = up(fx, "pos");
        k::native_rope_apply(x.as<float>(), x.as<float>(), fx.param_i("rows"), fx.param_i("dim"), fx.param_i("n_rot"),
                             (float)fx.param_f("freq_base"), pos.as<int32_t>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("x_out", x);
        return true;
    }
    if (e == "split_q") {
        Dev qf = up(fx, "q_full");
        Dev q((size_t)pT * 24 * 256 * 4);
        pf::split_q(qf.as<float>(), q.as<float>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("q", q);
        return true;
    }
    if (e == "gate_attn") {
        Dev attn = up(fx, "attn"), qf = up(fx, "q_full");
        Dev o16((size_t)pT * 24 * 256 * 2);
        pf::gate_attn(attn.as<float>(), qf.as<float>(), o16.as<uint16_t>(), pT, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("out16", o16);
        return true;
    }
    if (e == "to_f16" || e == "to_bf16") {
        Dev x = up(fx, "x");
        Dev y(x.bytes / 2);
        if (e == "to_f16") pf::to_f16(x.as<float>(), y.as<uint16_t>(), fx.param_i("n"), stream());
        else pf::to_bf16(x.as<float>(), y.as<uint16_t>(), fx.param_i("n"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("y", y);
        return true;
    }
    if (e == "round_f16") {
        Dev x = up(fx, "x");
        Dev y(x.bytes);
        pf::round_f16(x.as<float>(), y.as<float>(), fx.param_i("n"), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("y", y);
        return true;
    }
    if (e == "kv_append") {
        Dev K = up(fx, "K"), V = up(fx, "V"), table = up(fx, "page_table");
        const int64_t rows = fx.param_i("pool_rows"), pos0 = fx.param_i("pos0"), ps = fx.param_i("page_size");
        const int mode = (int)fx.param_i("mode");  // 16 f16 + host mirror, 8 int8
        if (mode == 16) {
            Dev kp((size_t)rows * 256 * 2), vp((size_t)rows * 256 * 2);
            k::KvHostPools host;
            Dev hkp((size_t)rows * 256 * 2), hvp((size_t)rows * 256 * 2);
            // The pools are partially written by design; the fixture's
            // expected contents are zero outside the written cells.
            for (Dev* d : {&kp, &vp, &hkp, &hvp}) CK(cudaMemset(d->p, 0, d->bytes));
            host.k_pool = hkp.as<uint16_t>();
            host.v_pool = hvp.as<uint16_t>();
            pf::kv_append(K.as<float>(), V.as<float>(), pT, pos0, table.as<int32_t>(), ps, kp.as<uint16_t>(),
                          vp.as<uint16_t>(), nullptr, nullptr, nullptr, nullptr, stream(), &host, nullptr);
            CK(cudaStreamSynchronize(g_stream));
            out_bytes("k_pool", kp);
            out_bytes("v_pool", vp);
            out_bytes("host_k_pool", hkp);
            out_bytes("host_v_pool", hvp);
        } else {
            Dev kq((size_t)rows * 256), vq((size_t)rows * 256), ks((size_t)rows * 4 * 2), vs((size_t)rows * 4 * 2);
            for (Dev* d : {&kq, &vq, &ks, &vs}) CK(cudaMemset(d->p, 0, d->bytes));
            pf::kv_append(K.as<float>(), V.as<float>(), pT, pos0, table.as<int32_t>(), ps, nullptr, nullptr,
                          kq.as<int8_t>(), vq.as<int8_t>(), ks.as<uint16_t>(), vs.as<uint16_t>(), stream(), nullptr,
                          nullptr);
            CK(cudaStreamSynchronize(g_stream));
            out_bytes("k_q", kq);
            out_bytes("v_q", vq);
            out_bytes("k_scale", ks);
            out_bytes("v_scale", vs);
        }
        return true;
    }
    if (e == "qsa_attn_batch") {
        Dev kp = up(fx, "k_pool"), vp = up(fx, "v_pool"), table = up(fx, "page_table");
        Dev q = up(fx, "q"), ids = up(fx, "ids"), steps = up(fx, "steps");
        const int64_t cap = fx.param_i("cap");
        const int64_t nq = fx.param_i("nq", 2);
        const k::QsaShapes s = k::qsa_real_shapes();
        k::QsaAttnPools pools;
        pools.k_pool = kp.as<uint16_t>();
        pools.v_pool = vp.as<uint16_t>();
        pools.page_table = table.as<int32_t>();
        const uint64_t stride = k::qsa_decode_attn_scratch_floats(cap, s);
        Dev scratch((size_t) stride * (size_t) nq * 4);
        CK(cudaMemset(scratch.p, 0, scratch.bytes));
        Dev attn((size_t) nq * 24 * 256 * 4);
        k::qsa_decode_attn_batch(q.as<float>(), pools, ids.as<int32_t>(), steps.as<int32_t>(), cap, s,
                                 scratch.as<float>(), attn.as<float>(), (int) nq, stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("attn", attn);
        return true;
    }
    if (e == "qsa_block_scores") {
        Dev pooled = up(fx, "pooled"), dead = up(fx, "dead"), qidx = up(fx, "q_idx"), steps = up(fx, "steps");
        const int64_t nq = fx.param_i("nq"), max_blocks = fx.param_i("max_blocks");
        Dev scores((size_t)nq * (size_t)max_blocks * 4);
        if (fx.params_f.size()) scores.fill_f32((float)fx.param_f("out_fill", 0.0), (size_t)nq * (size_t)max_blocks);
        k::qsa_block_scores(pooled.as<float>(), dead.as<float>(), qidx.as<float>(), steps.as<int32_t>(), (int)nq,
                            max_blocks, k::qsa_real_shapes(), scores.as<float>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("scores", scores);
        return true;
    }
    if (e == "bf16_gemv_fp32_mmvf_multi") {
        Dev x = up(fx, "x"), w = up(fx, "w");
        const int64_t n_in = fx.param_i("n_in"), n_out = fx.param_i("n_out"), n_tok = fx.param_i("n_tok");
        const int64_t ldx = fx.param_i("ldx"), ldy = fx.param_i("ldy");
        Dev y((size_t)n_tok * (size_t)ldy * 4);
        k::bf16_gemv_fp32_mmvf_multi(x.as<float>(), ldx, w.as<uint16_t>(), y.as<float>(), ldy, n_in, n_out, (int)n_tok,
                                     stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("y", y);
        return true;
    }
    if (e == "qsa_block_topk") {
        Dev scores = up(fx, "scores"), steps = up(fx, "steps");
        const int64_t nq = fx.param_i("nq"), max_blocks = fx.param_i("max_blocks"), cap = fx.param_i("cap");
        Dev ids((size_t)nq * (size_t)cap * 4);
        k::qsa_block_topk(scores.as<float>(), steps.as<int32_t>(), (int)nq, max_blocks, cap, k::qsa_real_shapes(),
                          ids.as<int32_t>(), stream());
        CK(cudaStreamSynchronize(g_stream));
        out_bytes("ids", ids);
        return true;
    }
    return false;
}

int run_one(const std::string& path) {
    mqfx::Fixture fx;
    std::string err;
    if (!mqfx::read_fixture(path, fx, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    CK(cudaStreamCreate(&g_stream));
    std::map<std::string, std::vector<uint8_t>> results;
    if (!dispatch(fx, results)) {
        std::printf("SKIP %s %s.%s (no dispatch in runner yet)\n", fx.entry.c_str(), fx.entry.c_str(), fx.case_name.c_str());
        return 3;
    }
    bool ok = true;
    double worst = 0;
    for (const auto& want : fx.outputs) {
        auto it = results.find(want.name);
        if (it == results.end()) { std::fprintf(stderr, "  no result for output %s\n", want.name.c_str()); ok = false; continue; }
        double me = 0;
        if (!compare(fx, want, it->second, me)) ok = false;
        worst = std::max(worst, me);
    }
    std::printf("%s %s.%s max_err=%.3g\n", ok ? "PASS" : "FAIL", fx.entry.c_str(), fx.case_name.c_str(), worst);
    std::fflush(stdout);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: metal_kernel_parity <fixture_dir> [--entry <name>] [--list] [--no-fork]\n"); return 2; }
    const std::string dir = argv[1];
    std::string only_entry;
    bool list_only = false;
    bool no_fork = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--entry") == 0 && i + 1 < argc) only_entry = argv[++i];
        if (std::strcmp(argv[i], "--list") == 0) list_only = true;
        // --no-fork: run cases in-process. Needed on macOS when the runner's
        // parent context cannot hand a live MTLCompilerService XPC connection
        // to forked children (post-fork compiler lookup fails); pair with
        // --entry so a stub abort's blast radius stays one entry.
        if (std::strcmp(argv[i], "--no-fork") == 0) no_fork = true;
    }
    DIR* d = opendir(dir.c_str());
    if (!d) { std::fprintf(stderr, "cannot open fixture dir %s\n", dir.c_str()); return 2; }
    std::vector<std::string> files;
    while (dirent* de = readdir(d)) {
        std::string n = de->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".mqfx") == 0) files.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    int passed = 0, failed = 0, skipped = 0, crashed = 0;
    for (const auto& path : files) {
        if (list_only) {
            mqfx::Fixture fx;
            std::string err;
            if (mqfx::read_fixture(path, fx, err))
                std::printf("%s %s %s\n", fx.family.c_str(), fx.entry.c_str(), fx.case_name.c_str());
            continue;
        }
        if (!only_entry.empty() && path.find(only_entry + ".") == std::string::npos) continue;
        if (no_fork) {
            const int rc = run_one(path);
            if (rc == 0) ++passed;
            else if (rc == 3) ++skipped;
            else ++failed;
            std::fflush(stdout);
            continue;
        }
        pid_t pid = fork();
        if (pid == 0) {
            alarm(120);
            _exit(run_one(path));
        }
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            const int rc = WEXITSTATUS(status);
            if (rc == 0) ++passed;
            else if (rc == 3) ++skipped;
            else ++failed;
        } else {
            ++crashed;
            ++failed;
            // Name the case whose child died on a signal (the stub signature).
            mqfx::Fixture fx;
            std::string err;
            if (mqfx::read_fixture(path, fx, err))
                std::printf("FAIL %s.%s (child killed by signal %d)\n", fx.entry.c_str(), fx.case_name.c_str(), WTERMSIG(status));
        }
        std::fflush(stdout);
    }
    if (!list_only) std::printf("metal_kernel_parity: %d passed, %d failed (%d by signal), %d skipped of %zu fixtures\n",
                                passed, failed, crashed, skipped, files.size());
    return failed > 125 ? 125 : failed;
}
