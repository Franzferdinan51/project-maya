// src/hy4-metal/hy4_metal_model.hpp - the Hy4-Flash forward for the Metal port, in two
// implementations of the same spec (src/hy4/hy4_kernels.cuh + hy4_model.cu's block recipe):
//
//   * RefForward  - a scalar CPU golden (hy4_metal_ref.cpp), written straight from the kernel
//                   semantics; the stage-3 gate compares the ggml forward against it.
//   * GgmlForward - the production forward (hy4_metal_ggml.cpp): ggml graphs over the fetched
//                   ggml-metal backend, the fork's GLM/MM3 pattern.  Weights stay in their pack
//                   types; the latent KV cache is F16 rows, as upstream's cache is.
//
// Shared here (deterministic indexing / format primitives, not numerics under test): the pack
// weight access + dequantization, the Engram n-gram row gather (upstream gathers on the host
// too), the F8_E4M3 decode (upstream's own f8e4m3), and the trace sink both forwards fill at
// the same named points (upstream's dbg names).
#pragma once

#include "../hy4/hy4_pack.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace hym {

// ------------------------------------------------------------------ trace
struct Trace {
    std::map<std::string, std::vector<float>> at;
    void put(const std::string& name, const float* p, size_t n) {
        if (this) at[name].assign(p, p + n);
    }
    void put(const std::string& name, const std::vector<float>& v) {
        if (this) at[name] = v;
    }
};

// ------------------------------------------------------------------ weights
// A loaded pack: name -> the pack's tensor (raw bytes, ggml type), plus a dequantized F32
// copy on demand (the golden's weights; toy packs only - the production path never builds it).
struct Weights {
    hy4::Pack pk;
    const hy4::Geo* g = nullptr;
    std::map<std::string, const hy4::Tensor*> by_name;
    std::map<std::string, std::vector<float>> f32_cache;

    bool load(const std::string& pack_dir, std::string& err) {
        if (!pk.load(pack_dir, err)) return false;
        g = &pk.g;
        for (const auto& [name, t] : pk.t) by_name[name] = &t;
        return true;
    }
    const hy4::Tensor* tensor(const std::string& name) const {
        const auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : it->second;
    }
    bool has(const std::string& name) const { return tensor(name) != nullptr; }
    const uint8_t* data(const hy4::Tensor& t) const { return pk.data(t); }

    static ggml_type ggml_type_of(const std::string& type, bool& ok) {
        ok = true;
        if (type == "F32") return GGML_TYPE_F32;
        if (type == "F16") return GGML_TYPE_F16;
        if (type == "BF16") return GGML_TYPE_BF16;
        if (type == "Q8_0") return GGML_TYPE_Q8_0;
        if (type == "Q6_K") return GGML_TYPE_Q6_K;
        if (type == "Q5_K") return GGML_TYPE_Q5_K;
        if (type == "Q4_K") return GGML_TYPE_Q4_K;
        if (type == "IQ2_XXS") return GGML_TYPE_IQ2_XXS;
        if (type == "IQ3_XXS") return GGML_TYPE_IQ3_XXS;
        if (type == "IQ2_XS") return GGML_TYPE_IQ2_XS;
        if (type == "IQ2_S") return GGML_TYPE_IQ2_S;
        if (type == "IQ3_S") return GGML_TYPE_IQ3_S;
        if (type == "IQ4_XS") return GGML_TYPE_IQ4_XS;
        if (type == "Q8_1") return GGML_TYPE_Q8_1;
        ok = false;
        return GGML_TYPE_F32;
    }

    // The tensor's values as F32 (row-major, ne0 fastest), dequantized by ggml itself on the
    // CPU backend.  Cached.  nullptr (+ err) on an unknown type.
    const std::vector<float>* f32(const std::string& name, std::string& err) {
        const auto it = f32_cache.find(name);
        if (it != f32_cache.end()) return &it->second;
        const hy4::Tensor* t = tensor(name);
        if (!t) {
            err = "hy4-metal: no tensor " + name;
            return nullptr;
        }
        bool ok = false;
        const ggml_type gt = ggml_type_of(t->type, ok);
        if (!ok) {
            err = "hy4-metal: tensor " + name + " has type " + t->type + " (no ggml type)";
            return nullptr;
        }
        const int64_t ne[4] = {std::max<int64_t>(t->n(0), 1), std::max<int64_t>(t->n(1), 1),
                               std::max<int64_t>(t->n(2), 1), 1};
        size_t n = 1;
        for (int i = 0; i < 4; ++i) n *= (size_t) ne[i];
        ggml_init_params ip{64 * 1024 * 1024, nullptr, true};
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* src = ggml_new_tensor_4d(ctx, gt, ne[0], ne[1], ne[2], ne[3]);
        ggml_tensor* dst = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        ggml_cgraph* gr = ggml_new_graph(ctx);
        ggml_build_forward_expand(gr, ggml_cpy(ctx, src, dst));
        ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(src, data(*t), 0, ggml_nbytes(src));
        if (ggml_backend_graph_compute(be, gr) != GGML_STATUS_SUCCESS) {
            err = "hy4-metal: dequant of " + name + " failed";
            ggml_backend_free(be);
            ggml_free(ctx);
            return nullptr;
        }
        std::vector<float>& out = f32_cache[name];
        out.resize(n);
        ggml_backend_tensor_get(dst, out.data(), 0, n * sizeof(float));
        ggml_backend_free(be);
        ggml_free(ctx);
        return &out;
    }
};

// ------------------------------------------------------------------ F8 + Engram gather
inline float f8_e4m3_to_f32(uint8_t x) {   // (upstream's f8e4m3, hy4_kernels.cu)
    const int ax = x & 0x7f;
    if (ax == 0x7f) return NAN;
    const int e = (ax >> 3) & 0xf, m = ax & 7;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return (x & 0x80) ? -v : v;
}

struct EgRows {
    std::vector<uint8_t> bytes;   // [T][cols][head_dim] F8_E4M3
    std::vector<float> valid;     // [T]
};

// The n-gram rows of a window (upstream engram_row, hy4_model.cu): for the token at global
// position p, toks[0] is its mapped id (the pad when unmapped - and the token is then dead),
// toks[s] the mapped id s positions back (the pad before the start / unmapped); the rolling
// hash picks a table row a (column, position-in-group) at a time.
inline EgRows engram_gather(const hy4::Pack& pk, int module, const std::vector<int>& history, int64_t p0,
                            int T) {
    const hy4::Engram& eg = pk.eg;
    const int n = eg.ngram, cols = eg.cols(), hd = eg.head_dim;
    EgRows out;
    out.bytes.resize((size_t) T * cols * hd);
    out.valid.assign((size_t) T, 0.0f);
    const hy4::Tensor* tab = pk.get("per_layer_token_embd.weight");
    if (!tab || module < 0) return out;
    const uint8_t* table = pk.data(*tab);
    const uint64_t* mult = eg.mult.data() + (size_t) module * n;
    const uint64_t* primes = eg.primes.data() + (size_t) module * cols;
    const uint64_t* offs = eg.offsets.data() + (size_t) module * cols;
    auto cid = [&](int64_t tok) -> int64_t {
        return tok < 0 || tok >= eg.vocab ? -1 : (int64_t) eg.token_map[(size_t) tok];
    };
    for (int t = 0; t < T; ++t) {
        const int64_t p = p0 + t;
        uint64_t toks[8];
        const int64_t c0 = cid(history[(size_t) p]);
        const bool dead = c0 < 0;
        toks[0] = (uint64_t) (dead ? eg.pad : c0);
        for (int sft = 1; sft < n; ++sft) {
            const int64_t c = p - sft < 0 ? -1 : cid(history[(size_t) (p - sft)]);
            toks[sft] = (uint64_t) (c < 0 ? eg.pad : c);
        }
        uint64_t rolling = toks[0] * mult[0];
        for (int oo = 1; oo < n; ++oo) {
            rolling ^= toks[oo] * mult[oo];
            for (int h = 0; h < eg.heads; ++h) {
                const int col = (oo - 1) * eg.heads + h;
                const uint64_t row = rolling % primes[(size_t) col] + offs[(size_t) col];
                std::memcpy(out.bytes.data() + ((size_t) t * cols + col) * hd, table + row * (uint64_t) hd,
                            (size_t) hd);
            }
        }
        out.valid[(size_t) t] = dead ? 0.0f : 1.0f;
    }
    return out;
}

// The Engram norms' eps: the pack's engram_eps when present, else FLT_EPSILON (upstream's
// default, hy4_model.cu - it is NOT the model's rms eps).
inline float engram_eps(const hy4::Pack& pk) {
    return pk.has("engram_eps") ? (float) pk.num("engram_eps") : 1.1920929e-07f;
}

// ------------------------------------------------------------------ the forwards
class Forward {
public:
    virtual ~Forward() = default;
    // Process T tokens at the current position (the state's history grows by them).  On
    // success the state has advanced; logits_out holds the window's logits - all T rows
    // [T][n_vocab] for the reference and for a ggml forward built with want_all_logits,
    // else just the last row [n_vocab].
    virtual bool run(const int* ids, int T, float* logits_out, Trace* trace, std::string& err) = 0;
    virtual void reset() = 0;
    virtual int64_t pos() const = 0;
    // Rewind the state to position n (0 <= n <= pos()): the caches are positional, so the
    // rows past n are simply overwritten by whatever is fed next; the history (which the
    // Engram gather reads by absolute position) is truncated to match.  The serve loop's
    // conversation-prefix reuse rides on this.
    virtual void rewind_to(int64_t n) = 0;
};

// The serve protocol (serve/server.py's StrataEngine): INFO/READY, then GEN lines in,
// T/DONE lines out.  Defined in hy4_metal_serve.cpp; declared here so the engine main
// stays a thin CLI.
int serve_hy4(Forward& fwd, const hy4::Geo& g, int64_t max_ctx, int chunk);

std::unique_ptr<Forward> make_ref_forward(Weights& w, int64_t max_ctx);
std::unique_ptr<Forward> make_ggml_forward(Weights& w, ggml_backend_t be, int64_t max_ctx, bool want_all_logits);

}  // namespace hym
