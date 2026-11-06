// src/metal/mm3_dflash2.cpp - Mm3DFlash2 (mm3_dflash2.hpp).  ggml graphs on the Metal
// backend, the same idioms as mm3_dspark.cpp: weights in a persistent backend context,
// one scheduler, per-call graphs in a scratch context.  The math follows SGLang's
// dflash.py and this repo's GLM DFlash2 port (glm_metal_dflash.cpp); the spec is
// DFLASH2-SPEC.md.
//
// Differences from the DSpark head in the next file over: the fc reads SIX taps
// (dflash_config.target_layer_ids, [5, 16, 26, 37, 47, 58] here), the norms are plain
// Qwen3 RMSNorm (weight as stored - no Gemma +1), every attention and MLP sublayer is
// wrapped in a grouped dynamic convolution, and the tokens come from the candidate
// selector's lattice walk instead of a Markov chain.  All checkpoint weights are
// BF16 and are widened to F32 at the load (this ggml's Metal library has no
// BF16xBF16 mul_mat).
#include "mm3_dflash2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace maya::metal {
namespace {

constexpr int64_t kChunk = 1024;      // context rows a graph (CUDA's max_rows shape)
constexpr size_t kGraphCtxBytes = 256 * 1024 * 1024;
constexpr int kGraphNodes = 8192;

// ---------------------------------------------------------------- the safetensors file
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;   // absolute file offsets of the data
};

struct StFile {
    std::vector<uint8_t> bytes;   // the whole file (a draft is ~4 GB; it is read once)
    std::map<std::string, StTensor> tensors;

    bool open(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            err = "dflash2: cannot open " + path;
            return false;
        }
        const std::streamoff size = f.tellg();
        bytes.resize((size_t) size);
        f.seekg(0);
        if (!f.read((char*) bytes.data(), size)) {
            err = "dflash2: cannot read " + path;
            return false;
        }
        if (bytes.size() < 8) {
            err = "dflash2: " + path + " is truncated";
            return false;
        }
        uint64_t hlen = 0;
        std::memcpy(&hlen, bytes.data(), 8);
        if (8 + hlen > bytes.size()) {
            err = "dflash2: " + path + " has a bad header";
            return false;
        }
        return parse_header(std::string((const char*) bytes.data() + 8, (size_t) hlen),
                            8 + hlen, err);
    }

    // a small scanner for the header's flat {"name": {"dtype": .., "shape": [..],
    // "data_offsets": [a, b]}} objects - enough for safetensors headers, nothing more
    bool parse_header(const std::string& h, uint64_t data0, std::string& err) {
        size_t i = 0;
        auto ws = [&] { while (i < h.size() && (h[i] == ' ' || h[i] == '\n' || h[i] == '\t' || h[i] == '\r')) ++i; };
        auto str = [&](std::string& out) -> bool {
            if (i >= h.size() || h[i] != '"') return false;
            ++i;
            out.clear();
            while (i < h.size() && h[i] != '"') {
                if (h[i] == '\\' && i + 1 < h.size()) ++i;
                out += h[i++];
            }
            ++i;
            return true;
        };
        auto num = [&](int64_t& out) -> bool {
            ws();
            const size_t s = i;
            if (i < h.size() && h[i] == '-') ++i;
            while (i < h.size() && h[i] >= '0' && h[i] <= '9') ++i;
            if (s == i) return false;
            out = std::stoll(h.substr(s, i - s));
            return true;
        };
        ws();
        if (i >= h.size() || h[i] != '{') {
            err = "dflash2: bad safetensors header";
            return false;
        }
        ++i;
        for (;;) {
            ws();
            if (i < h.size() && h[i] == '}') break;
            std::string name;
            if (!str(name)) {
                err = "dflash2: bad safetensors header (name)";
                return false;
            }
            ws();
            if (i >= h.size() || h[i] != ':') return false;
            ++i;
            ws();
            if (name == "__metadata__") {   // skip its object
                int depth = 0;
                do {
                    if (h[i] == '{') ++depth;
                    else if (h[i] == '}') --depth;
                    ++i;
                } while (i < h.size() && depth > 0);
            } else {
                if (i >= h.size() || h[i] != '{') return false;
                ++i;
                StTensor t;
                uint64_t a = 0, b = 0;
                for (;;) {
                    ws();
                    if (i < h.size() && h[i] == '}') {
                        ++i;
                        break;
                    }
                    std::string key;
                    if (!str(key)) return false;
                    ws();
                    if (h[i] != ':') return false;
                    ++i;
                    ws();
                    if (key == "dtype") {
                        if (!str(t.dtype)) return false;
                    } else if (key == "shape" || key == "data_offsets") {
                        if (h[i] != '[') return false;
                        ++i;
                        std::vector<int64_t> vals;
                        for (;;) {
                            ws();
                            if (i < h.size() && h[i] == ']') {
                                ++i;
                                break;
                            }
                            int64_t v = 0;
                            if (!num(v)) return false;
                            vals.push_back(v);
                            ws();
                            if (i < h.size() && h[i] == ',') ++i;
                        }
                        if (key == "shape") t.shape = vals;
                        else if (vals.size() == 2) {
                            a = (uint64_t) vals[0];
                            b = (uint64_t) vals[1];
                        }
                    } else {   // an unknown field: skip a scalar
                        while (i < h.size() && h[i] != ',' && h[i] != '}') ++i;
                    }
                    ws();
                    if (i < h.size() && h[i] == ',') ++i;
                }
                t.begin = data0 + a;
                t.end = data0 + b;
                tensors[name] = t;
            }
            ws();
            if (i < h.size() && h[i] == ',') ++i;
        }
        return true;
    }

    const uint8_t* data(const StTensor& t) const { return bytes.data() + t.begin; }
    size_t size(const StTensor& t) const { return (size_t) (t.end - t.begin); }
};

// a config.json value hunt: the file is small and flat enough that "key": value scans
// are unambiguous for the keys read here (dflash_config's keys appear nowhere else)
bool cfg_num(const std::string& text, const std::string& key, double& out) {
    const std::string pat = "\"" + key + "\"";
    const size_t at = text.find(pat);
    if (at == std::string::npos) return false;
    const size_t colon = text.find(':', at + pat.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n')) ++i;
    const size_t s = i;
    if (i < text.size() && (text[i] == '-' || text[i] == '+')) ++i;
    while (i < text.size() && ((text[i] >= '0' && text[i] <= '9') || text[i] == '.' || text[i] == 'e' || text[i] == 'E' || text[i] == '-' || text[i] == '+')) ++i;
    if (s == i) return false;
    out = std::stod(text.substr(s, i - s));
    return true;
}

bool cfg_ints(const std::string& text, const std::string& key, std::vector<int>& out) {
    const std::string pat = "\"" + key + "\"";
    const size_t at = text.find(pat);
    if (at == std::string::npos) return false;
    const size_t lb = text.find('[', at);
    const size_t rb = text.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;
    out.clear();
    std::string cur;
    for (size_t i = lb + 1; i <= rb; ++i) {
        const char c = i < rb ? text[i] : ',';
        if (c == ',' || c == ' ' || c == '\n') {
            if (!cur.empty()) {
                out.push_back(std::stoi(cur));
                cur.clear();
            }
        } else cur += c;
    }
    return !out.empty();
}

float bf16_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// DFlash2's grouped dynamic convolution over a block's rows (SGLang _grouped_conv,
// ggml form from glm_metal_dflash.cpp): dyn is the kernel projection of the sublayer
// input [2*taps*G][B]; the coefficient of (tap, group) is shared by the group's
// channels and added to the learned base kernel; the tap shift zero-fills the block's
// first rows - a block never reads the previous block.
ggml_tensor* dconv(ggml_context* ctx, ggml_tensor* x, ggml_tensor* dyn, ggml_tensor* base, int side, int taps,
                   int group) {
    const int64_t E = x->ne[0], B = x->ne[1], G = E / group;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    ggml_tensor* coef = ggml_reshape_4d(ctx, ggml_cont(ctx, dyn), G, taps, 2, B);
    coef = ggml_cont(ctx, ggml_view_3d(ctx, coef, G, taps, B, coef->nb[1], coef->nb[3], (size_t) side * coef->nb[2]));
    coef = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, coef, 1, G, taps, B), group, G, taps, B);
    ggml_tensor* bs = ggml_reshape_4d(ctx, ggml_view_1d(ctx, base, E * taps, (size_t) side * base->nb[2]), group, G,
                                      taps, 1);
    ggml_tensor* wt = ggml_add(ctx, coef, bs);
    ggml_tensor* out = nullptr;
    for (int64_t tap = 0; tap < taps; ++tap) {
        ggml_tensor* v = x;
        if (tap > 0) {
            ggml_tensor* z = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, std::min<int64_t>(tap, B)), 0.0f);
            v = tap < B ? ggml_concat(ctx, z, ggml_view_2d(ctx, x, E, B - tap, x->nb[1], 0), 1) : z;
        }
        ggml_tensor* w = ggml_reshape_2d(
            ctx, ggml_cont(ctx, ggml_view_4d(ctx, wt, group, G, 1, B, wt->nb[1], wt->nb[2], wt->nb[3],
                                             (size_t) tap * wt->nb[2])),
            E, B);
        ggml_tensor* term = ggml_mul(ctx, w, v);
        out = out ? ggml_add(ctx, out, term) : term;
    }
    return out;
}

}  // namespace

struct Mm3DFlash2::Impl {
    Mm3DraftGeo g;
    int top_k = 0, conv_group = 16, conv_taps = 2;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* wctx = nullptr;         // weights + caches (backend-owned)
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_tensor *emb = nullptr, *head = nullptr;   // the target's (borrowed)
    int64_t max_ctx = 0, hi = 0;          // hi: one past the highest context position fed

    // weights (ggml layout: a safetensors [N][K] matrix is ne {K, N}; mul_mat contracts K)
    ggml_tensor* fc = nullptr;                       // F32 [NT*D][D]
    ggml_tensor *hidden_norm = nullptr, *norm = nullptr;   // F32 {D, 1}, plain (no +1)
    ggml_tensor *sel_hid = nullptr;                  // F32 [D][R]
    ggml_tensor *sel_pred = nullptr, *sel_succ = nullptr;   // F32 [R][V] (see the loader)
    struct L {
        ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
        ggml_tensor *gt = nullptr, *up = nullptr, *dn = nullptr;
        ggml_tensor *in_n = nullptr, *post_n = nullptr, *qn = nullptr, *kn = nullptr;
        ggml_tensor *attn_base = nullptr, *attn_kproj = nullptr;   // the attention conv
        ggml_tensor *mlp_base = nullptr, *mlp_kproj = nullptr;     // the MLP conv
        ggml_tensor *kc = nullptr, *vc = nullptr;   // BF16 [KV][max_ctx]
    };
    std::vector<L> layers;
    ggml_tensor* ctx_acc = nullptr;   // F32 [D][kChunk]

    std::vector<float> last_hid, last_logits;

    ~Impl() {
        if (sched) ggml_backend_sched_free(sched);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
    }

    ggml_tensor* new_wt(ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor* t = ggml_new_tensor_2d(wctx, type, ne0, ne1);
        ggml_set_name(t, "dflash2_w");
        return t;
    }

    // one graph: build in a scratch context, allocate on the scheduler, upload inputs, run
    bool compute(ggml_context* gctx, ggml_cgraph* gf,
                 const std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>>& up,
                 std::string& err) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            err = "dflash2: cannot allocate the compute graph";
            return false;
        }
        for (const auto& [t, d] : up) ggml_backend_tensor_set(t, d.data(), 0, d.size());
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            err = "dflash2: the compute graph failed";
            return false;
        }
        (void) gctx;
        return true;
    }

    struct Graph {
        ggml_context* ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> up;
        explicit Graph(size_t bytes = kGraphCtxBytes) {
            ggml_init_params ip{(size_t) bytes, nullptr, true};
            ctx = ggml_init(ip);
            gf = ggml_new_graph_custom(ctx, kGraphNodes, false);
        }
        ~Graph() { if (ctx) ggml_free(ctx); }
        ggml_tensor* in(ggml_type type, int64_t ne0, int64_t ne1, std::vector<uint8_t> data) {
            ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
            ggml_set_input(t);
            up.emplace_back(t, std::move(data));
            return t;
        }
        ggml_tensor* in1(ggml_type type, int64_t ne0, std::vector<uint8_t> data) {
            ggml_tensor* t = ggml_new_tensor_1d(ctx, type, ne0);
            ggml_set_input(t);
            up.emplace_back(t, std::move(data));
            return t;
        }
    };

    // plain RMSNorm: the stored weight as-is (Qwen3 style, no +1)
    ggml_tensor* rms(ggml_context* c, ggml_tensor* x, ggml_tensor* w) {
        x = ggml_rms_norm(c, x, g.eps);
        return w ? ggml_mul(c, x, w) : x;
    }

    // per-head plain norm over the head dim of a [hd][heads][rows] tensor
    ggml_tensor* head_norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w) {
        x = ggml_rms_norm(c, x, g.eps);
        return ggml_mul(c, x, ggml_view_2d(c, w, w->ne[0], 1, w->nb[1], 0));
    }

    ggml_tensor* rope(ggml_context* c, ggml_tensor* x, ggml_tensor* pos) {
        return ggml_rope_ext(c, x, pos, nullptr, g.n_rot, GGML_ROPE_TYPE_NEOX, 1048576,
                             g.theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }
};

bool Mm3DFlash2::load(const std::string& dir, ggml_tensor* emb, ggml_tensor* head, int64_t max_ctx,
                      std::string& err) {
    std::string base = dir;
    {
        std::ifstream probe(base + "/config.json");
        if (!probe) base = dir + "/dflash2";
    }
    std::ifstream cf(base + "/config.json");
    if (!cf) {
        err = "dflash2: no config.json in " + dir;
        return false;
    }
    const std::string cfg((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    auto p = std::make_unique<Impl>();
    Mm3DraftGeo& g = p->g;
    double v = 0;
    auto num = [&](const char* key, double dflt) {
        return cfg_num(cfg, key, v) ? v : dflt;
    };
    g.dim = (int) num("hidden_size", 0);
    g.vocab = (int) num("vocab_size", 0);
    g.n_layers = (int) num("num_hidden_layers", 0);
    g.n_head = (int) num("num_attention_heads", 0);
    g.n_kv = (int) num("num_key_value_heads", g.n_head);
    g.hd = (int) num("head_dim", g.n_head ? g.dim / std::max(g.n_head, 1) : 0);
    g.n_rot = g.hd;   // DFlash2 ropes the full head dim (llama.cpp's dflash, the GLM port)
    g.ff = (int) num("intermediate_size", 0);
    g.eps = (float) num("rms_norm_eps", 1e-6);
    g.theta = (float) num("rope_theta", 10000.0);
    g.window = (int) num("sliding_window", 4096);
    // dflash_config's keys appear nowhere else in the file, so flat scans are exact
    g.block = (int) num("block_size", 8);
    g.mask_id = (int) num("mask_token_id", -1);
    g.rank = (int) num("selector_rank", 0);
    p->top_k = (int) num("selector_top_k", 0);
    p->conv_group = (int) num("conv_group_size", 16);
    p->conv_taps = (int) num("conv_kernel_size", 2);
    cfg_ints(cfg, "target_layer_ids", g.taps);
    if (g.dim <= 0 || g.vocab <= 0 || g.n_layers <= 0 || g.hd <= 0 || g.ff <= 0 || g.block < 2 ||
        g.block > 16 || g.rank <= 0 || g.taps.empty() || g.n_head % std::max(g.n_kv, 1) != 0 ||
        g.mask_id < 0 || g.mask_id >= g.vocab || p->top_k < 1 || p->top_k > 64 ||
        g.dim % p->conv_group != 0 || max_ctx < 1) {
        err = "dflash2: the config in " + base + " has no DFlash2 draft this engine covers";
        return false;
    }
    const int D = g.dim, V = g.vocab, R = g.rank, NT = (int) g.taps.size();
    const int Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff;
    const int G = D / p->conv_group, CT = p->conv_taps;

    StFile st;
    if (!st.open(base + "/model.safetensors", err)) return false;
    auto get = [&](const std::string& name) -> const StTensor* {
        const auto it = st.tensors.find(name);
        return it == st.tensors.end() ? nullptr : &it->second;
    };

    p->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!p->backend) {
        err = "dflash2: no GPU backend";
        return false;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend* bufs[2] = {p->backend, cpu};
    p->sched = ggml_backend_sched_new(bufs, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    ggml_init_params ip{16 * 1024 * 1024, nullptr, true};
    p->wctx = ggml_init(ip);
    p->emb = emb;
    p->head = head;
    p->max_ctx = max_ctx;

    auto bf16 = [&](const std::string& name, int64_t ne0, int64_t ne1, ggml_tensor*& out) -> bool {
        const StTensor* t = get(name);
        if (!t || t->dtype != "BF16" || st.size(*t) != (size_t) ne0 * ne1 * 2) {
            err = "dflash2: " + name + ": not the BF16 tensor expected";
            return false;
        }
        out = p->new_wt(GGML_TYPE_F32, ne0, ne1);   // F32 storage: this ggml's Metal library has
        return true;                                // no BF16 x BF16 mul_mat kernel (it aborts)
    };
    auto normv = [&](const std::string& name, int64_t n, ggml_tensor*& out) -> bool {
        const StTensor* t = get(name);
        if (!t || t->dtype != "BF16" || st.size(*t) != (size_t) n * 2) {
            err = "dflash2: " + name + ": not the BF16 vector expected";
            return false;
        }
        out = p->new_wt(GGML_TYPE_F32, n, 1);
        return true;
    };
    // a conv base kernel: safetensors [2][taps][D]; stored ggml [D][taps][2], whose
    // linear layout is byte-identical, so the upload below is a straight widening copy
    auto convbase = [&](const std::string& name, ggml_tensor*& out) -> bool {
        const StTensor* t = get(name);
        if (!t || t->dtype != "BF16" || t->shape != std::vector<int64_t>({2, CT, D}) ||
            st.size(*t) != (size_t) 2 * CT * D * 2) {
            err = "dflash2: " + name + ": not the BF16 base kernel expected";
            return false;
        }
        out = ggml_new_tensor_3d(p->wctx, GGML_TYPE_F32, D, CT, 2);
        ggml_set_name(out, "dflash2_w");
        return true;
    };

    bool ok = bf16("fc.weight", (int64_t) NT * D, D, p->fc) &&
              normv("hidden_norm.weight", D, p->hidden_norm) && normv("norm.weight", D, p->norm) &&
              bf16("candidate_selector.hidden_projection.weight", D, R, p->sel_hid) &&
              // the codebooks arrive [V][R]; a ggml [R][V] tensor over the same widened
              // bytes gathers a token's row as a column in ggml_get_rows - no transpose
              bf16("candidate_selector.predecessor_codebook", R, V, p->sel_pred) &&
              bf16("candidate_selector.successor_codebook", R, V, p->sel_succ);
    p->layers.resize((size_t) g.n_layers);
    for (int l = 0; ok && l < g.n_layers; ++l) {
        const std::string P = "layers." + std::to_string(l) + ".";
        Impl::L& L = p->layers[(size_t) l];
        ok = bf16(P + "self_attn.q_proj.weight", D, Q, L.q) && bf16(P + "self_attn.k_proj.weight", D, KV, L.k) &&
             bf16(P + "self_attn.v_proj.weight", D, KV, L.v) && bf16(P + "self_attn.o_proj.weight", Q, D, L.o) &&
             bf16(P + "mlp.gate_proj.weight", D, F, L.gt) && bf16(P + "mlp.up_proj.weight", D, F, L.up) &&
             bf16(P + "mlp.down_proj.weight", F, D, L.dn) &&
             normv(P + "input_layernorm.weight", D, L.in_n) &&
             normv(P + "post_attention_layernorm.weight", D, L.post_n) &&
             normv(P + "self_attn.q_norm.weight", g.hd, L.qn) &&
             normv(P + "self_attn.k_norm.weight", g.hd, L.kn) &&
             convbase(P + "attention_conv.base_kernel", L.attn_base) &&
             bf16(P + "attention_conv.kernel_projection.weight", D, 2 * CT * G, L.attn_kproj) &&
             convbase(P + "mlp_conv.base_kernel", L.mlp_base) &&
             bf16(P + "mlp_conv.kernel_projection.weight", D, 2 * CT * G, L.mlp_kproj);
        if (ok) {
            L.kc = p->new_wt(GGML_TYPE_BF16, KV, max_ctx);
            L.vc = p->new_wt(GGML_TYPE_BF16, KV, max_ctx);
        }
    }
    if (ok) p->ctx_acc = p->new_wt(GGML_TYPE_F32, D, kChunk);
    if (!ok) return false;

    p->wbuf = ggml_backend_alloc_ctx_tensors(p->wctx, p->backend);
    if (!p->wbuf) {
        err = "dflash2: cannot allocate the weights";
        return false;
    }
    auto up_bf16 = [&](const std::string& name, ggml_tensor* t) {
        const uint16_t* b = (const uint16_t*) st.data(*get(name));
        std::vector<float> f((size_t) ggml_nelements(t));
        for (size_t i = 0; i < f.size(); ++i) f[i] = bf16_f32(b[i]);
        ggml_backend_tensor_set(t, f.data(), 0, f.size() * 4);
    };
    up_bf16("fc.weight", p->fc);
    up_bf16("hidden_norm.weight", p->hidden_norm);
    up_bf16("norm.weight", p->norm);
    up_bf16("candidate_selector.hidden_projection.weight", p->sel_hid);
    up_bf16("candidate_selector.predecessor_codebook", p->sel_pred);
    up_bf16("candidate_selector.successor_codebook", p->sel_succ);
    for (int l = 0; l < g.n_layers; ++l) {
        const std::string P = "layers." + std::to_string(l) + ".";
        Impl::L& L = p->layers[(size_t) l];
        up_bf16(P + "self_attn.q_proj.weight", L.q);
        up_bf16(P + "self_attn.k_proj.weight", L.k);
        up_bf16(P + "self_attn.v_proj.weight", L.v);
        up_bf16(P + "self_attn.o_proj.weight", L.o);
        up_bf16(P + "mlp.gate_proj.weight", L.gt);
        up_bf16(P + "mlp.up_proj.weight", L.up);
        up_bf16(P + "mlp.down_proj.weight", L.dn);
        up_bf16(P + "input_layernorm.weight", L.in_n);
        up_bf16(P + "post_attention_layernorm.weight", L.post_n);
        up_bf16(P + "self_attn.q_norm.weight", L.qn);
        up_bf16(P + "self_attn.k_norm.weight", L.kn);
        up_bf16(P + "attention_conv.base_kernel", L.attn_base);
        up_bf16(P + "attention_conv.kernel_projection.weight", L.attn_kproj);
        up_bf16(P + "mlp_conv.base_kernel", L.mlp_base);
        up_bf16(P + "mlp_conv.kernel_projection.weight", L.mlp_kproj);
    }
    this->p = std::move(p);
    return true;
}

bool Mm3DFlash2::context(const float* taps, int T, int64_t pos0, std::string& err) {
    Impl& p = *this->p;
    const Mm3DraftGeo& g = p.g;
    const int D = g.dim, KV = g.n_kv * g.hd, NT = (int) g.taps.size();
    if (pos0 + T > p.max_ctx) {
        err = "dflash2: the context runs past the cache";
        return false;
    }
    for (int a = 0; a < T; a += (int) kChunk) {
        const int n = (int) std::min<int64_t>(kChunk, T - a);
        {   // ctx_acc[:, :n] = sum_j fc_j . taps_j
            Impl::Graph gr;
            ggml_tensor* acc = nullptr;
            for (int j = 0; j < NT; ++j) {
                const float* rows = taps + ((size_t) j * T + a) * D;
                ggml_tensor* t = gr.in(GGML_TYPE_F32, D, n,
                                       std::vector<uint8_t>((const uint8_t*) rows,
                                                            (const uint8_t*) rows + (size_t) n * D * 4));
                ggml_tensor* fcj = ggml_view_2d(gr.ctx, p.fc, D, D, p.fc->nb[1],
                                                (size_t) j * D * ggml_type_size(GGML_TYPE_F32));
                ggml_tensor* part = ggml_mul_mat(gr.ctx, fcj, t);
                acc = acc ? ggml_add(gr.ctx, acc, part) : part;
            }
            ggml_tensor* dest = ggml_view_2d(gr.ctx, p.ctx_acc, D, n, p.ctx_acc->nb[1], 0);
            ggml_build_forward_expand(gr.gf, ggml_cpy(gr.ctx, acc, dest));
            if (!p.compute(gr.ctx, gr.gf, gr.up, err)) return false;
        }
        {   // the chunk's K/V into every layer's cache at pos0 + a (no attention, no
            // convolution - context positions only ever appear as keys/values)
            Impl::Graph gr;
            ggml_tensor* accv = ggml_view_2d(gr.ctx, p.ctx_acc, D, n, p.ctx_acc->nb[1], 0);
            ggml_tensor* xn = p.rms(gr.ctx, accv, p.hidden_norm);
            std::vector<int32_t> pv((size_t) n);
            for (int i = 0; i < n; ++i) pv[(size_t) i] = (int32_t) (pos0 + a + i);
            ggml_tensor* pos = gr.in1(GGML_TYPE_I32, n,
                                      std::vector<uint8_t>((const uint8_t*) pv.data(),
                                                           (const uint8_t*) pv.data() + (size_t) n * 4));
            for (const Impl::L& L : p.layers) {
                ggml_tensor* K = ggml_mul_mat(gr.ctx, L.k, xn);   // [KV][n]
                ggml_tensor* V = ggml_mul_mat(gr.ctx, L.v, xn);
                ggml_tensor* k3 = ggml_reshape_3d(gr.ctx, K, g.hd, g.n_kv, n);
                k3 = p.rope(gr.ctx, p.head_norm(gr.ctx, k3, L.kn), pos);
                ggml_tensor* kd = ggml_view_2d(gr.ctx, L.kc, KV, n, L.kc->nb[1],
                                               (size_t) (pos0 + a) * KV * ggml_type_size(GGML_TYPE_BF16));
                ggml_build_forward_expand(gr.gf, ggml_cpy(gr.ctx, ggml_reshape_2d(gr.ctx, k3, KV, n), kd));
                ggml_tensor* vd = ggml_view_2d(gr.ctx, L.vc, KV, n, L.vc->nb[1],
                                               (size_t) (pos0 + a) * KV * ggml_type_size(GGML_TYPE_BF16));
                ggml_build_forward_expand(gr.gf, ggml_cpy(gr.ctx, V, vd));
            }
            if (!p.compute(gr.ctx, gr.gf, gr.up, err)) return false;
        }
    }
    p.hi = std::max(p.hi, pos0 + T);
    return true;
}

bool Mm3DFlash2::propose(int32_t anchor, int64_t pos, int n, std::vector<int32_t>& drafts,
                         std::vector<float>& confs, std::string& err) {
    Impl& p = *this->p;
    const Mm3DraftGeo& g = p.g;
    const int D = g.dim, V = g.vocab, R = g.rank, B = g.block;
    const int Q = g.n_head * g.hd, KV = g.n_kv * g.hd, G = g.n_head / g.n_kv;
    const int K = p.top_k, S = B - 1;   // draft slots = the block's mask rows
    drafts.clear();
    confs.clear();
    if (pos + B > p.max_ctx || pos < 0) return true;   // no room: no drafts, no error
    const int want = std::max(0, std::min(n, S));
    if (want == 0) return true;
    const int64_t hi = pos + B - 1, lo = std::max<int64_t>(0, hi - g.window + 1);
    const int64_t Twin = hi - lo + 1;
    ggml_tensor *hid_t = nullptr, *logits_t = nullptr, *cand_t = nullptr, *unary_t = nullptr,
                *gate_t = nullptr, *succ_t = nullptr, *pred_t = nullptr;
    std::vector<int32_t> cand((size_t) K * S);
    std::vector<float> un((size_t) K * S), ga((size_t) R * S), su((size_t) R * K * S),
        pr((size_t) R * (1 + K * (S - 1)));
    {
        Impl::Graph gr;
        std::vector<int32_t> ids((size_t) B, g.mask_id);
        ids[0] = anchor;
        ggml_tensor* idt = gr.in1(GGML_TYPE_I32, B,
                                  std::vector<uint8_t>((const uint8_t*) ids.data(),
                                                       (const uint8_t*) ids.data() + (size_t) B * 4));
        std::vector<int32_t> pv((size_t) B);
        for (int i = 0; i < B; ++i) pv[(size_t) i] = (int32_t) (pos + i);
        ggml_tensor* post = gr.in1(GGML_TYPE_I32, B,
                                   std::vector<uint8_t>((const uint8_t*) pv.data(),
                                                        (const uint8_t*) pv.data() + (size_t) B * 4));
        ggml_tensor* h = ggml_cpy(gr.ctx, ggml_get_rows(gr.ctx, p.emb, idt),
                                  ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, D, B));
        const float scale = 1.0f / std::sqrt((float) g.hd);
        for (const Impl::L& L : p.layers) {
            ggml_tensor* xn = p.rms(gr.ctx, h, L.in_n);
            ggml_tensor* dyn = ggml_mul_mat(gr.ctx, L.attn_kproj, xn);
            ggml_tensor* xc = dconv(gr.ctx, xn, dyn, L.attn_base, 0, p.conv_taps, p.conv_group);
            ggml_tensor* q = ggml_mul_mat(gr.ctx, L.q, xc);
            ggml_tensor* k = ggml_mul_mat(gr.ctx, L.k, xc);
            ggml_tensor* v = ggml_mul_mat(gr.ctx, L.v, xc);
            ggml_tensor* q3 = ggml_reshape_3d(gr.ctx, q, g.hd, g.n_head, B);
            q3 = p.rope(gr.ctx, p.head_norm(gr.ctx, q3, L.qn), post);
            ggml_tensor* k3 = ggml_reshape_3d(gr.ctx, k, g.hd, g.n_kv, B);
            k3 = p.rope(gr.ctx, p.head_norm(gr.ctx, k3, L.kn), post);
            ggml_tensor* kd = ggml_view_2d(gr.ctx, L.kc, KV, B, L.kc->nb[1],
                                           (size_t) pos * KV * ggml_type_size(GGML_TYPE_BF16));
            // the attention below must read the caches THROUGH these cpy results: a view of
            // the base tensor carries no dependency on the write (llama.cpp's KV pattern)
            ggml_tensor* kwr = ggml_cpy(gr.ctx, ggml_reshape_2d(gr.ctx, k3, KV, B), kd);
            ggml_build_forward_expand(gr.gf, kwr);
            ggml_tensor* vd = ggml_view_2d(gr.ctx, L.vc, KV, B, L.vc->nb[1],
                                           (size_t) pos * KV * ggml_type_size(GGML_TYPE_BF16));
            ggml_tensor* vwr = ggml_cpy(gr.ctx, v, vd);
            ggml_build_forward_expand(gr.gf, vwr);
            std::vector<ggml_tensor*> groups;
            for (int gi = 0; gi < g.n_kv; ++gi) {
                ggml_tensor* kw = ggml_view_2d(gr.ctx, kwr, g.hd, Twin, L.kc->nb[1],
                                               (size_t) ((lo - pos) * KV + gi * g.hd) * ggml_type_size(GGML_TYPE_BF16));
                ggml_tensor* kf = ggml_cpy(gr.ctx, kw, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, g.hd, Twin));
                ggml_tensor* vw = ggml_view_2d(gr.ctx, vwr, g.hd, Twin, L.vc->nb[1],
                                               (size_t) ((lo - pos) * KV + gi * g.hd) * ggml_type_size(GGML_TYPE_BF16));
                ggml_tensor* vf = ggml_cpy(gr.ctx, vw, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, g.hd, Twin));
                ggml_tensor* vt = ggml_cont(gr.ctx, ggml_transpose(gr.ctx, vf));
                ggml_tensor* qg = ggml_view_3d(gr.ctx, q3, g.hd, G, B, q3->nb[1], q3->nb[2],
                                               (size_t) gi * G * q3->nb[1]);
                ggml_tensor* sc = ggml_mul_mat(gr.ctx, kf, qg);   // [Twin][G][B]
                sc = ggml_soft_max(gr.ctx, ggml_scale(gr.ctx, sc, scale));
                groups.push_back(ggml_mul_mat(gr.ctx, vt, sc));   // [hd][G][B]
            }
            ggml_tensor* att = groups[0];
            for (size_t i = 1; i < groups.size(); ++i) att = ggml_concat(gr.ctx, att, groups[i], 1);
            att = ggml_cont(gr.ctx, ggml_reshape_2d(gr.ctx, att, Q, B));
            ggml_tensor* o = ggml_mul_mat(gr.ctx, L.o, att);
            o = dconv(gr.ctx, o, dyn, L.attn_base, 1, p.conv_taps, p.conv_group);
            h = ggml_add(gr.ctx, h, o);
            ggml_tensor* xn2 = p.rms(gr.ctx, h, L.post_n);
            ggml_tensor* dyn2 = ggml_mul_mat(gr.ctx, L.mlp_kproj, xn2);
            ggml_tensor* fc = dconv(gr.ctx, xn2, dyn2, L.mlp_base, 0, p.conv_taps, p.conv_group);
            ggml_tensor* act = ggml_swiglu_split(gr.ctx, ggml_mul_mat(gr.ctx, L.gt, fc),
                                                 ggml_mul_mat(gr.ctx, L.up, fc));
            ggml_tensor* mo = ggml_mul_mat(gr.ctx, L.dn, act);
            mo = dconv(gr.ctx, mo, dyn2, L.mlp_base, 1, p.conv_taps, p.conv_group);
            h = ggml_add(gr.ctx, h, mo);
        }
        hid_t = p.rms(gr.ctx, h, p.norm);
        ggml_set_output(hid_t);
        ggml_build_forward_expand(gr.gf, hid_t);
        ggml_tensor* hidb = ggml_cpy(gr.ctx, hid_t, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_BF16, D, B));
        ggml_tensor* hidf = ggml_cpy(gr.ctx, hidb, ggml_new_tensor_2d(gr.ctx, GGML_TYPE_F32, D, B));
        logits_t = ggml_mul_mat(gr.ctx, p.head, hidf);
        ggml_set_output(logits_t);
        ggml_build_forward_expand(gr.gf, logits_t);
        // the candidate selector, in-graph up to the lattice inputs (the walk is host-side)
        ggml_tensor* yr = ggml_view_2d(gr.ctx, hid_t, D, S, hid_t->nb[1], hid_t->nb[1]);   // mask rows
        ggml_tensor* lg = ggml_cont(gr.ctx, ggml_view_2d(gr.ctx, logits_t, V, S, logits_t->nb[1], logits_t->nb[1]));
        cand_t = ggml_cont(gr.ctx, ggml_argsort_top_k(gr.ctx, lg, K));                    // [K][S]
        unary_t = ggml_get_rows(gr.ctx, ggml_reshape_3d(gr.ctx, lg, 1, V, S), cand_t);    // [1][K][S]
        gate_t = ggml_mul_mat(gr.ctx, p.sel_hid, yr);                                     // [R][S]
        succ_t = ggml_get_rows(gr.ctx, p.sel_succ, ggml_reshape_1d(gr.ctx, cand_t, (int64_t) K * S));
        ggml_tensor* anch = gr.in1(GGML_TYPE_I32, 1,
                                   std::vector<uint8_t>((const uint8_t*) &anchor, (const uint8_t*) &anchor + 4));
        ggml_tensor* pids = anch;
        if (S > 1) {
            ggml_tensor* rest = ggml_reshape_1d(
                gr.ctx, ggml_cont(gr.ctx, ggml_view_2d(gr.ctx, cand_t, K, S - 1, cand_t->nb[1], 0)),
                (int64_t) K * (S - 1));
            pids = ggml_concat(gr.ctx, anch, rest, 0);
        }
        pred_t = ggml_get_rows(gr.ctx, p.sel_pred, pids);                                 // [R][1 + K*(S-1)]
        for (ggml_tensor* t : {cand_t, unary_t, gate_t, succ_t, pred_t}) {
            ggml_set_output(t);
            ggml_build_forward_expand(gr.gf, t);
        }
        if (!p.compute(gr.ctx, gr.gf, gr.up, err)) return false;
        p.last_hid.assign((size_t) B * D, 0.0f);
        p.last_logits.assign((size_t) B * V, 0.0f);
        ggml_backend_tensor_get(hid_t, p.last_hid.data(), 0, p.last_hid.size() * 4);
        ggml_backend_tensor_get(logits_t, p.last_logits.data(), 0, p.last_logits.size() * 4);
        ggml_backend_tensor_get(cand_t, cand.data(), 0, cand.size() * 4);
        ggml_backend_tensor_get(unary_t, un.data(), 0, un.size() * 4);
        ggml_backend_tensor_get(gate_t, ga.data(), 0, ga.size() * 4);
        ggml_backend_tensor_get(succ_t, su.data(), 0, su.size() * 4);
        ggml_backend_tensor_get(pred_t, pr.data(), 0, pr.size() * 4);
    }

    // the lattice walk (the GLM port's host walk): slot i scores its candidates by
    // their head logit plus the transition from the previous pick; the confidence is
    // the pick's softmax share of the slot's lattice scores
    const float* pv = pr.data();   // the anchor's predecessor row
    for (int i = 0; i < want; ++i) {
        double smax = -INFINITY;
        int best = 0;
        std::vector<double> s((size_t) K);
        for (int j = 0; j < K; ++j) {
            const float* sv = su.data() + ((size_t) i * K + (size_t) j) * R;
            const float* gv = ga.data() + (size_t) i * R;
            double acc = un[(size_t) i * K + (size_t) j];
            for (int r = 0; r < R; ++r) acc += (double) pv[r] * gv[r] * sv[r];
            s[(size_t) j] = acc;
            if (acc > smax) {
                smax = acc;
                best = j;
            }
        }
        double z = 0.0;
        for (int j = 0; j < K; ++j) z += std::exp(s[(size_t) j] - smax);
        drafts.push_back(cand[(size_t) i * K + (size_t) best]);
        confs.push_back((float) (1.0 / z));
        if (i + 1 < want) pv = pr.data() + (size_t) (1 + i * K + best) * R;   // the pick's predecessor row
    }
    return true;
}

Mm3DFlash2::Mm3DFlash2() = default;
Mm3DFlash2::~Mm3DFlash2() = default;
const Mm3DraftGeo& Mm3DFlash2::geo() const { return p->g; }
int64_t Mm3DFlash2::max_ctx() const { return p->max_ctx; }
int Mm3DFlash2::top_k() const { return p->top_k; }
int Mm3DFlash2::conv_group() const { return p->conv_group; }
int Mm3DFlash2::conv_taps() const { return p->conv_taps; }
void Mm3DFlash2::reset_context() { p->hi = 0; }
int64_t Mm3DFlash2::context_hi() const { return p->hi; }
const std::vector<float>& Mm3DFlash2::last_hid() const { return p->last_hid; }
const std::vector<float>& Mm3DFlash2::last_logits() const { return p->last_logits; }

}  // namespace maya::metal
