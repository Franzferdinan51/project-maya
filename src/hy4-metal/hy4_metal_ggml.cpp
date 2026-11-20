// src/hy4-metal/hy4_metal_ggml.cpp - the Hy4-Flash trunk forward as ggml graphs over the
// fetched ggml backend (ggml-metal in production; ggml-cpu selectable for debugging).
//
// The block math is upstream's (hy4_model.cu's recipe, hy4_kernels.cuh's semantics), expressed
// with ggml ops: weights keep their pack types and are consumed by ggml_mul_mat directly; the
// latent KV cache is F16 rows [kv_lora normed | rope part] per position, the indexer's
// value|gate ring is F16, its pooled keys F32 - the same stored forms as upstream.
//
// Deliberate simplifications (semantics-preserving, noted in the run record):
//   * the KV cache is linear over max_ctx positions (upstream rings it past n_swa + chunk);
//     identical results wherever upstream's window fits its ring.
//   * the indexer picks and the router picks are computed on the host from graph-produced
//     scores / logits (upstream: device kernels); the picks are data, the math is the same.
//   * a window runs as a handful of graphs per layer with small readbacks between them (the
//     picks, the traces) - correctness first; speed is measured honestly at stage 6.
// The MTP block is NOT forwarded: this is the trunk + head.  Draft/speculative requests must
// stop loudly at the caller, never silently fall back.
#include "hy4_metal_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>

#include <algorithm>
#include <chrono>
#include <dispatch/dispatch.h>
#include <limits>

namespace hym {
namespace {

constexpr float kMaskOff = -1e30f;

class Ggml : public Forward {
public:
    Ggml(Weights& w, ggml_backend_t be, int64_t max_ctx, bool all_logits, double eg_cache_gb = 1.0)
        : w_(w), g_(*w.g), be_(be), max_ctx_(max_ctx), all_logits_(all_logits),
          egc_((size_t) w.pk.eg.head_dim, eg_cache_gb) {}

    ~Ggml() override {
        for (ggml_backend_buffer_t b : bufs_) ggml_backend_buffer_free(b);
        for (ggml_context* ctx : wctx_) ggml_free(ctx);
        if (cctx_) ggml_free(cctx_);
    }

    bool init(std::string& err) {
        verbose_ = std::getenv("HY4_METAL_VERBOSE") != nullptr;
        if (!load_ctx(-1, err)) return false;
        for (int il = 0; il < g_.n_layer; ++il)
            if (!load_ctx(il, err)) return false;
        cctx_ = ggml_init({64 * 1024 * 1024, nullptr, true});
        kv_.assign((size_t) g_.n_layer, nullptr);
        ivg_.assign((size_t) g_.n_layer, nullptr);
        pooled_.assign((size_t) g_.n_layer, nullptr);
        for (int il = 0; il < g_.n_layer; ++il) {
            kv_[(size_t) il] = ggml_new_tensor_2d(cctx_, GGML_TYPE_F16, g_.kv_row(), max_ctx_);
            if (is_ix(il)) {
                ivg_[(size_t) il] = ggml_new_tensor_2d(cctx_, GGML_TYPE_F16, 2 * g_.ix_dim, max_ctx_);
                pooled_[(size_t) il] =
                    ggml_new_tensor_2d(cctx_, GGML_TYPE_F32, g_.ix_dim, max_ctx_ / g_.ix_kpool + 1);
            }
        }
        if (!ggml_backend_alloc_ctx_tensors(cctx_, be_)) {
            err = "hy4-metal: cache allocation failed";
            return false;
        }
        bufs_.push_back(kv_[0]->buffer);
        zero_caches();
        return true;
    }

    void reset() override {
        pos_ = 0;
        hist_.clear();
        zero_caches();
    }
    int64_t pos() const override { return pos_; }
    void rewind_to(int64_t n) override {
        pos_ = n;
        hist_.resize((size_t) n);
    }

    TierInfo tier_info() const override {
        TierInfo ti;
        ti.eg = egst_;
        ti.resident_gb = resident_bytes_ / 1e9;
        ti.eg_cache_rows = (int64_t) egc_.size();
        ti.eg_cache_cap = (int64_t) egc_.capacity();
        ti.eg_row_bytes = egc_.row_bytes();
        int mods = 0;
        for (int il = 0; il < g_.n_layer; ++il)
            if (w_.pk.eg.module_of(il) >= 0) mods++;
        ti.engram_lookups = mods * w_.pk.eg.cols();
        return ti;
    }

    bool run(const int* ids, int T, float* logits_out, Trace* trace, std::string& err) override {
        trace_ = trace;
        if (pos_ + T > max_ctx_) {
            err = "hy4-metal: window past max_ctx";
            return false;
        }
        for (int t = 0; t < T; ++t) hist_.push_back(ids[t]);
        prof_.begin("window_total");
        if (T == 1 && !all_logits_) {
            // the decode window keeps its tensors, buffer and gallocr across tokens: only
            // the graphs rebuild (their shapes are the picks' and the position's). Every
            // window tensor is rewritten before it is read, so nothing stale carries over.
            if (!dwin_) dwin_ = std::make_unique<Win>();
            if (!window(*dwin_, ids, T, err)) return false;
            if (!head(*dwin_, T, 1, logits_out, err)) return false;
        } else {
            dwin_.reset();
            Win win;
            if (!window(win, ids, T, err)) return false;
            const int want = all_logits_ ? T : 1;
            if (!head(win, T, want, logits_out, err)) return false;
        }
        prof_.end("window_total");
        if (++prof_.windows % 100 == 0) prof_.dump();
        pos_ += T;
        return true;
    }

private:
    bool is_swa(int il) const { return g_.swa[(size_t) il] != 0; }
    bool is_ix(int il) const { return !is_swa(il) && g_.indexer[(size_t) il] != 0 && g_.ix_top_k > 0; }
    bool is_dense(int il) const { return il < g_.n_dense_lead; }

    void zero_caches() {
        for (ggml_tensor* t : kv_) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        for (ggml_tensor* t : ivg_)
            if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        for (ggml_tensor* t : pooled_)
            if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }

    bool load_ctx(int il, std::string& err) {
        ggml_context* ctx = ggml_init({256 * 1024 * 1024, nullptr, true});
        std::vector<std::pair<ggml_tensor*, const hy4::Tensor*>> fill;
        auto want = [&](const std::string& name, bool required) -> bool {
            const hy4::Tensor* pt = w_.tensor(name);
            if (!pt) {
                if (required) err = "hy4-metal: pack has no tensor " + name;
                return !required;
            }
            bool ok = false;
            const ggml_type gt = Weights::ggml_type_of(pt->type, ok);
            if (!ok) {
                err = "hy4-metal: tensor " + name + " has type " + pt->type + " (no ggml type)";
                return false;
            }
            ggml_tensor* t = ggml_new_tensor_3d(ctx, gt, std::max<int64_t>(pt->n(0), 1),
                                                std::max<int64_t>(pt->n(1), 1), std::max<int64_t>(pt->n(2), 1));
            wt_[name] = t;
            fill.push_back({t, pt});
            return true;
        };
        auto group = [&](const std::string& p, std::initializer_list<const char*> names) {
            for (const char* n : names)
                if (!want(p + n, true)) return false;
            return true;
        };
        if (il < 0) {
            if (!group("", {"token_embd.weight", "output.weight", "output_norm.weight", "output_hc_fn.weight",
                            "output_hc_base.weight", "output_hc_scale.weight"}))
                return false;
        } else {
            const std::string p = "blk." + std::to_string(il) + ".";
            if (!group(p, {"attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight",
                           "attn_v_b.weight", "attn_output.weight", "attn_gate.weight", "attn_norm.weight",
                           "ffn_norm.weight", "post_attention_norm.weight", "post_ffw_norm.weight",
                           "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_sinks.weight",
                           "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                           "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight"}))
                return false;
            if (is_ix(il) &&
                !group(p, {"indexer.attn_q_b.weight", "indexer.attn_k.weight", "indexer_compressor_value.weight",
                           "indexer_compressor_gate.weight", "indexer.proj.weight", "indexer.k_norm.weight",
                           "indexer_compressor_ape.weight", "indexer_compressor_norm.weight"}))
                return false;
            if (w_.pk.eg.module_of(il) >= 0 &&
                !group(p, {"engram_kv.weight", "engram_kv_b.weight", "engram_gate.weight",
                           "engram_lookup_norm.weight", "engram_pre_gamma.weight", "engram_hc_scale.weight"}))
                return false;
            if (is_dense(il)) {
                if (!group(p, {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})) return false;
            } else {
                if (!group(p, {"ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_routed_down.weight",
                               "ffn_routed_up.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                               "ffn_down_shexp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                               "ffn_down_exps.weight"}))
                    return false;
            }
        }
        if (!ggml_backend_alloc_ctx_tensors(ctx, be_)) {
            err = "hy4-metal: weight allocation failed";
            return false;
        }
        bufs_.push_back(fill.front().first->buffer);
        wctx_.push_back(ctx);
        for (auto& [t, pt] : fill) {
            ggml_backend_tensor_set(t, w_.data(*pt), 0, ggml_nbytes(t));
            resident_bytes_ += (double) pt->bytes;
            // (the bytes now live in the backend buffer; left mapped they count twice in RAM
            // and squeeze the Engram table's pages out under pressure - the table is the one
            // tenant this engine keeps reading from its file, a token at a time)
            mm3::os::drop_mapped(w_.data(*pt), pt->bytes);
        }
        return true;
    }

    ggml_tensor* W(const std::string& name) const { return wt_.at(name); }

    // ------------------------------------------------------------------ window plumbing
    // ggml_backend_alloc_ctx_tensors attaches ONE backend buffer per context that the
    // context does NOT own. It must be freed explicitly (recorded at allocation time -
    // after graph building, tensor->buffer also names gallocr/cache/weight buffers that
    // belong to someone else) or ggml-metal's device teardown asserts at exit.
    struct Win {
        ggml_context* ctx = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        std::map<std::string, ggml_tensor*> t;
        ~Win() {
            if (alloc) ggml_gallocr_free(alloc);
            if (buf) ggml_backend_buffer_free(buf);
            if (ctx) ggml_free(ctx);
        }
        ggml_tensor* get(const std::string& n) const { return t.at(n); }
    };

    ggml_tensor* mk(Win& win, const std::string& name, ggml_type type, int64_t n0, int64_t n1 = 1,
                    int64_t n2 = 1) {
        auto it = win.t.find(name);
        if (it != win.t.end()) {
            ggml_tensor* t = it->second;
            if (t->type == type && t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2) return t;
        }
        ggml_tensor* t = ggml_new_tensor_3d(win.ctx, type, n0, n1, n2);
        win.t[name] = t;
        return t;
    }
    ggml_tensor* konst(Win& win, const std::string& name, const float* vals, int64_t n) {
        // the section's scratch, not the window: a reused window tensor would keep the
        // data pointer a previous graph's gallocr assigned it, and the next graph would
        // allocate over that region (constants are graph inputs, fresh each section)
        ggml_tensor* t = ggml_new_tensor_1d(bc_ ? bc_ : win.ctx, GGML_TYPE_F32, n);
        pending_consts_.push_back({t, std::vector<float>(vals, vals + n)});
        return t;
    }
    void flush_consts() {
        for (auto& [t, v] : pending_consts_) ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
        pending_consts_.clear();
        for (auto& [t, v] : pending_iconsts_) ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
        pending_iconsts_.clear();
    }

    // Compute a graph in win.ctx with the given roots (each expanded in turn).  The graph is
    // allocated, the pending constants written into their fresh buffers, then computed.
    bool run_graph(Win& win, ggml_cgraph* gr, std::string& err) {
        if (verbose_) std::fprintf(stderr, "[hy4-metal] graph\n");
        prof_.begin("galloc");
        if (!win.alloc) win.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be_));
        if (!ggml_gallocr_alloc_graph(win.alloc, gr)) {
            err = "hy4-metal: graph allocation failed";
            return false;
        }
        prof_.end("galloc");
        flush_consts();
        prof_.begin("gcompute");
        if (ggml_backend_graph_compute(be_, gr) != GGML_STATUS_SUCCESS) {
            err = "hy4-metal: graph compute failed";
            return false;
        }
        prof_.end("gcompute");
        prof_.nodes += ggml_graph_n_nodes(gr);
        return true;
    }
    bool compute(Win& win, std::initializer_list<ggml_tensor*> roots, std::string& err) {
        ggml_cgraph* gr = ggml_new_graph_custom(bc_ ? bc_ : win.ctx, 65536, false);
        for (ggml_tensor* r : roots) ggml_build_forward_expand(gr, r);
        return run_graph(win, gr, err);
    }
    bool compute(Win& win, const std::vector<ggml_tensor*>& roots, std::string& err) {
        ggml_cgraph* gr = ggml_new_graph_custom(bc_ ? bc_ : win.ctx, 65536, false);
        for (ggml_tensor* r : roots) ggml_build_forward_expand(gr, r);
        return run_graph(win, gr, err);
    }
    void readback(ggml_tensor* t, float* dst) const {
        prof_.begin("readback");
        ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t));
        prof_.end("readback");
    }

    // HY4_PROFILE=1: where a window's milliseconds go, dumped to stderr every 100 windows.
    // Off by default; the only cost when off is one bool test per measured span.
    struct Prof {
        const char* env = ::getenv("HY4_PROFILE");
        bool on = env != nullptr;
        std::string path = (env && env[0] && env[0] != '1') ? env : "";
        std::map<std::string, std::pair<double, int64_t>> acc;
        std::map<std::string, std::chrono::steady_clock::time_point> starts_;
        int64_t windows = 0, nodes = 0;
        // one start per key: spans nest (window_total holds gcompute holds nothing, but
        // ix_host holds readback), and a single shared start let the inner span steal the
        // outer's beginning
        void begin(const char* k) { if (on) starts_[k] = std::chrono::steady_clock::now(); }
        void end(const char* k) {
            if (!on) return;
            auto it = starts_.find(k);
            if (it == starts_.end()) return;
            auto& e = acc[k];
            e.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - it->second).count();
            e.second += 1;
            starts_.erase(it);
        }
        void dump() {
            if (!on) return;
            std::FILE* f = path.empty() ? stderr : std::fopen(path.c_str(), "a");
            if (!f) f = stderr;
            std::fprintf(f, "[hy4-profile] windows=%lld nodes=%lld", (long long) windows, (long long) nodes);
            for (const auto& [k, v] : acc)
                std::fprintf(f, " %s=%.1f/%lld", k.c_str(), v.first, (long long) v.second);
            std::fprintf(f, "\n");
            if (f != stderr) std::fclose(f);
            acc.clear();
            windows = 0;
            nodes = 0;
        }
    };
    mutable Prof prof_;

    // Immutable weight rows the host math needs (hc base/scale, attention sinks, the router
    // bias): read once at first use, not a GPU sync per layer per token. They are the weights
    // themselves - caching them cannot change a result. Keyed by tensor pointer: the weight
    // buffers live for the engine's whole life.
    std::map<const ggml_tensor*, std::vector<float>> wc_;
    const float* wconst(ggml_tensor* t, size_t n) {
        auto it = wc_.find(t);
        if (it == wc_.end()) {
            std::vector<float> v(n);
            readback(t, v.data());
            it = wc_.emplace(t, std::move(v)).first;
        }
        return it->second.data();
    }
    void trace_put(const std::string& name, ggml_tensor* t) {
        if (!trace_) return;
        std::vector<float> v((size_t) ggml_nelements(t));
        readback(t, v.data());
        trace_->put(name, v);
    }

    // ------------------------------------------------------------------ op helpers
    // ggml's binary ops broadcast one side only; expand both operands to a common shape.
    ggml_tensor* expand_to(ggml_context* c, ggml_tensor* t, int64_t n0, int64_t n1, int64_t n2) {
        if (t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2) return t;
        return ggml_repeat(c, t, ggml_new_tensor_3d(c, t->type, n0, n1, n2));
    }

    ggml_tensor* rms_w(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
        ggml_tensor* r = ggml_rms_norm(c, x, eps);
        if (w) r = ggml_mul(c, r, ggml_reshape_3d(c, w, w->ne[0], 1, 1));
        return r;
    }

    // hc_pre: cur_out [E, T] (folded norm when norm_w), post_out [hc, T]. The roots form
    // appends to a caller's graph (the attention span fuses it with what follows); the
    // wrapper runs it alone (the ffn side).
    void hc_pre_roots(ggml_context* c, std::vector<ggml_tensor*>& roots, Win& win, ggml_tensor* x,
                      const std::string& pfx, ggml_tensor* norm_w, ggml_tensor* cur_out, ggml_tensor* post_out) {
        const int64_t E = g_.n_embd, hc = g_.hc, T = x->ne[2];
        ggml_tensor* fn = W(pfx + "_fn.weight");
        ggml_tensor* base = W(pfx + "_base.weight");
        ggml_tensor* scale = W(pfx + "_scale.weight");
        const float* hbase = wconst(base, (size_t) 2 * hc);
        const float* hscl = wconst(scale, 2);
        const float s0 = hscl[0], s1 = hscl[1];
        const float heps = (float) g_.hc_eps;
        ggml_tensor* flat = ggml_reshape_2d(c, x, hc * E, T);
        ggml_tensor* r = ggml_rms_norm(c, flat, (float) g_.eps);
        ggml_tensor* mixes = ggml_mul_mat(c, ggml_reshape_2d(c, fn, hc * E, 2 * hc), r);   // [2hc, T]
        auto mixout = [&](int64_t j0, float s, float mag) {
            ggml_tensor* m = ggml_view_2d(c, mixes, hc, T, mixes->nb[1], (size_t) j0 * mixes->nb[0]);
            ggml_tensor* b = konst(win, pfx + std::to_string(j0), hbase + j0, hc);
            ggml_tensor* v = ggml_add(c, ggml_scale(c, m, s), ggml_reshape_3d(c, b, hc, 1, 1));
            v = ggml_sigmoid(c, v);
            if (mag != 1.0f) v = ggml_scale(c, v, mag);
            return ggml_add(c, v, konst(win, "heps", &heps, 1));
        };
        ggml_tensor* pre2 = mixout(0, s0, 1.0f);   // [hc, T]
        ggml_tensor* pre = ggml_view_3d(c, pre2, 1, hc, T, pre2->nb[0], pre2->nb[1], 0);
        ggml_tensor* xp = ggml_mul(c, x, pre);                          // [E, hc, T]
        ggml_tensor* cur = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_cont(c, ggml_permute(c, xp, 1, 0, 2, 3))), E, T);
        if (norm_w) cur = rms_w(c, cur, norm_w, (float) g_.eps);
        roots.push_back(ggml_cpy(c, cur, cur_out));
        roots.push_back(ggml_cpy(c, mixout(hc, s1, (float) g_.hc_mag), post_out));
    }
    bool hc_pre(Win& win, ggml_tensor* x, const std::string& pfx, ggml_tensor* norm_w, ggml_tensor* cur_out,
                ggml_tensor* post_out, std::string& err) {
        ggml_context* c = build_begin();
        std::vector<ggml_tensor*> roots;
        hc_pre_roots(c, roots, win, x, pfx, norm_w, cur_out, post_out);
        const bool ok = compute(win, roots, err);
        build_end();
        return ok;
    }

    void hc_post_roots(ggml_context* c, std::vector<ggml_tensor*>& roots, ggml_tensor* x, ggml_tensor* y,
                       ggml_tensor* post, ggml_tensor* norm_w) {
        ggml_tensor* yn = rms_w(c, y, norm_w, (float) g_.eps);
        const int64_t E = yn->ne[0], hc = post->ne[0], T = yn->ne[1];
        ggml_tensor* d = ggml_mul(c, expand_to(c, ggml_reshape_3d(c, yn, E, 1, T), E, hc, T),
                                  expand_to(c, ggml_reshape_3d(c, post, 1, hc, T), E, hc, T));
        roots.push_back(ggml_cpy(c, ggml_add(c, x, d), x));
    }
    bool hc_post_norm(Win& win, ggml_tensor* x, ggml_tensor* y, ggml_tensor* post, ggml_tensor* norm_w,
                      std::string& err) {
        ggml_context* c = build_begin();
        std::vector<ggml_tensor*> roots;
        hc_post_roots(c, roots, x, y, post, norm_w);
        const bool ok = compute(win, roots, err);
        build_end();
        return ok;
    }

    ggml_tensor* rope_ext(ggml_context* c, ggml_tensor* a, ggml_tensor* pos, int n_dims, float base) {
        return ggml_rope_ext(c, a, pos, nullptr, n_dims, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 0.0f,
                             0.0f);
    }

    // ------------------------------------------------------------------ the window + head
    bool window(Win& win, const int* ids, int T, std::string& err);
    bool layer(Win& win, int il, int64_t p0, std::string& err);
    bool engram(Win& win, int il, int module, int64_t p0, std::string& err);
    bool attention(Win& win, int il, int64_t p0, std::string& err);
    bool ffn(Win& win, int il, std::string& err);
    bool head(Win& win, int T, int want, float* logits_out, std::string& err);

    Weights& w_;
    const hy4::Geo& g_;
    ggml_backend_t be_;
    int64_t max_ctx_;
    bool all_logits_;
    std::vector<ggml_context*> wctx_;
    std::vector<ggml_backend_buffer_t> bufs_;
    std::map<std::string, ggml_tensor*> wt_;
    ggml_context* cctx_ = nullptr;
    std::vector<ggml_tensor*> kv_, ivg_, pooled_;
    int64_t pos_ = 0;
    std::vector<int> hist_;
    Trace* trace_ = nullptr;
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> pending_consts_;
    std::vector<std::pair<ggml_tensor*, std::vector<int32_t>>> pending_iconsts_;
    // H2 (Hermes review 2026-11-19): graph building used to allocate every graph and
    // intermediate in the window context, which is freed only at window end - at real
    // geometry (45 layers, a graph per indexer group, G4's per-token expert chains)
    // that exhausts any fixed context mid-window.  Sections now build in a scratch
    // context that is freed as soon as the section's graphs have computed; sections
    // communicate only through the persistent window/cache tensors, never through
    // intermediates, so nothing outlives its section.
    ggml_context* bc_ = nullptr;
    ggml_context* build_begin() {
        build_end();
        bc_ = ggml_init({512 * 1024 * 1024, nullptr, true});
        return bc_;
    }
    void build_end() {
        if (bc_) {
            ggml_free(bc_);
            bc_ = nullptr;
        }
    }
    int T_ = 0;   // the live window's size (layer() reads it)
    bool verbose_ = false;
    std::vector<std::vector<int>> keys_;   // the live window's indexer picks, a token at a time
    EgCache egc_;                          // the Engram table's RAM tier (the Strata row-cache pattern)
    EgStats egst_;
    double resident_bytes_ = 0.0;          // weight bytes copied into Metal buffers: all resident
    std::unique_ptr<Win> dwin_;            // the reused decode window (T==1)
};

bool Ggml::window(Win& win, const int* ids, int T, std::string& err) {
    const hy4::Geo& g = g_;
    const int E = g.n_embd, hc = g.hc;
    const int64_t p0 = pos_;
    T_ = T;
    if (!win.ctx) {
    win.ctx = ggml_init({1024 * 1024 * 1024, nullptr, true});
    int Hmax = 0;
    for (int64_t h : g.heads) Hmax = std::max(Hmax, (int) h);
    const int FFmax = std::max(g.n_ff, g.n_ff_exp);
    const int egw = std::max(1, w_.pk.eg.cols()) * std::max(1, w_.pk.eg.head_dim);
    ggml_tensor* ids_t = mk(win, "ids", GGML_TYPE_I32, T);
    ggml_tensor* pos_t = mk(win, "pos", GGML_TYPE_I32, T);
    ggml_tensor* x = mk(win, "x", GGML_TYPE_F32, E, hc, T);
    mk(win, "xa", GGML_TYPE_F32, E, T);
    mk(win, "xf", GGML_TYPE_F32, E, T);
    mk(win, "post", GGML_TYPE_F32, hc, T);
    mk(win, "y", GGML_TYPE_F32, E, T);
    mk(win, "qr", GGML_TYPE_F32, g.q_lora, T);
    mk(win, "q", GGML_TYPE_F32, g.k_mla, Hmax, T);
    mk(win, "kvr", GGML_TYPE_F32, g.kv_row(), T);
    mk(win, "gg", GGML_TYPE_F32, g.v_mla, Hmax, T);
    mk(win, "qcat", GGML_TYPE_F32, g.kv_row(), Hmax, T);
    mk(win, "obpre", GGML_TYPE_F32, g.kv_lora, Hmax, T);
    mk(win, "ob", GGML_TYPE_F32, g.v_mla, Hmax, T);
    mk(win, "rl", GGML_TYPE_F32, g.n_expert, T);
    mk(win, "lat", GGML_TYPE_F32, g.n_lat, T);
    mk(win, "latout", GGML_TYPE_F32, g.n_lat, T);
    mk(win, "gu", GGML_TYPE_F32, FFmax, T);
    mk(win, "hd", GGML_TYPE_F32, FFmax, T);
    mk(win, "ik", GGML_TYPE_F32, g.ix_dim, T);
    mk(win, "ivt", GGML_TYPE_F32, 2 * g.ix_dim, T);
    mk(win, "iq", GGML_TYPE_F32, g.ix_dim, g.ix_heads, T);
    mk(win, "wt", GGML_TYPE_F32, g.ix_heads, T);
    mk(win, "egrows", GGML_TYPE_F32, std::max(1, w_.pk.eg.head_dim), std::max(1, w_.pk.eg.cols()), T);
    mk(win, "egvalid", GGML_TYPE_F32, T);
    mk(win, "kvo", GGML_TYPE_F32, (hc + 1) * E, T);
    mk(win, "hcur", GGML_TYPE_F32, E, T);
    mk(win, "logits", GGML_TYPE_F32, g.n_vocab, T);
    prof_.begin("win_alloc");
    if (!ggml_backend_alloc_ctx_tensors(win.ctx, be_)) {
        err = "hy4-metal: window allocation failed";
        return false;
    }
    prof_.end("win_alloc");
    win.buf = win.t.begin()->second->buffer;
    }
    ggml_tensor* ids_t = win.get("ids");
    ggml_tensor* pos_t = win.get("pos");
    ggml_tensor* x = win.get("x");
    {
        std::vector<int32_t> iv((size_t) T), pv((size_t) T);
        for (int t = 0; t < T; ++t) {
            iv[(size_t) t] = ids[t];
            pv[(size_t) t] = (int32_t) (p0 + t);
        }
        ggml_backend_tensor_set(ids_t, iv.data(), 0, (size_t) T * 4);
        ggml_backend_tensor_set(pos_t, pv.data(), 0, (size_t) T * 4);
    }
    {
        ggml_context* c = build_begin();
        ggml_tensor* emb = ggml_get_rows(c, ggml_reshape_2d(c, W("token_embd.weight"), E, g.n_vocab), ids_t);
        ggml_tensor* xr = ggml_repeat(c, ggml_reshape_3d(c, emb, E, 1, T), x);
        const bool ok = compute(win, {ggml_cpy(c, xr, x)}, err);
        build_end();
        if (!ok) return false;
    }
    for (int il = 0; il < g.n_layer; ++il)
        if (!layer(win, il, p0, err)) return false;
    return true;
}

bool Ggml::layer(Win& win, int il, int64_t p0, std::string& err) {
    if (verbose_) std::fprintf(stderr, "[hy4-metal] layer %d\n", il);
    const int egm = w_.pk.eg.module_of(il);
    if (egm >= 0 && !engram(win, il, egm, p0, err)) return false;
    if (!attention(win, il, p0, err)) return false;
    if (!ffn(win, il, err)) return false;
    return true;
}

bool Ggml::engram(Win& win, int il, int module, int64_t p0, std::string& err) {
    if (verbose_) std::fprintf(stderr, "[hy4-metal] engram %d\n", il);
    const hy4::Geo& g = g_;
    const int E = g.n_embd, hc = g.hc, T = T_;
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto egt0 = std::chrono::steady_clock::now();
    prof_.begin("engram_host");
    const EgRows rows = engram_gather(w_.pk, module, hist_, p0, T, &egc_, &egst_);
    const int cols = w_.pk.eg.cols(), hd = w_.pk.eg.head_dim;
    ggml_tensor* egrows = win.get("egrows");
    {
        // the gather's row (t, c, d) sits at (size_t)(t * cols + c) * hd + d in bytes; the
        // tensor's (d, c, t) sits at d + hd * (c + cols * t) - the same linear order
        std::vector<float> host((size_t) T * cols * hd);
        for (int t = 0; t < T; ++t)
            for (int cd = 0; cd < cols * hd; ++cd)
                host[(size_t) t * cols * hd + cd] = f8_e4m3_to_f32(rows.bytes[(size_t) t * cols * hd + cd]);
        ggml_backend_tensor_set(egrows, host.data(), 0, host.size() * 4);
    }
    prof_.end("engram_host");
    egst_.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - egt0).count();
    const float eeps = engram_eps(w_.pk);
    ggml_context* c = build_begin();
    ggml_tensor* valid = win.get("egvalid");
    ggml_backend_tensor_set(valid, rows.valid.data(), 0, (size_t) T * 4);
    {
        ggml_tensor* r = ggml_mul(
            c, ggml_rms_norm(c, egrows, eeps),
            ggml_reshape_3d(c, ggml_reshape_2d(c, W(p + "engram_lookup_norm.weight"), hd, cols), hd, cols, 1));
        r = ggml_mul(c, r, ggml_reshape_3d(c, valid, 1, 1, T));
        ggml_tensor* kvo = ggml_mul_mat(c, W(p + "engram_kv.weight"), ggml_reshape_2d(c, r, cols * hd, T));
        if (!compute(win, {ggml_cpy(c, kvo, win.get("kvo"))}, err)) return false;
    }
    {
        ggml_tensor* kvo = win.get("kvo");
        ggml_tensor* x = win.get("x");
        ggml_tensor* kv2 = ggml_add(c, kvo, ggml_reshape_3d(c, W(p + "engram_kv_b.weight"), (hc + 1) * E, 1, 1));
        ggml_tensor* keys = ggml_view_3d(c, kv2, E, hc, T, (size_t) E * 4, kv2->nb[1], 0);
        ggml_tensor* val = ggml_view_3d(c, kv2, E, 1, T, kv2->nb[1], kv2->nb[1], (size_t) hc * E * 4);
        ggml_tensor* xn = ggml_rms_norm(c, x, eeps);
        ggml_tensor* kn = ggml_rms_norm(c, keys, eeps);
        ggml_tensor* prod = ggml_mul(c, ggml_mul(c, xn, kn),
                                     ggml_reshape_3d(c, W(p + "engram_gate.weight"), E, hc, 1));
        ggml_tensor* dot = ggml_scale(c, ggml_sum_rows(c, prod),
                                      (float) (1.0 / std::sqrt((double) E)));   // [1, hc, T]
        ggml_tensor* gate = ggml_sigmoid(
            c, ggml_mul(c, ggml_sgn(c, dot),
                        ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, dot), 1e-6f, 3.4e38f))));
        ggml_tensor* vn = ggml_mul(c, ggml_rms_norm(c, val, eeps),
                                   ggml_reshape_3d(c, W(p + "engram_pre_gamma.weight"), E, 1, 1));
        ggml_tensor* gate_full = ggml_add(c, expand_to(c, gate, E, hc, T),
                                          expand_to(c, ggml_reshape_3d(c, W(p + "engram_hc_scale.weight"), E, hc, 1),
                                                    E, hc, T));
        ggml_tensor* delta = ggml_mul(c, gate_full, vn);
        delta = ggml_mul(c, delta, ggml_reshape_3d(c, valid, 1, 1, T));
        if (!compute(win, {ggml_cpy(c, ggml_add(c, x, delta), x)}, err)) return false;
    }
    trace_put("engram_out-" + std::to_string(il), win.get("x"));
    build_end();
    return true;
}

bool Ggml::attention(Win& win, int il, int64_t p0, std::string& err) {
    if (verbose_) std::fprintf(stderr, "[hy4-metal] attention %d\n", il);
    const hy4::Geo& g = g_;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::string ils = std::to_string(il);
    const int E = g.n_embd, H = (int) g.heads[(size_t) il], T = T_;
    const int QL = g.q_lora, KL = g.kv_lora, R = g.rope, KR = g.kv_row(), NOPE = g.nope(), KMLA = g.k_mla,
              VM = g.v_mla;
    const bool ix = is_ix(il);
    const float base = is_swa(il) ? (float) g.rope_base_swa : (float) g.rope_base;
    ggml_tensor* x = win.get("x"), *xa = win.get("xa"), *post = win.get("post"), *y = win.get("y");
    ggml_tensor* pos_t = win.get("pos");
    ggml_tensor* qr = win.get("qr"), *q = win.get("q"), *kvr = win.get("kvr"), *ggt = win.get("gg");
    // G2+G3: qcat, the scores over JUST the visible rows, the latent outputs, the gate,
    // the output projection. A SWA layer's visible set is one contiguous window (a view,
    // no mask); an indexer layer's is its picks (a gather, deduplicated exactly as the
    // mask's set was). The sink row still appends. Appends to the caller's graph.
    auto g2g3 = [&](ggml_context* c, std::vector<ggml_tensor*>& roots) {
        const int64_t P = p0 + T;
        const float* sinks = wconst(W(p + "attn_sinks.weight"), (size_t) H);
        const float ascale = (float) (1.0 / std::sqrt((double) KMLA));
        ggml_tensor* qcat = win.get("qcat");
        ggml_tensor* pe = ggml_view_3d(c, q, R, H, T, q->nb[1], q->nb[2], (size_t) NOPE * 4);
        ggml_tensor* per = rope_ext(c, pe, pos_t, R, base);
        roots.push_back(ggml_cpy(c, per, ggml_view_3d(c, qcat, R, H, T, qcat->nb[1], qcat->nb[2], (size_t) KL * 4)));
        ggml_tensor* K16 = ggml_view_2d(c, kv_[(size_t) il], KR, P, kv_[(size_t) il]->nb[1], 0);
        ggml_tensor* V16 = ggml_view_2d(c, kv_[(size_t) il], KL, P, kv_[(size_t) il]->nb[1], 0);
        ggml_tensor* zeros = konst(win, "zeros" + std::to_string(KL), std::vector<float>((size_t) KL, 0.0f).data(), KL);
        ggml_tensor* sinkrow = konst(win, "sinks" + ils, sinks, H);
        {
            // the q absorption for all heads in one batched matmul
            ggml_tensor* kb = W(p + "attn_k_b.weight");   // [NOPE, KL, H]
            ggml_tensor* qn = ggml_permute(c, ggml_view_3d(c, q, NOPE, H, T, q->nb[1], q->nb[2], 0), 0, 2, 1, 3);
            ggml_tensor* qabs = ggml_mul_mat(c, kb, ggml_cont(c, qn));   // [KL, T, H]
            ggml_tensor* dst = ggml_view_3d(c, qcat, KL, H, T, qcat->nb[1], qcat->nb[2], 0);
            roots.push_back(ggml_cpy(c, ggml_permute(c, qabs, 0, 2, 1, 3), dst));
        }
        ggml_tensor* obpre = win.get("obpre");
        for (int t = 0; t < T; ++t) {
            const int64_t pos = p0 + t;
            ggml_tensor *K, *Vf;
            int64_t n;
            if (ix) {
                std::vector<int> ks = keys_[(size_t) t];
                std::sort(ks.begin(), ks.end());
                ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
                n = (int64_t) ks.size();
                ggml_tensor* kid = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
                pending_iconsts_.push_back({kid, std::vector<int32_t>(ks.begin(), ks.end())});
                K = ggml_cpy(c, ggml_get_rows(c, K16, kid), ggml_new_tensor_2d(c, GGML_TYPE_F32, KR, n));
                Vf = ggml_cpy(c, ggml_get_rows(c, V16, kid), ggml_new_tensor_2d(c, GGML_TYPE_F32, KL, n));
            } else {
                const int64_t lo = std::max<int64_t>(0, pos - g.n_swa + 1);
                n = pos - lo + 1;
                ggml_tensor* Kw16 = ggml_view_2d(c, kv_[(size_t) il], KR, n, kv_[(size_t) il]->nb[1],
                                                 (size_t) lo * kv_[(size_t) il]->nb[1]);
                ggml_tensor* Vw16 = ggml_view_2d(c, kv_[(size_t) il], KL, n, kv_[(size_t) il]->nb[1],
                                                 (size_t) lo * kv_[(size_t) il]->nb[1]);
                K = ggml_cpy(c, Kw16, ggml_new_tensor_2d(c, GGML_TYPE_F32, KR, n));
                Vf = ggml_cpy(c, Vw16, ggml_new_tensor_2d(c, GGML_TYPE_F32, KL, n));
            }
            ggml_tensor* Vz = ggml_concat(c, Vf, ggml_reshape_2d(c, zeros, KL, 1), 1);
            ggml_tensor* qt = ggml_view_3d(c, qcat, KR, H, 1, qcat->nb[1], qcat->nb[2],
                                           (size_t) t * qcat->nb[2]);
            ggml_tensor* sc = ggml_scale(c, ggml_mul_mat(c, K, qt), ascale);
            ggml_tensor* S2 = ggml_concat(c, sc, ggml_reshape_3d(c, sinkrow, 1, H, 1), 0);
            ggml_tensor* probs = ggml_soft_max(c, S2);
            ggml_tensor* o = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, Vz)), probs);
            roots.push_back(ggml_cpy(
                c, o, ggml_view_3d(c, obpre, KL, H, 1, obpre->nb[1], obpre->nb[2], (size_t) t * obpre->nb[2])));
        }
        ggml_tensor* ob = win.get("ob");
        {
            // v_b for all heads in one batched matmul
            ggml_tensor* vb = W(p + "attn_v_b.weight");   // [KL, VM, H]
            ggml_tensor* obp = ggml_view_3d(c, obpre, KL, H, T, obpre->nb[1], obpre->nb[2], 0);
            ggml_tensor* oh = ggml_permute(c, obp, 0, 2, 1, 3);   // [KL, T, H]
            ggml_tensor* vo = ggml_mul_mat(c, vb, ggml_cont(c, oh));   // [VM, T, H]
            ggml_tensor* odst = ggml_view_3d(c, ob, VM, H, T, ob->nb[1], ob->nb[2], 0);
            roots.push_back(ggml_cpy(c, ggml_permute(c, vo, 0, 2, 1, 3), odst));
        }
        ggml_tensor* gated = ggml_mul(c, ggml_cont(c, ggml_view_3d(c, ob, VM, H, T, ob->nb[1], ob->nb[2], 0)),
                                      ggml_sigmoid(c, ggml_view_3d(c, ggt, VM, H, T, ggt->nb[1], ggt->nb[2], 0)));
        ggml_tensor* gatedc = ggml_cpy(c, gated, ggml_new_tensor_3d(c, GGML_TYPE_F32, VM, H, T));
        ggml_tensor* yv = ggml_mul_mat(c, W(p + "attn_output.weight"),
                                       ggml_reshape_2d(c, gatedc, (int64_t) H * VM, T));
        roots.push_back(ggml_cpy(c, yv, y));
    };
    // ---- span 1: the hc fold, the projections, the cache store, the indexer's prep -
    // and for a SWA layer the whole attention besides (nothing needs the host until the
    // router): one graph commit instead of four a layer a token.
    {
        ggml_context* c = build_begin();
        std::vector<ggml_tensor*> roots;
        hc_pre_roots(c, roots, win, x, p + "hc_attn", W(p + "attn_norm.weight"), xa, post);
        ggml_tensor* qa = ggml_mul_mat(c, W(p + "attn_q_a.weight"), xa);
        ggml_tensor* qrn = rms_w(c, qa, W(p + "attn_q_a_norm.weight"), (float) g.eps);
        ggml_tensor* qfull = ggml_mul_mat(c, W(p + "attn_q_b.weight"), qrn);
        ggml_tensor* kvfull = ggml_mul_mat(c, W(p + "attn_kv_a_mqa.weight"), xa);
        ggml_tensor* gfull = ggml_mul_mat(c, W(p + "attn_gate.weight"), xa);
        roots.push_back(ggml_cpy(c, qrn, qr));
        roots.push_back(ggml_cpy(c, ggml_reshape_3d(c, qfull, KMLA, H, T),
                                 ggml_view_3d(c, q, KMLA, H, T, q->nb[1], q->nb[2], 0)));
        roots.push_back(ggml_cpy(c, kvfull, kvr));
        roots.push_back(ggml_cpy(c, ggml_reshape_3d(c, gfull, VM, H, T),
                                 ggml_view_3d(c, ggt, VM, H, T, ggt->nb[1], ggt->nb[2], 0)));
        // the cache row: [rmsnorm(latent) | rope(rope part)] as F16
        ggml_tensor* lat = ggml_view_2d(c, kvr, KL, T, kvr->nb[1], 0);
        ggml_tensor* latn = rms_w(c, lat, W(p + "attn_kv_a_norm.weight"), (float) g.eps);
        ggml_tensor* rpart = ggml_view_3d(c, kvr, R, 1, T, kvr->nb[1], kvr->nb[1], (size_t) KL * 4);
        ggml_tensor* roped = rope_ext(c, rpart, pos_t, R, base);
        ggml_tensor* row = ggml_concat(c, ggml_reshape_3d(c, latn, KL, 1, T), roped, 0);
        ggml_tensor* dst = ggml_view_2d(c, kv_[(size_t) il], KR, T, kv_[(size_t) il]->nb[1],
                                        (size_t) p0 * kv_[(size_t) il]->nb[1]);
        roots.push_back(ggml_cpy(c, ggml_reshape_2d(c, row, KR, T), dst));
        if (ix) {
            ggml_tensor* ikn = rms_w(c, ggml_mul_mat(c, W(p + "indexer.attn_k.weight"), xa),
                                     W(p + "indexer.k_norm.weight"), (float) g.eps);
            roots.push_back(ggml_cpy(c, ikn, win.get("ik")));
            ggml_tensor* ivt = ggml_concat(c, ggml_mul_mat(c, W(p + "indexer_compressor_value.weight"), ikn),
                                           ggml_mul_mat(c, W(p + "indexer_compressor_gate.weight"), ikn), 0);
            roots.push_back(ggml_cpy(c, ivt, win.get("ivt")));
            ggml_tensor* idst = ggml_view_2d(c, ivg_[(size_t) il], 2 * g.ix_dim, T, ivg_[(size_t) il]->nb[1],
                                             (size_t) p0 * ivg_[(size_t) il]->nb[1]);
            roots.push_back(ggml_cpy(c, ivt, idst));
            ggml_tensor* iq3 = ggml_reshape_3d(c, ggml_mul_mat(c, W(p + "indexer.attn_q_b.weight"), qrn),
                                               g.ix_dim, g.ix_heads, T);
            ggml_tensor* tail = ggml_view_3d(c, iq3, R, g.ix_heads, T, iq3->nb[1], iq3->nb[2],
                                             (size_t) (g.ix_dim - R) * 4);
            ggml_tensor* headp = ggml_view_3d(c, iq3, g.ix_dim - R, g.ix_heads, T, iq3->nb[1], iq3->nb[2], 0);
            ggml_tensor* iqf = ggml_concat(c, headp, rope_ext(c, tail, pos_t, R, base), 0);
            roots.push_back(ggml_cpy(c, iqf, win.get("iq")));
            ggml_tensor* wt = ggml_scale(c, ggml_mul_mat(c, W(p + "indexer.proj.weight"), xa),
                                         (float) (1.0 / std::sqrt((double) g.ix_dim * g.ix_heads)));
            roots.push_back(ggml_cpy(c, wt, win.get("wt")));
        }
        if (!ix) {
            g2g3(c, roots);
            hc_post_roots(c, roots, x, y, post, W(p + "post_attention_norm.weight"));
        }
        ggml_cgraph* gr = ggml_new_graph_custom(c, 65536, false);
        for (ggml_tensor* r : roots) ggml_build_forward_expand(gr, r);
        if (!run_graph(win, gr, err)) {
            err = "hy4-metal: attention span 1 failed";
            return false;
        }
        build_end();
    }
    trace_put("attn_norm-" + ils, xa);
    trace_put("q_resid-" + ils, qr);
    // ---- the indexer's pools, then the host-side picks, then the attention span
    if (ix) {
        const int D = g.ix_dim, KP = g.ix_kpool;
        const int64_t g0 = p0 / KP, g1 = (p0 + T) / KP;
        for (int64_t gr = g0; gr < g1; ++gr) {
            ggml_context* c = build_begin();
            ggml_tensor* src = ggml_view_2d(c, ivg_[(size_t) il], 2 * D, KP, ivg_[(size_t) il]->nb[1],
                                            (size_t) (gr * KP) * ivg_[(size_t) il]->nb[1]);
            ggml_tensor* valf = ggml_cpy(c, ggml_view_2d(c, src, D, KP, src->nb[1], 0),
                                         ggml_new_tensor_2d(c, GGML_TYPE_F32, D, KP));
            ggml_tensor* gatef = ggml_cpy(c, ggml_view_2d(c, src, D, KP, src->nb[1], (size_t) D * 2),
                                          ggml_new_tensor_2d(c, GGML_TYPE_F32, D, KP));
            ggml_tensor* ape = ggml_reshape_2d(c, W(p + "indexer_compressor_ape.weight"), D, KP);
            ggml_tensor* lg = ggml_add(c, gatef, ape);
            ggml_tensor* wgt = ggml_permute(c, ggml_soft_max(c, ggml_cont(c, ggml_permute(c, lg, 1, 0, 2, 3))), 1, 0, 2, 3);
            ggml_tensor* praw = ggml_reshape_1d(
                c, ggml_sum_rows(c, ggml_cont(c, ggml_permute(c, ggml_mul(c, ggml_cont(c, wgt), valf), 1, 0, 2, 3))), D);
            ggml_tensor* pnorm = rms_w(c, ggml_reshape_2d(c, praw, D, 1), W(p + "indexer_compressor_norm.weight"),
                                       (float) g.eps);
            ggml_tensor* tail = ggml_view_3d(c, pnorm, R, 1, 1, pnorm->nb[1], pnorm->nb[1],
                                             (size_t) (D - R) * 4);
            const int32_t gpos = (int32_t) (gr * KP);
            ggml_tensor* post1 = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
            pending_iconsts_.push_back({post1, {gpos}});
            ggml_tensor* headp = ggml_view_2d(c, pnorm, D - R, 1, pnorm->nb[1], 0);
            ggml_tensor* full = ggml_concat(c, headp, ggml_reshape_2d(c, rope_ext(c, tail, post1, R, base), R, 1), 0);
            ggml_tensor* dst = ggml_view_2d(c, pooled_[(size_t) il], D, 1, pooled_[(size_t) il]->nb[1],
                                            (size_t) gr * pooled_[(size_t) il]->nb[1]);
            const bool ok = compute(win, {ggml_cpy(c, full, dst)}, err);
            build_end();
            if (!ok) return false;
        }
        // the picks: only the live prefix of the pool is read back, and the groups score
        // in parallel - the same double arithmetic per group (the picks must not move).
        prof_.begin("ix_host");
        const int IH = g.ix_heads;
        std::vector<float> iqv((size_t) T * IH * D), wtv((size_t) T * IH);
        readback(win.get("iq"), iqv.data());
        readback(win.get("wt"), wtv.data());
        const int64_t nv_max = (p0 + T - 1) / KP;
        std::vector<float> poolv((size_t) nv_max * D);
        if (nv_max > 0) {
            prof_.begin("readback");
            ggml_backend_tensor_get(pooled_[(size_t) il], poolv.data(), 0, (size_t) nv_max * D * 4);
            prof_.end("readback");
        }
        keys_.assign((size_t) T, {});
        const int top = g.ix_top_k / KP - 1;
        for (int t = 0; t < T; ++t) {
            const int64_t pos = p0 + t, nv = pos / KP;
            const float* iq_t = &iqv[(size_t) t * IH * D];
            const float* wt_t = &wtv[(size_t) t * IH];
            std::vector<std::pair<float, int64_t>> scored((size_t) nv);
            auto score_gr = [&](int64_t gr) {
                double s = 0;
                for (int h = 0; h < IH; ++h) {
                    double d = 0;
                    for (int i = 0; i < D; ++i)
                        d += (double) iq_t[(size_t) h * D + i] * poolv[(size_t) gr * D + i];
                    s += std::max(d, 0.0) * wt_t[h];
                }
                scored[(size_t) gr] = {(float) s, gr};
            };
            if (nv >= 64) {
                dispatch_apply((size_t) nv, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                               ^(size_t gr) { score_gr((int64_t) gr); });
            } else {
                for (int64_t gr = 0; gr < nv; ++gr) score_gr(gr);
            }
            std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            std::vector<int64_t> picks;
            for (int64_t i = 0; i < std::min<int64_t>(nv, top); ++i) picks.push_back(scored[(size_t) i].second);
            std::sort(picks.begin(), picks.end());
            for (int64_t gr : picks)
                for (int k = 0; k < KP; ++k) keys_[(size_t) t].push_back((int) (gr * KP + k));
            for (int64_t pp = (pos / KP) * KP; pp <= pos; ++pp) keys_[(size_t) t].push_back((int) pp);
        }
        prof_.end("ix_host");
        if (trace_) {
            std::vector<float> kk((size_t) T * (g.ix_top_k + KP), -1.0f);
            for (int t = 0; t < T; ++t)
                for (size_t i = 0; i < keys_[(size_t) t].size(); ++i)
                    kk[(size_t) t * (g.ix_top_k + KP) + i] = (float) keys_[(size_t) t][i];
            trace_->put("ix_keys-" + ils, kk);
        }
        ggml_context* c = build_begin();
        std::vector<ggml_tensor*> roots;
        g2g3(c, roots);
        hc_post_roots(c, roots, x, y, post, W(p + "post_attention_norm.weight"));
        ggml_cgraph* gr = ggml_new_graph_custom(c, 65536, false);
        for (ggml_tensor* r : roots) ggml_build_forward_expand(gr, r);
        if (!run_graph(win, gr, err)) {
            err = "hy4-metal: attention span 2 failed";
            return false;
        }
        build_end();
    }
    trace_put("attn_y-" + ils, y);
    trace_put("hc_attn_out-" + ils, x);
    return true;
}
bool Ggml::ffn(Win& win, int il, std::string& err) {
    if (verbose_) std::fprintf(stderr, "[hy4-metal] ffn %d\n", il);
    const hy4::Geo& g = g_;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::string ils = std::to_string(il);
    const int E = g.n_embd, T = T_;
    ggml_context* c = win.ctx;
    ggml_tensor* x = win.get("x"), *xf = win.get("xf"), *post = win.get("post"), *y = win.get("y");
    if (is_dense(il)) {
        // a dense layer needs nothing from the host: the whole ffn is one graph commit
        c = build_begin();
        std::vector<ggml_tensor*> roots;
        hc_pre_roots(c, roots, win, x, p + "hc_ffn", W(p + "ffn_norm.weight"), xf, post);
        ggml_tensor* gu = ggml_mul_mat(c, W(p + "ffn_gate.weight"), xf);
        ggml_tensor* uu = ggml_mul_mat(c, W(p + "ffn_up.weight"), xf);
        ggml_tensor* hd = ggml_mul(c, ggml_silu(c, gu), uu);
        ggml_tensor* yv = ggml_mul_mat(c, W(p + "ffn_down.weight"), hd);
        ggml_tensor* hdv = ggml_view_2d(c, win.get("hd"), g.n_ff, T, win.get("hd")->nb[1], 0);
        roots.push_back(ggml_cpy(c, hd, hdv));
        roots.push_back(ggml_cpy(c, yv, y));
        hc_post_roots(c, roots, x, y, post, W(p + "post_ffw_norm.weight"));
        ggml_cgraph* gr = ggml_new_graph_custom(c, 65536, false);
        for (ggml_tensor* r : roots) ggml_build_forward_expand(gr, r);
        if (!run_graph(win, gr, err)) {
            err = "hy4-metal: dense ffn failed";
            return false;
        }
        build_end();
        trace_put("ffn_y-" + ils, y);
        trace_put("l_out-" + ils, x);
        return true;
    }
    const int LAT = g.n_lat, FFE = g.n_ff_exp, NU = g.n_used, NE = g.n_expert;
    // The whole MoE ffn in ONE graph: the hc fold, the router, the picks IN-GRAPH
    // (sigmoid + correction bias + argsort_top_k, the mm3/Qwen pattern - no host round-trip,
    // so nothing forces a commit between the router and the experts), the grouped experts,
    // the shared expert and the hc residual. The picks' tie order is the sort kernel's;
    // the e2e token parity and the real-prompt coherence gate verify it against the old
    // host sigmoid+bias+sort.
    c = build_begin();
    const float limit = (float) g.limit[(size_t) il];
    ggml_tensor* latout = win.get("latout");
    std::vector<ggml_tensor*> roots;
    hc_pre_roots(c, roots, win, x, p + "hc_ffn", W(p + "ffn_norm.weight"), xf, post);
    ggml_tensor* rl = ggml_mul_mat(c, W(p + "ffn_gate_inp.weight"), xf);      // [NE, T]
    ggml_tensor* lat = ggml_mul_mat(c, W(p + "ffn_routed_down.weight"), xf);  // [LAT, T]
    ggml_tensor* sh = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, W(p + "ffn_gate_shexp.weight"), xf)),
                               ggml_mul_mat(c, W(p + "ffn_up_shexp.weight"), xf));
    ggml_tensor* hdv = ggml_view_2d(c, win.get("hd"), FFE, T, win.get("hd")->nb[1], 0);
    roots.push_back(ggml_cpy(c, rl, win.get("rl")));
    roots.push_back(ggml_cpy(c, sh, hdv));
    // the picks and their weights, in-graph
    ggml_tensor* sig = ggml_sigmoid(c, rl);                                   // [NE, T]
    ggml_tensor* biased = ggml_add(c, sig, ggml_reshape_3d(c, W(p + "exp_probs_b.bias"), NE, 1, 1));
    // argsort_top_k is a strided VIEW of the sorted rows (stride NE between tokens), so it
    // must be made contiguous before the ids layout (ids[j + NU*t]) and the gather read it
    ggml_tensor* top = ggml_cont(c, ggml_argsort_top_k(c, biased, NU));        // [NU, T] i32, desc
    ggml_tensor* rw_raw = ggml_get_rows(c, ggml_reshape_3d(c, sig, 1, NE, T), top);   // [1, NU, T]
    ggml_tensor* rw2 = ggml_reshape_2d(c, rw_raw, NU, T);
    ggml_tensor* rw = ggml_reshape_3d(c, rw_raw, NU, 1, T);
    if (g.w_norm) {
        ggml_tensor* wsum = ggml_clamp(c, ggml_sum_rows(c, rw2), 6.103515625e-5f, 3.4e38f);   // [1, T], the 2^-14 floor
        rw = ggml_div(c, rw, ggml_reshape_3d(c, wsum, 1, 1, T));
    }
    rw = ggml_scale(c, rw, (float) g.w_scale);
    // the routed experts (grouped matmuls), the per-token accumulation in the old order
    ggml_tensor* ge = W(p + "ffn_gate_exps.weight"), *ue = W(p + "ffn_up_exps.weight"),
                *de = W(p + "ffn_down_exps.weight");
    ggml_tensor* lat3 = ggml_reshape_3d(c, lat, LAT, 1, T);
    ggml_tensor* gv = ggml_mul_mat_id(c, ge, lat3, top);   // [FFE, NU, T]
    ggml_tensor* uv = ggml_mul_mat_id(c, ue, lat3, top);
    if (limit > 0) {
        gv = ggml_clamp(c, gv, -3.4e38f, limit);
        uv = ggml_clamp(c, uv, -limit, limit);
    }
    ggml_tensor* h = ggml_mul(c, ggml_silu(c, gv), uv);
    ggml_tensor* eo = ggml_mul_mat_id(c, de, h, top);      // [LAT, NU, T]
    eo = ggml_mul(c, eo, ggml_reshape_3d(c, rw, 1, NU, T));
    for (int t = 0; t < T; ++t) {
        ggml_tensor* acc = nullptr;
        for (int j = 0; j < NU; ++j) {
            ggml_tensor* eo_j = ggml_view_2d(c, eo, LAT, 1, eo->nb[1],
                                             ((size_t) j + (size_t) NU * t) * LAT * 4);
            acc = acc ? ggml_add(c, acc, eo_j) : eo_j;
        }
        roots.push_back(ggml_cpy(
            c, acc, ggml_view_2d(c, latout, LAT, 1, latout->nb[1], (size_t) t * LAT * 4)));
    }
    ggml_tensor* yv = ggml_add(c, ggml_mul_mat(c, W(p + "ffn_routed_up.weight"), latout),
                               ggml_mul_mat(c, W(p + "ffn_down_shexp.weight"), hdv));
    roots.push_back(ggml_cpy(c, yv, y));
    hc_post_roots(c, roots, x, y, post, W(p + "post_ffw_norm.weight"));
    ggml_cgraph* gr = ggml_new_graph_custom(c, 65536 + (size_t) T * NU * 16, false);
    for (ggml_tensor* r : roots) ggml_build_forward_expand(gr, r);
    if (!run_graph(win, gr, err)) {
        err = "hy4-metal: ffn failed";
        return false;
    }
    build_end();
    if (trace_) {
        // the picks/weights/scores now live on the GPU; read them back for the trace only
        std::vector<int32_t> topv((size_t) NU * T);
        std::vector<float> rwv((size_t) NU * T), rlv((size_t) NE * T);
        ggml_backend_tensor_get(top, topv.data(), 0, topv.size() * 4);
        ggml_backend_tensor_get(rw, rwv.data(), 0, rwv.size() * 4);
        ggml_backend_tensor_get(win.get("rl"), rlv.data(), 0, rlv.size() * 4);
        const float* biasv = wconst(W(p + "exp_probs_b.bias"), (size_t) NE);
        std::vector<float> tr_ids, tr_rw, tr_sc;
        for (int t = 0; t < T; ++t) {
            auto sigf = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
            for (int e = 0; e < NE; ++e) tr_sc.push_back(sigf(rlv[(size_t) t * NE + e]) + biasv[e]);
            for (int j = 0; j < NU; ++j) {
                tr_ids.push_back((float) topv[(size_t) j + (size_t) NU * t]);
                tr_rw.push_back(rwv[(size_t) j + (size_t) NU * t]);
            }
        }
        trace_->put("router_ids-" + ils, tr_ids);
        trace_->put("router_rw-" + ils, tr_rw);
        trace_->put("router_sc-" + ils, tr_sc);
    }
    trace_put("ffn_latout-" + ils, win.get("latout"));
    trace_put("ffn_y-" + ils, y);
    trace_put("l_out-" + ils, x);
    return true;
}

bool Ggml::head(Win& win, int T, int want, float* logits_out, std::string& err) {
    if (verbose_) std::fprintf(stderr, "[hy4-metal] head\n");
    const hy4::Geo& g = g_;
    const int E = g.n_embd, hc = g.hc;
    ggml_context* c = build_begin();
    ggml_tensor* x = win.get("x");
    ggml_tensor* fn = W("output_hc_fn.weight");
    const float* hbase = wconst(W("output_hc_base.weight"), (size_t) hc);
    const float s0 = *wconst(W("output_hc_scale.weight"), 1), heps = (float) g.hc_eps;
    ggml_tensor* flat = ggml_reshape_2d(c, x, hc * E, T);
    ggml_tensor* r = ggml_rms_norm(c, flat, (float) g.eps);
    ggml_tensor* mixes = ggml_mul_mat(c, ggml_reshape_2d(c, fn, hc * E, hc), r);   // [hc, T]
    ggml_tensor* b = konst(win, "hbase", hbase, hc);
    ggml_tensor* pre = ggml_add(c, ggml_sigmoid(c, ggml_add(c, ggml_scale(c, mixes, s0),
                                                            ggml_reshape_3d(c, b, hc, 1, 1))),
                                konst(win, "heps", &heps, 1));
    ggml_tensor* prev = ggml_view_3d(c, pre, 1, hc, T, pre->nb[0], pre->nb[1], 0);
    ggml_tensor* cur = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_cont(c, ggml_permute(c, ggml_mul(c, x, prev), 1, 0, 2, 3))), E, T);
    ggml_tensor* hn = rms_w(c, cur, W("output_norm.weight"), (float) g.eps);
    ggml_tensor* hnw = ggml_view_2d(c, hn, E, want, hn->nb[1], (size_t) (T - want) * hn->nb[1]);
    ggml_tensor* lg = ggml_mul_mat(c, ggml_reshape_2d(c, W("output.weight"), E, g.n_vocab), hnw);
    ggml_tensor* lgdst = ggml_view_2d(c, win.get("logits"), g.n_vocab, want, win.get("logits")->nb[1],
                                      (size_t) (T - want) * win.get("logits")->nb[1]);
    const bool ok = compute(win, {ggml_cpy(c, hn, win.get("hcur")), ggml_cpy(c, lg, lgdst)}, err);
    build_end();
    if (!ok) return false;
    if (trace_) {
        trace_put("hcur", win.get("hcur"));
        if (all_logits_) trace_put("logits", win.get("logits"));
    }
    // the win logits are [n_vocab, T] feature-major; the wanted rows are the last `want` columns
    std::vector<float> lv((size_t) want * g.n_vocab);
    ggml_backend_tensor_get(win.get("logits"), lv.data(), (size_t) (T - want) * g.n_vocab * 4,
                            (size_t) want * g.n_vocab * 4);
    for (int t = 0; t < want; ++t)
        std::memcpy(logits_out + (size_t) t * g.n_vocab, &lv[(size_t) t * g.n_vocab],
                    (size_t) g.n_vocab * sizeof(float));
    return true;
}

}  // namespace

std::unique_ptr<Forward> make_ggml_forward(Weights& w, ggml_backend_t be, int64_t max_ctx, bool want_all_logits,
                                           double eg_cache_gb) {
    auto f = std::make_unique<Ggml>(w, be, max_ctx, want_all_logits, eg_cache_gb);
    std::string err;
    if (!f->init(err)) return nullptr;
    return f;
}

}  // namespace hym

