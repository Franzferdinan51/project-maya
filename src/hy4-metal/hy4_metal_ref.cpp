// src/hy4-metal/hy4_metal_ref.cpp - the scalar CPU golden for the Hy4-Flash trunk forward.
//
// Written directly from the upstream kernel semantics (src/hy4/hy4_kernels.cuh's contracts and
// hy4_kernels.cu's bodies, hy4_model.cu's block recipe).  Everything is F32 except the latent
// KV cache and the indexer's value|gate ring, which are F16 rows exactly as upstream stores
// them.  This is the reference the ggml/Metal forward is gated against (stage 3); it is a
// second implementation of the spec, not a port of the ggml graph code.
#include "hy4_metal_model.hpp"

#include <algorithm>
#include <limits>

namespace hym {
namespace {

struct W {
    const float* p = nullptr;
    int64_t n0 = 1, n1 = 1, n2 = 1;
    float at(int64_t i0, int64_t i1 = 0, int64_t i2 = 0) const { return p[i0 + n0 * (i1 + n1 * i2)]; }
};

// y[n] = W x, W [k][n] (ne0 = k contiguous)
void matvec(const W& w, const float* x, float* y) {
    for (int64_t j = 0; j < w.n1; ++j) {
        double s = 0;
        const float* col = w.p + j * w.n0;
        for (int64_t i = 0; i < w.n0; ++i) s += (double) col[i] * x[i];
        y[j] = (float) s;
    }
}
// y[n] = W_b x for batch b of W [k][n][b]
void matvec_b(const W& w, int64_t b, const float* x, float* y) { matvec(W{w.p + b * w.n0 * w.n1, w.n0, w.n1}, x, y); }

void rmsnorm(const float* x, int n, const float* w, float eps, float* y) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = (float) (1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * (w ? w[i] : 1.0f);
}

// rotate the consecutive pairs of x[off, off + n_rot) at position pos (ggml NORM / upstream rope)
void rope(float* x, int off, int n_rot, int64_t pos, float base) {
    for (int i = 0; i < n_rot / 2; ++i) {
        const double th = (double) pos * std::pow((double) base, -2.0 * i / n_rot);
        const float c = (float) std::cos(th), s = (float) std::sin(th);
        const float a = x[off + 2 * i], b = x[off + 2 * i + 1];
        x[off + 2 * i] = a * c - b * s;
        x[off + 2 * i + 1] = a * s + b * c;
    }
}

float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

class Ref : public Forward {
public:
    Ref(Weights& w, int64_t max_ctx) : w_(w), g_(*w.g), max_ctx_(max_ctx) {
        const int L = g_.n_layer;
        kv_.assign((size_t) L, {});
        ivg_.assign((size_t) L, {});
        pooled_.assign((size_t) L, {});
        for (int il = 0; il < L; ++il) {
            kv_[(size_t) il].assign((size_t) max_ctx_ * g_.kv_row(), 0);
            ivg_[(size_t) il].assign((size_t) max_ctx_ * 2 * g_.ix_dim, 0);
            pooled_[(size_t) il].assign((size_t) (max_ctx_ / g_.ix_kpool + 1) * g_.ix_dim, 0.0f);
        }
    }

    void reset() override {
        pos_ = 0;
        hist_.clear();
        for (auto& v : kv_) std::fill(v.begin(), v.end(), 0);
        for (auto& v : ivg_) std::fill(v.begin(), v.end(), 0);
        for (auto& v : pooled_) std::fill(v.begin(), v.end(), 0.0f);
    }
    int64_t pos() const override { return pos_; }
    void rewind_to(int64_t n) override {
        pos_ = n;
        hist_.resize((size_t) n);
    }

    bool run(const int* ids, int T, float* logits_out, Trace* trace, std::string& err) override {
        const hy4::Geo& g = g_;
        if (g.ix_dim > 4096 || g.kv_row() > 4096 || w_.pk.eg.head_dim > 1024 || g.rope > 512) {
            err = "hy4-metal ref: the golden's scratch is sized for toy geometry";
            return false;
        }
        const int E = g.n_embd, hc = g.hc;
        const int64_t p0 = pos_;
        if (p0 + T > max_ctx_) {
            err = "hy4-metal ref: window past max_ctx";
            return false;
        }
        for (int t = 0; t < T; ++t) hist_.push_back(ids[t]);
        // the embedding, every hyper-connection stream a copy
        std::vector<float> x((size_t) T * hc * E);
        {
            const std::vector<float>* emb = w_.f32("token_embd.weight", err);
            if (!emb) return false;
            for (int t = 0; t < T; ++t)
                for (int s = 0; s < hc; ++s)
                    std::memcpy(&x[((size_t) t * hc + s) * E], &(*emb)[(size_t) ids[t] * E], E * sizeof(float));
        }
        for (int il = 0; il < g.n_layer; ++il)
            if (!layer(il, x, T, p0, trace, err)) return false;
        // the head: hc_head, output norm, the output projection
        const std::vector<float>* fn = w_.f32("output_hc_fn.weight", err);
        const std::vector<float>* base = w_.f32("output_hc_base.weight", err);
        const std::vector<float>* scale = w_.f32("output_hc_scale.weight", err);
        const std::vector<float>* onorm = w_.f32("output_norm.weight", err);
        const std::vector<float>* out = w_.f32("output.weight", err);
        if (!fn || !base || !scale || !onorm || !out) return false;
        std::vector<float> hcur((size_t) T * E);
        for (int t = 0; t < T; ++t) {
            const float* xt = &x[(size_t) t * hc * E];
            std::vector<float> r((size_t) hc * E);
            rmsnorm(xt, hc * E, nullptr, g.eps, r.data());
            float pre[64];
            for (int s = 0; s < hc; ++s) {
                double m = 0;
                for (int i = 0; i < hc * E; ++i) m += (double) (*fn)[(size_t) s * hc * E + i] * r[(size_t) i];
                pre[s] = sigmoid((float) m * (*scale)[0] + (*base)[(size_t) s]) + g.hc_eps;
            }
            for (int d = 0; d < E; ++d) {
                double s = 0;
                for (int k = 0; k < hc; ++k) s += (double) pre[k] * xt[(size_t) k * E + d];
                hcur[(size_t) t * E + d] = (float) s;
            }
        }
        rmsnorm_rows(hcur, T, E, onorm->data());
        if (trace) trace->put("hcur", hcur);
        const W ow{out->data(), E, g.n_vocab};
        for (int t = 0; t < T; ++t) matvec(ow, &hcur[(size_t) t * E], logits_out + (size_t) t * g.n_vocab);
        if (trace) trace->put("logits", logits_out, (size_t) T * g.n_vocab);
        pos_ += T;
        return true;
    }

private:
    void rmsnorm_rows(std::vector<float>& v, int rows, int n, const float* w) {
        for (int r = 0; r < rows; ++r) rmsnorm(&v[(size_t) r * n], n, w, g_.eps, &v[(size_t) r * n]);
    }

    // hc_pre: returns cur [T][E] (normed with norm_w when given) and post [T][hc]
    void hc_pre(const std::vector<float>& x, int T, const std::string& pfx, const float* norm_w,
                std::vector<float>& cur, std::vector<float>& post, std::string& err, bool& ok) {
        const hy4::Geo& g = g_;
        const int E = g.n_embd, hc = g.hc;
        const std::vector<float>* fn = w_.f32(pfx + "_fn.weight", err);
        const std::vector<float>* base = w_.f32(pfx + "_base.weight", err);
        const std::vector<float>* scale = w_.f32(pfx + "_scale.weight", err);
        if (!fn || !base || !scale) {
            ok = false;
            return;
        }
        cur.assign((size_t) T * E, 0.0f);
        post.assign((size_t) T * hc, 0.0f);
        std::vector<float> r((size_t) hc * E);
        for (int t = 0; t < T; ++t) {
            const float* xt = &x[(size_t) t * hc * E];
            rmsnorm(xt, hc * E, nullptr, g.eps, r.data());
            float pre[64];
            for (int j = 0; j < 2 * hc; ++j) {
                double m = 0;
                for (int i = 0; i < hc * E; ++i) m += (double) (*fn)[(size_t) j * hc * E + i] * r[(size_t) i];
                if (j < hc)
                    pre[j] = sigmoid((float) m * (*scale)[0] + (*base)[(size_t) j]) + g.hc_eps;
                else
                    post[(size_t) t * hc + (j - hc)] =
                        sigmoid((float) m * (*scale)[1] + (*base)[(size_t) j]) * g.hc_mag + g.hc_eps;
            }
            for (int d = 0; d < E; ++d) {
                double s = 0;
                for (int k = 0; k < hc; ++k) s += (double) pre[k] * xt[(size_t) k * E + d];
                cur[(size_t) t * E + d] = (float) s;
            }
        }
        if (norm_w)
            for (int t = 0; t < T; ++t) rmsnorm(&cur[(size_t) t * E], E, norm_w, g.eps, &cur[(size_t) t * E]);
    }

    void hc_post_norm(std::vector<float>& x, int T, const std::vector<float>& y, const std::vector<float>& post,
                      const float* norm_w) {
        const int E = g_.n_embd, hc = g_.hc;
        std::vector<float> yn((size_t) T * E);
        for (int t = 0; t < T; ++t) rmsnorm(&y[(size_t) t * E], E, norm_w, g_.eps, &yn[(size_t) t * E]);
        for (int t = 0; t < T; ++t)
            for (int s = 0; s < hc; ++s)
                for (int d = 0; d < E; ++d)
                    x[((size_t) t * hc + s) * E + d] += yn[(size_t) t * E + d] * post[(size_t) t * hc + s];
    }

    const std::vector<float>* getw(const std::string& name, std::string& err, bool& ok) {
        const std::vector<float>* v = w_.f32(name, err);
        if (!v) ok = false;
        return v;
    }
    W mat(const std::string& name, std::string& err, bool& ok) {
        const hy4::Tensor* t = w_.tensor(name);
        const std::vector<float>* v = getw(name, err, ok);
        if (!v || !t) return {};
        return W{v->data(), t->n(0), std::max<int64_t>(t->n(1), 1), std::max<int64_t>(t->n(2), 1)};
    }

    bool layer(int il, std::vector<float>& x, int T, int64_t p0, Trace* trace, std::string& err) {
        const hy4::Geo& g = g_;
        const int E = g.n_embd, hc = g.hc, H = (int) g.heads[(size_t) il];
        const int KL = g.kv_lora, R = g.rope, KR = g.kv_row(), nope = g.nope(), KMLA = g.k_mla, VM = g.v_mla;
        const bool swa = g.swa[(size_t) il] != 0, ix = !swa && g.indexer[(size_t) il] != 0 && g.ix_top_k > 0;
        const bool dense = il < g.n_dense_lead;
        const float base = swa ? (float) g.rope_base_swa : (float) g.rope_base;
        const std::string p = "blk." + std::to_string(il) + ".";
        bool ok = true;
        auto W_ = [&](const std::string& n) { return mat(p + n, err, ok); };
        auto V_ = [&](const std::string& n) { return getw(p + n, err, ok); };

        // ---- Engram (its module's rows into the streams, before the attention mix)
        const int egm = w_.pk.eg.module_of(il);
        if (egm >= 0) {
            const EgRows rows = engram_gather(w_.pk, egm, hist_, p0, T);
            const int cols = w_.pk.eg.cols(), hd = w_.pk.eg.head_dim;
            const std::vector<float>* ln = V_("engram_lookup_norm.weight");
            const std::vector<float>* kvw = V_("engram_kv.weight");
            const std::vector<float>* kvb = V_("engram_kv_b.weight");
            const std::vector<float>* egate = V_("engram_gate.weight");
            const std::vector<float>* gamma = V_("engram_pre_gamma.weight");
            const std::vector<float>* hcs = V_("engram_hc_scale.weight");
            if (!ok) return false;
            const float eeps = engram_eps(w_.pk);
            std::vector<float> r((size_t) T * cols * hd);
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < cols; ++c) {
                    float row[1024];
                    for (int d = 0; d < hd; ++d)
                        row[d] = f8_e4m3_to_f32(rows.bytes[((size_t) t * cols + c) * hd + d]);
                    rmsnorm(row, hd, nullptr, eeps, row);
                    for (int d = 0; d < hd; ++d)
                        r[((size_t) t * cols + c) * hd + d] = row[d] * (*ln)[(size_t) c * hd + d] * rows.valid[(size_t) t];
                }
            const W egkv{kvw->data(), (int64_t) cols * hd, (int64_t) (hc + 1) * E};
            std::vector<float> kvo((size_t) T * (hc + 1) * E);
            for (int t = 0; t < T; ++t) matvec(egkv, &r[(size_t) t * cols * hd], &kvo[(size_t) t * (hc + 1) * E]);
            for (int t = 0; t < T; ++t) {
                if (rows.valid[(size_t) t] == 0.0f) continue;
                const float* kt = &kvo[(size_t) t * (hc + 1) * E];
                float* xt = &x[(size_t) t * hc * E];
                // the value's rms factor
                double sv = 0;
                for (int d = 0; d < E; ++d) {
                    const float v = kt[(size_t) hc * E + d] + (*kvb)[(size_t) hc * E + d];
                    sv += (double) v * v;
                }
                const float rv = (float) (1.0 / std::sqrt(sv / E + eeps));
                float gate_s[64];
                for (int s = 0; s < hc; ++s) {
                    double sx = 0, sk = 0, sd = 0;
                    for (int d = 0; d < E; ++d) {
                        const float xv = xt[(size_t) s * E + d];
                        const float kvv = kt[(size_t) s * E + d] + (*kvb)[(size_t) s * E + d];
                        sx += (double) xv * xv;
                        sk += (double) kvv * kvv;
                        sd += (double) xv * kvv * (*egate)[(size_t) s * E + d];
                    }
                    const double dot = sd / std::sqrt(sx / E + eeps) / std::sqrt(sk / E + eeps) / std::sqrt((double) E);
                    const double mag = std::sqrt(std::max(std::fabs(dot), 1e-6));
                    gate_s[s] = sigmoid((float) (dot < 0 ? -mag : mag));
                }
                for (int d = 0; d < E; ++d) {
                    const float sig = (kt[(size_t) hc * E + d] + (*kvb)[(size_t) hc * E + d]) * rv * (*gamma)[(size_t) d];
                    for (int s = 0; s < hc; ++s)
                        xt[(size_t) s * E + d] += sig * (gate_s[s] + (*hcs)[(size_t) s * E + d]);
                }
            }
            if (trace) trace->put("engram_out-" + std::to_string(il), x);
        }

        // ---- attention
        const std::vector<float>* attn_norm = V_("attn_norm.weight");
        std::vector<float> xa, post;
        hc_pre(x, T, p + "hc_attn", attn_norm ? attn_norm->data() : nullptr, xa, post, err, ok);
        if (!ok) return false;
        if (trace) trace->put("attn_norm-" + std::to_string(il), xa);
        const W q_a = W_("attn_q_a.weight"), q_b = W_("attn_q_b.weight"), kv_a = W_("attn_kv_a_mqa.weight");
        const W k_b = W_("attn_k_b.weight"), v_b = W_("attn_v_b.weight"), wo = W_("attn_output.weight");
        const W gate = W_("attn_gate.weight");
        const std::vector<float>* q_a_norm = V_("attn_q_a_norm.weight");
        const std::vector<float>* kv_a_norm = V_("attn_kv_a_norm.weight");
        const std::vector<float>* sinks = V_("attn_sinks.weight");
        if (!ok) return false;
        const int QL = g.q_lora;
        std::vector<float> qr((size_t) T * QL), q((size_t) T * H * KMLA), kvr((size_t) T * KR), gg((size_t) T * H * VM);
        for (int t = 0; t < T; ++t) {
            std::vector<float> qa((size_t) QL);
            matvec(q_a, &xa[(size_t) t * E], qa.data());
            matvec(kv_a, &xa[(size_t) t * E], &kvr[(size_t) t * KR]);
            matvec(gate, &xa[(size_t) t * E], &gg[(size_t) t * H * VM]);
            rmsnorm(qa.data(), QL, q_a_norm->data(), g.eps, &qr[(size_t) t * QL]);
            matvec(q_b, &qr[(size_t) t * QL], &q[(size_t) t * H * KMLA]);
            // the cache row: the normed latent, then the rotated rope part, stored F16
            float row[4096];
            rmsnorm(&kvr[(size_t) t * KR], KL, kv_a_norm->data(), g.eps, row);
            std::memcpy(row + KL, &kvr[(size_t) t * KR] + KL, R * sizeof(float));
            rope(row, KL, R, p0 + t, base);
            uint16_t* dst = &kv_[(size_t) il][(size_t) (p0 + t) * KR];
            for (int i = 0; i < KR; ++i) dst[i] = ggml_fp32_to_fp16(row[i]);
        }
        if (trace) trace->put("q_resid-" + std::to_string(il), qr);

        // ---- the indexer (its ring and pools first: the attention below reads the picks)
        std::vector<std::vector<int>> keys((size_t) T);
        if (ix) {
            const int Dd = g.ix_dim, IH = g.ix_heads, KP = g.ix_kpool;
            const W ixqb = W_("indexer.attn_q_b.weight"), ixk = W_("indexer.attn_k.weight");
            const W ixval = W_("indexer_compressor_value.weight"), ixgate = W_("indexer_compressor_gate.weight");
            const W ixproj = W_("indexer.proj.weight");
            const std::vector<float>* ixkn = V_("indexer.k_norm.weight");
            const std::vector<float>* ape = V_("indexer_compressor_ape.weight");
            const std::vector<float>* ixpn = V_("indexer_compressor_norm.weight");
            if (!ok) return false;
            std::vector<float> iq((size_t) T * IH * Dd), wt((size_t) T * IH);
            const float wsc = (float) (1.0 / std::sqrt((double) Dd * IH));
            for (int t = 0; t < T; ++t) {
                std::vector<float> ik((size_t) Dd);
                matvec(ixk, &xa[(size_t) t * E], ik.data());
                rmsnorm(ik.data(), Dd, ixkn->data(), g.eps, ik.data());
                std::vector<float> ivt((size_t) 2 * Dd);
                matvec(ixval, ik.data(), ivt.data());
                matvec(ixgate, ik.data(), ivt.data() + Dd);
                uint16_t* dst = &ivg_[(size_t) il][(size_t) (p0 + t) * 2 * Dd];
                for (int i = 0; i < 2 * Dd; ++i) dst[i] = ggml_fp32_to_fp16(ivt[(size_t) i]);
                matvec(ixqb, &qr[(size_t) t * QL], &iq[(size_t) t * IH * Dd]);
                for (int h = 0; h < IH; ++h) rope(&iq[((size_t) t * IH + h) * Dd], Dd - R, R, p0 + t, base);
                matvec(ixproj, &xa[(size_t) t * E], &wt[(size_t) t * IH]);
                for (int h = 0; h < IH; ++h) wt[(size_t) t * IH + h] *= wsc;
            }
            // the groups this window completes: [p0/KP, (p0+T)/KP)
            const int64_t g0 = p0 / KP, g1 = (p0 + T) / KP;
            for (int64_t gr = g0; gr < g1; ++gr) {
                float acc[4096] = {0};
                for (int d = 0; d < Dd; ++d) {
                    float lg[64], mx = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < KP; ++k) {
                        const float gv = ggml_fp16_to_fp32(ivg_[(size_t) il][((size_t) (gr * KP + k)) * 2 * Dd + Dd + d]);
                        lg[k] = gv + (*ape)[(size_t) k * Dd + d];
                        mx = std::max(mx, lg[k]);
                    }
                    double den = 0;
                    for (int k = 0; k < KP; ++k) den += std::exp((double) lg[k] - mx);
                    for (int k = 0; k < KP; ++k) {
                        const float wgt = (float) (std::exp((double) lg[k] - mx) / den);
                        acc[d] += wgt * ggml_fp16_to_fp32(ivg_[(size_t) il][((size_t) (gr * KP + k)) * 2 * Dd + d]);
                    }
                }
                float* pr = &pooled_[(size_t) il][(size_t) gr * Dd];
                rmsnorm(acc, Dd, ixpn->data(), g.eps, pr);
                rope(pr, Dd - R, R, gr * KP, base);
            }
            const int top = g.ix_top_k / KP - 1;
            for (int t = 0; t < T; ++t) {
                const int64_t pos = p0 + t, nv = pos / KP;
                std::vector<std::pair<float, int64_t>> scored;
                for (int64_t gr = 0; gr < nv; ++gr) {
                    double s = 0;
                    for (int h = 0; h < IH; ++h) {
                        double d = 0;
                        for (int i = 0; i < Dd; ++i)
                            d += (double) iq[((size_t) t * IH + h) * Dd + i] * pooled_[(size_t) il][(size_t) gr * Dd + i];
                        s += std::max(d, 0.0) * wt[(size_t) t * IH + h];
                    }
                    scored.push_back({(float) s, gr});
                }
                std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
                    return a.first != b.first ? a.first > b.first : a.second < b.second;
                });
                std::vector<int64_t> picks;
                for (int64_t i = 0; i < std::min<int64_t>(nv, top); ++i) picks.push_back(scored[(size_t) i].second);
                std::sort(picks.begin(), picks.end());
                for (int64_t gr : picks)
                    for (int k = 0; k < KP; ++k) keys[(size_t) t].push_back((int) (gr * KP + k));
                for (int64_t pp = (pos / KP) * KP; pp <= pos; ++pp) keys[(size_t) t].push_back((int) pp);
            }
            if (trace) {
                std::vector<float> kk((size_t) T * (g.ix_top_k + KP), -1.0f);
                for (int t = 0; t < T; ++t)
                    for (size_t i = 0; i < keys[(size_t) t].size(); ++i)
                        kk[(size_t) t * (g.ix_top_k + KP) + i] = (float) keys[(size_t) t][i];
                trace->put("ix_keys-" + std::to_string(il), kk);
            }
        }

        // ---- attention over the latent cache
        const float scale = (float) (1.0 / std::sqrt((double) KMLA));
        std::vector<float> ob((size_t) T * H * VM);
        std::vector<float> qcat((size_t) H * KR);
        for (int t = 0; t < T; ++t) {
            const int64_t pos = p0 + t;
            for (int h = 0; h < H; ++h) {
                float* qc = &qcat[(size_t) h * KR];
                matvec_b(k_b, h, &q[((size_t) t * H + h) * KMLA], qc);
                float pe[512];
                std::memcpy(pe, &q[((size_t) t * H + h) * KMLA] + nope, R * sizeof(float));
                rope(pe, 0, R, pos, base);
                std::memcpy(qc + KL, pe, R * sizeof(float));
            }
            std::vector<int> ks;
            if (ix) {
                ks = keys[(size_t) t];
            } else {
                const int64_t lo = std::max<int64_t>(0, pos - g.n_swa + 1);
                for (int64_t pp = lo; pp <= pos; ++pp) ks.push_back((int) pp);
            }
            std::vector<float> crow((size_t) KR);
            for (int h = 0; h < H; ++h) {
                std::vector<float> sc(ks.size());
                float mx = (*sinks)[(size_t) h];
                for (size_t i = 0; i < ks.size(); ++i) {
                    const uint16_t* rowp = &kv_[(size_t) il][(size_t) ks[i] * KR];
                    double d = 0;
                    for (int j = 0; j < KR; ++j) d += (double) qcat[(size_t) h * KR + j] * ggml_fp16_to_fp32(rowp[j]);
                    sc[i] = (float) (d * scale);
                    mx = std::max(mx, sc[i]);
                }
                double den = std::exp((double) (*sinks)[(size_t) h] - mx);
                for (float v : sc) den += std::exp((double) v - mx);
                std::vector<float> o((size_t) KL, 0.0f);
                for (size_t i = 0; i < ks.size(); ++i) {
                    const float pr = (float) (std::exp((double) sc[i] - mx) / den);
                    const uint16_t* rowp = &kv_[(size_t) il][(size_t) ks[i] * KR];
                    for (int j = 0; j < KL; ++j) o[(size_t) j] += pr * ggml_fp16_to_fp32(rowp[j]);
                }
                matvec_b(v_b, h, o.data(), &ob[((size_t) t * H + h) * VM]);
            }
        }
        for (size_t i = 0; i < ob.size(); ++i) ob[i] *= sigmoid(gg[i]);
        std::vector<float> y((size_t) T * E);
        for (int t = 0; t < T; ++t) matvec(wo, &ob[(size_t) t * H * VM], &y[(size_t) t * E]);
        if (trace) trace->put("attn_y-" + std::to_string(il), y);
        const std::vector<float>* post_attn = V_("post_attention_norm.weight");
        if (!ok) return false;
        hc_post_norm(x, T, y, post, post_attn->data());
        if (trace) trace->put("hc_attn_out-" + std::to_string(il), x);

        // ---- FFN
        const std::vector<float>* ffn_norm = V_("ffn_norm.weight");
        std::vector<float> xf, post2;
        hc_pre(x, T, p + "hc_ffn", ffn_norm ? ffn_norm->data() : nullptr, xf, post2, err, ok);
        if (!ok) return false;
        std::vector<float> y2((size_t) T * E);
        if (dense) {
            const W fg = W_("ffn_gate.weight"), fu = W_("ffn_up.weight"), fd = W_("ffn_down.weight");
            if (!ok) return false;
            const int FF = g.n_ff;
            std::vector<float> hd((size_t) T * FF);
            for (int t = 0; t < T; ++t) {
                std::vector<float> gu((size_t) FF), uu((size_t) FF);
                matvec(fg, &xf[(size_t) t * E], gu.data());
                matvec(fu, &xf[(size_t) t * E], uu.data());
                for (int i = 0; i < FF; ++i) hd[(size_t) t * FF + i] = gu[(size_t) i] * sigmoid(gu[(size_t) i]) * uu[(size_t) i];
                matvec(fd, &hd[(size_t) t * FF], &y2[(size_t) t * E]);
            }
        } else {
            const W router = W_("ffn_gate_inp.weight"), rd = W_("ffn_routed_down.weight"), ru = W_("ffn_routed_up.weight");
            const W sg = W_("ffn_gate_shexp.weight"), su = W_("ffn_up_shexp.weight"), sd = W_("ffn_down_shexp.weight");
            const W ge = W_("ffn_gate_exps.weight"), ue = W_("ffn_up_exps.weight"), de = W_("ffn_down_exps.weight");
            const std::vector<float>* rbias = V_("exp_probs_b.bias");
            if (!ok) return false;
            const int LAT = g.n_lat, FFE = g.n_ff_exp, NU = g.n_used, NE = g.n_expert;
            const float limit = (float) g.limit[(size_t) il];
            std::vector<float> latout((size_t) T * LAT, 0.0f);
            std::vector<float> trace_ids, trace_rw, trace_sc;
            for (int t = 0; t < T; ++t) {
                std::vector<float> rl((size_t) NE);
                matvec(router, &xf[(size_t) t * E], rl.data());
                std::vector<std::pair<float, int>> cand((size_t) NE);
                for (int e = 0; e < NE; ++e) cand[(size_t) e] = {sigmoid(rl[(size_t) e]) + (*rbias)[(size_t) e], e};
                for (int e = 0; e < NE; ++e) trace_sc.push_back(cand[(size_t) e].first);
                std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
                    return a.first != b.first ? a.first > b.first : a.second < b.second;
                });
                std::vector<float> lat((size_t) LAT);
                matvec(rd, &xf[(size_t) t * E], lat.data());
                float wsum = 0;
                std::vector<float> ws((size_t) NU);
                for (int j = 0; j < NU; ++j) {
                    ws[(size_t) j] = sigmoid(rl[(size_t) cand[(size_t) j].second]);
                    wsum += ws[(size_t) j];
                }
                for (int j = 0; j < NU; ++j) {
                    float wgt = ws[(size_t) j];
                    if (g.w_norm) wgt /= std::max(wsum, 6.103515625e-5f);   // upstream's 2^-14 floor
                    wgt *= (float) g.w_scale;
                    trace_ids.push_back((float) cand[(size_t) j].second);
                    trace_rw.push_back(wgt);
                    const int e = cand[(size_t) j].second;
                    std::vector<float> gu((size_t) FFE), uu((size_t) FFE), hh((size_t) FFE), eo((size_t) LAT);
                    matvec_b(ge, e, lat.data(), gu.data());
                    matvec_b(ue, e, lat.data(), uu.data());
                    for (int i = 0; i < FFE; ++i) {
                        float gv = gu[(size_t) i], uv = uu[(size_t) i];
                        if (limit > 0) {
                            gv = std::min(gv, limit);
                            uv = std::max(-limit, std::min(limit, uv));
                        }
                        hh[(size_t) i] = gv * sigmoid(gv) * uv;
                    }
                    matvec_b(de, e, hh.data(), eo.data());
                    for (int i = 0; i < LAT; ++i) latout[(size_t) t * LAT + i] += wgt * eo[(size_t) i];
                }
                std::vector<float> sh((size_t) FFE), sh2((size_t) FFE), sho((size_t) E);
                matvec(sg, &xf[(size_t) t * E], sh.data());
                matvec(su, &xf[(size_t) t * E], sh2.data());
                for (int i = 0; i < FFE; ++i) sh[(size_t) i] = sh[(size_t) i] * sigmoid(sh[(size_t) i]) * sh2[(size_t) i];
                matvec(sd, sh.data(), sho.data());
                std::vector<float> ro((size_t) E);
                matvec(ru, &latout[(size_t) t * LAT], ro.data());
                for (int i = 0; i < E; ++i) y2[(size_t) t * E + i] = ro[(size_t) i] + sho[(size_t) i];
            }
            if (trace) {
                trace->put("router_ids-" + std::to_string(il), trace_ids);
                trace->put("router_rw-" + std::to_string(il), trace_rw);
                trace->put("router_sc-" + std::to_string(il), trace_sc);
                trace->put("ffn_latout-" + std::to_string(il), latout);
            }
        }
        if (trace) trace->put("ffn_y-" + std::to_string(il), y2);
        const std::vector<float>* post_ffn = V_("post_ffw_norm.weight");
        if (!ok) return false;
        hc_post_norm(x, T, y2, post2, post_ffn->data());
        if (trace) trace->put("l_out-" + std::to_string(il), x);
        return true;
    }

    Weights& w_;
    const hy4::Geo& g_;
    int64_t max_ctx_;
    int64_t pos_ = 0;
    std::vector<int> hist_;
    std::vector<std::vector<uint16_t>> kv_, ivg_;
    std::vector<std::vector<float>> pooled_;
};

}  // namespace

std::unique_ptr<Forward> make_ref_forward(Weights& w, int64_t max_ctx) {
    return std::make_unique<Ref>(w, max_ctx);
}

}  // namespace hym
