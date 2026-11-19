// src/hy4-metal/metal_hy4_block_test.cpp - the Hy4 Metal port's stage 3 gate: the ggml/Metal
// trunk forward against the scalar CPU golden (hy4_metal_ref.cpp), on the complete-block
// fixture pack (tools/hy4_metal_block_fixture.py via the real packer).  Both forwards run
// the same windows - a 9-token prompt (the indexed layer's selection truncates, the SWA
// window truncates, the Engram layer fires), then greedy decode steps with one forced
// unmapped token (the Engram dead-token path) - and every named intermediate (upstream's
// dbg points) plus the logits must agree within the cross-backend band, the discrete
// picks (indexer keys, router ids) and the greedy ids exactly.
//
// Usage: metal_hy4_block_test <pack-dir>
#include "hy4_metal_model.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

double rel_l2(const float* a, const float* b, size_t n) {   // ||a - b|| / ||b||
    double d = 0, r = 0;
    for (size_t i = 0; i < n; ++i) {
        d += (double) (a[i] - b[i]) * (a[i] - b[i]);
        r += (double) b[i] * b[i];
    }
    return std::sqrt(d) / std::max(std::sqrt(r), 1e-9);
}

void check_matrix(const char* name, const std::vector<float>& ref, const std::vector<float>& gpu, int T, int F,
                  double tol) {
    if ((int) ref.size() != T * F || (int) gpu.size() != T * F) {
        std::fprintf(stderr, "FAIL %-24s size ref %zu gpu %zu want %d\n", name, ref.size(), gpu.size(), T * F);
        ++failures;
        return;
    }
    // ref is [T][F] token-major; the ggml tensors are [F][T] feature-major: (f, t) at f + F*t
    std::vector<float> gt((size_t) T * F);
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < F; ++f) gt[(size_t) t * F + f] = gpu[(size_t) f + (size_t) F * t];
    const double e = rel_l2(gt.data(), ref.data(), (size_t) T * F);
    std::fprintf(stderr, "  %-24s rel L2 %.3e (tol %.1e)%s\n", name, e, tol, e <= tol ? "" : "  FAIL");
    if (e > tol) ++failures;
}

void check_host(const char* name, const std::vector<float>& ref, const std::vector<float>& gpu, double tol,
                bool exact) {
    if (ref.size() != gpu.size()) {
        std::fprintf(stderr, "FAIL %-24s size ref %zu gpu %zu\n", name, ref.size(), gpu.size());
        ++failures;
        return;
    }
    if (exact) {
        for (size_t i = 0; i < ref.size(); ++i)
            if (ref[i] != gpu[i]) {
                std::fprintf(stderr, "FAIL %-24s [%zu] ref %.1f gpu %.1f\n", name, i, ref[i], gpu[i]);
                ++failures;
                return;
            }
        std::fprintf(stderr, "  %-24s exact\n", name);
        return;
    }
    const double e = rel_l2(gpu.data(), ref.data(), ref.size());
    std::fprintf(stderr, "  %-24s rel L2 %.3e (tol %.1e)%s\n", name, e, tol, e <= tol ? "" : "  FAIL");
    if (e > tol) ++failures;
}

// Router picks must match exactly, except a swapped pair whose selection scores are
// within the tie band on BOTH sides: the two forwards differ by F16-cache rounding
// (~1e-3), so a genuine tie can flip. The router_sc traces arbitrate; a wider gap is a
// real selection bug and fails.
int check_router_ids(const char* name, const std::vector<float>& ref, const std::vector<float>& gpu,
                     const std::vector<float>& ref_sc, const std::vector<float>& gpu_sc, int NU, int NE) {
    int flips = 0;
    if (ref.size() != gpu.size() || ref_sc.size() != gpu_sc.size() || ref_sc.empty()) {
        std::fprintf(stderr, "FAIL %-24s size ref %zu gpu %zu sc %zu/%zu\n", name, ref.size(), gpu.size(),
                     ref_sc.size(), gpu_sc.size());
        ++failures;
        return 0;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] == gpu[i]) continue;
        const size_t t = i / (size_t) NU;
        const int ra = (int) ref[i], gb = (int) gpu[i];
        const float dr = std::fabs(ref_sc[t * NE + ra] - ref_sc[t * NE + gb]);
        const float dg = std::fabs(gpu_sc[t * NE + ra] - gpu_sc[t * NE + gb]);
        if (dr <= 0.02f && dg <= 0.02f) {
            std::fprintf(stderr, "  %-24s [%zu] tie flip ref %d gpu %d (score gaps %.4f/%.4f)\n", name, i, ra,
                         gb, dr, dg);
            ++flips;
            continue;
        }
        std::fprintf(stderr, "FAIL %-24s [%zu] ref %.1f gpu %.1f (score gaps %.4f/%.4f)\n", name, i, ref[i],
                     gpu[i], dr, dg);
        ++failures;
        return flips;
    }
    std::fprintf(stderr, "  %-24s exact\n", name);
    return flips;
}

int argmax(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: metal_hy4_block_test <pack-dir>\n");
        return 2;
    }
    hym::Weights w;
    std::string err;
    if (!w.load(argv[1], err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const hy4::Geo& g = w.pk.g;
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) {
        std::fprintf(stderr, "no GPU backend\n");
        return 1;
    }
    auto ref = hym::make_ref_forward(w, g.context);
    auto gpu = hym::make_ggml_forward(w, be, g.context, true);
    if (!ref || !gpu) {
        std::fprintf(stderr, "forward init failed\n");
        return 1;
    }
    const int E = g.n_embd, V = g.n_vocab;
    auto run_pair = [&](const int* ids, int T, const char* what) -> std::vector<float> {
        hym::Trace tr, tg;
        std::vector<float> lr((size_t) T * V), lg((size_t) T * V);
        if (!ref->run(ids, T, lr.data(), &tr, err)) {
            std::fprintf(stderr, "ref run (%s): %s\n", what, err.c_str());
            ++failures;
            return lr;
        }
        if (!gpu->run(ids, T, lg.data(), &tg, err)) {
            std::fprintf(stderr, "gpu run (%s): %s\n", what, err.c_str());
            ++failures;
            return lr;
        }
        std::fprintf(stderr, "%s (T=%d):\n", what, T);
        // after a proven tie flip the two forwards legitimately take different experts, so
        // this window's comparisons downstream of the flip are meaningless (the flip touches
        // no cache state, so later windows still compare cleanly)
        bool bad_tail = false;
        for (int il = 0; il < g.n_layer; ++il) {
            const std::string s = std::to_string(il);
            auto has = [&](const std::string& n) { return tr.at.count(n) && tg.at.count(n); };
            if (bad_tail) {
                std::fprintf(stderr, "  (layer %d skipped: downstream of a tie flip)\n", il);
                continue;
            }
            if (has("engram_out-" + s))
                check_matrix(("engram_out-" + s).c_str(), tr.at["engram_out-" + s], tg.at["engram_out-" + s], T,
                             g.hc * E, 2e-2);
            check_matrix(("attn_norm-" + s).c_str(), tr.at["attn_norm-" + s], tg.at["attn_norm-" + s], T, E, 2e-2);
            check_matrix(("q_resid-" + s).c_str(), tr.at["q_resid-" + s], tg.at["q_resid-" + s], T, g.q_lora, 2e-2);
            if (has("ix_keys-" + s))
                check_host(("ix_keys-" + s).c_str(), tr.at["ix_keys-" + s], tg.at["ix_keys-" + s], 0, true);
            check_matrix(("attn_y-" + s).c_str(), tr.at["attn_y-" + s], tg.at["attn_y-" + s], T, E, 2e-2);
            check_matrix(("hc_attn_out-" + s).c_str(), tr.at["hc_attn_out-" + s], tg.at["hc_attn_out-" + s], T,
                         g.hc * E, 2e-2);
            if (has("router_ids-" + s)) {
                const int flips = check_router_ids(("router_ids-" + s).c_str(), tr.at["router_ids-" + s],
                                                   tg.at["router_ids-" + s], tr.at["router_sc-" + s],
                                                   tg.at["router_sc-" + s], g.n_used, g.n_expert);
                check_host(("router_sc-" + s).c_str(), tr.at["router_sc-" + s], tg.at["router_sc-" + s], 2e-2,
                           false);
                if (flips > 0) {
                    bad_tail = true;
                } else {
                    check_host(("router_rw-" + s).c_str(), tr.at["router_rw-" + s], tg.at["router_rw-" + s],
                               2e-2, false);
                    check_matrix(("ffn_latout-" + s).c_str(), tr.at["ffn_latout-" + s],
                                 tg.at["ffn_latout-" + s], T, g.n_lat, 2e-2);
                }
            }
            if (!bad_tail) {
                check_matrix(("ffn_y-" + s).c_str(), tr.at["ffn_y-" + s], tg.at["ffn_y-" + s], T, E, 2e-2);
                check_matrix(("l_out-" + s).c_str(), tr.at["l_out-" + s], tg.at["l_out-" + s], T, g.hc * E,
                             2e-2);
            }
        }
        if (!bad_tail) {
            check_matrix("hcur", tr.at["hcur"], tg.at["hcur"], T, E, 2e-2);
            check_matrix("logits", tr.at["logits"], tg.at["logits"], T, V, 3e-2);
            const double e = rel_l2(lg.data(), lr.data(), (size_t) T * V);
            std::fprintf(stderr, "  %-24s rel L2 %.3e (returned logits)\n", "logits_out", e);
            for (int t = 0; t < T; ++t) {
                const int a = argmax(lr.data() + (size_t) t * V, V), b = argmax(lg.data() + (size_t) t * V, V);
                if (a != b) {
                    std::fprintf(stderr, "FAIL greedy row %d: ref %d gpu %d\n", t, a, b);
                    ++failures;
                }
            }
        } else {
            std::fprintf(stderr, "  (hcur/logits skipped: downstream of a tie flip)\n");
        }
        return lr;
    };

    const int prompt[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    std::vector<float> logits = run_pair(prompt, 9, "prompt");
    int next_ref = argmax(logits.data() + 8 * V, V);
    for (int step = 0; step < 6; ++step) {
        int id = next_ref;
        if (step == 2) id = 268;   // unmapped in the Engram token map: the dead-token path
        char what[64];
        std::snprintf(what, sizeof what, "decode step %d (id %d)", step, id);
        logits = run_pair(&id, 1, what);
        next_ref = argmax(logits.data(), V);
    }
    // a second, shorter prompt after reset: the caches must clear
    ref->reset();
    gpu->reset();
    const int prompt2[] = {5, 15, 25, 35, 45};
    run_pair(prompt2, 5, "prompt after reset");
    std::fprintf(stderr, failures ? "BLOCK TEST FAILED (%d)\n" : "BLOCK TEST PASSED\n", failures);
    // orderly teardown: the forwards and the backend must die before ggml-metal's static
    // device cleanup, or its residue-set assert aborts the process at exit
    const int rc = failures ? 1 : 0;
    ref.reset();
    gpu.reset();
    ggml_backend_free(be);
    return rc;
}
