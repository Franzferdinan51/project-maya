// src/metal-qwen/runtime/session_body_test.cpp - the metal_session_test target:
// under STRATA_USE_METAL, session_capture / session_capture_token must STORE
// their layer sequences as bodies (not run them - there are no weights or
// kernels behind this geometry, so a body that ran at capture time would fail
// loudly here) and mint real execs for the slots; the frees must release both.
// The launch half needs a real model and is exercised by the engine, not here.
// See runtime/CAPTURE-NOTES.md.
#include "strata/core/session.hpp"

#include <cstdio>

using namespace strata::core;

static int fail(const char* what) {
    std::printf("FAIL: %s\n", what);
    return 1;
}

int main() {
    ModelGeometry g;
    g.n_layers = 4;   // layers 0-2 GDN, layer 3 QSA (qsa_interval 4)
    WeightTable wt;   // empty: capture under Metal never touches the table

    SessionState s;
    static float gdn_dummy[1];
    s.gdn_state = gdn_dummy;          // gdn_point_at only does pointer arithmetic at capture time
    s.qsa_states = new QsaState[1];   // the non-QSA layers borrow slot 0, as in a real session
    s.k = 10;

    float parts_dummy[1] = {0.0f};
    SessionGraphs gr;
    std::string err;
    if (!session_capture(wt, g, s, parts_dummy, gr, err, /*split=*/true)) {
        std::printf("capture fail: %s\n", err.c_str());
        return 1;
    }
    if (!gr.captured || gr.n != g.n_layers) return fail("capture did not complete all layers");
#ifdef STRATA_USE_METAL
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!gr.execs[l] || !gr.posts[l]) return fail("exec slot not minted");
        if (!gr.exec_bodies[l] || !gr.post_bodies[l]) return fail("pre/post body not stored");
        if (!gr.preA_bodies[l] || !gr.preB_bodies[l]) return fail("split body not stored");
        for (int k = 0; k < 5; ++k)
            if (!gr.preP_bodies[k][l]) return fail("prefix body not stored");
    }
#endif
    // Idempotent: a second capture is a no-op and keeps the stored bodies.
    if (!session_capture(wt, g, s, parts_dummy, gr, err, /*split=*/true)) return fail("re-capture failed");
    if (!gr.captured || gr.n != g.n_layers) return fail("re-capture disturbed the graphs");

    // An uncaptured set refuses to replay, without launching anything.
    SessionGraphs empty;
    if (session_replay(g, 0, 0, s, empty, nullptr, err)) return fail("replay of uncaptured graphs succeeded");

    session_graphs_free(gr);
    if (gr.captured || gr.execs != nullptr) return fail("graphs_free did not reset the set");
#ifdef STRATA_USE_METAL
    if (gr.exec_bodies != nullptr || gr.post_bodies != nullptr || gr.preP_bodies[0] != nullptr)
        return fail("graphs_free did not release the bodies");
#endif

    // ---- the whole-token graph: same storage shape, one body
    static uint32_t flag_word = 0, seq_word = 0;
    Doorbell db;
    db.d_flag = &flag_word;
    db.d_seq = &seq_word;
    SessionState s2;
    s2.gdn_state = gdn_dummy;
    s2.qsa_states = s.qsa_states;
    s2.k = 10;
    s2.db = &db;
    void* y_miss = nullptr;
    if (cudaHostAlloc(&y_miss, 256, cudaHostAllocMapped) != cudaSuccess) return fail("host alloc");
    void* parts_dev = nullptr;
    if (cudaMalloc(&parts_dev, 256) != cudaSuccess) return fail("device alloc");
    TokenGraph tg;
    if (!session_capture_token(wt, g, s2, (float*) parts_dev, (const float*) y_miss, 256, tg, err)) {
        std::printf("token capture fail: %s\n", err.c_str());
        return 1;
    }
    if (!tg.captured || tg.n_layers != g.n_layers || !tg.exec) return fail("token capture incomplete");
#ifdef STRATA_USE_METAL
    if (!tg.body) return fail("token body not stored");
#endif
    // The wrong staging buffer is refused before anything launches.
    if (session_run_token(g, 0, 0, s2, tg, nullptr, nullptr, (float*) y_miss + 1, nullptr, err))
        return fail("run_token accepted a foreign staging buffer");
    token_graph_free(tg);
    if (tg.captured || tg.exec != nullptr) return fail("token_graph_free did not reset");
    cudaFree(parts_dev);
    cudaFreeHost(y_miss);
    delete[] s.qsa_states;

    std::printf("session_body_store: PASS (capture stores bodies + minted execs, frees release both)\n");
    return 0;
}
