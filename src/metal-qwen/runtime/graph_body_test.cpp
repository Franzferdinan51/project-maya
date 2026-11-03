// src/metal-qwen/runtime/graph_body_test.cpp - the metal_graph_test target: under
// STRATA_USE_METAL, GraphRegistry::record must STORE the body (not run it) and
// launch() must invoke it once per launch, with wait_ms completing. See
// runtime/CAPTURE-NOTES.md.
#include "strata/core/graph.hpp"
#include <cstdio>
int main() {
    cudaStream_t cs = nullptr;
    if (cudaStreamCreate(&cs) != cudaSuccess) { std::printf("stream fail\n"); return 1; }
    strata::core::GraphRegistry reg(cs);
    int calls = 0;
    std::string err;
    if (!reg.record(strata::core::LayerType::GDN, 1, [&] { ++calls; }, err)) { std::printf("record fail: %s\n", err.c_str()); return 1; }
    if (calls != 0) { std::printf("FAIL: body ran at record time (%d)\n", calls); return 1; }
    if (!reg.record(strata::core::LayerType::GDN, 1, [&] { ++calls; }, err)) { std::printf("re-record fail\n"); return 1; }
    if (!reg.launch(strata::core::LayerType::GDN, 1, 5000, err)) { std::printf("launch fail: %s\n", err.c_str()); return 1; }
    if (calls != 1) { std::printf("FAIL: calls=%d after 1 launch\n", calls); return 1; }
    if (!reg.launch(strata::core::LayerType::GDN, 1, 5000, err)) { std::printf("launch2 fail: %s\n", err.c_str()); return 1; }
    if (calls != 2) { std::printf("FAIL: calls=%d after 2 launches\n", calls); return 1; }
    std::printf("graph_body_invoke: PASS (record stores, launch replays, wait_ms completes)\n");
    return 0;
}
