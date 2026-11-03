// src/metal-qwen/stubs/prefill_moe_fused_rdna4.cpp - the RDNA4 fused-experts
// surface for the Metal build (include/strata/prefill/moe_fused_rdna4.hpp).
//
// Unreachable in this build, by caller evidence: the only callers of
// strata::prefill::rdna4::* in the tree are src/core/glm_prefill.cu and
// src/ds4/ds4_iq.cu, both excluded from the Metal build (GLM/DS4 scope,
// RISKS.md §3/§7), and src/prefill/prefill.cpp never calls it - the Qwen
// prompt path's experts go through prefill::Gemm. The kernels are
// WMMA/RDNA-specific (src/prefill/moe_fused_rdna4.cu), so there is nothing
// to port. The probes answer honestly (RISKS.md §4: available/supported
// false, sizes 0, after which callers fall back by design); the work
// entries stay loud stubs.

#include "strata/prefill/moe_fused_rdna4.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::rdna4 {
bool available() { return false; }
bool supported(int, int) { return false; }
size_t act_bytes(int64_t, int64_t) { return 0; }
void quantize(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::rdna4::quantize\n");
    std::abort();
}
void experts(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds, const int* host_bounds, void* tiles, const void* xa, const int* src, void* ha, float* out, void* stream) {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::rdna4::experts\n");
    std::abort();
}
void experts_gate_up(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds, const int* host_bounds, void* tiles, const void* xa, const int* src, void* ha, void* stream) {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::rdna4::experts_gate_up\n");
    std::abort();
}
void experts_down(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds, const int* host_bounds, void* tiles, const void* ha, float* out, void* stream) {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::rdna4::experts_down\n");
    std::abort();
}
void quantize_rows(const float* x, int64_t ldx, const int* idx, int64_t rows, int64_t cols, void* xa, void* stream) {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::rdna4::quantize_rows\n");
    std::abort();
}
}  // namespace strata::prefill::rdna4

