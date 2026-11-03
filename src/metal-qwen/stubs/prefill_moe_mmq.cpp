// src/metal-qwen/stubs/prefill_moe_mmq.cpp - the MMQ complement for the
// Metal build (include/strata/prefill/moe_mmq.hpp).
//
// Reachability, from the sources: the Metal build does not define
// STRATA_PREFILL_MMQ (the strata_mmq library is CUDA/HIP-only in
// CMakeLists.txt), so src/prefill/prefill.cpp compiles its own no-MMQ
// fallback block for the entries it uses (built, supported, matrix_bytes,
// q8_bytes, quantize, Context, run, gather_native, gather_strata_q2,
// swiglu, iota - all honest: built() is false, so prefill's mmq_plan()
// disables the MMQ path and the experts go through prefill::Gemm). Those
// entries are therefore NOT defined here (defining them twice would be a
// duplicate symbol the day this TU is pulled in).
//
// What remains of the header is the complement below. Its only callers in
// the tree are src/prefill/moe_mmq.cu itself (not compiled here) and the
// GLM/DS4 translation units excluded from this build, so nothing in the
// Metal build reaches it today. The probes answer honestly (the fused
// GLU product and the ggml dequantizer do not exist in this build); the
// work entries stay loud stubs, per the stage-1 rule.

#include "strata/prefill/moe_mmq.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {

void pool_slab(int, void*, size_t) {
    // The MMQ kernels' scratch pool does not exist in this build; there is
    // nothing to lend. (No caller in the Metal build - see the file header.)
}

size_t q8_block_bytes() {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::mmq::q8_block_bytes\n");
    std::abort();
}
bool glu_supported(int, int64_t) { return false; }
void Context::run_glu(const Product&, float, void*) {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::mmq::Context::run_glu\n");
    std::abort();
}
bool to_f16_supported(int) { return false; }
bool to_f16(int, const void*, uint16_t*, int64_t, void*) {
    std::fprintf(stderr, "strata metal: unimplemented kernel entry: strata::prefill::mmq::to_f16\n");
    std::abort();
}

}  // namespace strata::prefill::mmq
