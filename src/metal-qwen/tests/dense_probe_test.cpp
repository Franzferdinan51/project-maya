// src/metal-qwen/tests/dense_probe_test.cpp - VM-runnable checks for the dense
// workstream's probe / pure-host entries (RISKS.md §4). These entries are host
// arithmetic and flag state ported from the CUDA TUs, so they are testable without a
// GPU: link this file against the metal-qwen TUs for native_mmvq, iq_kernels,
// dequant_bf16 and ple and run it. The cvec flag pair is covered in cvec_probe_test
// terms by the same pattern (it needs the compat runtime, so it runs on the Mac).
//
// Expected values are the CUDA TUs' own tables, written out independently here:
//   native_mmvq.cu native_mmvq_weight_bytes / native_q8_1_bytes / *_supported
//   iq_kernels.cu  iq_row_bytes / native_expert_layout / native_expert_scratch_bytes
//   dequant_bf16.cu geometry()
//   ple.cu         ple_block_scratch_bytes()

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/ple.hpp"

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
template <typename A, typename B>
void eq(A got, B want, const char* what) {
    if (!(got == want)) {
        std::fprintf(stderr, "FAIL: %s: got %lld want %lld\n", what, (long long) got, (long long) want);
        ++failures;
    }
}
}  // namespace

int main() {
    using namespace strata::kernels;

    // native_mmvq flag pair: default true, settable, per the CUDA TU.
    check(native_mmvq_multi_exact(), "multi_exact default true");
    native_mmvq_set_multi_exact(false);
    check(!native_mmvq_multi_exact(), "multi_exact set false");
    native_mmvq_set_multi_exact(true);
    check(native_mmvq_multi_exact(), "multi_exact set true");

    // native_q8_1_bytes: 36 bytes per 32 elements per column.
    eq(native_q8_1_bytes(2560), std::size_t(80 * 36), "q8_1_bytes 2560x1");
    eq(native_q8_1_bytes(2560, 4), std::size_t(4 * 80 * 36), "q8_1_bytes 2560x4");
    bool threw = false;
    try {
        (void) native_q8_1_bytes(100);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "q8_1_bytes rejects n_in % 32");
    threw = false;
    try {
        (void) native_q8_1_bytes(2560, 9);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "q8_1_bytes rejects ncols > 8");

    // native_mmvq_supported: the CUDA list (19 excluded, 42 included).
    for (int t : {2, 6, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42})
        check(native_mmvq_supported(t), "mmvq_supported true case");
    for (int t : {0, 1, 7, 19, 24, 43}) check(!native_mmvq_supported(t), "mmvq_supported false case");

    // native_mmvq_weight_bytes: rows of blocks.
    eq(native_mmvq_weight_bytes(42, 2560, 640), std::size_t(640 * 40 * 18), "weight_bytes Q2_0");
    eq(native_mmvq_weight_bytes(2, 2560, 640), std::size_t(640 * 80 * 18), "weight_bytes Q4_0");
    eq(native_mmvq_weight_bytes(14, 2560, 10), std::size_t(10 * 10 * 210), "weight_bytes Q6_K");
    eq(native_mmvq_weight_bytes(16, 2560, 10), std::size_t(10 * 10 * 66), "weight_bytes IQ2_XXS via iq_row_bytes");
    threw = false;
    try {
        (void) native_mmvq_weight_bytes(99, 2560, 1);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "weight_bytes rejects unknown type");

    // iq_supported / iq_row_bytes.
    for (int t : {11, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42}) check(iq_supported(t), "iq_supported true case");
    for (int t : {2, 6, 8, 12, 13, 14}) check(!iq_supported(t), "iq_supported false case");
    eq(iq_row_bytes(42, 2560), std::size_t(40 * 18), "iq_row_bytes Q2_0");
    // iq_row_bytes for every type, per 256 elements, against the ggml sizeof table
    // (ggml-common.h static_asserts): the full probe surface, not one sample.
    eq(iq_row_bytes(2, 256), std::size_t(8 * 18), "iq_row_bytes Q4_0");
    eq(iq_row_bytes(6, 256), std::size_t(8 * 22), "iq_row_bytes Q5_0");
    eq(iq_row_bytes(8, 256), std::size_t(8 * 34), "iq_row_bytes Q8_0");
    eq(iq_row_bytes(11, 256), std::size_t(110), "iq_row_bytes Q3_K");
    eq(iq_row_bytes(12, 256), std::size_t(144), "iq_row_bytes Q4_K");
    eq(iq_row_bytes(13, 256), std::size_t(176), "iq_row_bytes Q5_K");
    eq(iq_row_bytes(14, 256), std::size_t(210), "iq_row_bytes Q6_K");
    eq(iq_row_bytes(16, 256), std::size_t(66), "iq_row_bytes IQ2_XXS");
    eq(iq_row_bytes(17, 256), std::size_t(74), "iq_row_bytes IQ2_XS");
    eq(iq_row_bytes(18, 256), std::size_t(98), "iq_row_bytes IQ3_XXS");
    eq(iq_row_bytes(19, 256), std::size_t(50), "iq_row_bytes IQ1_S");
    eq(iq_row_bytes(20, 256), std::size_t(8 * 18), "iq_row_bytes IQ4_NL");
    eq(iq_row_bytes(21, 256), std::size_t(110), "iq_row_bytes IQ3_S");
    eq(iq_row_bytes(22, 256), std::size_t(82), "iq_row_bytes IQ2_S");
    eq(iq_row_bytes(23, 256), std::size_t(136), "iq_row_bytes IQ4_XS");
    eq(iq_row_bytes(29, 256), std::size_t(56), "iq_row_bytes IQ1_M");
    eq(iq_row_bytes(42, 256), std::size_t(4 * 18), "iq_row_bytes Q2_0 /256");
    eq(iq_row_bytes(23, 2560), std::size_t(10 * 136), "iq_row_bytes IQ4_XS");
    eq(iq_row_bytes(17, 2560), std::size_t(10 * 74), "iq_row_bytes IQ2_XS");
    eq(iq_row_bytes(7, 2560), std::size_t(0), "iq_row_bytes unknown = 0");
    check(!gemm_iq_f16_grouped_supported(0), "gemm_iq_f16_grouped_supported honest false");

    // native_expert_layout / scratch: [gate | up | down] blob arithmetic.
    const NativeExpertLayout L = native_expert_layout(42, 42, 2560, 640);
    eq(L.gu_row, std::size_t(40 * 18), "expert gu_row");
    eq(L.d_row, std::size_t(10 * 18), "expert d_row");
    eq(L.up_off, std::size_t(640) * L.gu_row, "expert up_off");
    eq(L.down_off, 2 * L.up_off, "expert down_off");
    eq(L.bytes, L.down_off + std::size_t(2560) * L.d_row, "expert bytes");
    eq(native_expert_scratch_bytes(4, 640),
       3 * ((std::size_t(4 * 640 * 4) + 255) & ~(std::size_t) 255) +
           ((std::size_t(4 * 20 * 36) + 255) & ~(std::size_t) 255),
       "expert scratch bytes");

    // dequant_bf16_supported geometry list.
    for (int t : {2, 6, 8, 11, 12, 13, 14, 20, 23, 42}) check(dequant_bf16_supported(t), "dqbf16 true case");
    for (int t : {0, 16, 17, 19, 29}) check(!dequant_bf16_supported(t), "dqbf16 false case");

    // PLE flag pair + scratch size (ple.cu formula at NG_HC_DIM 10240 etc.).
    check(!ple_native_postops_enabled(), "ple postops default false");
    ple_set_native_postops(true);
    check(ple_native_postops_enabled(), "ple postops set true");
    ple_set_native_postops(false);
    ple_set_native_bf16(true);  // setter must not abort; no getter in the header
    const std::size_t f = (5 * 10240 + 2560 + 4) * sizeof(float);
    const std::size_t q = (2560 / 32) * 34;
    const std::size_t e = 2560 * sizeof(uint16_t);
    eq(ple_block_scratch_bytes(), ((f + 15) & ~(std::size_t) 15) + ((q + 15) & ~(std::size_t) 15) + e + 256,
       "ple scratch bytes");
    check(ple_block_available(), "ple_block_available true on Metal");

    if (failures) {
        std::fprintf(stderr, "dense_probe_test: %d failures\n", failures);
        return 1;
    }
    std::printf("dense_probe_test: all checks passed\n");
    return 0;
}
