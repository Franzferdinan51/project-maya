// src/hy4-metal/metal_hy4_matvec_test.cpp - the Hy4 Metal port's stage 2 gate:
// the four weight formats of the Maya-S trunk - IQ2_XXS and IQ3_XXS (the routed
// experts), Q6_K (attention + head), Q8_0 (the shared expert) - multiplied on the
// Metal backend against the same bytes on the CPU backend.
//
// The stage 2 inventory (port plan SS7's open question) came out in ggml's favour:
// the ggml this tree already links for the GLM/MiniMax engines carries all four
// formats' Metal kernels (ggml-metal's mul_mv/mul_mm + dequantize tables), and the
// vendored third_party/ggml has none - so the port's forward is built as ggml
// graphs over that backend (the fork's GLM/MM3 pattern), and this test is the
// kernel layer's parity gate.
//
// Two checks a type, because the two backends' arithmetic differs by design:
// - the dequant probe: y = W . e_j extracts one dequantized column with a single
//   nonzero product (the basis vector is exact in the CPU path's Q8_0 activation
//   quantization too), so codebook, scales and signs must agree to F32 rounding -
//   this pins the semantics exactly;
// - the matvec: random activations, where the CPU path quantizes the activation
//   to Q8_0 (the same int8-activation arithmetic upstream's CUDA token path runs:
//   hy4_model.hpp's "GEMVs over int8 inputs") and the Metal path stages its own
//   rounding - agreement is statistical, bounded at 1% relative L2, a level a
//   wrong codebook or scale would blow past by orders of magnitude.
//
// The CPU backend is the reference: its IQ dequant is the same semantics the CUDA
// engine's ds4_iq CPU lane (src/ds4/ds4_iq_cpu.cpp) mirrors.

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

// deterministic fixture data (an LCG; no rand(), no files)
struct Rng {
    uint32_t s;
    float next() {   // uniform in [-1, 1)
        s = s * 1664525u + 1013904223u;
        return (float) ((s >> 8) & 0xFFFFFF) / (float) 0x800000 - 1.0f;
    }
};

bool run(ggml_backend_t be, ggml_type type, const std::vector<uint8_t>& wq, const std::vector<float>& x,
         int64_t K, int64_t N, std::vector<float>& out) {
    ggml_init_params ip{16 * 1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor* w = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_tensor* v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor* y = ggml_mul_mat(ctx, w, v);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    bool ok = buf != nullptr;
    if (ok) {
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(v, x.data(), 0, x.size() * sizeof(float));
        ggml_cgraph* gr = ggml_new_graph(ctx);
        ggml_build_forward_expand(gr, y);
        ok = ggml_backend_graph_compute(be, gr) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out.assign((size_t) N, 0.0f);
        ggml_backend_tensor_get(y, out.data(), 0, (size_t) N * sizeof(float));
    }
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

std::vector<uint8_t> quantize(ggml_type type, const std::vector<float>& wf, int64_t K, int64_t N) {
    std::vector<uint8_t> wq(ggml_row_size(type, K) * (size_t) N);
    // the IQ quantizers assert on a missing importance matrix in this ggml; a flat one
    // (a float a column, the same for every row) is the neutral choice - and immaterial
    // here either way: both backends multiply the same quantized bytes.
    const std::vector<float> imat((size_t) K, 1.0f);
    const size_t made = ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N, K, imat.data());
    return made == wq.size() ? wq : std::vector<uint8_t>{};
}

int probe(ggml_backend_t cpu, ggml_backend_t gpu, ggml_type type) {
    const int64_t K = 256, N = 2;
    Rng rng{0x44u + (uint32_t) type * 7919u};
    std::vector<float> wf((size_t) K * N);
    for (float& f : wf) f = rng.next();
    const std::vector<uint8_t> wq = quantize(type, wf, K, N);
    CHECK(!wq.empty());
    const int64_t cols[] = {0, 1, 37, 128, 255};
    double worst = 0;
    for (int64_t j : cols) {
        std::vector<float> x((size_t) K, 0.0f);
        x[(size_t) j] = 1.0f;
        std::vector<float> ref, got;
        CHECK(run(cpu, type, wq, x, K, N, ref));
        CHECK(run(gpu, type, wq, x, K, N, got));
        for (int64_t r = 0; r < N; ++r) {
            const double d = std::fabs((double) got[(size_t) r] - ref[(size_t) r]);
            worst = std::max(worst, d);
            CHECK(d < 1e-4 * (1.0 + std::fabs((double) ref[(size_t) r])));
        }
    }
    std::fprintf(stderr, "%-8s dequant probe: worst |diff| %.2e\n", ggml_type_name(type), worst);
    return 0;
}

int matvec(ggml_backend_t cpu, ggml_backend_t gpu, ggml_type type, int64_t K, int64_t N) {
    Rng rng{0x44u + (uint32_t) type * 7919u + (uint32_t) K};
    std::vector<float> wf((size_t) K * N), x((size_t) K);
    for (float& f : wf) f = rng.next();
    for (float& f : x) f = rng.next();
    const std::vector<uint8_t> wq = quantize(type, wf, K, N);
    CHECK(!wq.empty());

    std::vector<float> ref, got;
    CHECK(run(cpu, type, wq, x, K, N, ref));
    CHECK(run(gpu, type, wq, x, K, N, got));
    CHECK(ref.size() == got.size());

    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        num += d * d;
        den += (double) ref[i] * ref[i];
    }
    const double rel_l2 = std::sqrt(num / std::max(den, 1e-12));
    std::fprintf(stderr, "%-8s matvec K=%lld N=%lld: rel L2 %.2e\n", ggml_type_name(type),
                 (long long) K, (long long) N, rel_l2);
    CHECK(rel_l2 < 1e-2);
    return 0;
}

}  // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!cpu || !gpu) {
        std::fprintf(stderr, "a backend is missing (cpu %p, gpu %p)\n", (void*) cpu, (void*) gpu);
        return 1;
    }
    const ggml_type types[] = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};
    for (ggml_type t : types) {
        if (probe(cpu, gpu, t)) return 1;
        if (matvec(cpu, gpu, t, 1024, 8)) return 1;   // the prompt shape
        if (matvec(cpu, gpu, t, 256, 8)) return 1;    // the decode shape (one block-row)
    }
    std::fprintf(stderr, "metal_hy4_matvec_test: PASS\n");
    return 0;
}
