// metal_gemm_test - the prefill GEMM parity test for the Metal backend.
//
// Two surfaces, both against a double-precision host reference computed in
// this file (the parity-oracle pattern of src/metal-qwen/tests/, standalone
// because a GEMM's operands are matrices, not the .mqfx tensor fixtures):
//   1. the cuBLAS compat surface (include/strata/metal_compat/cublas_v2.h,
//      implemented in runtime/metal_blas.cpp): cublasGemmEx in the exact
//      shape prefill::Gemm calls it (OP_T x OP_N, FP16/BF16 in, FP32 out,
//      CUBLAS_COMPUTE_32F), plus cublasSgemm in all four transpose
//      combinations and cublasSgemmStridedBatched;
//   2. prefill::Gemm itself (bf16 / f16 / native), at the Qwen prompt
//      geometry from include/strata/prefill/gemm.hpp: Y[T, N] = X[T, K] .
//      W[N, K]^T with W row-major, ldy strides, and beta = 0 / 1.
//
// Inputs are exactly representable in the input format (random BF16/FP16
// bit patterns from a small integer range, FP32 values likewise), so the
// double reference is the true product; the FP32-accumulated result must
// land within a rounding bound of it. The bound used is
//   |got - ref| <= 1e-3 + 2e-5 * sum_p |a_p * b_p|
// (per element, the standard a priori FP32 GEMM bound with the sum of
// |terms| as the scale: K * eps_fp32 * sum|terms| with K <= 10240 is under
// 1.3e-3 * sum|terms| worst-case-ordered; cblas accumulates in a blocked
// order, and the check uses the looser of the two terms so it is a real
// bound, not a tuned fit).

#include "strata/metal_compat/cublas_v2.h"
#include "strata/metal_compat/cuda_runtime.h"
#include "strata/prefill/gemm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // A value in [-2, 2] on a 1/16 grid: exactly representable in FP32,
    // FP16 and BF16 alike.
    float small() { return (float) ((int) (next() % 65) - 32) * 0.0625f; }
};

uint16_t f32_to_bf16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    x += 0x7FFFu + ((x >> 16) & 1u);
    return (uint16_t) (x >> 16);
}
uint16_t f32_to_f16(float v) {  // values on the 1/16 grid in [-2, 2] are exact
    uint32_t x;
    std::memcpy(&x, &v, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int) ((x >> 23) & 0xFF) - 112;
    uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t) sign;  // never taken for our grid except 0
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
}
float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const int exp = (int) ((h >> 10) & 0x1Fu);
    const uint32_t man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0x1F) u = sign | 0x7F800000u | (man << 13);
    else if (exp == 0) u = sign | (man ? 0x33800000u : 0u);  // inputs never subnormal here except 0
    else u = sign | ((uint32_t) (exp - 15 + 127) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void check(const char* what, const std::vector<float>& got, const std::vector<double>& ref,
           const std::vector<double>& scale) {
    double worst = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double bound = 1e-3 + 2e-5 * scale[i];
        const double err = std::fabs((double) got[i] - ref[i]);
        if (err > bound) {
            if (worst == 0.0) wi = i;
            if (err > worst) { worst = err; wi = i; }
        }
    }
    if (worst > 0.0) {
        std::printf("FAIL %s: worst err %.6g at %zu (got %.9g ref %.9g)\n", what, worst, wi, got[wi], ref[wi]);
        ++g_fail;
    } else {
        std::printf("ok   %s\n", what);
    }
}

void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc failed\n"); std::exit(2); }
    return p;
}
void dev_put(void* dst, const void* src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { std::fprintf(stderr, "H2D failed\n"); std::exit(2); }
}
void dev_get(void* dst, const void* src, size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { std::fprintf(stderr, "D2H failed\n"); std::exit(2); }
}

// Reference for the cuBLAS call shape: column-major C = alpha op(A) op(B) +
// beta C0. A/B given as double matrices in their stored (column-major)
// layouts.
void ref_gemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, double alpha,
              const std::vector<double>& A, int lda, const std::vector<double>& B, int ldb, double beta,
              const std::vector<float>& C0, int ldc, std::vector<double>& ref, std::vector<double>& scale) {
    ref.assign((size_t) m * n, 0.0);
    scale.assign((size_t) m * n, 0.0);
    auto a_at = [&](int i, int p) { return ta == CUBLAS_OP_N ? A[(size_t) i + (size_t) p * lda] : A[(size_t) p + (size_t) i * lda]; };
    auto b_at = [&](int p, int j) { return tb == CUBLAS_OP_N ? B[(size_t) p + (size_t) j * ldb] : B[(size_t) j + (size_t) p * ldb]; };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double s = 0.0, sc = 0.0;
            for (int p = 0; p < k; ++p) {
                const double t = a_at(i, p) * b_at(p, j);
                s += t;
                sc += std::fabs(t);
            }
            ref[(size_t) i * n + j] = alpha * s + beta * (double) C0[(size_t) i + (size_t) j * ldc];
            scale[(size_t) i * n + j] = std::fabs(alpha) * sc + std::fabs(beta) * std::fabs((double) C0[(size_t) i + (size_t) j * ldc]);
        }
}

// Unpack a column-major result (with leading dimension) to row-major [m][n].
std::vector<float> unpack_cm(const std::vector<float>& c, int m, int n, int ldc) {
    std::vector<float> out((size_t) m * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) out[(size_t) i * n + j] = c[(size_t) i + (size_t) j * ldc];
    return out;
}

void test_sgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, const char* name) {
    const int m = 37, n = 23, k = 41;
    const int lda = (ta == CUBLAS_OP_N ? m : k) + 3, ldb = (tb == CUBLAS_OP_N ? k : n) + 2, ldc = m + 5;
    const int arows = ta == CUBLAS_OP_N ? m : k, acols = ta == CUBLAS_OP_N ? k : m;
    const int brows = tb == CUBLAS_OP_N ? k : n, bcols = tb == CUBLAS_OP_N ? n : k;
    Rng rng(1000 + (unsigned) ta * 10 + (unsigned) tb);
    std::vector<double> A((size_t) lda * acols, 0.0), B((size_t) ldb * bcols, 0.0);
    std::vector<float> Af(A.size()), Bf(B.size()), C((size_t) ldc * n, 0.0f), C0((size_t) ldc * n);
    for (int c = 0; c < acols; ++c)
        for (int r = 0; r < arows; ++r) { const float v = rng.small(); A[(size_t) r + (size_t) c * lda] = v; Af[(size_t) r + (size_t) c * lda] = v; }
    for (int c = 0; c < bcols; ++c)
        for (int r = 0; r < brows; ++r) { const float v = rng.small(); B[(size_t) r + (size_t) c * ldb] = v; Bf[(size_t) r + (size_t) c * ldb] = v; }
    for (auto& v : C) v = rng.small();
    C0 = C;
    const float alpha = 1.25f, beta = -0.5f;
    if (cublasSgemm(h, ta, tb, m, n, k, &alpha, Af.data(), lda, Bf.data(), ldb, &beta, C.data(), ldc) != CUBLAS_STATUS_SUCCESS) {
        std::printf("FAIL %s: cublasSgemm status\n", name);
        ++g_fail;
        return;
    }
    std::vector<double> ref, scale;
    ref_gemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C0, ldc, ref, scale);
    check(name, unpack_cm(C, m, n, ldc), ref, scale);
}

void test_sgemm_batched(cublasHandle_t h) {
    const int m = 9, n = 11, k = 13, batch = 3;
    Rng rng(77);
    std::vector<float> A((size_t) m * k * batch), B((size_t) k * n * batch), C((size_t) m * n * batch), C0;
    for (auto& v : A) v = rng.small();
    for (auto& v : B) v = rng.small();
    for (auto& v : C) v = rng.small();
    C0 = C;
    const float alpha = 1.0f, beta = 0.25f;
    const cublasStatus_t s = cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A.data(), m,
                                                       (long long) m * k, B.data(), k, (long long) k * n, &beta,
                                                       C.data(), m, (long long) m * n, batch);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("FAIL sgemm_batched: status %d\n", (int) s); ++g_fail; return; }
    std::vector<double> ref((size_t) m * n * batch), scale((size_t) m * n * batch);
    for (int b = 0; b < batch; ++b) {
        std::vector<double> Ad(A.begin() + b * m * k, A.begin() + (b + 1) * m * k), Bd(B.begin() + b * k * n, B.begin() + (b + 1) * k * n);
        std::vector<double> r1, s1;
        std::vector<float> c0b(C0.begin() + b * m * n, C0.begin() + (b + 1) * m * n);
        ref_gemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0, Ad, m, Bd, k, 0.25, c0b, m, r1, s1);
        std::copy(r1.begin(), r1.end(), ref.begin() + b * m * n);
        std::copy(s1.begin(), s1.end(), scale.begin() + b * m * n);
    }
    std::vector<float> got;
    for (int b = 0; b < batch; ++b) {
        const std::vector<float> cb(C.begin() + b * m * n, C.begin() + (b + 1) * m * n);
        const std::vector<float> ub = unpack_cm(cb, m, n, m);
        got.insert(got.end(), ub.begin(), ub.end());
    }
    check("sgemm_strided_batched", got, ref, scale);
}

// The exact call prefill::Gemm makes, spelled out at the surface: OP_T x
// OP_N with the weights as the A operand.
void test_gemmex_prefill_shape(cublasHandle_t h) {
    const int T = 19, N = 45, K = 33;  // m = N, n = T, k = K in cuBLAS terms
    Rng rng(4242);
    std::vector<uint16_t> W((size_t) N * K), X((size_t) T * K);
    std::vector<double> Wd(W.size()), Xd(X.size());
    for (size_t i = 0; i < W.size(); ++i) { const float v = rng.small(); W[i] = f32_to_bf16(v); Wd[i] = bf16_to_f32(W[i]); }
    for (size_t i = 0; i < X.size(); ++i) { const float v = rng.small(); X[i] = f32_to_bf16(v); Xd[i] = bf16_to_f32(X[i]); }
    // cuBLAS column-major buffers: W stored [K][N] col-major (lda = K) is the
    // transpose of the row-major [N][K] weight; X stored [K][T] col-major.
    std::vector<float> Y((size_t) N * T, -7.0f);
    const float alpha = 1.0f, beta = 0.0f;
    // Pass the row-major arrays directly, as gemm.cu does: its W pointer is
    // the row-major [N][K] array read as col-major [K][N] (the transpose),
    // its X pointer the row-major [T][K] array read as col-major [K][T].
    const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, W.data(), CUDA_R_16BF, K,
                                          X.data(), CUDA_R_16BF, K, &beta, Y.data(), CUDA_R_32F, N,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) { std::printf("FAIL gemmex_prefill_shape: status %d\n", (int) s); ++g_fail; return; }
    // Reference: Y[t][n] = sum_k X[t][k] * W[n][k], read back from the
    // col-major [N][T] result.
    std::vector<double> ref((size_t) T * N), scale((size_t) T * N);
    std::vector<float> got((size_t) T * N);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n) {
            double sum = 0.0, sc = 0.0;
            for (int k = 0; k < K; ++k) {
                const double term = Xd[(size_t) t * K + k] * Wd[(size_t) n * K + k];
                sum += term;
                sc += std::fabs(term);
            }
            ref[(size_t) t * N + n] = sum;
            scale[(size_t) t * N + n] = sc;
            got[(size_t) t * N + n] = Y[(size_t) n + (size_t) t * N];
        }
    check("gemmex_prefill_shape (bf16, OP_T x OP_N)", got, ref, scale);
}

// prefill::Gemm at prompt geometry: several (T, N, K) from the pack
// (n_embd 2560; expert gate/up 1280 x 2560; expert down 2560 x 640; the
// GDN qkv projection 10240 x 2560 at small T), bf16 and f16, beta 0 and 1,
// and one ldy-strided case.
void test_gemm_class() {
    struct Shape { int64_t T, N, K; const char* name; };
    const Shape shapes[] = {
        {1, 2560, 2560, "gemm 1x2560x2560"},
        {7, 2560, 2560, "gemm 7x2560x2560"},
        {64, 1280, 2560, "gemm expert gate/up 64x1280x2560"},
        {64, 2560, 640, "gemm expert down 64x2560x640"},
        {3, 10240, 2560, "gemm qkv 3x10240x2560"},
        {33, 640, 320, "gemm odd 33x640x320"},
    };
    for (const Shape& sh : shapes) {
        for (int fmt = 0; fmt < 2; ++fmt) {          // 0 bf16, 1 f16
            for (float beta : {0.0f, 1.0f}) {
                strata::prefill::Gemm g;
                std::string err;
                // scratch for native-sized dequant is not needed here; give
                // the bf16/f16 paths a modest scratch via init().
                if (!g.init(nullptr, 1 << 16, err)) { std::printf("FAIL %s: init: %s\n", sh.name, err.c_str()); ++g_fail; return; }
                Rng rng(9000 + (unsigned) sh.T * 131 + (unsigned) sh.N + fmt * 17 + (beta != 0.0f ? 5 : 0));
                std::vector<float> Xf((size_t) sh.T * sh.K), Wf((size_t) sh.N * sh.K);
                for (auto& v : Xf) v = rng.small();
                for (auto& v : Wf) v = rng.small();
                std::vector<uint16_t> X((size_t) sh.T * sh.K), W((size_t) sh.N * sh.K);
                for (size_t i = 0; i < X.size(); ++i) X[i] = fmt ? f32_to_f16(Xf[i]) : f32_to_bf16(Xf[i]);
                for (size_t i = 0; i < W.size(); ++i) W[i] = fmt ? f32_to_f16(Wf[i]) : f32_to_bf16(Wf[i]);
                auto dec = [&](uint16_t b) { return fmt ? (double) f16_to_f32(b) : (double) bf16_to_f32(b); };
                const int64_t ldy = sh.N + 4;
                std::vector<float> Y0((size_t) sh.T * ldy);
                for (auto& v : Y0) v = rng.small();
                void* dX = dev_alloc(X.size() * 2);
                void* dW = dev_alloc(W.size() * 2);
                void* dY = dev_alloc(Y0.size() * 4);
                dev_put(dX, X.data(), X.size() * 2);
                dev_put(dW, W.data(), W.size() * 2);
                dev_put(dY, Y0.data(), Y0.size() * 4);
                if (fmt) g.f16((const uint16_t*) dX, (const uint16_t*) dW, (float*) dY, sh.T, sh.N, sh.K, ldy, beta);
                else g.bf16((const uint16_t*) dX, (const uint16_t*) dW, (float*) dY, sh.T, sh.N, sh.K, ldy, beta);
                if (cudaStreamSynchronize(nullptr) != cudaSuccess) { std::printf("FAIL %s: sync\n", sh.name); ++g_fail; return; }
                std::vector<float> Y(Y0.size());
                dev_get(Y.data(), dY, Y.size() * 4);
                cudaFree(dX);
                cudaFree(dW);
                cudaFree(dY);
                std::vector<double> ref((size_t) sh.T * sh.N), scale((size_t) sh.T * sh.N);
                std::vector<float> got((size_t) sh.T * sh.N);
                for (int64_t t = 0; t < sh.T; ++t)
                    for (int64_t n = 0; n < sh.N; ++n) {
                        double s = 0.0, sc = 0.0;
                        for (int64_t k = 0; k < sh.K; ++k) {
                            const double term = dec(X[(size_t) t * sh.K + k]) * dec(W[(size_t) n * sh.K + k]);
                            s += term;
                            sc += std::fabs(term);
                        }
                        ref[(size_t) t * sh.N + n] = s + (double) beta * Y0[(size_t) t * ldy + n];
                        scale[(size_t) t * sh.N + n] = sc + std::fabs((double) beta * Y0[(size_t) t * ldy + n]);
                        got[(size_t) t * sh.N + n] = Y[(size_t) t * ldy + n];
                    }
                char label[128];
                std::snprintf(label, sizeof label, "%s %s beta=%g ldy", sh.name, fmt ? "f16" : "bf16", beta);
                check(label, got, ref, scale);
            }
        }
    }
}

// Gemm::native: Q8_0 blocks dequantized into the scratch, then the f16 GEMM.
void test_gemm_native() {
    const int64_t T = 5, N = 64, K = 64;  // two Q8_0 blocks per row
    strata::prefill::Gemm g;
    std::string err;
    if (!g.init(nullptr, N * K, err)) { std::printf("FAIL native: init: %s\n", err.c_str()); ++g_fail; return; }
    Rng rng(31337);
    // Q8_0 block: f16 scale + 32 int8 codes (34 bytes).
    std::vector<uint8_t> blocks((size_t) N * (K / 32) * 34);
    std::vector<double> Wd((size_t) N * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t b = 0; b < K / 32; ++b) {
            uint8_t* blk = blocks.data() + ((size_t) r * (K / 32) + b) * 34;
            const float d = 0.03125f * (1 + (int) (rng.next() % 4));
            const uint16_t db = f32_to_f16(d);
            blk[0] = (uint8_t) (db & 0xFF);
            blk[1] = (uint8_t) (db >> 8);
            for (int i = 0; i < 32; ++i) {
                const int q = (int) (rng.next() % 33) - 16;
                blk[2 + i] = (uint8_t) (int8_t) q;
                Wd[(size_t) r * K + b * 32 + i] = (double) q * (double) f16_to_f32(db);
            }
        }
    std::vector<uint16_t> X((size_t) T * K);
    std::vector<double> Xd(X.size());
    for (size_t i = 0; i < X.size(); ++i) { const float v = rng.small(); X[i] = f32_to_f16(v); Xd[i] = f16_to_f32(X[i]); }
    void* dX = dev_alloc(X.size() * 2);
    void* dW = dev_alloc(blocks.size());
    void* dY = dev_alloc((size_t) T * N * 4);
    dev_put(dX, X.data(), X.size() * 2);
    dev_put(dW, blocks.data(), blocks.size());
    constexpr int kQ8_0 = 8;
    g.native((const uint16_t*) dX, kQ8_0, dW, (float*) dY, T, N, K);
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) { std::printf("FAIL native: sync\n"); ++g_fail; return; }
    std::vector<float> Y((size_t) T * N);
    dev_get(Y.data(), dY, Y.size() * 4);
    cudaFree(dX);
    cudaFree(dW);
    cudaFree(dY);
    std::vector<double> ref((size_t) T * N), scale((size_t) T * N);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) {
            double s = 0.0, sc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                const double term = Xd[(size_t) t * K + k] * Wd[(size_t) n * K + k];
                s += term;
                sc += std::fabs(term);
            }
            ref[(size_t) t * N + n] = s;
            scale[(size_t) t * N + n] = sc;
        }
    check("gemm native Q8_0 5x64x64", Y, ref, scale);
}

}  // namespace

int main() {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::printf("FAIL cublasCreate\n"); return 1; }
    test_sgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, "sgemm N x N");
    test_sgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, "sgemm T x N");
    test_sgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, "sgemm N x T");
    test_sgemm(h, CUBLAS_OP_T, CUBLAS_OP_T, "sgemm T x T");
    test_sgemm_batched(h);
    test_gemmex_prefill_shape(h);
    cublasDestroy(h);
    test_gemm_class();
    test_gemm_native();
    std::printf("metal_gemm_test: %s\n", g_fail == 0 ? "all passed" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
