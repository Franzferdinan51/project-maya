// src/metal-qwen/runtime/metal_blas.cpp - the cuBLAS-shaped surface of
// include/strata/metal_compat/cublas_v2.h, implemented host-side over Apple
// Accelerate (cblas). See src/metal-qwen/PREFILL-NOTES.md for the surface
// inventory and the row/column-major convention reasoning.
//
// Semantics, matched to what the callers (src/prefill/gemm.cu) rely on:
//  * Column-major math, exactly as cuBLAS defines it:
//    C(m x n) = alpha * op(A)(m x k) * op(B)(k x n) + beta * C, with A, B, C
//    column-major at leading dimensions lda/ldb/ldc.
//  * CUBLAS_COMPUTE_32F: FP16/BF16 inputs are converted to FP32 exactly
//    (both formats widen to FP32 without loss), the product accumulates in
//    FP32 through cblas_sgemm, C is FP32. That is the same contract the
//    CUDA call makes (tensor-core inputs, FP32 accumulate and output).
//  * Stream: the Metal runtime's buffers are host-visible shared memory and
//    stream work is batched into the stream's current command buffer, in
//    stream order (src/metal-qwen/BATCH-NOTES.md). A host GEMM cannot be encoded into that queue, so the
//    implementation synchronizes the handle's stream before computing and
//    computes synchronously on the calling thread: every prior launch on
//    the stream has completed before the first input is read, and no later
//    launch on any stream is issued by this thread until the results are
//    written. That is stricter ordering than the async CUDA call provides,
//    never looser, for the single-threaded-per-stream way the prompt path
//    drives it.
//
// The generic path materializes op(A) and op(B) as FP32 row-major matrices
// and C as an FP32 row-major matrix, calls cblas_sgemm once in row-major,
// and scatters C back through ldc. Materializing (instead of the
// transpose-trick single call) keeps every transpose/leading-dimension
// combination on one obviously-correct path; the conversion cost is the
// known bring-up trade-off, noted in PREFILL-NOTES.md.

#include "strata/metal_compat/cublas_v2.h"

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

struct cublasContext {
    cudaStream_t stream = nullptr;
    void* workspace = nullptr;   // accepted, unused: the host path allocates
    size_t workspace_bytes = 0;  // its own scratch (see the header note)
    cublasMathMode_t math_mode = CUBLAS_DEFAULT_MATH;
};

namespace {

float load_elem(const void* base, cudaDataType type, size_t idx) {
    switch (type) {
        case CUDA_R_32F: return static_cast<const float*>(base)[idx];
        case CUDA_R_16F: {
            // IEEE half -> float, exact (every half value is a float value).
            const auto* p = static_cast<const uint16_t*>(base);
            const uint32_t h = p[idx];
            const uint32_t sign = (h & 0x8000u) << 16;
            const int exp = (int) ((h >> 10) & 0x1Fu);
            const uint32_t man = h & 0x3FFu;
            uint32_t u;
            if (exp == 0x1F) {
                u = sign | 0x7F800000u | (man << 13);
            } else if (exp == 0) {
                if (man == 0) {
                    u = sign;
                } else {  // subnormal half: normalize
                    int e = -1;
                    uint32_t mm = man;
                    while ((mm & 0x400u) == 0) { mm <<= 1; ++e; }
                    u = sign | ((uint32_t) (127 - 15 - e) << 23) | ((mm & 0x3FFu) << 13);
                }
            } else {
                u = sign | ((uint32_t) (exp - 15 + 127) << 23) | (man << 13);
            }
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }
        case CUDA_R_16BF: {
            const auto* p = static_cast<const uint16_t*>(base);
            uint32_t u = (uint32_t) p[idx] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }
        default: return 0.0f;
    }
}

bool supported_in(cudaDataType t) { return t == CUDA_R_32F || t == CUDA_R_16F || t == CUDA_R_16BF; }

// One column-major GEMM through the materialized row-major path. A/B element
// types may differ (GemmEx allows it; the caller never mixes, but the
// surface does not need the restriction).
cublasStatus_t gemm_core(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                         int k, float alpha, const void* A, cudaDataType at, int lda, const void* B, cudaDataType bt,
                         int ldb, float beta, float* C, int ldc) {
    if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    const int lda_min = (transa == CUBLAS_OP_N) ? m : k;
    const int ldb_min = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (lda_min > 1 ? lda_min : 1) || ldb < (ldb_min > 1 ? ldb_min : 1) || ldc < (m > 1 ? m : 1))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (k == 0) {  // C = beta * C
        if (beta == 0.0f) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) C[(size_t) i + (size_t) j * ldc] = 0.0f;
        } else if (beta != 1.0f) {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) C[(size_t) i + (size_t) j * ldc] *= beta;
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    if (!A || !B || !C) return CUBLAS_STATUS_INVALID_VALUE;

    // The stream's prior work must be complete before host reads/writes of
    // shared buffers (see the file header). A null handle stream means the
    // default stream.
    if (cudaStreamSynchronize(handle->stream) != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;

    std::vector<float> ma((size_t) m * k), mb((size_t) k * n), mc((size_t) m * n);
    // op(A) as row-major [m][k]: element (i, p) is A(i, p) in column-major
    // terms: transa N -> A stored [lda][k] column-major, A(i,p) = A[i + p*lda];
    // T -> A stored [lda][m] column-major holding A^T, op(A)(i,p) = A[p + i*lda];
    // C (conjugate transpose) is the transpose for real types.
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p) {
            const size_t src = (transa == CUBLAS_OP_N) ? (size_t) i + (size_t) p * lda
                                                       : (size_t) p + (size_t) i * lda;
            ma[(size_t) i * k + p] = load_elem(A, at, src);
        }
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j) {
            const size_t src = (transb == CUBLAS_OP_N) ? (size_t) p + (size_t) j * ldb
                                                       : (size_t) j + (size_t) p * ldb;
            mb[(size_t) p * n + j] = load_elem(B, bt, src);
        }
    if (beta != 0.0f)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) mc[(size_t) i * n + j] = C[(size_t) i + (size_t) j * ldc];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ma.data(), k, mb.data(), n, beta,
                mc.data(), n);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) C[(size_t) i + (size_t) j * ldc] = mc[(size_t) i * n + j];
    return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = new (std::nothrow) cublasContext();
    return *handle ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_ALLOC_FAILED;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId) {
    if (!handle || !streamId) return CUBLAS_STATUS_INVALID_VALUE;
    *streamId = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    handle->workspace = workspace;
    handle->workspace_bytes = workspaceSizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMathMode_t mode) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    // Accepted and recorded. CUBLAS_PEDANTIC_MATH and CUBLAS_DEFAULT_MATH are
    // the same computation here: the product is always FP32-accumulated
    // (there is no TF32/FP16-accumulate fast path to disallow).
    handle->math_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                            int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    (void) algo;  // one algorithm on this backend
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
    if (computeType != CUBLAS_COMPUTE_32F && computeType != CUBLAS_COMPUTE_32F_PEDANTIC)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (!supported_in(Atype) || !supported_in(Btype) || Ctype != CUDA_R_32F) return CUBLAS_STATUS_NOT_SUPPORTED;
    return gemm_core(handle, transa, transb, m, n, k, *static_cast<const float*>(alpha), A, Atype, lda, B, Btype,
                     ldb, *static_cast<const float*>(beta), static_cast<float*>(C), ldc);
}

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
                           const float* beta, float* C, int ldc) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
    return gemm_core(handle, transa, transb, m, n, k, *alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, *beta, C, ldc);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const float* alpha, const float* A, int lda,
                                         long long int strideA, const float* B, int ldb, long long int strideB,
                                         const float* beta, float* C, int ldc, long long int strideC, int batchCount) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
    if (batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        const cublasStatus_t s =
            gemm_core(handle, transa, transb, m, n, k, *alpha, A ? A + (size_t) b * strideA : nullptr, CUDA_R_32F,
                      lda, B ? B + (size_t) b * strideB : nullptr, CUDA_R_32F, ldb, *beta,
                      C ? C + (size_t) b * strideC : nullptr, ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"
