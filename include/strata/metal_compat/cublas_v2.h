// include/strata/metal_compat/cublas_v2.h - the cuBLAS-shaped surface for the
// Metal backend, shaped like include/strata/hip_compat/cublas_v2.h shapes it
// for HIP (real cuBLAS names and enum values; the implementation lives in
// src/metal-qwen/runtime/metal_blas.cpp, host-side over Accelerate).
//
// The surface is exactly what the host translation units compiled in the
// Metal build call (inventory in src/metal-qwen/PREFILL-NOTES.md): the only
// consumer is src/prefill/gemm.cu (compiled as C++ in this build), which
// uses cublasCreate / cublasDestroy / cublasSetStream / cublasSetWorkspace /
// cublasSetMathMode / cublasGemmEx. cublasSgemm and
// cublasSgemmStridedBatched are declared and implemented as well - they are
// part of the minimum surface RISKS.md §5 names (the GLM prompt TU that also
// calls them is excluded from this build, but the functions are real, not
// declarations without bodies).
#pragma once

#include <cstddef>

// ---- data types (real CUDA library_types values) ----
typedef enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_16F = 2,
    CUDA_R_8I = 3,
    CUDA_R_8U = 4,
    CUDA_R_32I = 6,
    CUDA_R_32U = 7,
    CUDA_R_16BF = 14,
} cudaDataType;

// ---- cuBLAS enums (real cublas_v2.h values) ----
typedef enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16,
} cublasStatus_t;

typedef enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
} cublasOperation_t;

typedef enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
} cublasComputeType_t;

typedef enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    // The tensor-op default: this backend has one algorithm, so it is the
    // default under both names (the HIP shape maps both to one value too).
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = -1,
} cublasGemmAlgo_t;

typedef enum cublasMathMode_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_PEDANTIC_MATH = 1,
    CUBLAS_TF32_TENSOR_OP_MATH = 2,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
} cublasMathMode_t;

struct cublasContext;
typedef struct cublasContext* cublasHandle_t;

// The compat cuda_runtime.h is force-included ahead of this header in the
// runtime target and is on the include path for the prefill TU; the stream
// type comes from there. Forward-declare the one stream call the handle
// implementation needs so this header also stands alone in the test TU.
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId);
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMathMode_t mode);

// C = alpha * op(A) * op(B) + beta * C, column-major (see PREFILL-NOTES.md
// for the convention reasoning). Supported: A/B in CUDA_R_16F, CUDA_R_16BF
// or CUDA_R_32F; C in CUDA_R_32F; compute CUBLAS_COMPUTE_32F (alpha/beta are
// const float*). Anything else returns CUBLAS_STATUS_NOT_SUPPORTED.
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                            int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo);

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
                           const float* beta, float* C, int ldc);

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const float* alpha, const float* A, int lda,
                                         long long int strideA, const float* B, int ldb, long long int strideB,
                                         const float* beta, float* C, int ldc, long long int strideC, int batchCount);

#ifdef __cplusplus
}
#endif
