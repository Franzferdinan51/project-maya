#pragma once
// Included only by STRATA_ENABLE_METAL builds. No profiler is attached to the Metal runtime;
// the entry points exist so sources that name them compile, and do nothing.
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);
#ifdef __cplusplus
}
#endif
