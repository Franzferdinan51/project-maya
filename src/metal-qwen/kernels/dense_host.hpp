// dense_host.hpp - shared host-side launch helpers for the dense family's
// metal-qwen TUs (header-only; each TU that launches MSL kernels includes it).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "strata/metal_compat/cuda_runtime.h"
#include "../runtime/metal_kernel.hpp"

namespace strata::kernels::dense_detail {

inline cudaStream_t stream_of(void* stream) { return (cudaStream_t) stream; }

// Launch on the entry point's stream; the synchronous CUDA entries pass
// sync=true and keep their "caller may read the output immediately" contract
// (the CUDA originals cudaDeviceSynchronize there too).
inline void launch_sync(cudaStream_t stream, const char* fn, uint32_t grid, uint32_t block,
                        const metal::KernelArg* args, std::size_t n_args, bool sync,
                        std::size_t threadgroup_memory = 0) {
    metal::launch(stream, fn, grid, 1, 1, block, 1, 1, args, n_args, threadgroup_memory);
    if (sync) cudaDeviceSynchronize();
}

[[noreturn]] inline void bad_shape(const char* what) {
    std::fprintf(stderr, "strata metal dense: %s\n", what);
    std::abort();
}

// threads_per_row for the split kernels: power of two, <= 1024 (the MSL
// threadgroup partial array), matching the CUDA kernels' reduction contract.
inline uint32_t checked_tpr(int threads_per_row, const char* what) {
    if (threads_per_row < 1 || threads_per_row > 1024 || (threads_per_row & (threads_per_row - 1)) != 0)
        bad_shape(what);
    return (uint32_t) threads_per_row;
}

}  // namespace strata::kernels::dense_detail
