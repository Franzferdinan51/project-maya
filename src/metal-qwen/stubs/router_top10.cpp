// src/metal-qwen/stubs/router_top10.cpp - Metal implementation for include/strata/kernels/router_top10.hpp.
// Replaces the stage-1 loud stub. Host launcher over the MSL kernels in
// src/metal-qwen/kernels/moe.metal via strata::metal::launch; semantics mirror
// src/kernels/cuda/router_top10.cu (see that file and the header for the contract).

#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../runtime/metal_kernel.hpp"

namespace strata::kernels {

void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > 1024) {  // the MSL kernel's threadgroup arrays; real geometries are 256/512
        std::fprintf(stderr, "router_top10: n_expert %d is past the Metal kernel's 1024\n", n_expert);
        std::exit(1);
    }
    struct Args {
        int32_t n_expert, k;
    };
    const Args a{n_expert, k};
    const metal::KernelArg args[] = {metal::buf(logits), metal::buf(ids), metal::buf(weights), metal::val(a)};
    metal::launch((cudaStream_t) stream, "router_top10_k", (uint32_t) n_tokens * 256u, 1, 1, 256, 1, 1, args, 4);
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
