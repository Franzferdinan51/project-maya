// src/metal-qwen/stubs/router_sigmoid.cpp - Metal implementation for include/strata/kernels/router_sigmoid.hpp.
// Replaces the stage-1 loud stub. Host launcher over the MSL kernels in
// src/metal-qwen/kernels/moe.metal via strata::metal::launch; semantics mirror
// src/kernels/cuda/router_sigmoid.cu (see that file and the header for the contract).

#include "strata/kernels/router_sigmoid.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../runtime/metal_kernel.hpp"

namespace strata::kernels {

void router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k,
                    float w_scale, bool norm_w, int* ids, float* weights, void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_sigmoid: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > 1024) {  // the MSL kernel's threadgroup arrays (the CUDA kernel's cap is 512)
        std::fprintf(stderr, "router_sigmoid: n_expert %d is past the Metal kernel's 1024\n", n_expert);
        std::exit(1);
    }
    struct Args {
        int32_t n_expert, k;
        float w_scale;
        int32_t norm_w, has_bias;
    };
    const Args a{n_expert, k, w_scale, norm_w ? 1 : 0, bias != nullptr ? 1 : 0};
    // With no bias the buffer slot is never read (has_bias == 0); pass logits to keep
    // the argument list well-formed.
    const metal::KernelArg args[] = {metal::buf(logits), metal::buf(bias ? bias : logits), metal::buf(ids),
                                     metal::buf(weights), metal::val(a)};
    metal::launch((cudaStream_t) stream, "router_sigmoid_k", (uint32_t) n_tokens * 256u, 1, 1, 256, 1, 1, args,
                  5);
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
