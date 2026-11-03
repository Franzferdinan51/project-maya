// src/metal-qwen/runtime/dispatch_smoke_main.cpp - the metal_dispatch_smoke target:
// allocate three vectors through the Metal runtime, launch kernels/smoke.metal's
// smoke_add through strata::metal::launch, and check every output on the host (the
// runtime's buffers are shared memory, so the results are read back in place).
// The second half is the dispatch-batching regression (BATCH-NOTES.md): several
// hundred smoke_inc launches on one stream with NO sync until a commit point -
// the exact enqueue shape that deadlocked when every launch created (and a
// saturated queue blocked) its own command buffer.
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "metal_kernel.hpp"
#include "strata/metal_compat/cuda_runtime.h"

int main() {
    constexpr uint32_t n = 256;
    float *a = nullptr, *b = nullptr, *out = nullptr;
    if (cudaMalloc((void**) &a, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &b, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &out, n * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "metal_dispatch_smoke: allocation failed\n");
        return 1;
    }
    for (uint32_t i = 0; i < n; ++i) {
        a[i] = (float) i;
        b[i] = 1000.0f - (float) i;
        out[i] = -1.0f;
    }
    // An interior pointer, not just base pointers: out + 0 exercises the base case and
    // the second launch below writes through an offset pointer (out2 = out + n/2 filled
    // from a2 = a + n/2) to prove the allocation map resolves offsets too.
    const strata::metal::KernelArg args[] = {strata::metal::buf(a), strata::metal::buf(b),
                                             strata::metal::buf(out), strata::metal::val(n)};
    strata::metal::launch(nullptr, "smoke_add", n, 1, 1, 64, 1, 1, args, 4);
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "metal_dispatch_smoke: synchronize failed\n");
        return 1;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (out[i] != 1000.0f) {
            std::fprintf(stderr, "metal_dispatch_smoke: out[%u] = %f, want 1000\n", i, out[i]);
            return 1;
        }
    }
    const uint32_t half = n / 2;
    const strata::metal::KernelArg args2[] = {strata::metal::buf(a + half), strata::metal::buf(b + half),
                                              strata::metal::buf(out + half), strata::metal::val(half)};
    strata::metal::launch(nullptr, "smoke_add", half, 1, 1, 64, 1, 1, args2, 4);
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) return 1;
    for (uint32_t i = half; i < n; ++i) {
        if (out[i] != 1000.0f) {
            std::fprintf(stderr, "metal_dispatch_smoke: offset out[%u] = %f, want 1000\n", i, out[i]);
            return 1;
        }
    }
    // ---- batching regression -------------------------------------------------
    // smoke_inc adds exactly 1 per launch, so the final value counts the
    // launches that actually executed, in order. No sync of any kind inside
    // the bursts: reaching the end of each burst at all is the first assert
    // (the old one-buffer-per-launch dispatch blocked on buffer creation
    // once the queue saturated, mid-burst, forever).
    float* acc = nullptr;
    if (cudaMalloc((void**) &acc, n * sizeof(float)) != cudaSuccess) return 1;
    for (uint32_t i = 0; i < n; ++i) acc[i] = 0.0f;
    cudaStream_t bs = nullptr;
    if (cudaStreamCreate(&bs) != cudaSuccess) return 1;
    const strata::metal::KernelArg iargs[] = {strata::metal::buf(acc), strata::metal::val(n)};
    auto burst = [&](int count) {
        for (int k = 0; k < count; ++k) strata::metal::launch(bs, "smoke_inc", n, 1, 1, 64, 1, 1, iargs, 2);
    };
    auto check_acc = [&](float want, const char* phase) {
        for (uint32_t i = 0; i < n; ++i) {
            if (acc[i] != want) {
                std::fprintf(stderr, "metal_dispatch_smoke: batch %s: acc[%u] = %f, want %f\n", phase, i,
                             acc[i], want);
                return false;
            }
        }
        return true;
    };
    // Burst 1: 600 launches, committed by the event record; the event sync is
    // the wait. Proves an event observes every launch enqueued before it.
    burst(600);
    cudaEvent_t ev = nullptr;
    if (cudaEventCreate(&ev) != cudaSuccess || cudaEventRecord(ev, bs) != cudaSuccess ||
        cudaEventSynchronize(ev) != cudaSuccess) {
        std::fprintf(stderr, "metal_dispatch_smoke: batch event phase failed\n");
        return 1;
    }
    if (!check_acc(600.0f, "event")) return 1;
    // Burst 2: 2100 launches - over the 1024-encoder safety valve twice, so
    // this also spans several committed buffers - waited on by a DEVICE sync,
    // which must flush and wait non-default streams too.
    burst(2100);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    if (!check_acc(2700.0f, "device-sync")) return 1;
    // Burst 3: 10 launches, no sync at all: stream destruction is the commit
    // point (and waits, as in CUDA), so the results are readable right after.
    burst(10);
    if (cudaStreamDestroy(bs) != cudaSuccess) return 1;
    if (!check_acc(2710.0f, "destroy")) return 1;
    cudaEventDestroy(ev);
    cudaFree(acc);

    std::printf("metal_dispatch_smoke: PASS (%u elements, base + interior pointers; "
                "batch bursts 600+2100+10 launches, event/device-sync/destroy commits)\n",
                n);
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    return 0;
}
