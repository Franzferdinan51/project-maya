// src/metal-qwen/runtime/metal_dispatch.mm - strata::metal::launch over the runtime:
// the embedded MSL library (common.metalh + kernels/*.metal, generated into
// metal_kernels_source.h by cmake/embed_metal_kernels.cmake) compiled once, pipelines
// cached by function name, buffer arguments resolved through the runtime's allocation
// map. Every failure aborts loudly with its cause printed.
//
// Launches encode into the stream's CURRENT command buffer (BATCH-NOTES.md),
// one compute encoder per launch; the runtime commits that buffer at its
// commit points. One buffer per launch blocked mid-body once a queue held
// too many uncompleted buffers, and a body's early spin kernels cannot
// complete while the host is still enqueueing - a self-deadlock.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include "metal_internal.hpp"
#include "metal_kernel.hpp"
#include "metal_kernels_source.h"

namespace strata::metal {
namespace {

[[noreturn]] void dispatch_fail(const char* fn, const std::string& why) {
    std::fprintf(stderr, "strata metal: kernel dispatch failed (%s): %s\n", fn ? fn : "?", why.c_str());
    std::abort();
}

id<MTLLibrary> library() {
    static id<MTLLibrary> lib = nil;
    static std::once_flag once;
    std::call_once(once, [] {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) metal_internal::device_handle();
        if (!dev) dispatch_fail("<library>", "no Metal device");
        NSString* src = [[NSString alloc] initWithBytes:kMetalKernelsSource
                                                 length:kMetalKernelsSourceLen
                                               encoding:NSUTF8StringEncoding];
        if (!src) dispatch_fail("<library>", "embedded MSL source is not valid UTF-8");
        NSError* err = nil;
        lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            std::fprintf(stderr, "strata metal: MSL library compile failed:\n%s\n",
                         err ? [[err localizedDescription] UTF8String] : "(no log)");
            std::abort();
        }
    });
    return lib;
}

id<MTLComputePipelineState> pipeline(const char* fn) {
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    const auto it = cache.find(fn);
    if (it != cache.end()) return it->second;
    id<MTLFunction> f = [library() newFunctionWithName:[NSString stringWithUTF8String:fn]];
    if (!f) dispatch_fail(fn, "function not found in the MSL library");
    NSError* err = nil;
    id<MTLDevice> dev = (__bridge id<MTLDevice>) metal_internal::device_handle();
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) {
        std::fprintf(stderr, "strata metal: pipeline creation failed for %s:\n%s\n", fn,
                     err ? [[err localizedDescription] UTF8String] : "(no log)");
        std::abort();
    }
    cache.emplace(fn, p);
    return p;
}

}  // namespace

void launch(cudaStream_t stream, const char* fn, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
            uint32_t block_x, uint32_t block_y, uint32_t block_z, const KernelArg* args, size_t n_args,
            size_t threadgroup_memory_bytes) {
    if (!fn) dispatch_fail(fn, "null function name");
    if (block_x == 0 || block_y == 0 || block_z == 0) dispatch_fail(fn, "zero threadgroup dimension");
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) dispatch_fail(fn, "zero grid dimension");
    if (metal_internal::stream_is_capturing(stream))
        dispatch_fail(fn, "launch during stream capture: the capture layer must invoke bodies, "
                          "not record dispatches (see runtime/CAPTURE-NOTES.md)");
    metal_internal::note_launch_name(fn);
    // Gated stream (a flag-wait interleave is outstanding): record the launch
    // for the runtime watcher's replay instead of encoding it into a buffer
    // that would sit live-but-uncommitted (metal_runtime.mm, "flag waits").
    if (metal_internal::stream_try_defer_launch(stream, fn, grid_x, grid_y, grid_z, block_x, block_y,
                                                block_z, args, n_args, threadgroup_memory_bytes))
        return;
    id<MTLComputePipelineState> p = pipeline(fn);
    if (!metal_internal::stream_queue(stream)) dispatch_fail(fn, "stream has no command queue");
    // Encode into the stream's current buffer. Encoding is serialised by the
    // runtime's one encode mutex: two threads launching on the same stream
    // (or, cheaply, on any streams) can never interleave encoders on a shared
    // buffer, and the runtime's commit points take the same mutex, so a
    // commit can never land under an open encoder. Encoding never waits on
    // the GPU, so the mutex cannot deadlock against GPU work - it is
    // released as soon as the encoder is closed.
    std::lock_guard<std::recursive_mutex> encode_guard(metal_internal::encode_mutex());
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) metal_internal::stream_current_buffer(stream);
    if (!cb) dispatch_fail(fn, "stream has no current command buffer");
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (size_t i = 0; i < n_args; ++i) {
        const KernelArg& a = args[i];
        if (a.is_buffer) {
            void* raw = nullptr;
            size_t off = 0;
            if (!metal_internal::lookup_buffer(a.data, &raw, &off)) {
                char buf[128];
                std::snprintf(buf, sizeof buf, "argument %zu: pointer %p is not in a tracked allocation", i,
                              a.data);
                dispatch_fail(fn, buf);
            }
            [enc setBuffer:(__bridge id<MTLBuffer>) raw offset:off atIndex:i];
        } else {
            if (!a.data || a.size == 0) dispatch_fail(fn, "by-value argument with no bytes");
            [enc setBytes:a.data length:a.size atIndex:i];
        }
    }
    if (threadgroup_memory_bytes > 0) [enc setThreadgroupMemoryLength:threadgroup_memory_bytes atIndex:0];
    [enc dispatchThreads:MTLSizeMake(grid_x, grid_y, grid_z)
        threadsPerThreadgroup:MTLSizeMake(block_x, block_y, block_z)];
    [enc endEncoding];
    // No commit here: the buffer goes out at the runtime's next commit point
    // for this stream (or at the encoder safety valve, which this call
    // accounts for). Ordering is unchanged - encoders in one buffer, and
    // buffers on one queue, execute in submission order.
    metal_internal::stream_note_encoder(stream);
}

}  // namespace strata::metal
