// launch_shim.mm - TEST-ONLY implementation of the kernel dispatch contract
// (src/metal-qwen/runtime/metal_kernel.hpp) plus the small slice of the
// compat CUDA surface the qsa host launchers use (cudaMalloc / cudaMemcpy /
// cudaDeviceSynchronize / cudaGetErrorString). It exists so the parity
// runner can drive the REAL host launcher TUs and MSL kernels on a Mac
// before the runtime workstream's launch() lands; it is not part of the
// engine build. Buffer resolution mirrors the contract: a contents pointer
// resolves to the allocation that contains it, plus the offset inside it.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "strata/metal_compat/cuda_runtime.h"
#include "../../src/metal-qwen/runtime/metal_kernel.hpp"

namespace {
id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_queue = nil;
std::map<void*, id<MTLBuffer>> g_allocs;  // base contents ptr -> buffer
std::vector<id<MTLCommandBuffer>> g_pending;
std::map<std::string, id<MTLComputePipelineState>> g_pipes;
std::string g_kernel_dir;

void ensure_device() {
    if (!g_dev) {
        g_dev = MTLCreateSystemDefaultDevice();
        g_queue = [g_dev newCommandQueue];
        if (!g_dev) {
            std::fprintf(stderr, "shim: no Metal device\n");
            std::exit(1);
        }
    }
}

void load_library() {
    if (!g_pipes.empty()) return;
    // prelude + every .metal in the kernels dir, like the runtime contract
    std::string src;
    auto slurp = [&](const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "shim: cannot read %s\n", path.c_str());
            std::exit(1);
        }
        char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) src.append(buf, n);
        std::fclose(f);
        src += "\n";
    };
    slurp(g_kernel_dir + "/common.metalh");
    const char* files[] = {"rope.metal",       "native_qsa.metal", "qsa_score.metal", "qsa_indexer.metal",
                           "qsa.metal",        "qsa_attn.metal",   "native_flash_attn.metal", "kv.metal"};
    for (const char* f : files) slurp(g_kernel_dir + "/" + f);
    NSError* err = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                             options:nil
                                               error:&err];
    if (!lib) {
        std::fprintf(stderr, "shim: MSL compile failed:\n%s\n",
                     err ? [[err localizedDescription] UTF8String] : "?");
        std::exit(1);
    }
    for (NSString* name in [lib functionNames]) {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        NSError* perr = nil;
        id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithFunction:fn error:&perr];
        if (!ps) {
            std::fprintf(stderr, "shim: pipeline %s failed: %s\n", [name UTF8String],
                         perr ? [[perr localizedDescription] UTF8String] : "?");
            std::exit(1);
        }
        g_pipes[[name UTF8String]] = ps;
    }
}

bool resolve(const void* p, id<MTLBuffer>* buf, size_t* off) {
    auto it = g_allocs.upper_bound(const_cast<void*>(p));
    if (it == g_allocs.begin()) return false;
    --it;
    char* base = (char*) it->first;
    if ((const char*) p < base || (const char*) p >= base + [it->second length]) return false;
    *buf = it->second;
    *off = (size_t) ((const char*) p - base);
    return true;
}
}  // namespace

extern "C" void qsa_shim_init(const char* kernel_dir) {
    ensure_device();
    g_kernel_dir = kernel_dir;
    load_library();
}

const char* cudaGetErrorString(cudaError_t) { return "shim"; }
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaMalloc(void** out, size_t size) {
    ensure_device();
    id<MTLBuffer> b = [g_dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!b) return cudaErrorMemoryAllocation;
    g_allocs[[b contents]] = b;
    *out = [b contents];
    return cudaSuccess;
}
cudaError_t cudaFree(void* p) {
    auto it = g_allocs.find(p);
    if (it != g_allocs.end()) g_allocs.erase(it);
    return cudaSuccess;
}
cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind_t) {
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind_t kind, cudaStream_t) {
    return cudaMemcpy(dst, src, n, kind);
}
cudaError_t cudaDeviceSynchronize(void) {
    for (id<MTLCommandBuffer> cb : g_pending) [cb waitUntilCompleted];
    g_pending.clear();
    return cudaSuccess;
}

namespace strata::metal {
void launch(cudaStream_t, const char* fn, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by,
            uint32_t bz, const KernelArg* args, size_t n_args, size_t tg_mem) {
    ensure_device();
    load_library();
    auto it = g_pipes.find(fn);
    if (it == g_pipes.end()) {
        std::fprintf(stderr, "shim: kernel %s not found\n", fn);
        std::exit(1);
    }
    if (tg_mem != 0) {
        std::fprintf(stderr, "shim: threadgroup_memory_bytes unsupported (kernel %s)\n", fn);
        std::exit(1);
    }
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:it->second];
    for (size_t i = 0; i < n_args; ++i) {
        if (args[i].is_buffer) {
            id<MTLBuffer> b = nil;
            size_t off = 0;
            if (!resolve(args[i].data, &b, &off)) {
                std::fprintf(stderr, "shim: kernel %s arg %zu: pointer %p not in any allocation\n", fn, i,
                             args[i].data);
                std::exit(1);
            }
            [enc setBuffer:b offset:off atIndex:i];
        } else {
            [enc setBytes:args[i].data length:args[i].size atIndex:i];
        }
    }
    // The contract's grid is in THREADS: dispatchThreads gives exactly
    // that many threads (non-uniform final threadgroups), which is what
    // the kernels' bounds guards are written for.
    MTLSize grid = MTLSizeMake(gx, gy, gz);
    MTLSize block = MTLSizeMake(bx, by, bz);
    [enc dispatchThreads:grid threadsPerThreadgroup:block];
    [enc endEncoding];
    [cb commit];
    g_pending.push_back(cb);
}
}  // namespace strata::metal
