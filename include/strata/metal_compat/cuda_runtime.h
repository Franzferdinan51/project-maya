#pragma once
// The Metal backend's CUDA-runtime-shaped surface. Included only by STRATA_ENABLE_METAL builds
// (cmake/metal_qwen_backend.cmake force-includes it into every engine translation unit, the way
// cmake/hip_backend.cmake force-includes the HIP layer's copy). There is no vendor runtime
// underneath: the functions declared here are implemented by src/metal-qwen/runtime/
// metal_runtime.mm on top of Metal buffers in shared (unified) memory, so a "device" pointer is
// a host pointer and every copy kind is a plain copy.
//
// CUDART_VERSION is deliberately reported as 11080: src/core/expert_cache.cpp gates its CUDA
// driver-API virtual-memory path (dlopen'd cuMem*) on CUDART_VERSION >= 12050 and keeps a plain
// allocation path below it. The Metal runtime has no VMM, so the plain path is the honest one.
#ifndef CUDART_VERSION
#define CUDART_VERSION 11080
#endif
// CUDA's annotation avoids per-thread copies of const kernel parameters; the host compiler
// passes the same values without it.
#ifndef __grid_constant__
#define __grid_constant__
#endif
// This build has no device compilation pass: every translation unit is host code, so the
// CUDA execution-space qualifiers in shared headers are inert here, exactly as they are in
// the host pass of a CUDA build.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __managed__
#define __managed__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorNotReady = 34,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorUnknown = 999,
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
} cudaMemcpyKind_t;

struct strata_metal_stream;
struct strata_metal_event;
struct strata_metal_graph;
struct strata_metal_graph_exec;
struct strata_metal_graph_node;
typedef struct strata_metal_stream* cudaStream_t;
typedef struct strata_metal_event* cudaEvent_t;
typedef struct strata_metal_graph* cudaGraph_t;
typedef struct strata_metal_graph_exec* cudaGraphExec_t;
typedef struct strata_metal_graph_node* cudaGraphNode_t;

#define cudaStreamLegacy ((cudaStream_t) 0x1)
#define cudaStreamPerThread ((cudaStream_t) 0x2)
#define cudaStreamDefault ((cudaStream_t) 0x0)
#define cudaStreamNonBlocking 0x01

typedef enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
} cudaStreamCaptureMode_t;

typedef enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
} cudaStreamCaptureStatus_t;

typedef enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
} cudaFuncAttribute_t;

typedef enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
    cudaGraphNodeTypeExtSemaphoreSignal = 8,
    cudaGraphNodeTypeExtSemaphoreWait = 9,
    cudaGraphNodeTypeMemAlloc = 10,
    cudaGraphNodeTypeMemFree = 11,
    cudaGraphNodeTypeCount = 12,
} cudaGraphNodeType_t;

typedef struct cudaKernelNodeParams {
    void* func;
    unsigned int gridDim[3];
    unsigned int blockDim[3];
    unsigned int sharedMemBytes;
    void** kernelParams;
    void** extra;
} cudaKernelNodeParams_t;

typedef enum cudaDeviceAttr {
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrClockRate = 13,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 88,
    cudaDevAttrIntegrated = 18,
} cudaDeviceAttr_t;

#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04

#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02

typedef struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int multiProcessorCount;
    int warpSize;
    int maxThreadsPerBlock;
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;
    int integrated;
    int canMapHostMemory;
    size_t sharedMemPerMultiprocessor;
    int maxThreadsPerMultiProcessor;
    int asyncEngineCount;
    int unifiedAddressing;
    char gcnArchName[256];  // unused on Metal; present so shared code that names it still compiles
} cudaDeviceProp_t;

// ---- errors / device -------------------------------------------------------
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr_t attr, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDriverGetVersion(int* driverVersion);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

// ---- memory ----------------------------------------------------------------
// All allocations are Metal buffers in shared storage: the returned pointer is the buffer's
// host-visible contents pointer, valid on both sides of the (unified) memory system.
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind_t kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind_t kind,
                            cudaStream_t stream);
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                         size_t height, cudaMemcpyKind_t kind);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, cudaMemcpyKind_t kind,
                              cudaStream_t stream);
cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                           size_t count);
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                                size_t count, cudaStream_t stream);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);

// ---- streams / events --------------------------------------------------------
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* userData);

cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);

// ---- graphs ------------------------------------------------------------------
// Stage 1 of the Metal backend: capture brackets a sequence of kernel launches whose stubs
// stop the process loudly before a graph can complete, so these entry points model the API
// state machine (capture -> instantiate -> launch) without recording work. Closure-recording
// capture arrives with the Stage 2 runtime; see src/metal-qwen/runtime/metal_runtime.mm.
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode_t mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus_t* status);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType_t* type);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams_t* params);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* graphExec, cudaGraph_t graph,
                                 cudaGraphNode_t* errorNode, char* logBuffer, size_t bufferSize);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);

cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute_t attr, int value);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize);

#ifdef __cplusplus
}  // extern "C"

// cudaMalloc's typed form, as the engine's sources call it.
template <typename T>
inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(devPtr), size);
}
template <typename T>
inline cudaError_t cudaMallocHost(T** ptr, size_t size) {
    return cudaMallocHost(reinterpret_cast<void**>(ptr), size);
}
template <typename T>
inline cudaError_t cudaHostAlloc(T** ptr, size_t size, unsigned int flags) {
    return cudaHostAlloc(reinterpret_cast<void**>(ptr), size, flags);
}
template <typename Kernel>
inline cudaError_t cudaFuncSetAttribute(Kernel kernel, cudaFuncAttribute_t attribute, int value) {
    return cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
template <typename Kernel>
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, Kernel func,
                                                                  int blockSize,
                                                                  size_t dynamicSMemSize) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,
                                                         reinterpret_cast<const void*>(func),
                                                         blockSize, dynamicSMemSize);
}
// The flags-only instantiate form (CUDA's C++ overload; src/core/mtp.cpp calls it).
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                        unsigned long long flags) {
    (void) flags;
    return cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0);
}
// The default-stream convenience forms (CUDA's C++ overloads; src/program/generate.cpp's
// bandwidth probe calls them).
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                   cudaMemcpyKind_t kind) {
    return cudaMemcpyAsync(dst, src, count, kind, nullptr);
}
inline cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count) {
    return cudaMemsetAsync(devPtr, value, count, nullptr);
}
inline cudaError_t cudaEventRecord(cudaEvent_t event) {
    return cudaEventRecord(event, nullptr);
}
#endif
