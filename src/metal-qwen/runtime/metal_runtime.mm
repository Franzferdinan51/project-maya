// The Metal backend's runtime: the CUDA-runtime-shaped surface declared in
// include/strata/metal_compat/cuda_runtime.h, implemented over Metal buffers in shared
// (unified) memory. A "device" allocation is an MTLBuffer whose contents pointer the host
// also reads and writes directly, so every cudaMemcpy kind is a plain copy and pinned /
// registered host memory needs no registration at all.
//
// Stage 1 scope (see QWEN-PR-INVENTORY.md §5): allocations, copies, streams and events are
// real; graphs model the API state machine without recording work (the kernel stubs stop
// the process during capture, before a graph can complete). Closure-recording capture is
// the Stage 2 runtime's job.
//
// Dispatch batching (BATCH-NOTES.md): kernel launches encode into the stream's CURRENT
// command buffer, committed only at the runtime's commit points (synchronize, event
// record, host-func, stream query, graph launch, stream destroy, and a 1024-encoder
// safety valve) - never one buffer per launch, because Metal blocks command-buffer
// creation once a queue holds too many uncompleted buffers and a captured body is
// hundreds of launches whose early spin kernels cannot complete mid-body.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <deque>
#include <vector>

#include "strata/metal_compat/cuda_runtime.h"

#include "metal_internal.hpp"
#include "metal_kernel.hpp"

namespace {

id<MTLDevice> device() {
    static id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d;
}

// contents pointer -> the buffer that owns it (retained by the map)
std::unordered_map<void*, id<MTLBuffer>>& allocations() {
    static std::unordered_map<void*, id<MTLBuffer>> m;
    return m;
}
std::mutex& allocations_mu() {
    static std::mutex mu;
    return mu;
}

thread_local cudaError_t last_error = cudaSuccess;

cudaError_t fail(cudaError_t e) {
    last_error = e;
    return e;
}

uint64_t physical_memory() {
    return [[NSProcessInfo processInfo] physicalMemory];
}

}  // namespace

// Safety valve: a stream's current buffer is committed once it holds this many
// encoders, so a single body invocation spans very few buffers no matter how many
// launches it makes (BATCH-NOTES.md §2).
static constexpr uint32_t kBatchEncoderLimit = 1024;

struct strata_metal_event;  // defined below; streams hold deferred event records

struct strata_metal_stream {
    id<MTLCommandQueue> queue = nil;
    id<MTLCommandBuffer> last = nil;     // most recent committed buffer (for synchronize)
    id<MTLCommandBuffer> current = nil;  // uncommitted buffer being encoded into
    uint32_t current_encoders = 0;       // encoders in `current` (the safety valve)
    bool capturing = false;
    // Flag waits (see the "flag waits" section): buffers encoded after a wait
    // are held UNCOMMITTED in this FIFO until the runtime's watcher sees the
    // wait's flag condition hold; the pump commits them strictly in order.
    struct HeldBuffer {
        id<MTLCommandBuffer> cb = nil;
        const volatile uint32_t* a = nullptr;  // flag word (conditional entries)
        const volatile uint32_t* b = nullptr;  // or: skip word; eq: seq word
        uint32_t value = 0;                    // ge/or target
        int kind = 0;                          // 0 ge, 1 or, 2 eq
        bool conditional = false;              // false: commit as soon as it reaches the head
    };
    std::deque<HeldBuffer> held;
    bool pending_cond = false;  // a wait was registered: the NEXT flushed buffer carries its condition
    HeldBuffer pending{};       // the condition `pending_cond` refers to
    // Deferred encode (same section): while a stream is gated, launches and
    // other buffer ops are recorded here as ops instead of being encoded -
    // Metal throttles LIVE command buffers per queue (creating one blocks
    // when too many are uncompleted), so a gated segment must not become a
    // buffer until the pump is about to commit it. The watcher replays the
    // ops one segment ahead of the pump.
    struct DeferredOp {
        int kind = 0;  // 0 launch, 1 wait marker, 2 host func, 3 event record
        // launch
        std::string fn;
        uint32_t grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
        std::vector<strata::metal::KernelArg> args;
        std::vector<std::vector<uint8_t>> owned;  // by-value arg bytes (args[i].data points into these)
        size_t threadgroup_memory = 0;
        // wait marker (the condition that tags the segment FOLLOWING it)
        const volatile uint32_t* a = nullptr;
        const volatile uint32_t* b = nullptr;
        uint32_t value = 0;
        int cond_kind = 0;
        // host func
        void (*host_fn)(void*) = nullptr;
        void* host_arg = nullptr;
        // event record
        strata_metal_event* event = nullptr;
    };
    std::deque<DeferredOp> deferred;
    bool seg_open = false;  // the watcher is mid-replay of one segment
};

struct strata_metal_event {
    std::atomic<bool> complete{true};
    std::atomic<uint64_t> generation{0};  // bumped per record; stale handlers no-op
    std::chrono::steady_clock::time_point stamp{};
    id<MTLCommandBuffer> buffer = nil;  // the buffer whose completion fires this event
    bool timing_disabled = false;
    bool deferred = false;               // recorded on a gated stream; materializes at replay
    strata_metal_stream* stream = nullptr;  // the gated stream (while deferred)
};

struct strata_metal_graph {
    bool placeholder = true;
};
struct strata_metal_graph_exec {
    bool placeholder = true;
};
struct strata_metal_graph_node {
    bool placeholder = true;
};

namespace {

// Every live stream (the default stream included), so cudaDeviceSynchronize can
// flush and wait ALL of them - the CUDA meaning of a device sync, and the only
// honest one under batching (a non-default stream's work may sit entirely in its
// uncommitted current buffer).
std::mutex& streams_mu() {
    static std::mutex mu;
    return mu;
}
std::vector<strata_metal_stream*>& streams() {
    static std::vector<strata_metal_stream*> v;
    return v;
}
void register_stream(strata_metal_stream* s) {
    std::lock_guard<std::mutex> g(streams_mu());
    streams().push_back(s);
}
void unregister_stream(strata_metal_stream* s) {
    std::lock_guard<std::mutex> g(streams_mu());
    auto& v = streams();
    v.erase(std::remove(v.begin(), v.end(), s), v.end());
}
std::vector<strata_metal_stream*> streams_snapshot() {
    std::lock_guard<std::mutex> g(streams_mu());
    return streams();
}

strata_metal_stream* default_stream() {
    static strata_metal_stream* s = [] {
        auto* st = new strata_metal_stream();
        st->queue = [device() newCommandQueue];
        register_stream(st);
        return st;
    }();
    return s;
}

strata_metal_stream* resolve(cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        return default_stream();
    return stream;
}

// Commit the stream's current buffer, if any, and make it the stream's tail:
// after a flush, synchronize/query mean "everything enqueued so far". Returns
// the buffer that now tails the stream (the flushed one, or the previous tail).
// Serializes flush/pump/register/destroy handling of the held FIFOs; the
// watcher thread pumps under it too. (Per-stream encoding itself stays under
// the dispatch layer's encode mutex; bodies are single-threaded per stream.)
std::mutex& pump_mu() { static std::mutex m; return m; }

bool held_condition_met(const strata_metal_stream::HeldBuffer& h) {
    if (!h.conditional) return true;
    if (h.kind == 2) return *h.a == *h.b;
    if (*h.a >= h.value) return true;
    return h.kind == 1 && *h.b == h.value;
}

// Commit held buffers from the head while their conditions hold. Committing
// is the ONLY gate release: a held buffer has never been submitted, so it
// cannot age against the GPU watchdog while it waits (VSTALL-NOTES.md).
void stream_pump(strata_metal_stream* s) {
    while (!s->held.empty() && held_condition_met(s->held.front())) {
        id<MTLCommandBuffer> cb = s->held.front().cb;
        s->held.pop_front();
        [cb commit];
        s->last = cb;
    }
}

// Replay state (watcher thread only): while the watcher encodes one deferred
// segment, flushes inside that encode (a host-func attach is a commit point)
// must tag their buffers with the SEGMENT's condition, not the stream's
// pending one.
thread_local bool tl_in_replay = false;
thread_local bool tl_tag_set = false;
thread_local strata_metal_stream::HeldBuffer tl_tag{};

// A stream is gated while any part of a flag-wait interleave is outstanding:
// a condition waiting to tag the next buffer, held buffers, deferred ops, or
// a segment being replayed. Call with pump_mu held.
bool defer_active_locked(strata_metal_stream* s) {
    return s->pending_cond || !s->held.empty() || !s->deferred.empty() || s->seg_open;
}

// A commit point. The flushed buffer joins the held FIFO whenever anything is
// queued ahead of it or a wait condition tags it; otherwise it commits
// directly (the fast path every non-interleave stream always takes). Inside
// a watcher replay, the segment's condition (tl_tag) is the tag.
id<MTLCommandBuffer> stream_flush(strata_metal_stream* s) {
    // Never commit (or detach) the current buffer under an open encoder:
    // serialize with the dispatch layer's encoding (recursive - a launch's
    // own valve flush re-enters on the same thread).
    std::lock_guard<std::recursive_mutex> encode_lk(strata::metal_internal::encode_mutex());
    if (!s->current) return s->last;
    id<MTLCommandBuffer> cb = s->current;
    s->current = nil;
    s->current_encoders = 0;
    std::lock_guard<std::mutex> lk(pump_mu());
    const bool have_tag = tl_in_replay ? tl_tag_set : s->pending_cond;
    if (s->held.empty() && !have_tag && s->deferred.empty() && !s->seg_open) {
        [cb commit];
        s->last = cb;
        return cb;
    }
    strata_metal_stream::HeldBuffer h;
    h.cb = cb;
    if (have_tag) {
        const strata_metal_stream::HeldBuffer& t = tl_in_replay ? tl_tag : s->pending;
        h.a = t.a; h.b = t.b; h.value = t.value; h.kind = t.kind; h.conditional = true;
        if (!tl_in_replay) s->pending_cond = false;
    }
    s->held.push_back(h);
    stream_pump(s);
    return cb;
}

// Wait until the stream's gated work has all been submitted (held FIFO and
// deferred queue drained by the watcher). A synchronize on a gated stream
// means what CUDA means: everything enqueued so far completes, and gated
// segments complete once their flags are raised - by the time a caller
// syncs, the interleave protocol has it serving (or has already served)
// every layer the segments wait on, so in every live flow this finishes as
// fast as the watcher pumps. Never called from the watcher.
//
// The one exception is teardown after a fatal interleave error: the flags
// will never be raised (the serving loop has exited), and an unbounded wait
// here would hang the verifier's destructor on abandoned work - the exact
// failure this design exists to remove. So the wait is bounded by PROGRESS,
// not by the clock: any commit, replay step, or tag change resets it, and
// only a stream that has made no progress at all for a long stretch (30 s -
// far beyond any live drain, which completes in well under a second once
// its flags are published) is treated as abandoned. The sync then returns
// with the gated work still queued, and the stream's destruction discards
// it, which is the CUDA context-teardown end state for the same situation.
void stream_drain(strata_metal_stream* s) {
    auto last_progress = std::chrono::steady_clock::now();
    size_t seen_held = SIZE_MAX, seen_deferred = SIZE_MAX;
    void* seen_last = nullptr;
    bool seen_pending = false;
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(pump_mu());
            stream_pump(s);
            if (!defer_active_locked(s)) return;
            if (s->held.size() != seen_held || s->deferred.size() != seen_deferred ||
                (void*) s->last != seen_last || s->pending_cond != seen_pending) {
                seen_held = s->held.size();
                seen_deferred = s->deferred.size();
                seen_last = (void*) s->last;
                seen_pending = s->pending_cond;
                last_progress = std::chrono::steady_clock::now();
            }
        }
        if (std::chrono::steady_clock::now() - last_progress > std::chrono::seconds(30)) return;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// The stream's current (uncommitted) command buffer, created on demand. This is
// the ONLY place dispatch-path code asks the queue for a buffer, and it runs at
// most once per kBatchEncoderLimit encoders or per commit point - the bounded
// creation rate that keeps a queue from saturating mid-body (BATCH-NOTES.md).
id<MTLCommandBuffer> stream_current(strata_metal_stream* s) {
    if (!s->current) {
        s->current = [s->queue commandBuffer];
        s->current_encoders = 0;
    }
    return s->current;
}

}  // namespace

extern "C" {

// ---- errors / device -------------------------------------------------------

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidDevicePointer: return "invalid device pointer";
        case cudaErrorInvalidMemcpyDirection: return "invalid copy direction";
        case cudaErrorNotReady: return "not ready";
        case cudaErrorStreamCaptureUnsupported: return "stream capture unsupported";
        default: return "unknown error";
    }
}

cudaError_t cudaGetLastError(void) {
    cudaError_t e = last_error;
    last_error = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError(void) { return last_error; }

cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return fail(cudaErrorInvalidValue);
    *count = device() ? 1 : 0;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* dev) {
    if (!dev) return fail(cudaErrorInvalidValue);
    *dev = 0;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int dev) {
    return dev == 0 ? cudaSuccess : fail(cudaErrorInvalidValue);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t* prop, int dev) {
    if (!prop || dev != 0 || !device()) return fail(cudaErrorInvalidValue);
    std::memset(prop, 0, sizeof(*prop));
    const char* name = [[device() name] UTF8String];
    std::snprintf(prop->name, sizeof(prop->name), "%s", name ? name : "Apple GPU");
    // The engine's device check requires compute capability >= 7.5 (device.cu). The Metal
    // backend has no compute capability; 9.0 is the compat layer's contract value, the same
    // kind of mapping the HIP layer makes when it reports a gfx arch as a CUDA device.
    prop->major = 9;
    prop->minor = 0;
    prop->totalGlobalMem = (size_t) physical_memory();
    prop->sharedMemPerBlock = 32768;  // Metal's per-threadgroup memory ceiling class
    prop->multiProcessorCount = 0;    // Metal exposes no multiprocessor count; nothing in
                                      // the engine computes on it (print + stubbed prefill)
    prop->warpSize = 32;              // Metal SIMD-group width on Apple GPUs
    prop->maxThreadsPerBlock = 1024;
    prop->integrated = 1;
    prop->canMapHostMemory = 1;
    prop->unifiedAddressing = 1;
    prop->asyncEngineCount = 1;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr_t attr, int dev) {
    if (!value || dev != 0) return fail(cudaErrorInvalidValue);
    cudaDeviceProp_t p{};
    cudaGetDeviceProperties(&p, dev);
    switch (attr) {
        case cudaDevAttrComputeCapabilityMajor: *value = p.major; return cudaSuccess;
        case cudaDevAttrComputeCapabilityMinor: *value = p.minor; return cudaSuccess;
        case cudaDevAttrMultiProcessorCount: *value = p.multiProcessorCount; return cudaSuccess;
        case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = 32768; return cudaSuccess;
        case cudaDevAttrClockRate: *value = 0; return cudaSuccess;
        case cudaDevAttrMemoryClockRate: *value = 0; return cudaSuccess;
        case cudaDevAttrGlobalMemoryBusWidth: *value = 0; return cudaSuccess;
        case cudaDevAttrCanUseHostPointerForRegisteredMem: *value = 1; return cudaSuccess;
        case cudaDevAttrIntegrated: *value = 1; return cudaSuccess;
        default: return fail(cudaErrorInvalidValue);
    }
}

cudaError_t cudaDeviceSynchronize(void) {
    // Flush every stream first (so all queues start draining), then wait each
    // stream's tail. Previously this waited only on the default stream's last
    // buffer, which under batching could leave another stream's work not even
    // submitted; CUDA device sync means the whole device.
    const std::vector<strata_metal_stream*> all = streams_snapshot();
    for (strata_metal_stream* s : all) stream_flush(s);
    for (strata_metal_stream* s : all) stream_drain(s);
    for (strata_metal_stream* s : all)
        if (s->last) [s->last waitUntilCompleted];
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int* v) {
    if (!v) return fail(cudaErrorInvalidValue);
    *v = 11080;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int* v) {
    if (!v) return fail(cudaErrorInvalidValue);
    *v = CUDART_VERSION;
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) {
    if (!free_b || !total_b) return fail(cudaErrorInvalidValue);
    // Unified memory: the total is the machine's RAM; "free" is what the device has not
    // currently allocated (an approximation the engine uses for arena sizing heuristics).
    *total_b = (size_t) physical_memory();
    uint64_t alloc = device() ? [device() currentAllocatedSize] : 0;
    *free_b = *total_b > alloc ? (size_t) (*total_b - alloc) : 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetPCIBusId(char* buf, int len, int dev) {
    if (!buf || len <= 0 || dev != 0) return fail(cudaErrorInvalidValue);
    std::snprintf(buf, (size_t) len, "metal:0");
    return cudaSuccess;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
    if (least) *least = 0;
    if (greatest) *greatest = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(int* can, int, int) {
    if (can) *can = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int, unsigned int) {
    return fail(cudaErrorInvalidValue);  // one device; there is no peer
}

// ---- memory ------------------------------------------------------------------

namespace {
cudaError_t alloc_buffer(void** out, size_t size) {
    if (!out) return fail(cudaErrorInvalidValue);
    if (size == 0) size = 1;
    id<MTLBuffer> buf = [device() newBufferWithLength:size
                                              options:MTLResourceStorageModeShared];
    if (!buf) return fail(cudaErrorMemoryAllocation);
    void* p = [buf contents];
    {
        std::lock_guard<std::mutex> g(allocations_mu());
        allocations()[p] = buf;
    }
    *out = p;
    return cudaSuccess;
}

cudaError_t free_buffer(void* p) {
    if (!p) return cudaSuccess;
    std::lock_guard<std::mutex> g(allocations_mu());
    auto it = allocations().find(p);
    if (it == allocations().end()) return fail(cudaErrorInvalidDevicePointer);
    allocations().erase(it);  // releases the buffer
    return cudaSuccess;
}
}  // namespace

cudaError_t cudaMalloc(void** devPtr, size_t size) { return alloc_buffer(devPtr, size); }
cudaError_t cudaMallocHost(void** ptr, size_t size) { return alloc_buffer(ptr, size); }
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) {
    return alloc_buffer(ptr, size);
}
cudaError_t cudaFree(void* p) { return free_buffer(p); }
cudaError_t cudaFreeHost(void* p) { return free_buffer(p); }

cudaError_t cudaHostRegister(void*, size_t, unsigned int) {
    return cudaSuccess;  // shared memory: every host pointer is already device-visible
}
cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }
cudaError_t cudaHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int) {
    if (!devPtr) return fail(cudaErrorInvalidValue);
    *devPtr = hostPtr;
    return cudaSuccess;
}

// Batching audit (BATCH-NOTES.md §3): every copy/memset in this section is a
// HOST-side immediate operation over shared memory - none of them creates a
// command buffer or enqueues GPU work, so none joins the current-buffer
// discipline. Their ordering contract is unchanged: bytes are visible when
// the call returns, and a caller that needs a kernel's results first reaches
// a commit point (sync/query/event) before copying, exactly as before.
// GRFOLD exception: the ASYNC device-to-device copies below DO join the
// stream as copy kernels (memcpy_bytes_k / memcpy2d_bytes_k) - their CUDA
// callers (the QSA q/gate split in the verify, layer and MTP bodies) enqueue
// them between kernels on the same stream with no commit point in between,
// and an immediate host copy reads the source before those kernels land.
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind_t) {
    if (count && (!dst || !src)) return fail(cudaErrorInvalidValue);
    strata::metal_internal::note_launch_name("memcpy");
    std::memcpy(dst, src, count);
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind_t kind,
                            cudaStream_t stream) {
    // GRFOLD: a device-to-device copy must observe its stream position. The
    // immediate host copy below reads the source at call time, but kernels
    // already enqueued on the stream may still be writing it (the QSA q/gate
    // split in the verify/layer/MTP bodies is a 2D case of the same hazard).
    // Route D2D through a copy kernel launched on the caller's stream; every
    // other kind keeps the immediate path its callers' choreography expects
    // (host memory is stable, and their device-side consumers synchronize).
    if (kind == cudaMemcpyDeviceToDevice && stream != nullptr && count > 0) {
        const uint64_t n = (uint64_t) count;
        const ::strata::metal::KernelArg args[] = {::strata::metal::buf(dst), ::strata::metal::buf(src),
                                                   ::strata::metal::val(n)};
        const uint32_t grid = (uint32_t) std::min<uint64_t>(n, 1u << 20);
        ::strata::metal::launch(stream, "memcpy_bytes_k", grid, 1, 1, 256, 1, 1, args, 3);
        return cudaSuccess;
    }
    return cudaMemcpy(dst, src, count, cudaMemcpyDefault);  // shared memory: already visible
}
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                         size_t height, cudaMemcpyKind_t) {
    if (!dst || !src) return fail(cudaErrorInvalidValue);
    strata::metal_internal::note_launch_name("memcpy");
    for (size_t row = 0; row < height; ++row)
        std::memcpy((char*) dst + row * dpitch, (const char*) src + row * spitch, width);
    return cudaSuccess;
}
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, cudaMemcpyKind_t kind, cudaStream_t stream) {
    // GRFOLD: as in cudaMemcpyAsync - a D2D 2D copy (the QSA q/gate split) is
    // launched as a stream-ordered kernel, not performed at call time.
    if (kind == cudaMemcpyDeviceToDevice && stream != nullptr && width > 0 && height > 0) {
        const uint64_t dp = (uint64_t) dpitch, sp = (uint64_t) spitch, w = (uint64_t) width, h = (uint64_t) height;
        const ::strata::metal::KernelArg args[] = {::strata::metal::buf(dst),  ::strata::metal::buf(src),
                                                   ::strata::metal::val(dp), ::strata::metal::val(sp),
                                                   ::strata::metal::val(w),  ::strata::metal::val(h)};
        const uint64_t total = w * h;
        const uint32_t grid = (uint32_t) std::min<uint64_t>(total, 1u << 20);
        ::strata::metal::launch(stream, "memcpy2d_bytes_k", grid, 1, 1, 256, 1, 1, args, 6);
        return cudaSuccess;
    }
    return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault);
}
cudaError_t cudaMemcpyPeer(void* dst, int, const void* src, int, size_t count) {
    return cudaMemcpy(dst, src, count, cudaMemcpyDefault);
}
cudaError_t cudaMemcpyPeerAsync(void* dst, int, const void* src, int, size_t count,
                                cudaStream_t) {
    return cudaMemcpy(dst, src, count, cudaMemcpyDefault);
}
cudaError_t cudaMemset(void* p, int value, size_t count) {
    if (!p && count) return fail(cudaErrorInvalidValue);
    strata::metal_internal::note_launch_name("memset");
    std::memset(p, value, count);
    return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void* p, int value, size_t count, cudaStream_t) {
    return cudaMemset(p, value, count);
}

// ---- streams / events --------------------------------------------------------

cudaError_t cudaStreamCreate(cudaStream_t* out) { return cudaStreamCreateWithFlags(out, 0); }

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* out, unsigned int) {
    if (!out) return fail(cudaErrorInvalidValue);
    auto* s = new strata_metal_stream();
    s->queue = [device() newCommandQueue];
    if (!s->queue) {
        delete s;
        return fail(cudaErrorMemoryAllocation);
    }
    register_stream(s);
    *out = s;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* out, unsigned int flags, int) {
    return cudaStreamCreateWithFlags(out, flags);  // one queue class on Metal
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    strata_metal_stream* s = resolve(stream);
    if (s == default_stream()) return cudaSuccess;  // the default stream is process-lived
    // Commit point: destruction submits the current buffer rather than dropping
    // it, and (as in CUDA) waits for the stream's work to finish first - the
    // stream object owns `current`/`last`, so nothing may outlive it unwaited.
    // EXCEPT flag-held buffers: those were never submitted, and discarding
    // them is the honest teardown for work gated on flags that will never be
    // raised (an aborted verify window). It is what keeps destruction from
    // waiting on an unreleasable wait - the gated segments simply never
    // execute, the same observable end state as CUDA context teardown. The
    // wait below then only covers the last COMMITTED buffer, which contains
    // gate-free compute and always finishes.
    stream_flush(s);
    {
        std::lock_guard<std::mutex> lk(pump_mu());
        s->held.clear();
        s->pending_cond = false;
        // Deferred ops die with the stream. An event whose record was among
        // them can never fire on its (discarded) work; resolve it complete
        // rather than leaving a waiter spinning on a dead stream - context
        // teardown semantics, as with the held buffers above.
        for (auto& op : s->deferred)
            if (op.kind == 3 && op.event) {
                op.event->deferred = false;
                op.event->stream = nullptr;
                op.event->complete.store(true);
            }
        s->deferred.clear();
    }
    if (s->last) [s->last waitUntilCompleted];
    unregister_stream(s);
    delete stream;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    strata_metal_stream* s = resolve(stream);
    // Commit point: the current buffer becomes the tail, and waiting on it
    // waits for everything enqueued so far (a queue runs its buffers in
    // commit order). No current buffer: the previous tail is the wait.
    // Gated work first: held/deferred segments are part of "everything
    // enqueued so far" - drain submits them as their flags land.
    stream_flush(s);
    stream_drain(s);
    if (s->last) [s->last waitUntilCompleted];
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    strata_metal_stream* s = resolve(stream);
    // Commit point (a submission flush, not a wait): the engine's interleave
    // loops use Query exactly this way - session_run_token's "one flush, so
    // WDDM submits the graph now", Verifier::run's watchdog poll - and under
    // batching it is what starts a just-invoked body's batched buffer while
    // the host goes on to the pool work that releases its spin kernels. The
    // answer stays honest: uncommitted work is flushed by the query itself,
    // then success is reported only if the tail buffer has COMPLETED.
    stream_flush(s);
    {
        // Gated work outstanding (held buffers, deferred ops, a segment in
        // replay) is stream work that has not completed - the tail buffer
        // being done does not make the stream done. (Verifier::run reads a
        // Success here as "the graph finished"; with spins everything was
        // committed up front so the tail WAS the whole body. Held commits
        // break that identity, and the query must say so.)
        std::lock_guard<std::mutex> lk(pump_mu());
        if (defer_active_locked(s)) return fail(cudaErrorNotReady);
    }
    if (!s->last || [s->last status] == MTLCommandBufferStatusCompleted) return cudaSuccess;
    return fail(cudaErrorNotReady);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t event, unsigned int) {
    return event ? cudaEventSynchronize(event) : cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* userData) {
    if (!fn) return fail(cudaErrorInvalidValue);
    strata_metal_stream* s = resolve(stream);
    if (!tl_in_replay) {
        // Gated stream: the attach is recorded as a deferred op at this exact
        // stream position and materialized by the watcher's replay.
        std::lock_guard<std::mutex> lk(pump_mu());
        if (defer_active_locked(s)) {
            strata_metal_stream::DeferredOp op;
            op.kind = 2;
            op.host_fn = fn;
            op.host_arg = userData;
            s->deferred.push_back(std::move(op));
            return cudaSuccess;
        }
    }
    // Commit point: the host function is a completion handler on the stream's
    // current buffer when one is open - it fires when ALL work enqueued before
    // it (this buffer's encoders, and every earlier buffer on the serial
    // queue) has completed, which is the CUDA ordering - and the buffer is
    // committed by this call. With no work open, an empty marker buffer
    // carries the handler, as before.
    std::lock_guard<std::recursive_mutex> encode_lk(strata::metal_internal::encode_mutex());
    if (s->current) {
        [s->current addCompletedHandler:^(id<MTLCommandBuffer>) { fn(userData); }];
        stream_flush(s);
    } else {
        id<MTLCommandBuffer> cb = [s->queue commandBuffer];
        [cb addCompletedHandler:^(id<MTLCommandBuffer>) { fn(userData); }];
        [cb commit];
        s->last = cb;
    }
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* out) { return cudaEventCreateWithFlags(out, 0); }

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* out, unsigned int flags) {
    if (!out) return fail(cudaErrorInvalidValue);
    auto* e = new strata_metal_event();
    e->timing_disabled = (flags & cudaEventDisableTiming) != 0;
    *out = e;
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return fail(cudaErrorInvalidValue);
    strata_metal_stream* s = resolve(stream);
    if (!tl_in_replay) {
        // Gated stream: record is a deferred op at this stream position; the
        // event materializes when the watcher's replay reaches it, so it can
        // only fire after the gated work ahead of it completes. Until then
        // the event honestly reports not-complete.
        std::lock_guard<std::mutex> lk(pump_mu());
        if (defer_active_locked(s)) {
            strata_metal_stream::DeferredOp op;
            op.kind = 3;
            op.event = event;
            s->deferred.push_back(std::move(op));
            event->complete.store(false);
            event->deferred = true;
            event->stream = s;
            return cudaSuccess;
        }
    }
    event->deferred = false;
    event->stream = nullptr;
    std::lock_guard<std::recursive_mutex> encode_lk(strata::metal_internal::encode_mutex());
    // Commit point: the event must observe ALL work enqueued before it, so the
    // completion handler goes on the stream's current buffer itself when one
    // is open (committing it), and on a fresh empty marker buffer otherwise.
    // Either way the event's buffer is committed by this call and queued after
    // every earlier buffer, so the event can only fire after all prior work.
    // Query/synchronize stay honest: `complete` is false from here until that
    // buffer's handler runs at its completion - never earlier.
    event->complete.store(false);
    const uint64_t gen = event->generation.fetch_add(1) + 1;
    id<MTLCommandBuffer> cb = s->current ? s->current : [s->queue commandBuffer];
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
      // A re-record supersedes this handler (the session loop re-records its
      // probe event every layer, possibly before the previous buffer fired):
      // only the latest generation may stamp/complete the event.
      if (event->generation.load() != gen) return;
      event->stamp = std::chrono::steady_clock::now();
      event->complete.store(true);
    }];
    if (s->current) stream_flush(s);
    else {
        [cb commit];
        s->last = cb;
    }
    event->buffer = cb;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) return fail(cudaErrorInvalidValue);
    // A deferred record materializes only once its stream's gated work ahead
    // of it has been submitted and completed; drain the stream (which waits
    // for the interleave's flags, raised by the serving side) first.
    if (event->deferred && event->stream) stream_drain(event->stream);
    while (!event->complete.load()) std::this_thread::yield();
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!event) return fail(cudaErrorInvalidValue);
    if (event->deferred) return fail(cudaErrorNotReady);  // not yet materialized
    return event->complete.load() ? cudaSuccess : fail(cudaErrorNotReady);
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms || !start || !end) return fail(cudaErrorInvalidValue);
    *ms = std::chrono::duration<float, std::milli>(end->stamp - start->stamp).count();
    return cudaSuccess;
}

// ---- graphs (Stage 1: state machine only; see the header comment) ------------

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode_t) {
    strata_metal_stream* s = resolve(stream);
    s->capturing = true;
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    strata_metal_stream* s = resolve(stream);
    s->capturing = false;
    if (!graph) return fail(cudaErrorInvalidValue);
    *graph = new strata_metal_graph();
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus_t* status) {
    if (!status) return fail(cudaErrorInvalidValue);
    *status = resolve(stream)->capturing ? cudaStreamCaptureStatusActive
                                         : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    delete graph;
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    delete exec;
    return cudaSuccess;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t*, size_t* numNodes) {
    if (!graph || !numNodes) return fail(cudaErrorInvalidValue);
    *numNodes = 0;
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, cudaGraphNode_t*,
                                 char*, size_t) {
    if (!exec || !graph) return fail(cudaErrorInvalidValue);
    *exec = new strata_metal_graph_exec();
    return cudaSuccess;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType_t* type) {
    if (!node || !type) return fail(cudaErrorInvalidValue);
    *type = cudaGraphNodeTypeEmpty;  // Stage 1 graphs hold no recorded nodes
    return cudaSuccess;
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams_t*) {
    return fail(cudaErrorInvalidValue);  // no kernel nodes exist to describe (see above)
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    if (!exec) return fail(cudaErrorInvalidValue);
    // A graph/body boundary is a commit point: graphs here hold no recorded
    // work (the capture layer invokes bodies directly, CAPTURE-NOTES.md), so
    // there is nothing of the graph's own to submit - but work enqueued
    // around the launch must not sit uncommitted past the boundary.
    stream_flush(resolve(stream));
    return cudaSuccess;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t) {
    return exec ? cudaSuccess : fail(cudaErrorInvalidValue);
}

cudaError_t cudaFuncSetAttribute(const void*, cudaFuncAttribute_t, int) { return cudaSuccess; }

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void*, int,
                                                          size_t) {
    if (!numBlocks) return fail(cudaErrorInvalidValue);
    *numBlocks = 1;  // no occupancy model on the compat layer; callers use it as a sizing hint
    return cudaSuccess;
}

cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

}  // extern "C"

// ---- flag waits: host-flag gating by held commit ----------------------------
//
// The engine's interleave protocol (verify.cpp, session.cpp) has the GPU wait
// mid-stream for a flag word in host-shared memory that the host raises after
// its CPU-pool work. Two measured facts shape this implementation
// (VSTALL-NOTES.md):
//   1. An in-kernel spin can NEVER observe a host write made after the spin
//      started: Metal's memory model does not include host writes to a
//      running kernel - a relaxed atomic load in an empty loop body is
//      loop-invariant in practice (a preset flag passes; a write 100 ms
//      later is invisible indefinitely).
//   2. A command buffer that waits in-stream (spin or MTLSharedEvent gate)
//      is aborted by the OS GPU watchdog after a few seconds
//      (kIOGPUCommandBufferCallbackErrorTimeout) - shorter than real pool
//      service times, so no submitted buffer may wait at all.
// The CUDA contract being honored is the observable one: work after the wait
// does not execute until the flag condition holds, and stream order is
// preserved. Here the wait is an ENCODE-TIME boundary instead: registering a
// wait flushes the segment encoded so far (it commits in order) and tags the
// NEXT segment with the condition; that segment - and everything queued
// behind it - is held UNCOMMITTED until a runtime watcher thread, polling
// the flag words host-side (the direction shared memory works), sees the
// condition and the pump commits it. An uncommitted buffer has not been
// submitted, so it cannot age against the watchdog, and destroying a stream
// simply discards its held buffers - which is also what makes verifier
// teardown after an aborted window terminate (the discarded segments never
// execute).
// While a stream is gated, the body keeps "encoding" at full speed - into
// the deferred-op queue, not into buffers: Metal throttles LIVE command
// buffers per queue (creating one blocks once too many are uncompleted), so
// a held segment must not become a buffer until its turn is near. The
// watcher replays the recorded ops into real buffers one segment ahead of
// the pump. Kernel launches defer at the dispatch choke point; host-func
// attaches and event records defer at their public entries, all at their
// exact stream positions.
namespace {

// Replay one deferred segment on the watcher thread: pop the segment's ops
// (up to the next wait marker) under pump_mu, encode them WITHOUT the lock
// (encoding takes the dispatch layer's encode mutex; holding pump_mu across
// it would invert against the body's valve flush), then seal the segment's
// buffer with the segment's condition. The segment's condition is the
// stream's pending tag for the first segment, thereafter the wait marker at
// the segment's head. Live buffers stay bounded: at most one held segment is
// ever waiting to be encoded behind (plus the executing ones).
void replay_one_segment(strata_metal_stream* s) {
    std::vector<strata_metal_stream::DeferredOp> ops;
    strata_metal_stream::HeldBuffer tag;
    bool have_tag = false;
    {
        std::lock_guard<std::mutex> lk(pump_mu());
        // A pending tag with nothing deferred is still a segment - an EMPTY
        // one (a wait that ended a body). It must be sealed too, or its
        // condition is never pumped and a following synchronize waits on a
        // tag nobody evaluates.
        if (s->seg_open || (s->deferred.empty() && !s->pending_cond) || s->held.size() > 1) return;
        if (s->pending_cond) {
            tag = s->pending;
            tag.conditional = true;
            have_tag = true;
            s->pending_cond = false;
        } else if (!s->deferred.empty() && s->deferred.front().kind == 1) {
            const auto& m = s->deferred.front();
            tag.a = m.a; tag.b = m.b; tag.value = m.value; tag.kind = m.cond_kind; tag.conditional = true;
            have_tag = true;
            s->deferred.pop_front();
        }
        s->seg_open = true;
        while (!s->deferred.empty() && s->deferred.front().kind != 1) {
            ops.push_back(std::move(s->deferred.front()));
            s->deferred.pop_front();
        }
    }
    tl_in_replay = true;
    tl_tag_set = have_tag;
    tl_tag = tag;
    cudaStream_t self = (cudaStream_t) s;
    if (ops.empty() && have_tag && !s->current) (void) stream_current(s);  // the empty segment's buffer
    for (auto& op : ops) {
        switch (op.kind) {
            case 0:
                strata::metal::launch(self, op.fn.c_str(), op.grid[0], op.grid[1], op.grid[2], op.block[0],
                                      op.block[1], op.block[2],
                                      op.args.empty() ? nullptr : op.args.data(), op.args.size(),
                                      op.threadgroup_memory);
                break;
            case 2:
                cudaLaunchHostFunc(self, op.host_fn, op.host_arg);
                break;
            case 3:
                cudaEventRecord((cudaEvent_t) op.event, self);
                break;
            default:
                break;
        }
    }
    stream_flush(s);  // seal: tags with tl_tag (or queues behind the FIFO)
    tl_in_replay = false;
    tl_tag_set = false;
    {
        std::lock_guard<std::mutex> lk(pump_mu());
        s->seg_open = false;
        stream_pump(s);
    }
}

void watcher_main() {
    for (;;) {
        bool any = false;
        {
            const std::vector<strata_metal_stream*> all = streams_snapshot();
            for (strata_metal_stream* s : all) {
                bool gated;
                {
                    std::lock_guard<std::mutex> lk(pump_mu());
                    stream_pump(s);
                    gated = defer_active_locked(s);
                }
                if (gated) {
                    any = true;
                    replay_one_segment(s);
                }
            }
        }
        // Conditions in this protocol are raised within milliseconds; poll
        // briskly while anything is gated, sleep when the world is quiet.
        if (any) std::this_thread::sleep_for(std::chrono::microseconds(50));
        else std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void watcher_start_once() {
    static std::once_flag once;
    std::call_once(once, [] { std::thread(watcher_main).detach(); });
}

// Register a flag wait at the caller's exact stream position. On an ungated
// stream: flush (the pre-wait segment commits in FIFO order), and the
// condition becomes the pending tag for the next buffer. On an already gated
// stream the body is being recorded, not encoded, so the wait is recorded as
// a marker op in the deferred queue at exactly this position.
// ORDER IS THE PROTOCOL: the flush comes FIRST - tagging the pre-wait
// segment would hold the very work (e.g. a layer's doorbell publish) the
// host needs in order to raise this wait's flag: a circular wait.
void stream_wait_register(cudaStream_t stream, const uint32_t* a, const uint32_t* b, uint32_t value,
                          int kind) {
    strata_metal_stream* s = resolve(stream);
    watcher_start_once();
    {
        std::lock_guard<std::mutex> lk(pump_mu());
        if (defer_active_locked(s)) {
            strata_metal_stream::DeferredOp op;
            op.kind = 1;
            op.a = (const volatile uint32_t*) a;
            op.b = (const volatile uint32_t*) b;
            op.value = value;
            op.cond_kind = kind;
            s->deferred.push_back(std::move(op));
            return;
        }
    }
    stream_flush(s);
    std::lock_guard<std::mutex> lk(pump_mu());
    s->pending.a = (const volatile uint32_t*) a;
    s->pending.b = (const volatile uint32_t*) b;
    s->pending.value = value;
    s->pending.kind = kind;
    s->pending_cond = true;
}

}  // namespace

// ---- internal surface for the other Metal TUs (metal_internal.hpp) ------------

#include "metal_internal.hpp"

namespace strata::metal_internal {

bool lookup_buffer(const void* ptr, void** out_mtl_buffer, size_t* out_offset) {
    if (!ptr || !out_mtl_buffer || !out_offset) return false;
    std::lock_guard<std::mutex> g(allocations_mu());
    const char* p = (const char*) ptr;
    for (const auto& kv : allocations()) {
        const char* base = (const char*) kv.first;
        const size_t len = [kv.second length];
        if (p >= base && p < base + len) {
            *out_mtl_buffer = (__bridge void*) kv.second;
            *out_offset = (size_t) (p - base);
            return true;
        }
    }
    return false;
}

void* stream_queue(cudaStream_t stream) {
    return (__bridge void*) resolve(stream)->queue;
}

bool stream_is_capturing(cudaStream_t stream) { return resolve(stream)->capturing; }

void* stream_current_buffer(cudaStream_t stream) {
    return (__bridge void*) stream_current(resolve(stream));
}

void stream_note_encoder(cudaStream_t stream) {
    strata_metal_stream* s = resolve(stream);
    // The safety valve: commit a full current buffer mid-body so creation of
    // the next one starts from a bounded in-flight count (BATCH-NOTES.md).
    if (s->current && ++s->current_encoders >= kBatchEncoderLimit) stream_flush(s);
}

void stream_flush_handle(cudaStream_t stream) { stream_flush(resolve(stream)); }

std::recursive_mutex& encode_mutex() {
    static std::recursive_mutex m;
    return m;
}

void stream_wait_flag(cudaStream_t stream, const uint32_t* flag, uint32_t value) {
    stream_wait_register(stream, flag, nullptr, value, 0);
}

void stream_wait_flag_or(cudaStream_t stream, const uint32_t* flag, uint32_t value, const uint32_t* skip) {
    stream_wait_register(stream, flag, skip, value, 1);
}

void stream_wait_doorbell(cudaStream_t stream, const uint32_t* flag, const uint32_t* seq) {
    stream_wait_register(stream, flag, seq, 0, 2);
}

bool stream_defer_active(cudaStream_t stream) {
    strata_metal_stream* s = resolve(stream);
    std::lock_guard<std::mutex> lk(pump_mu());
    return defer_active_locked(s);
}

// Record a kernel launch instead of encoding it, when the stream is gated.
// By-value argument bytes are deep-copied (the caller's values die with its
// stack frame); buffer arguments keep their allocation pointer - buffer
// CONTENTS are read by the kernel at execution time, exactly as on CUDA, so
// a producer that writes the buffer before the gate opens is observed.
bool stream_try_defer_launch(cudaStream_t stream, const char* fn, uint32_t grid_x, uint32_t grid_y,
                             uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z,
                             const metal::KernelArg* args, size_t n_args, size_t threadgroup_memory) {
    if (tl_in_replay) return false;  // the watcher's own replay encodes for real
    strata_metal_stream* s = resolve(stream);
    std::lock_guard<std::mutex> lk(pump_mu());
    if (!defer_active_locked(s)) return false;
    strata_metal_stream::DeferredOp op;
    op.kind = 0;
    op.fn = fn;
    op.grid[0] = grid_x; op.grid[1] = grid_y; op.grid[2] = grid_z;
    op.block[0] = block_x; op.block[1] = block_y; op.block[2] = block_z;
    op.threadgroup_memory = threadgroup_memory;
    op.args.reserve(n_args);
    op.owned.reserve(n_args);
    for (size_t i = 0; i < n_args; ++i) {
        const metal::KernelArg& a = args[i];
        if (a.is_buffer) {
            op.owned.emplace_back();
            op.args.push_back(a);
        } else {
            const auto* first = (const uint8_t*) a.data;
            op.owned.emplace_back(first, first + a.size);
            op.args.push_back(metal::KernelArg{nullptr, a.size, false});
        }
    }
    for (size_t i = 0; i < n_args; ++i)
        if (!op.args[i].is_buffer) op.args[i].data = op.owned[i].data();
    s->deferred.push_back(std::move(op));
    return true;
}

void stream_note_commit(cudaStream_t stream, void* mtl_command_buffer) {
    // Kept contract for any caller that commits a buffer out-of-band: that
    // buffer becomes the stream's tail, so synchronize/query wait on it. If
    // it was the stream's current buffer, the current slot is cleared - the
    // buffer is committed now and must not be encoded into or re-committed.
    strata_metal_stream* s = resolve(stream);
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) mtl_command_buffer;
    if (s->current == cb) {
        s->current = nil;
        s->current_encoders = 0;
    }
    s->last = cb;
}

void* device_handle() { return (__bridge void*) device(); }

namespace {
std::vector<std::string>*& name_sink() {
    static std::vector<std::string>* sink = nullptr;
    return sink;
}
}  // namespace

void launch_name_recording(std::vector<std::string>* sink) { name_sink() = sink; }

void note_launch_name(const char* name) {
    if (name_sink() && name) name_sink()->emplace_back(name);
}

}  // namespace strata::metal_internal
