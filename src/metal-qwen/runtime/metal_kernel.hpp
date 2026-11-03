// src/metal-qwen/runtime/metal_kernel.hpp - the kernel dispatch contract every
// metal-qwen kernel workstream codes against (QWEN-NATIVE-STATE.md, kernel dispatch
// contract). Implemented by metal_dispatch.mm over the runtime's allocation map.
//
// The MSL library is src/metal-qwen/kernels/common.metalh (a shared prelude, prepended)
// plus every src/metal-qwen/kernels/*.metal, concatenated and compiled once at first
// use; functions are looked up by name. Each .metal file is self-contained apart from
// the prelude. Buffer arguments resolve through the runtime's allocation map
// (contents pointer -> MTLBuffer + offset). Launch is async on the stream's queue;
// compile errors print the MSL log and abort loudly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "strata/metal_compat/cuda_runtime.h"

namespace strata::metal {

struct KernelArg {
    const void* data;
    size_t size;
    bool is_buffer;
};

/// An MTLBuffer argument: `device_ptr` is a contents pointer (base or interior) of an
/// allocation made through the Metal runtime (cudaMalloc and friends).
inline KernelArg buf(const void* device_ptr) { return KernelArg{device_ptr, 0, true}; }

/// A by-value argument: its bytes are copied into the command encoder at launch time.
/// The referenced value must stay alive until launch() returns (launch encodes
/// synchronously, so a named local at the call site is the intended shape).
template <typename T>
inline KernelArg val(const T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "KernelArg::val needs a trivially copyable type");
    return KernelArg{&v, sizeof(T), false};
}

/// Launch MSL function `fn`. grid_* are totals IN THREADS; block_* are the threadgroup
/// dimensions. Arguments bind in order (buffer index == argument index); a by-value
/// argument binds with setBytes at its index. threadgroup_memory_bytes, when nonzero,
/// sets the kernel's threadgroup memory length at index 0 (the single allocation the
/// ported kernels use). Any failure - unknown function, unmapped buffer pointer, MSL
/// compile error - prints the cause and aborts: a dispatch that cannot run must never
/// silently do nothing.
///
/// The launch is ENCODED into the stream's current command buffer, not
/// committed per launch (src/metal-qwen/BATCH-NOTES.md): it executes when the
/// runtime next commits that stream (synchronize, event record, host-func,
/// stream query, graph launch, stream destroy, or the 1024-encoder valve), in
/// stream order with everything enqueued before and after it. Callers that
/// read results on the host must reach one of those commit points and wait,
/// exactly as the CUDA contract already requires a sync before a readback.
void launch(cudaStream_t stream, const char* fn, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
            uint32_t block_x, uint32_t block_y, uint32_t block_z, const KernelArg* args, size_t n_args,
            size_t threadgroup_memory_bytes = 0);

}  // namespace strata::metal
