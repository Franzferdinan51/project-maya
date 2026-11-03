// src/metal-qwen/runtime/metal_internal.hpp - the runtime's internal surface for the
// other Metal translation units (kernel dispatch today). Not a public header: only
// src/metal-qwen includes it. The void* handles are Objective-C objects (id<MTLBuffer>,
// id<MTLCommandQueue>, id<MTLCommandBuffer>, id<MTLDevice>) bridged across the C++ /
// Objective-C++ boundary; the owning translation unit casts them back.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "strata/metal_compat/cuda_runtime.h"

#include "metal_kernel.hpp"  // metal::KernelArg, for the deferred-launch surface below

namespace strata::metal_internal {

/// The MTLBuffer that owns `ptr` (any interior pointer of a tracked allocation) and the
/// byte offset of `ptr` inside it. Buffers are retained by the runtime's allocation map;
/// a command buffer that encodes the buffer keeps it alive until completion. Returns
/// false when no tracked allocation contains the pointer.
bool lookup_buffer(const void* ptr, void** out_mtl_buffer, size_t* out_offset);

/// The stream's command queue (the default stream's queue for the null/legacy handles).
void* stream_queue(cudaStream_t stream);

/// Whether the stream is between cudaStreamBeginCapture and cudaStreamEndCapture.
bool stream_is_capturing(cudaStream_t stream);

/// Remember `mtl_command_buffer` as the stream's most recent commit, so stream
/// synchronize/query mean "everything enqueued so far" for dispatched work too.
/// If the buffer was the stream's current buffer it stops being current (it is
/// committed now). Kept for out-of-band committers; dispatch no longer uses it
/// (it encodes into the current buffer, below).
void stream_note_commit(cudaStream_t stream, void* mtl_command_buffer);

/// The stream's CURRENT (uncommitted) command buffer, created on demand
/// (BATCH-NOTES.md): dispatch encodes into this buffer instead of creating
/// one buffer per launch. The buffer is committed only at the runtime's
/// commit points, or by the safety valve via stream_note_encoder.
void* stream_current_buffer(cudaStream_t stream);

/// Note that one encoder was encoded into the stream's current buffer;
/// applies the safety valve (commits the buffer at the encoder limit).
void stream_note_encoder(cudaStream_t stream);

/// Commit the stream's current buffer, if any (a commit point).
void stream_flush_handle(cudaStream_t stream);

/// Flag waits (metal_runtime.mm, "flag waits" section): gate the stream's
/// subsequent work on a host-raised flag word in shared memory - the backend
/// form of the engine's device flag spins, which neither an in-kernel spin
/// nor an in-stream event wait can express on Metal (VSTALL-NOTES.md). The
/// runtime holds the post-wait buffers uncommitted until its watcher sees
/// the condition; stream order does the rest. stream_wait_flag: released
/// when *flag >= value; stream_wait_flag_or: also when *skip == value;
/// stream_wait_doorbell: when *flag == *seq (both words read live).
void stream_wait_flag(cudaStream_t stream, const uint32_t* flag, uint32_t value);
void stream_wait_flag_or(cudaStream_t stream, const uint32_t* flag, uint32_t value, const uint32_t* skip);
void stream_wait_doorbell(cudaStream_t stream, const uint32_t* flag, const uint32_t* seq);

/// The one mutex that serializes ENCODING everywhere: dispatch encodes under
/// it, and the runtime's commit points (stream_flush, event record, host-func
/// attach) take it before touching a stream's current buffer - a commit must
/// never land on a buffer with an encoder still open, which became a real
/// race once the flag-wait watcher started encoding (replay) while the
/// body's thread polls the stream with query flushes. Recursive: a launch
/// holds it across its own valve flush.
std::recursive_mutex& encode_mutex();

/// Deferred encode (same section): while a stream is gated by a flag wait,
/// kernel launches are recorded (stream_try_defer_launch returns true) and
/// replayed by the runtime's watcher one segment ahead of the commit pump,
/// so gated segments never become live command buffers early - Metal
/// throttles live buffers per queue and creating one blocks when too many
/// are uncompleted. stream_defer_active reports the gated state.
bool stream_defer_active(cudaStream_t stream);
bool stream_try_defer_launch(cudaStream_t stream, const char* fn, uint32_t grid_x, uint32_t grid_y,
                             uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z,
                             const metal::KernelArg* args, size_t n_args, size_t threadgroup_memory);

/// The shared MTLDevice.
void* device_handle();

/// Launch-name recording — the Metal substitute for graph node introspection
/// (cudaGraphGetNodes / cudaGraphKernelNodeGetParams, which have no analogue
/// here: a stored capture body IS the sequence, so its launches are where the
/// names live). While a sink is installed, every kernel dispatch and every
/// runtime copy/memset appends its name, in launch order. The capture layer
/// (src/core/verify.cpp) installs the sink around a stored body's first
/// launch. One sink at a time, process-wide; a debug facility, not a hot path.
void launch_name_recording(std::vector<std::string>* sink);
void note_launch_name(const char* name);

}  // namespace strata::metal_internal
