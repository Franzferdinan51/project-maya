# BATCH NOTES — stream-level command-buffer batching (dispatch workstream)

Companion to `SYNCFIX-NOTES.md` / `runtime/CAPTURE-NOTES.md`. This file records
the fix for the second end-to-end generation hang: a stack sample of the hung
run showed the main thread inside `Verifier::record_window ->
native_expert_grouped_based -> strata::metal::launch ->
-[AGXG17XFamilyCommandQueue commandBuffer] ->
_dispatch_semaphore_wait_slow`, forever, at 0% CPU.

## 1. The bug

`metal::launch` created **one `MTLCommandBuffer` per kernel launch** and
committed it immediately. A verify-window body invocation is hundreds of
launches. Metal bounds the number of uncompleted command buffers a queue
will hand out and **blocks buffer creation when saturated**.

The window body legitimately contains device spin kernels
(`wait_flag_ge_k` and friends) whose release depends on host writes that
happen **after the body returns** (the `Verifier::run` interleave: pool work,
flag A/B raises). So while the host is still enqueueing the later part of a
body, the early buffers cannot complete — the queue fills with
uncompletable buffers, creation of the next one blocks, and the host thread
that must eventually release the spins is the thread blocked in creation.
Self-deadlock, deterministically, mid-body.

CUDA stream launches are effectively unbounded, which is why the engine's
design assumes free enqueueing. The parity harness never hit this: its
pattern is launch-then-sync per fixture entry, so the queue never held more
than a buffer or two.

## 2. The fix: one CURRENT buffer per stream

Each stream owns a **current command buffer** in the runtime
(`metal_runtime.mm`: `current` / `current_encoders` on the stream object;
`stream_current` / `stream_flush` helpers; internal surface in
`metal_internal.hpp`).

- `strata::metal::launch` (metal_dispatch.mm) encodes into the stream's
  current buffer — created on demand, **one compute encoder per launch**
  (encoders are not the bounded resource) — instead of creating and
  committing a buffer per launch. Encoding is serialised by one mutex so
  two threads can never interleave encoders on a shared buffer; encoding
  never waits on the GPU, so the mutex cannot deadlock against a spin
  kernel.
- The current buffer is **committed only at the commit points** in §3, or
  by the safety valve: at **1024 encoders** the current buffer is committed
  and a fresh one opened (`stream_note_encoder`). 1024 is far above any
  single kernel's needs and far below any body size that matters; it exists
  so an unbounded enqueue run (a body that never reaches a commit point)
  still spans a bounded number of buffers.
- Ordering is unchanged: encoders within a buffer execute in encode order,
  and a queue executes its buffers in commit order — the same stream order
  CUDA guarantees and the engine already relies on.

### Why a body can no longer saturate the queue (the in-flight bound)

Between two commit points, a stream creates **at most
ceil(encoders / 1024) command buffers**. A verify window body is hundreds
of launches: that is **one** buffer for the whole window (two if it ever
crossed 1024 launches), where the old dispatch created hundreds. Buffer
creation can only block when the queue already holds Metal's maximum of
uncompleted buffers; reaching that maximum now needs
`limit × 1024` uncompleted encoders with no commit point in between — no
body in this engine is within orders of magnitude of that. And the commit
points that end a body (§3) are exactly the calls the engine's drivers
already make immediately after invoking a body, so the batched buffer is
submitted before the host's interleave loop starts polling for GPU
progress: the spin kernels then run, observe the host's flag writes in
shared memory, and release, in stream order, inside the same buffer.

## 3. Commit points

| Commit point | Where | Why it must commit |
|---|---|---|
| `cudaStreamSynchronize(stream)` | runtime | Flush current, wait on it as the tail: "everything enqueued so far has completed". (Previously this created an extra empty tail buffer per sync; the flushed current buffer *is* the tail now.) |
| `cudaDeviceSynchronize()` | runtime | CUDA device sync means the whole device. It now flushes **all** registered streams, then waits each tail. Previously it waited only on the default stream's last buffer — under batching that could leave another stream's work not even submitted. The registry also makes destroy-time removal well-defined. |
| `cudaEventRecord(event, stream)` | runtime | An event must observe all work enqueued before it. The completion handler is attached to the stream's **current buffer itself** (committed by this call), or to a fresh empty marker buffer when no work is open — queued after every earlier buffer either way. |
| `cudaEventSynchronize` / `cudaEventQuery` honesty | runtime | Record *is* a commit point, so there is no "recorded but uncommitted" window: `complete` is false from record until the event buffer's handler fires at completion, never earlier. `cudaStreamQuery` and event query are unchanged in what they report (§4). A generation counter per event makes a re-record supersede the previous record's handler (the session loop re-records its probe event every layer, possibly before the previous buffer completed; a stale handler must not stamp/complete the new record). |
| `cudaLaunchHostFunc(stream, …)` | runtime | The host function must run after all prior stream work. With work open, its completion handler goes on the current buffer and the call commits it (the verify pool's `fetch_dma` flag-B raise rides this). With no work open, an empty marker buffer carries it, as before. |
| `cudaStreamQuery(stream)` | runtime | A submission flush, not a wait — this is an established engine idiom, not a new invention: `session_run_token` calls it with the comment "one flush, so WDDM submits the graph now", `Verifier::run` calls it right after launching the window and again in its watchdog poll, and `CapturedGraph::wait_ms`'s header note makes the same point for event queries. Under batching this call is what *starts* a just-invoked body's buffer while the host goes on to the interleave work that releases its spins. The answer stays honest: success only when the (now submitted) tail buffer has completed. |
| Graph/body boundary: `cudaGraphLaunch(exec, stream)` | runtime | Graphs hold no recorded work on this backend (the capture layer invokes bodies directly), so there is nothing of the graph's own to submit; the flush guarantees work enqueued around a launch does not sit uncommitted past the boundary. In practice bodies are committed even earlier/again by the layer's own following call — `GraphRegistry::launch` records its completion event right after invoking the body, the session loop records its probe event after each `pre`, MTP/verify drivers sync — all rows above. |
| `cudaStreamDestroy(stream)` | runtime | Destruction submits the current buffer rather than dropping it, and waits for the stream's work (the CUDA shape), since the stream object owns `current`/`last`. The default stream stays process-lived and exempt, as before. |
| Safety valve: 1024 encoders | dispatch accounting (`stream_note_encoder`) | Bounds buffers-per-body (§2) even for an enqueue run with no commit point at all. |

Deliberately **not** commit points: `cudaStreamEndCapture` (capture records
nothing on this backend — dispatch during capture still aborts loudly, and
that check runs *before* any buffer is touched, unchanged), `cudaGraphUpload`
(no work exists to submit), `cudaStreamWaitEvent` (kept as the pre-existing
host wait on the event, which is stricter than CUDA and cannot be fooled by
batching: events are only ever recorded committed, §3 row 3).

## 4. Audit of the other buffer consumers in `metal_runtime.mm`

| Consumer | Before | Now |
|---|---|---|
| Kernel dispatch (`metal_dispatch.mm`) | 1 buffer per launch, committed per launch, `stream_note_commit` per launch | Encodes into the stream's current buffer; valve accounting per encoder. `stream_note_commit` is **kept** with its contract intact (an out-of-band committed buffer becomes the tail; if it was the current buffer, the current slot is cleared) — synchronize/query still wait on the right buffer for any caller that uses it. |
| `commit_tail` (sync helper) | Created + committed an extra empty buffer on every `cudaStreamSynchronize` | **Removed.** The flushed current buffer is the tail; with nothing open, the previous tail is the wait. One less buffer per sync, same meaning. |
| `cudaEventRecord` | Own empty buffer per record, queued behind the launches' buffers | Handler on the current buffer when work is open (one less buffer, same "after all prior work" point in the serial queue); marker buffer otherwise; generation guard added (§3). Event timestamps remain command-buffer **completion** timestamps (CAPTURE-NOTES' validated semantics — never a fake 0 ms). |
| `cudaLaunchHostFunc` | Own empty buffer per call | Handler on the current buffer when work is open (committed by the call); marker buffer otherwise. Still a completion handler, never an immediate call (CAPTURE-NOTES). |
| `cudaMemcpy` / `cudaMemcpyAsync` / `cudaMemcpy2D[Async]` / `cudaMemcpyPeer[Async]` / `cudaMemset[Async]` | Host-side `memcpy`/`memset` over shared memory; **no command buffer, no GPU work enqueued** | **Unchanged**, by audit: there is nothing to fold into the current-buffer discipline. Their contract (bytes visible at return; reach a commit point before reading kernel results) is stated in a comment at the section head. The known in-body uses are host-produced staging refills (SYNCFIX-NOTES §3 class ii). |
| cuBLAS (`metal_blas.cpp`) | Host compute after `cudaStreamSynchronize(handle->stream)` | Unchanged behaviour — that sync is a commit point, so all batched producers of the GEMM's inputs are submitted and waited before the host reads them. Its header comment now says "batched" instead of "one command buffer per launch". |
| Graphs (`cudaGraphLaunch` / `Upload` / instantiate…) | No-ops returning success | `cudaGraphLaunch` additionally flushes the launch stream (§3); the rest are unchanged no-ops — still honest statements about an object that holds no work. |

Behaviour the tests encode, and how it is preserved:

- **Dispatch smoke** (launch + host readback): launches now wait for the
  smoke's own `cudaStreamSynchronize` to commit them — same results, read
  after the same sync the test always did.
- **Parity runner**: its per-entry pattern is launch-then-`cudaStreamSynchronize`
  on its stream; the sync commits the batch and waits. Kernel order within
  the batch is encode order, arithmetic is per-kernel — results must stay
  bit-identical, and §5's full sweep is the check.
- **Graph/session tests**: bodies invoke → following event record / sync
  commits (§3). `wait_ms` still polls a completion event whose timestamp is
  real.
- **GEMM test**: GEMM still syncs its stream before host compute (above).
- **Launch-during-capture**: still `dispatch_fail` before any buffer state
  is touched — a capturing stream never even creates a current buffer.
- **`cudaStreamQuery` as a flush**: see §3 — load-bearing for
  `session_run_token` and `Verifier::run`, and the reason a batched spin
  window still interleaves with its host loop.

## 5. Regression + Mac verification

New regression, inside the existing `metal_dispatch_smoke` ctest (no new
target): a `smoke_inc` kernel (`out[i] += 1`, so the result *counts* the
launches that executed) launched on a fresh stream in three no-sync bursts —

1. **600 launches**, committed by `cudaEventRecord` + `cudaEventSynchronize`
   → every element must read exactly 600.
2. **2100 launches** (crosses the 1024-encoder valve twice, spanning several
   buffers), waited on by `cudaDeviceSynchronize` → 2700. Also proves a
   device sync flushes/waits a non-default stream.
3. **10 launches**, no sync at all, committed by `cudaStreamDestroy` → 2710.

Reaching the end of burst 1 at all is the headline assertion: under the old
dispatch the burst's buffer creations blocked mid-burst once the queue
saturated.

Mac verification (staging `~/qwen-native-staging/ws-batch/`, fresh build
dir, build lock held, `maya` + `metal_kernel_parity` + the four metal test
executables built clean):

- `ctest -R 'metal|e2e'`: **6/6 PASS** (incl. the extended
  `metal_dispatch_smoke` above).
- Full fixture sweep, `--no-fork`: **112/112 PASS** — unchanged results
  (the parity pattern is launch-then-sync per entry; batching must be, and
  is, bit-identical).

The coordinator re-runs the end-to-end generation probe after merge; that
probe is the run that hung in both §1 shapes (sync deadlock first, buffer
saturation second).
