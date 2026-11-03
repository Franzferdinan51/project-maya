# Verify-window stall — diagnosis and fix

## The failure

Coordinator's generation run (merged tree, real Q2_0 pack, `--spec 2 --mtp`,
26-token prompt, greedy): prefill completed, the verify interleave ran for
~5 minutes, then `verify: timed out at layer 39` — the GPU never rang layer
39's bell within the 20 s per-layer watchdog. Teardown then hung forever in
`Verifier::~Verifier → cudaStreamSynchronize → waitUntilCompleted` on the
abandoned window buffer.

## Which window / layer / flag

Reproduced on the Mac (staging `ws-vstall`, `STRATA_VERIFY_DEBUG=1`, the
coordinator's exact probe). In the reproduction the stall presented at the
**first verify window** (T=1, pos0=25): the host waited at layer step 1 and
the GPU had rung **0 bells**; flags at the timeout: served 0, plan (A) 0,
copies (B) 0. The presentation layer differs from the coordinator's (39)
because of *which* spin loses its race (below); the mechanism is the same.
All flag waits live in `post(l)`; bell *l+1* is published at the end of
`pre(l+1)`, so a timeout "at layer *l+1*" means the stuck wait is in
`post(l)` — in the coordinator's run, `post(38)`'s M-flag spin gated bell 40,
reported as layer 39. Layer 39 being a QSA layer is coincidence: QSA layers
3, 7, …, 35 passed in the same run, and in the reproduction the stall hit
layer 0, which is not QSA.

## Evidence chain

1. **The tail command buffer sat in `MTLCommandBufferStatusScheduled`
   forever, no error** — its first flag spin never released. Launch tracing
   of the window buffer: doorbell publish is launch #24; the first spin
   (`wait_flag_ge` on flag A) is #36. Nothing before the doorbell can block.
2. **No other stream in the process had a stuck buffer** (all-streams dump)
   — the verify queue's head buffer is the blocker; not a cross-stream
   host-func ordering problem.
3. **Task candidates (a), (b), (d) eliminated by construction + trace:**
   the pre-wait segment contains no spin before the doorbell (not d); the
   buffer was committed and scheduled (not b); the flag-B host function rides
   the *copy* stream, not the verify stream, and layer 0's flag A is a plain
   host publish, no host function involved (not a).
4. **Micro-test (decisive).** One stream, one buffer containing
   `doorbell_publish` then `wait_flag_ge(flag, 1)`; the host polls the bell,
   then raises the flag:
   - GPU→host works: the bell is visible host-side within 0.1 s mid-buffer.
   - Host→GPU fails: with `flag = 0` when the spin starts, a host write of
     `flag = 1` is **never** observed — not after 100 ms, 2 s, or 10 s, in
     any formulation tried (relaxed atomic load; non-const `atomic_uint`;
     atomic RMW read; device fence in the body; a heartbeat store in the
     body — the heartbeat word stayed 0, i.e. the loop is compiled/served
     as loop-invariant). Preset `flag = 1` before launch: instant pass.
     A delay sweep showed the apparent releases in early variants were
     races where the write landed before the spin's first load.
   - A fresh probe kernel on a *fresh stream* launched 1 s after the raise
     never ran either — a spinning kernel also starves the device.
5. **The workaround that isn't:** gating the wait with an `MTLSharedEvent`
   signaled by a host watcher works for waits up to ~4 s, then the buffer
   dies: status 5, `kIOGPUCommandBufferCallbackErrorTimeout` — the OS GPU
   watchdog aborts any command buffer whose execution (including in-stream
   waiting) exceeds a few seconds. Real pool service times exceed that
   (the coordinator's run: seconds per layer), so no submitted buffer may
   wait at all.
6. **The second resource wall:** per-segment command buffers accumulate as
   *live* objects; creating one blocks (`-[MTLCommandQueue commandBuffer]`
   semaphore) once too many are uncompleted — encoding a 144-segment window
   deadlocked at creation. Gated segments must not become buffers early.

## Root cause

The violated contract is the one the engine's whole interleave rests on:
**on CUDA, a device-side wait on a host-mapped flag observes host writes
made after the wait began.** Metal's memory model gives no such guarantee:
a host write to shared memory is not visible to a kernel already running —
the flag load is loop-invariant in practice. The backend (this runtime) was
implementing `wait_flag_ge` / `wait_flag_ge_or` / `doorbell_wait` as
in-kernel spins (`wait_flag_ge_k`, `wait_flag_ge_or_k`, `doorbell_wait_k`),
i.e. it claimed a CUDA contract it cannot honor in that form. Spins release
only when the host write happens to land before the spin's first load —
which is why the interleave
...[truncated 6095 chars]