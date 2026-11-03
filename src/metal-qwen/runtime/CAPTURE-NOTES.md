# Capture on Metal — design notes (replaces the closure-recording plan)

Metal has no stream capture and no graph object. A runtime-level fake that
"succeeds" would replay nothing — the worst kind of wrong (RISKS.md §1). So
capture is patched at the capture layer, not shimmed at the runtime, and the
runtime's graph API functions stay the honest state machine they are (an empty
graph instantiates, launches as a no-op, and reports 0 nodes — true statements
about an object that holds no work).

## Landed in this workstream

- `GraphRegistry` / `CapturedGraph` (`src/core/graph.cpp`, `graph.hpp`): under
  `STRATA_USE_METAL`, `record()` keeps the layer body (it already arrives as a
  `std::function`) in the `CapturedGraph` and runs nothing at record time;
  `launch()` invokes the body on the live stream and records the completion
  event `wait_ms()` polls. Correct by construction: the capture contract
  (graph.hpp notes 2–3) already requires every per-step value to live in fixed
  device buffers rather than kernel arguments, so invoking the body computes
  exactly what a replay would. Cost: the per-call host submission graphs exist
  to save (~0.1 ms/layer class) — a Stage-3 performance question, not a
  correctness one.
- `MtpDrafter` (`src/core/mtp.cpp`, `mtp.hpp`): the four capture wrappers
  (`capture_prefill`, `capture_prefill_dev`, `capture_round`, `capture_step`)
  store their exact launch sequence as a body under Metal and mint a real
  (empty) graph exec for the slot, so the already-captured checks and the
  destructor keep the CUDA shape. The four launch sites go through
  `MTP_LAUNCH`, which invokes the body on Metal and `cudaGraphLaunch`es
  elsewhere. A body whose `record_forward` fails at launch aborts loudly
  (`mtp_body_fail`) — there is no caller left to return an error to at that
  point, and a silently half-run draft round would corrupt the draft chain.
- Kernel dispatch (`metal_dispatch.mm`) refuses to launch while a stream is
  capturing: with the capture layer above, a dispatch inside a capture can
  only come from an unpatched capture path, and recording nothing while
  reporting success is precisely the failure this design exists to avoid.
- `SessionGraphs` / `TokenGraph` (`src/core/session.cpp`, `session.hpp`):
  `session_capture`'s existing capture lambda stores each per-layer pre/post
  (and split preA/preB/preP) sequence as a body in an array parallel to its
  exec array, under `STRATA_USE_METAL`, and mints a real (empty) exec for the
  slot; `session_capture_token` stores the whole-token sequence the same way.
  The bodies take the launch stream (`std::function<void(void*)>`) — session
  graphs have no stream of their own, unlike the registry's and the
  drafter's, and the ordering against the caller's own work (the parts copy,
  the doorbell probe) is the launch stream's. All eleven launch sites route
  through `SESSION_LAUNCH`. A pre body re-points the GDN state
  (`gdn_point_at`) before the layer call: the CUDA graph baked that pointer,
  so it is part of the sequence. A body that fails at launch aborts loudly
  (`session_body_fail`), the `mtp_body_fail` rule.
- `Verifier` (`src/core/verify.cpp`, `verify.hpp`): `capture` /
  `capture_commit` store the window / commit sequence as a body (the commit
  sequence factored into `record_commit`, mirroring `record_window`) and
  mint empty execs for the slots; the two launch sites go through
  `VERIFY_LAUNCH`. The node introspection has no Metal analogue, so the
  `STRATA_VERIFY_NODES` kernel-name accounting comes from the stored
  sequence itself: the runtime records launch names
  (`metal_internal::launch_name_recording` — kernel dispatches plus the
  runtime's copies/memsets) around the body's first launch, and the report
  prints in the CUDA path's shape. First launch rather than capture time,
  necessarily: nothing launches at capture time in this design, and a dry
  recording pass would execute `record_window`'s state mutations (KV
  appends, the GDN recurrence) an extra time. Until a window's first launch
  the runtime still truthfully reports 0 nodes.
- `generate.cpp`'s remaining direct graph calls ride on the two items above:
  it calls only `session_capture` / the replay functions / the `Verifier`
  and `TokenGraph` entry points, so it needed no changes of its own.

## Validated runtime semantics (RISKS.md §9) — do not regress

- Events are command-buffer completion timestamps (host clock at completion),
  never a constant 0 ms: stage timing and the draft-length cost models read
  `cudaEventElapsedTime` and would silently corrupt their decisions on fake
  timings.
- `cudaLaunchHostFunc` is a command-buffer completion handler, not an
  immediate call: the verify window relies on that ordering.

## Designed-for, not built (RISKS.md §5)

`prefill::Gemm`'s cuBLAS surface is real work for the prompt path, separate
from capture: declare `metal_compat/cublas_v2.h` (the `hip_compat` shape) and
implement the minimum — `cublasGemmEx` for FP16/BF16→FP32 and
`cublasSgemm(StridedBatched)` — over MPS/Accelerate in the runtime TU.
Declaring without implementing is only honest while `gemm.cu` stays excluded,
which removes the prompt path's GEMMs entirely; schedule it with the prefill
stage, not with bring-up.
