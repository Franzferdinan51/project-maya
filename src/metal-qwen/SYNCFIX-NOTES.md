# SYNCFIX NOTES — the mid-body stream-sync deadlock (grouped workstream follow-up)

Companion to `GROUPED-NOTES.md`. This file records the fix for the first
end-to-end generation hang (stack sample of the killed run, 87 min):
main thread in `Verifier::run -> verify_graph_launch -> body invoke ->
Verifier::record_window -> grouped lambda ->
strata::kernels::native_expert_grouped -> cudaStreamSynchronize ->
-[MTLCommandBuffer waitUntilCompleted]`, forever.

## 1. The CUDA structure (what the original actually does)

`native_expert_grouped` in `src/kernels/cuda/iq_kernels.cu` contains **no
host synchronisation and no host readback, anywhere**:

- It launches `native_gu_kernel` ONCE with grid
  `((2*n_ff + GU_ROWS-1)/GU_ROWS, cap_groups)`, then the swiglu and
  quantize kernels over all entries, then `native_down_kernel` ONCE with
  grid `((n_embd+7)/8, cap_groups)` — all on the caller's stream.
- The kernels read the plan on the device: `if (g >= *n_groups) return;`,
  `grp_start[g] / grp_start[g+1]` for the entry range, and
  `(const uint8_t*) grp_ptr[g]` for the blob. `check()` after each stage
  is `cudaGetLastError` only — asynchronous.
- In `verify.cpp` the grouped lambda is called from `post()` inside
  `record_window`, i.e. **inside the captured region** on CUDA (stream
  capture) — the call was never outside the body. Under capture the
  launcher only *records*; nothing executes, so no sync question arises
  there either.

So the faithful mirror is: single launch per stage, fixed worst-case grid
(`cap_groups`), plan read on the device. The Metal host-plan launcher
(one sync + one launch per group) was the divergence.

Why the sync deadlocks on Metal specifically: bodies here are *executed*
at launch (`runtime/CAPTURE-NOTES.md`). Each `metal::launch` commits its
own command buffer to the stream's queue, in order. The verify body's
queue therefore contains `wait_flag_ge_k` spin buffers (flag A, then flag
B, then the pool flag) that complete only when the host loop in
`Verifier::run` — which runs **after** the body returns — publishes the
plan / raises the flags. `cudaStreamSynchronize` in this runtime commits
a tail buffer behind everything already queued and waits for it: mid-body
it waits for a spin buffer whose release depends on the very host thread
that is blocked in the sync. Self-deadlock, deterministically, at the
first grouped call of the first window.

## 2. The fix

### `native_expert_grouped_based` (new entry, `iq_kernels.hpp`)

Same signature plus `(const void* blob_base, uint64_t blob_span)`. On CUDA
it is exactly `native_expert_grouped` (device pointers dereference; the
base is redundant). On Metal (`stubs/iq_kernels.cpp` +
`kernels/native_expert.metal`):

- One launch per stage — `native_gu_grouped_k` with grid
  `(ceil(2*n_ff/8)*256 threads, cap_groups)`, the shared swiglu +
  `quantize_q8_1_k` over all entries, `native_down_grouped_k` with grid
  `(ceil(n_embd/8)*256, cap_groups)` — the CUDA grids. Kernels read
  `n_groups` / `grp_start` / `grp_ptr` from the bound plan buffers and
  self-assign (`g = threadgroup_position_in_grid.y`), exactly like the
  CUDA kernels. Scratch carving is the shared helper, byte-identical to
  the CUDA TU (and unchanged for the plain entry).
- **No host sync, no host readback, at any point.** The only stream state
  the launcher touches is appending three command buffers to the stream's
  queue, in order. It therefore cannot wait on anything a later host
  action must release; the deadlock cannot recur through this entry.
- The one Metal-forced difference: an MSL kernel cannot dereference a
  contents address belonging to another buffer (that is why the first
  device-plan attempt and the S2 grouped kernels' raw cast read zeros).
  So a blob is addressed as an **offset inside one bound base
  allocation**: the kernel computes `grp_ptr[g] - base_value` (the
  pointer VALUE minus the base's contents address — integer arithmetic
  on values the plan legitimately carries) and indexes the bound base
  buffer. A group whose pointer lands outside
  `[base, base + span)` contributes nothing instead of reading wild.
  Every real caller satisfies the range by construction:
  - verify VRAM groups: all pointers are `cache_base + slot offset`
    (pool publish and `resident_plan` both build them that way) — base =
    `hits_.cache_base`, span from `hits_.n_slots / slot_off / blob`.
  - verify PCIe groups: after staging, all pointers are
    `stage + q * blob_bytes` (mode 0 writes them so directly; modes 1/2
    are rebased by `rebase_ptrs`) — base = the group's staging region,
    span = `stage_per * max_blob`.

`verify.cpp`'s grouped lambda calls the based entry under
`STRATA_USE_METAL` with those two bases; other backends keep the plain
call. The plain `native_expert_grouped` (host plan + sync) is **kept**
for its remaining callers, which are host-side by construction:
`RemoteExperts::launch` (pool thread, own stream, `finish()` syncs that
stream by design) and the parity runner. Its header comment in the stub
now states the restriction.

### `resident_plan` (verify_kernels) — same bug, fixed

Was: host-executed after `sync_stream`, reading `ids` and writing the
plan on the host. Reachable from `record_window`'s `pre()` whenever
`STRATA_VERIFY_DEVICE_PLAN=1` (off by default, so not the probe's hang,
but the identical self-deadlock for any layer past the first). Now: a
one-thread device kernel (`resident_plan_k` in `verify.metal`) performing
the layout's exact integer operations in stream order — it only *writes*
pointer values as integers, so nothing needs the host. Byte-identical
plan (fixture-checked, §4).

### `fetch_blobs` (verify_kernels) — same bug, fixed

Was: host-executed after `sync_stream`, reading `*n` / `src[k]` and
memcpy-ing the PCIe blobs into staging. Called from `post()` in
`pcie_mode == 2` — which is what generate's default `--pcie-mode auto`
selects — **after** the flag-B spin buffer, so it would have been the
next hang after the grouped fix, on every window. On Metal the staging
copy is now owned by the pool side, which is the one host actor that
legitimately runs at the right moment:

- `expert_source.cpp` (Metal only): the plan publish hands the fetch
  list to `Verifier::fetch_dma` for **every** pcie mode (previously modes
  1/2 passed 0, expecting the in-body kernel). `fetch_dma` copies
  host→staging on the copy stream and only then raises flag B — and the
  body's flag-B wait precedes the grouped PCIe call, so the blobs are in
  staging before any kernel can read them.
- `fetch_blobs` itself is a validating no-op on Metal; `rebase_ptrs`
  (already a device kernel) still points the plan at staging in stream
  order.
- Mode 1 ("direct") on Metal takes the staged route too (`verify.cpp`):
  its aliases point into unregistered (mmap) host memory that no Metal
  kernel can bind or dereference, so direct reading was never available
  on this backend; staging is the only correct form. CUDA behaviour of
  modes 0/1/2 is unchanged.

## 3. Sync audit — every launcher reachable from a captured body

Method: grep `src/metal-qwen/stubs/*.cpp`, `src/metal-qwen/kernels/*.hpp`
and `src/metal-qwen/runtime/` for `cudaStreamSynchronize`,
`cudaDeviceSynchronize`, `sync_stream`, `launch_sync` arguments, and host
dereferences of device pointers; then trace each hit's callers across the
body-executed TUs (`verify.cpp` record_window/record_commit, `session.cpp`
token/layer bodies, `mtp.cpp` bodies, `graph.cpp`/`layer.cpp` bodies).
Classes: (i) outside bodies only, (ii) inside bodies but provably
non-blocking, (iii) the bug.

| Site | Class | Evidence |
|---|---|---|
| `iq_kernels.cpp` `native_expert_grouped` sync + plan readback | **(iii) → fixed** | the hang itself; body callers moved to `native_expert_grouped_based` (§2). Remaining callers are host-side (RemoteExperts, parity runner). |
| `verify_kernels.cpp` `resident_plan` sync + host plan build | **(iii) → fixed** | called from `record_window` pre() under `STRATA_VERIFY_DEVICE_PLAN=1`; now a device kernel (§2). |
| `verify_kernels.cpp` `fetch_blobs` sync + readback + host memcpy | **(iii) → fixed** | called from `record_window` post() in the default pcie mode; now pool-side staging + no-op (§2). |
| `verify_kernels.cpp` all other entries (`wait_flag_*`, `copy_*`, `gdn_*`, `embedding_gather_dev`, `broadcast_streams`, `mtp_select`, `ident_hits`, `gather_rows`, `map_ids`, `row_top_prob`, `window_ids`, `dense_steps`, `rebase_ptrs`, `gpu_stamp`) | (ii) | pure `metal::launch` / `launch_sync(..., stream == nullptr)`: the sync fires only when the entry's stream argument is null; body invocations always pass the body stream (`cs_`, the session launch stream). `gpu_stamp` samples the host clock but *writes it via a kernel* in stream order — no readback, no wait. |
| dense family + most stubs via `dense_detail::launch_sync` (`bf16_gemv`, `dequant_*`, `iq_*`, `native_*`, `ple`, `quantize_act`, `s2_gemv_q8`, `sampler`, `shared_expert`, `router_*`, `elementwise`, `cvec_apply`, `gdn`, `qsa` helpers, …) | (ii) | same pattern: sync argument is literally `stream == nullptr` or `false` at every call site; bodies pass non-null streams. `qsa.cpp`'s helper, `router_sigmoid.cpp`, `router_top10.cpp`, `shared_expert.cpp` spell it as `if (stream == nullptr) cudaDeviceSynchronize()`. A null-stream sync in this runtime waits only on the *default* stream's last buffer, never on a body queue — so even a hypothetical null call could not self-deadlock a body (it would be a contract misuse for other reasons). |
| No-stream synchronous entries: `dequant_s2`, `s2_gemv`, `s2_split`, `s_gemv`, `s_gemv_split` (launch on the null stream + `cudaDeviceSynchronize`, mirroring their CUDA contract) | (i) | no caller in any Metal-compiled engine TU — grep finds only the `*_parity.cpp` benches and `concurrent_main` / `overlap_main` bench mains, none in the Metal build. Body paths use the stream forms (`s_gemv_split_async` with sync=false, `s2_gemv_q8`, …). |
| `cvec.cpp` `cudaDeviceSynchronize` in `cvec_upload` / `cvec_set_enabled` | (i) | host configuration APIs (load / toggle), called from setup code, never from a record_* body; the body entry `cvec_apply` is async-only. |
| `runtime/metal_blas.cpp` cuBLAS GEMM: `cudaStreamSynchronize(handle->stream)` + host compute | (i), latent risk flagged | reachable only from the prompt path's `prefill::Gemm` surface — host-driven, never captured; `gemm.cu` is excluded from the Metal build and prefill runs its no-MMQ fallback. No body TU calls cuBLAS. If a future body path ever routed through cuBLAS it would be class (iii) (sync + host readback of shared buffers mid-body). |
| runtime `cudaStreamSynchronize` / `cudaDeviceSynchronize` / `cudaEventSynchronize` implementations | (i) | the mechanism itself. Engine call sites are the drivers' *post-body* completion waits (`Verifier::run`'s final sync, `commit()` after `VERIFY_LAUNCH` returns, session token/stage final syncs, MTP launch-then-sync, destructors, `read_head_logits`) — outside bodies by construction; the verify interleave (pool loop between launch and final sync) is designed around exactly this. After this fix, no stub reachable from a body calls them mid-body. |
| runtime `cudaMemcpy[Async]` = immediate host memcpy over shared buffers | (ii) | non-blocking by design (shared memory; CAPTURE-NOTES). In-body engine uses are host→mapped staging refills whose contents are host-produced; device→device uses in `layer.cpp`'s doorbell handoff are pre-existing shared-memory semantics, not a wait — noted for completeness, not a sync. |
| `moe_grouped_s2` launcher (`s2_expert_grouped.cpp`) | (ii) for sync | no sync anywhere (pure launches, plan read on device) — cannot deadlock. Its *addressing* is the separate known flag in §5. |

## 4. Fixtures

- Existing 4 `native_expert_grouped` fixtures: unchanged, still pass
  (the plain entry's semantics are untouched).
- New `native_expert_grouped_based` fixtures: the same 4 cases (same
  shapes, types, reference chain, tolerances) with all blobs packed into
  one arena allocation at a 256-aligned stride, `grp_ptr` filled with the
  slices' contents addresses — the entry exercised exactly the way
  `record_window` calls it (device plan, fixed cap_groups grid, no sync
  before the call in the runner).
- New `resident_plan` fixtures (3 cases, bitwise): two-groups with
  slot-stride pointers, three-groups with `slot_off`, and a
  not-resident case that must write nothing but `*skip = 0` (plan buffer
  sentinel-checked). `cache_base` is passed as 0 — the entry only writes
  the value — so expected ptr words are the raw slot offsets.
- Generator drift note: regenerating the whole fixture set on the VM
  reproduces 99/105 committed fixtures byte-identically; 6 unrelated
  pre-existing fixtures (`gate_attn`, `gdn_recurrence`, `kv_append`,
  `round_f16`, `swiglu_interleaved`, `to_f16`) differ under regeneration
  and were left untouched — only the 7 new fixtures were added.

## 5. The `moe_grouped_s2` flag (GROUPED-NOTES §1) — mapping, not rewritten

`s2_gu_grouped_k` / `s2_down_grouped_k` cast `grp_ptr[g]` to a device
pointer inside MSL, which reads zeros on this backend (no fixture
coverage). It is **not** a sync bug (its launcher never syncs) and it is
not on the native pack's generate path (it is the non-native S2 arm of
the same verify/session calls), so it is not rewritten here. The based
form generalises to it directly, if a non-native pack ever needs it:
both its real call contexts already have a single natural base —
`hits.cache_base` (session token bodies and the verify VRAM call) and
the verifier staging region (the verify PCIe call) — so the kernels
would take `(base buffer, base_value, base_span)` and compute
`blob = base + (grp_ptr[g] - base_value)`, the exact transform
`native_gu_grouped_k` performs, with the same fixed-grid device plan the
S2 kernels already use. It needs its own fixture first (today it has
none): generate one S2 case in both plain and based form, like §4's
grouped pair.

## 6. Mac verification (2026-11-03, staging `~/qwen-native-staging/ws-syncfix/`)

Fresh build dir, build lock held, `maya` + `metal_kernel_parity` (+ the
ctest executables) built clean. Results:

- `native_expert_grouped` (existing fixtures, unchanged): **4/4 PASS**.
- `native_expert_grouped_based` (new fixtures): **4/4 PASS**, with
  per-case max_err identical to the plain entry's (0, 0.5, 0, 1.02e+03) —
  the two launchers agree case-for-case.
- `resident_plan` (new fixtures): **3/3 PASS**, bitwise.
- Full sweep, `--no-fork`: **112/112 PASS** (105 prior + 7 new), 0 by
  signal.
- `ctest -R 'metal|e2e'`: **6/6 PASS**.
