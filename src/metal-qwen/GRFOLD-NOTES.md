# GRFOLD workstream notes (branch qwen-native-grfold)

## Scope

ROWFIX exonerated the whole post() FFN composition and named the remaining
surface: (1) the hyper-connection fold itself — `fused_gr_read`'s APPLY path
(bo_prev folded into R_out) as driven per token by the window body; (2) QSA
layers' in-window KV/attention. This workstream fixtures (1) at the exact
boundary, and — if the fixture passes — bisects inside the window body with
per-layer residual hashes (T=1 vs T=2 arms) to find the first layer whose
residual stream differs given identical inputs.

## What the fixture audit found before any run

- `FIXTURE-FORMAT.md` classified `fused_gr_read` / `fused_gr_read_multi` as
  "not-fixtureable — GLM engine family; never called on the Qwen path". That
  is wrong: the entry is on the Qwen path in three places —
  `src/core/layer.cpp` (both fused halves of `block_layer_pre`),
  `src/core/verify.cpp` (`gr_read_group`, both halves of every window layer),
  and `src/core/mtp.cpp`. Corrected in the table.
- Upstream's only coverage, `src/kernels/gr_parity.cpp`'s
  `fused_multi_lds_parity`, compares the CUDA multi against the CUDA single
  (the same semantics twice — no independent reference) and uses a SEPARATE
  R_out buffer. The verify body drives the fold with `R_out == R` (in-place),
  a shape no upstream case exercises. So the apply fold has never been
  checked against a scalar reference, on any backend.
- Static read of the Metal kernels (`kernels/gr.metal`) against
  `src/kernels/cuda/fused_gr.cu`: norm / down+inject GEMV / up all transcribe
  the CUDA math (gate `2*sigmoid(inj[c]/hc)`, R' folded on the fly in every
  stage, R_out written only under apply, mixed = mean_c xn*sigmoid(w_up.lo)).
  Stub argument order in `stubs/fused_gr.cpp` matches the MSL signatures.
  No static slip found — the fixture decides.

## The fixture: `fused_gr_read`

Generator `gen_fused_gr_read()` in `tests/gen_fixtures.cpp`, dispatch
`fused_gr_read` in `tests/metal_kernel_parity.cpp`. Real geometry only
(the kernel supports nothing else): N=2560, HC=4, LR=320. Reference is a
double-precision transcription of the CUDA fused kernel (see the generator
comment for the formula chain). Inputs are drawn so the fold cannot hide:
bo_prev at a real block-output scale (±1.2), inj_prev in ±3 so the per-stream
gates `2*sigmoid(inj/4)` span ≈0.64–1.36. R_out and inject_out are prefilled
with a sentinel (555.25): apply=0 must leave R_out untouched, a missing
w_inject must leave inject_out untouched — an unwanted or missing write is
itself the failure. Outputs compared: R_out, lo, rs, inject_out, mixed.

| case | shape |
|---|---|
| apply1_inject_t1 | single token, apply=1, w_inject present |
| apply0_inject_t1 | single token, apply=0 control |
| apply1_noinject_t1 | single token, apply=1, w_inject null |
| apply1_inject_t1_inplace | single token, apply=1, R_out == R |
| apply1_inject_t2_multi | fused_gr_read_multi T=2, separate R_out |
| apply1_inject_t2_inplace | fused_gr_read_multi T=2, R_out == R (the verify shape) |
| apply1_inject_t4_multi | fused_gr_read_multi T=4 |

Result: PENDING MAC RUN (the fixtures were generated and are present in
the Mac staging copy; the parity dispatch run is still owed — it was
queued behind the residual bisection below, which produced the sharper
evidence first. The fold is in any case cleared by the bisection:
stage 0/1/3 hashes below are the fold's outputs in situ.)

Note: regenerating the fixture set on this VM also rewrote 8 unrelated
committed fixtures (gate_attn, gdn_recurrence, kv_append.f16_host,
round_f16, swiglu_interleaved, to_f16, verify_moe_assembly small cases) —
a pre-existing generator/toolchain byte-instability, NOT touched by this
workstream (reverted; only the new fused_gr_read fixtures are added).

## Residual-hash bisection (STRATA_METAL_HASH_RESID)

Env-gated, off by default. `resid_hash` (verify.metal `resid_hash_k` + CUDA
twin in verify_kernels.cu): a deterministic order-independent integer
fingerprint, h = Σ (bits(x[i]) + φ)·(i+1)·FNV-prime mod 2^64, written in
stream order into a slot buffer. `Verifier::record_window` fills, per token:
stage 0 = R after gr_read(0), 1 = mixed after gr_read(0), 2 = bo after the
mixer, 3 = R after gr_read(1), 4 = bo after post()'s combine.
`Verifier::run` reads the slots back after the window sync and prints them
(first 40 windows). Comparing a T=1 arm against a T=2 arm over the same
committed tokens: the first (layer, stage) whose hash differs, given the
previous stage matched, is the fault's home — stage 0 differing while the
previous layer's stage 4 matched convicts the fold at gr_read(0); stage 2
differing with stage 1 matching convicts the mixer (GDN/QSA in-window);
stage 3 convicts the gr_read(1) fold; stage 4 the FFN composition with real
inputs.

Result: RUN on the Mac (staging ws-grfold, build-run17 binary), arms
`probe-t1` (`--mtp-max-t 1`) and `probe-t2` (default T=2), plus two
rerun variants (probe-t1cpu / probe-t2cpu logs). The deep instrumentation
(23 stages, per-entry parts hashes, plan/VAL dumps) is committed as
76c8a05. Verdict below — it MOVES the fault out of the fold entirely.

## Bisection verdict (2026-11-03, second grfold driver)

Outputs: T=1 arm reproduces the golden through position 13 (position 14
is the documented near-tie). T=2 arm diverges at position 3, as before.

Per-position hash comparison (same committed token, T=1 arm vs T=2 arm),
layer 0 first, stages in post() order:

- Position 25 (a T=1 window in both arms): every cell identical.
- Positions 26 and 27 (the two rows of the first T=2 window): stages
  0 (R after gr_read(0) — THE FOLD), 1 (mixed), 2 (bo after mixer),
  3 (R after gr_read(1)), 5 (mixed after gr_read(1) = FFN input),
  6 (router ids+weights), 7 (shared expert out), 9 (nat_xq) all MATCH
  at layer 0. **The hyper-connection fold is bit-exact across tilings;
  the fused_gr_read apply path is EXONERATED in situ**, as are the
  mixer, router, shared expert and the q8_1 quantization.
- The FIRST divergence is stage 4 (bo after post()'s combine) at
  layer 0, with stages 13 (parts after the mapped copy), 14 (hit_out
  before moe_hit_add) and 8 (parts after moe_hit_add) differing —
  i.e. inside the expert-parts production, exactly the composition
  ROWFIX fixtured — but with a twist the fixture cannot see (below).

### The plan rule, and why the fixture could not see this

`expert_pool_dispatch_multi` (src/core/expert_source.cpp) classifies
each DISTINCT routed expert: resident in the expert cache → GPU/VRAM;
otherwise a miss. The LAST `m = (nmiss * pcie_num) >> 8` misses in
routing order are promoted to the GPU via PCIe staging
(`--pcie-frac`, default 0.55 for native packs → pcie_num 141); the
remaining misses are computed by the CPU pool into ymiss. At layer 0 in
these runs the cache is cold (0 resident), so for a 1-token window:
entries 0..4 CPU, entries 5..9 GPU. For the 2-token window at pos0=26:
entries 0..8 CPU, entries 9..19 GPU (dst {9..19}, tok {0,1×10}).

Per-entry parts hashes (RP lines) for the pos0=26 windows, layer 0:

- Every entry computed by the SAME path in both arms is bit-identical
  across arms (token 26 e0..e4 CPU/CPU, e9 GPU/GPU; token 27 e5..e9
  GPU/GPU).
- Every entry whose path FLIPPED differs: token 26 e5..e8 (GPU in T=1,
  CPU in T=2) and token 27 e0..e4 (CPU in T=1, GPU in T=2).

The verify_moe_assembly fixture fabricates ymiss CPU rows as arbitrary
values and blobs as fabricated weights: it proves the ASSEMBLY correct
given correct inputs, but it never computes a real expert on the CPU
path nor fetches a real blob through fetch_dma/rebase_ptrs. The fault
therefore sits in one (or both) of the two real expert producers in a
multi-token group context — the CPU pool's `run_split_multi_native`
jobs, or the PCIe staging path (fetch_dma slot order / rebase_ptrs /
the grouped kernel reading staged blobs) — NOT in the fold, NOT in the
assembly, NOT in the plan arithmetic (the dst/tok/start2 the plan
publishes match the rule above exactly, verified against three
windows including the duplicate-expert window at pos0=30).

Caveat on magnitude: a hash flip is any bit difference. The pure-path
arms below are the discriminator between "one producer is wrong" and
"the producers round differently and the split change alone perturbs
the window" — ROWFIX's fixture measured the GPU grouped path at
max_err 0.03–0.08 vs the double-precision reference, so path rounding
is real and the arms decide whether it explains a position-3 flip.

### Pure-path arms (COMPLETED) — the tier suspicion does not survive them

- t1cpu (`--mtp-max-t 1 --pcie-frac 0`): GOLDEN through position 13.
- t1gpu (`--mtp-max-t 1 --pcie-frac 1`): GOLDEN through position 13.
- t2cpu (`--mtp-max-t 2 --pcie-frac 0`): FAULTS at position 3, token
  stream identical to the mixed arm.
- t2 mixed (default frac): FAULTS at position 3.

So neither expert producer is wrong in single-token windows, and the
CPU producer alone in a 2-token window still produces the fault. The
RP cross-path differences above are real (the two producers round
differently, and the split point moves with window size), but they are
NOT the position-3 fault. The earlier "first divergence at layer 0
stage 4" was an artifact of comparing arms whose tier splits differed:
with the producer held constant (t2cpu vs t1cpu), layer 0 is clean in
the fault window.

### Correct-baseline bisection (t2cpu vs t1cpu, both all-CPU)

- Window pos0=25 (T=1 in both arms): identical everywhere.
- Window pos0=26, token 0: NO differing stage at any of the 48 layers.
- Window pos0=26, token 1: first difference at **layer 3 — the first
  QSA (full-attention) layer — stage 2 (bo after the mixer)**. GDN
  layers 0–2 are bit-clean for both rows. Everything downstream is
  inherited.
- Window pos0=28, token 0: first difference also layer 3 bo — its
  layer-3 state (KV pool / indexer) was written by the fault window,
  and its own qcur is wrong again (below).

### Inside the QSA mixer (stages 15–23 added)

At layer 3, fault-window row 1 (and pos0=28 row 0):

- kcur, vcur (K/V GEMV + norm + RoPE + append): MATCH.
- sel (indexer block selection): MATCH.
- **qfull (Q GEMV output): MATCH.**
- **qcur (after the q/gate split + RMS norm + RoPE): DIFFERS** — the
  first wrong value in the whole window.
- attn, bo_mix: differ as consequences.

The fault is between the Q GEMV and qcur: the `cudaMemcpy2DAsync`
q/gate split, the RMS norm, or the RoPE — the only three operations
there. The norm/RoPE kernels are per-row independent and correct at
T=1 shapes; the split is the one operation whose CUDA semantics are
stream-ordered while our runtime's implementation is not:

### The runtime defect

`src/metal-qwen/runtime/metal_runtime.mm`: `cudaMemcpyAsync` and
`cudaMemcpy2DAsync` IGNORE their stream argument and perform an
immediate host `std::memcpy`. For host-to-device copies from stable
host memory that is safe; for a DEVICE-to-DEVICE copy enqueued between
two kernels on the same stream it is a data race by construction: the
copy reads the source at host call time, while the producing kernel
(the Q GEMV, native_mmvq) is still in flight — in multi-row windows
its later rows are exactly what gets read early/stale. The Q split is
the only D2D copy sitting between compute stages in the layer body
(verify.cpp qb and per-token branches; the same pattern exists in
layer.cpp's qsa_layer and mtp.cpp's MTP head). Single-token windows
are saved by timing (a 1-row GEMV lands before the host reaches the
copy); the vendored CUDA engine is saved by real stream semantics.

A host `cudaStreamSynchronize` before the copy is NOT a valid fix
here: the body thread is the same thread that must raise the expert
pool's flags, and the stream already holds spin kernels waiting on
those flags — syncing inside the body deadlocks it. The copy must
join the stream.

### The fix (backend only: runtime + kernels)

- `src/metal-qwen/kernels/misc.metal`: new `memcpy_bytes_k` and
  `memcpy2d_bytes_k` copy kernels (grid-stride, byte-exact).
- `src/metal-qwen/runtime/metal_runtime.mm`: async copies with kind
  `cudaMemcpyDeviceToDevice` and a non-null stream now launch the
  matching copy kernel ON that stream (stream-ordered, capture-safe:
  the launch records like every other kernel op). All other kinds
  keep the immediate host path their choreography relies on.
- Confirmation still owed from the run18 logs (stage 21 = qcur right
  after the split, stage 22 = after the norm): the prediction is
  stage 21 already differs. A `native_rope` fixture (4 cases incl.
  rows=64 with two distinct positions, the exact qb shape; the entry
  was "planned" in FIXTURE-FORMAT.md until now) arbitrates the RoPE
  kernel standalone.

## Gates

PENDING MAC RUN (Mac unreachable since ~06:29 ET 2026-11-03; when it
returns: read stages 21/22 from the ws-grfold run18 logs, run the
native_rope + fused_gr_read fixtures, fresh-build with the runtime
fix, sweep --no-fork, ctest -R 'metal|e2e', short capstone with the
bar MATCH through position 13, then the long capstone. Disk gate
before any engine run: 27 Gi free at the 06:17 ET check.)

## Resolution (2026-11-03, second session) - TWO faults, both fixed

### Fault 1 (the position-3 flip): the immediate-host-memcpy split

As recorded above: the runtime's async D2D copies ignored the stream
and memcpy'd at host call time, racing the Q GEMV; the fix (copy
kernels launched on the caller's stream, commit dbe6c95) was written
in the first session but never reached the Mac staging tree - the
staging copy predated it (grep: 0 hits for memcpy2d_bytes_k). Synced
and rebuilt 2026-11-03 ~08:1x ET. Value-level confirmation of the
diagnosis, from RESIDVAL2 samples at QSA layer 3, position 27 (row 1
of the first T=2 window), T=1 vs T=2 arms: qfull deviation 0 (the Q
GEMV output bit-identical), qcur-right-after-the-split deviation
1.68 (rel 1.0), kcur/vcur deviation 0. After the fix those samples
are bit-exact and the short capstone moves from diverge-at-3 to
diverge-at-6.

A hazard found on the way: commit 76c8a05's instrumentation had
SWAPPED production qcur math under STRATA_METAL_HASH_RESID
(native_qsa_rms_norm_weighted + native_rope_apply instead of
norm_rope, both branches). Every hash-gated probe therefore measured
a debug math path, not production. Removed (decontamination commit);
observers kept, math is production everywhere now. Lesson pinned:
instrumentation may observe, never substitute.

### Fault 2 (the position-6 flip): dangling temporary as ids stride

With fault 1 fixed, the --spec-corrupt arm (T=2 windows, commit(1)
only) matched the golden on all 32 tokens, while --spec-oracle
(commit(2)) still diverged at output position 6 - kept row-1 state.
Single-writer value probes (RESIDVAL4) at layer 3, position 27:
qcur bit-exact, attention output off by rel 0.45; K/V pool verified
bit-exact (hash over the paged region + raw page dump: cell 27
exact, cell 26 within batched-GEMV rounding), steps and the full
32-id selection identical. A new parity fixture (qsa_attn_batch,
n_q=2 vs a scalar reference) then failed in isolation at
attn[6144] - the first element of query row 1 - and its scratch
dump showed row 1's partials with l = width exactly and m equal to
the score of cell 0: uniform attention over cell 0, i.e. the chunk
kernel read all-zero selection ids for row 1.

Root cause: `KernelArg::val` binds its argument BY REFERENCE until
launch() encodes it (metal_kernel.hpp states the contract: a named
local at the call site is the intended shape).
`qsa_decode_attn`'s launch_attn passed `val((int) cap_ids)` - a
cast temporary that died at the end of the args-array statement.
The chunk kernel consequently read a garbage `cap` (the per-query
ids stride). Row 0 is immune (its ids offset is 0 under any
stride); every row z >= 1 indexed its selection far outside the
ids buffer and read zeros in practice. Fixed by binding
`const int cap_i = (int) cap_ids` at the call site. The same
defect class was fixed in gr.cpp (`val(0)` / `val(1)` literals ->
named locals); a tree-wide grep found no other temporary sites.

### Final gates (fixed build, Mac staging ws-grfold, 2026-11-03)

- qsa_attn_batch.nq2 + nq2_dup fixtures: PASS, max_err 2.38e-07
  (pre-fix: FAIL at attn[6144], err 1.62). Permanent regression pins.
- fused_gr_read: 7/7 PASS. native_rope: 3/4 - rows64_pos5_900
  misses its tolerance by 4e-7 on one element (3.03e-05 vs
  2.99e-05; pre-existing, an fp32 trig-precision edge at RoPE
  position 900, unrelated to either fault).
- Full parity sweep --no-fork: 130 passed, 1 failed (the
  native_rope case above) of 131 fixtures.
- ctest -R 'metal|e2e': 6/6 passed.
- SHORT CAPSTONE: **MATCH - all 32 greedy tokens identical** to the
  vendored-engine golden (bar was MATCH through position 13; the
  position-14 near-tie landed on the golden side in this run and
  is reported as part of the full match, not tuned for).
- LONG CAPSTONE: see QWEN-NATIVE-STATE.md log (ran after this note
  was written; verdict recorded there and in the merge commit).

### Addendum - the LONG capstone after the fix (2026-11-03)

Default-spec long run (5,983-token prompt): DIVERGE at position 7,
got 248046, expected 11 - the same position and token as before the
fix. But the long T=1 run (--mtp-max-t 1) matches the golden on all
32 tokens. So prefill state, indexer selection at width 2051, RoPE
at position ~6000, and single-row decode are all correct end to
end; what remains is a THIRD fault, specific to multi-row windows
at full selection width (2051 selected of ~5990 cells). The short
capstone cannot see it (width 28 = identity selection). Pinned
starting point for the next hunt: the batched indexer append /
batched topk in verify.cpp (per-row scores and queries through the
shared single-row QsaState buffers) at width 2051, and the chunk
kernel's output-merge invocation at that width (the row-merge
launch geometry changes: merge is per (z, head) over 33 chunks).
A mid-length prompt (~2,600 tokens, width saturated) with a fresh
vendored golden reproduces it with ~8-minute prefills.
