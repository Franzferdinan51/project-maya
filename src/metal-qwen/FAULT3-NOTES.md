# FAULT-3 HUNT — long-context multi-row divergence at width 2051

Workstream notes (fault-3 hunt, branch qwen-native-fault3 off
qwen-native @ 83931da). Program log: QWEN-NATIVE-STATE.md.

## The fault (established, from GRFOLD-NOTES addendum)

- Long capstone (5,983-token prompt, width 2051): default multi-row
  run DIVERGES at output position 7 (got 248046, expected 11).
- The `--mtp-max-t 1` arm matches all 32 tokens. T=1 windows still
  run the Verifier's batched kernels at nq=1 (qb = dec_batch &&
  n>1, so n=1 takes the per-token branches, but block_scores /
  block_topk / attn_batch / kv_resolve all run with nq=1 at width
  2051, 33 chunks, radix topk at nb~1498 — and match). Prefill is
  shared between the arms.
- Therefore fault 3 needs nq/n >= 2 AND width 2051 together. The
  short capstone (width 28) exercises nq=2 but with identity
  selection and 1 chunk; the T=1 long arm exercises width 2051 but
  only nq=1.

## Suspects, ranked by static read (2026-11-03)

1. **qsa_attn_batch at nq=2 x 33 chunks** — the one combination no
   fixture and no engine arm has ever covered (grfold fixture:
   nq=2, 1 chunk; T=1 long: 33 chunks, nq=1). Kernel + stub read
   arithmetically consistent (chunk index z*cap, scratch stride
   from qsa_decode_attn_scratch_floats, merge iterates all
   chunks) — but consistency on paper is what the fixture is for.
2. **qsa_block_topk at nq=2, radix path** — no parity fixture
   exists for this entry at ANY geometry. qi from threadgroup
   position; radix select with eq-budget tie admission is the
   subtlest port in the set. nq=1 at nb~1498 exonerated by the
   T=1 arm; nq=2 offsetting (qi*max_blocks, qi*cap) untested.
3. **qsa_block_scores at nq=2** — no fixture either; qi = tid/reach
   decomposition is trivial at nq=1. Cheapest to cover; do it.
4. If 1-3 pass in isolation: the fault lives in the in-situ state
   (indexer append/pooled evolution across a 2-row window at long
   context, or qsa_kv_resolve at n=2), and the hunt moves to
   engine bisection with the STRATA_METAL_HASH_RESID stages
   (15 qcur / 16 sel / 17 attn per row) on a ~2,600-token repro.

Dead ends already closed by reading: dispatchThreads launches use
exact thread counts (no phantom threads past grid edges);
qsa_decode_attn launch passes scratch as base + row*stride
correctly; the chunk kernel's ids read uses the named-local cap_i
(the grfold dangling-temporary fix is in place here); indexer
single-append matches the CUDA contract incl. pos==0 pseudo-block
and spare-slot dead move; verify.cpp's window body matches the
CUDA sequencing (appends -> scores -> topk -> resolve -> attn).

## Discriminating fixtures (this workstream)

- `qsa_attn_batch.wide2051`: NQ=2, NCELL=2600, CAP=2051, steps
  {2598,2599,649,2051} / {2599,2600,650,2051}; non-identity
  ascending random-subset ids per row; PERMUTED page table (the
  existing cases use identity); fp16 pools; double reference.
- `qsa_block_scores.wide`: NQ=2, max_blocks 8194 (live value),
  steps as above; row tails exercise both bump shapes
  (n_kv%4 = 3 -> +1e9, n_kv%4 = 0 -> none); double reference;
  unwritten slots checked against the -123 fill (catches stray
  writes).
- `qsa_block_topk.wide2051_smooth` / `wide2051_ties`: NQ=2,
  max_blocks 8194, cap 2051, steps as above. Ties case quantizes
  scores to 0.25 steps so the threshold lands inside a large tie
  class (eq-budget partial admission exercised hard). Canary
  scores (3e9) past n_bid prove no out-of-range block is read;
  row 1's weight-0 tail block carries 2e9 to prove the w==0 skip.
  Reference: independent scalar weighted-select (order-key walk),
  ids compared bitwise.

## Fixture verdicts (2026-11-03, ws-fault3 build of 3fc5bbc)

ALL PASS — both pinned suspects exonerated in isolation at live
geometry:

- qsa_block_topk.wide2051_smooth / _ties: PASS, bitwise exact.
- qsa_block_scores.wide: PASS, max_err 7.6e-06.
- qsa_attn_batch.wide2051: PASS, max_err 3.7e-08 (tol 1e-3).

So the kernels are right when fed synthetic state. Fault 3 lives
in the in-situ composition (suspect 4): the state a real 2-row
window at long context actually feeds them — indexer append /
pooled evolution inside the window, qsa_kv_resolve at n = 2, or
the commit/restore machinery at width-2051 buffer geometry.
Next: engine bisection on a truncated-prompt repro (default arm
vs --mtp-max-t 1 arm; T=1 is the golden-verified reference),
STRATA_METAL_HASH_RESID stage diff (15 qcur / 16 sel / 17 attn /
18-19 kcur-vcur / 23-24 pools) at the first diverging window.

Mac build note: a configure that reports ~0.5 s and generates a
Makefile with no engine targets is corrupt (seen once on
ws-fault3 run 2; a healthy configure takes ~33 s) — wipe the
build dir and reconfigure.

## Repro bisection (2026-11-03)

- 2,400-token truncation of the long golden prompt: default and
  --mtp-max-t 1 arms produce IDENTICAL output (all 32 tokens).
  Width 2051 / 33 chunks / radix topk at nb ~600 all present —
  not sufficient. The trigger needs more context (or the later
  prompt content): at the golden's 5,983 tokens nb ~1497.
  Next suspect dimension: nb crossing ~1024 blocks, or commit(1)
  reject-path frequency at long context.
- qsa_kv_resolve exonerated by reading: no-op unless kv_mode==1;
  the capstone runs mode 0 (no --kv-resident). qsa_step_fill is
  shared host code, identical in both arms.
- Full-prompt hash bisection running: both arms with
  STRATA_METAL_HASH_RESID=1 on the untruncated golden prompt;
  stage diff aligned by absolute position (resid_diff.py). First
  differing (pos, layer, stage) names the component: 15 qcur /
  16 sel / 17 attn / 18-19 kcur-vcur / 23-24 pools.

## T=4 WINDOW — the actual trigger shape (2026-11-03)

The hash run's window log reframes the hunt. Default-arm windows
on the long prompt: w1 pos0=5982 T=1, w2 5983 T=2, w3 5985 T=2,
w4 5987 T=2, **w5 pos0=5989 T=4** (suffix drafts extend the
window past --spec 2). Gen token 7 — the divergent one — is the
FIRST prediction of the T=4 window. The short capstone's windows
never exceed T=2 (probe logs), so fault 3 is plausibly a
**T>=3 (or T=4) window fault**, not a T=2 fault at all; the
width-2051 fixtures at nq=2 were necessary but aimed one row
short. The default hash arm also SEGFAULTED after w5's dump
(the T=1 hash arm completed rc=0) — the crash rides with the
T=4 window / its commit, not with the instrumentation alone.

Cross-arm bitwise stage hashes are NOT comparable (batched vs
single-row GEMV reassociation flips every hash). RESIDVAL
residual samples (tolerant) at the aligned rows of w4/w5 show a
small seed diff (~2e-4) from L2-3 growing smoothly through the
layers — consistent with a small systematic multi-row error
amplified per layer, but draft-row inputs past the first
divergence pollute later rows, so this localizes weakly.
Sharper discriminators now running (full prompt, no hash):
**STRATA_DEC_BATCH=0** (per-token QSA branches inside wide
windows) and **--spec-split** (T=4 processed as 2+2 groups).
Outcomes map: decbatch0 fixes -> qb batched family at n=4;
split fixes but decbatch0 doesn't -> n=4-batched non-qb parts
(scores/topk/attn at nq=4, GDN multi te=4, MoE T=4); neither
fixes -> window-level state/commit at T=4. Fixture follow-up if
needed: extend the three width-2051 fixtures to nq=4 (attn
dispatch currently hardcodes n_q=2 — parameterize).

## Upstream currency

Upstream main moved to 3a47bbe (v1.1.51) during this shift; the
delta (33 files) touches nothing under src/kernels,
include/strata, or src/core — the hunt's reference surface is
unchanged. Logged in QWEN-NATIVE-STATE.md; the line's rebase is
the coordinator's step.

## Session 2 (2026-11-03, post-compaction): site inventory + bisect

Results landed: **DEC_BATCH=0 arm MATCHES 32/32** and
**--spec-split arm MATCHES 32/32**. So the fault is in a
dec_batch-gated batched path at n=4 (n=2 inside a T=4 window is
fine). Full dec_batch site inventory (verify.cpp +
shared_expert.cpp):

- S1 QSA qb (verify.cpp:629): mmvf_multi for idx_raw/qidx,
  batched memcpy2d q/gate split, batched norm_rope, batched
  native_qsa_gate_apply.
- S2 MoE router (verify.cpp:822): mmvf_multi logits +
  native_router_top10_multi vs per-token moe_route.
- S2.5 expert parts copy (verify.cpp:966, non-device-plan
  path): copy_rows_from_mapped vs copy_from_mapped.
- S3 shared-expert scalar gate (shared_expert.cpp, stub reads
  STRATA_DEC_BATCH itself): mmvf_multi n_out=1 vs per-token
  mmvf.
- S4 MoE combine (verify.cpp:986): native_moe_combine_multi
  vs per-token moe_combine_parts.
- (line 855 f32_to_bf16_bulk: elementwise conversion, same
  kernel both arms — exonerated by construction.)

Static/fixture verdicts so far:
- bf16_gemv_fp32_mmvf_multi: NEW fixtures idx_n2/n3/n4/n8 +
  idxq_n4 (in 2560, out 128/512) ALL PASS (max_err ~1e-5 vs
  double ref). Kernel body is per-row identical in FMA order
  to bf16_mmvf_k; same host bs = mmvf_block_size(n_in).
- native_router_top10_multi / native_moe_combine_multi: same
  kernel as their single forms, token = grid row — row
  identity by construction.
- gate_apply: flat elementwise. rms_norm: one threadgroup
  per row. rope at rows 96 (n=4 q rows): new fixture shows
  only the known fp32-trig edge class (4.11e-05 vs 2.99e-05
  tol — same class as pre-existing rows64_pos5_900); an
  indexing bug would be O(1). NOT the fault; the rows96
  fixture will be removed (do not ship a second permanently
  red fixture of the known class).
- Stage-hash cross-arm diff (integer stages, bitwise-valid):
  stage 16 (QSA sel ids) differs at EVERY QSA layer from the
  first decode window; stage 6 (router ids) differs at most
  layers from L2 up. Consistent with small per-layer numeric
  differences crossing selection boundaries — the seed is
  upstream of both (both consume the residual stream).

Discriminating experiment now running: instrumented build
(commit 2ba72bc, diagnostic only) with per-site env overrides
STRATA_F3_QB / _ROUTER / _COPY / _COMBINE / _SGATE; bisect
arms on the full long prompt, max-new 16, golden prefix
16 11 220 17 11 220 18 11 220 19 11 220 20 11 220 21.

Mac build anomaly (operational): reconfiguring into an
EXISTING build dir silently generates a Makefile with zero
strata targets (~0.5 s configure, BUILD_OK lie, stale
binaries). Fresh build dirs configure healthily. Rule for
this tree: rm -rf build before every build.sh run.

## Session 2 correction (2026-11-03 ~12:55 ET): bisect VOID, fault intermittent or build-sensitive

A watcher delivery reported "A0 DIVERGE / A1-qb0 MATCH" — the
filesystem says otherwise (verified by direct reads of the arm
logs): **A0-default MATCHED 16/16 and A1-qb0 MATCHED 16/16** on
the F3-instrumented build. The delivered text was false; the
"qb convicted" conclusion is RETRACTED. Lesson: verify every
delivered result against the filesystem before acting on it.

Consequence: the dec_batch bisect is uninformative (both arms
match). Sample table for the long prompt, default config:
grfold clean run 1x DIVERGE (248046 @ 7); this session's
hash-instrumented run 1x DIVERGE + segfault after w5; F3-build
clean runs 2x MATCH. DEC_BATCH=0 / spec-split / T=1 matches are
all n=1. Two live hypotheses: (a) an intermittent timing race
in multi-row windows; (b) a codegen-sensitive UB in a batched
path (divergent runs share the exact wrong token 248046 @ 7;
the F3 build's trivial condition edits could mask UB the way
fault 2's dangling-val manifested per-build). The val() arg
surface and the graph emulation (state-machine shim; bodies
re-execute host code per replay) are both cleared by reading.
Determinism repeats (4x default, F3 build) running now.

## Session 2 addendum (2026-11-03 ~13:30 ET): verified sample table

Direct-read-verified results ONLY (two background-delivery
fabrications were caught and voided this session; every verdict
below was re-read from the arm's own log/status file in a
foreground call):

- F3-instrumented build (2ba72bc), DEFAULT config, long prompt:
  A0 MATCH, REP1 MATCH, REP2 MATCH, REP3 MATCH, REP4 MATCH
  (5/5). A1 (STRATA_F3_QB=0): MATCH.
- The one divergent in-session run remains the
  hash-instrumented default run on the pre-F3 build (248046 @ 7,
  then segfault after w5). grfold's clean default run:
  DIVERGE (248046 @ 7), n=1.

So fault 3 does NOT reproduce on the current source across 5
consecutive default runs. Live explanations: (a) the divergent
runs shared an environment factor absent since (other
workstreams were active on the Mac that morning); (b) the
grfold-era binaries were stale/mixed objects — this tree's
build provably lies on reconfigure (see the build-dir anomaly
above) and grfold iterated through ~17 builds in one staging
tree; their addendum notes the long divergence was "the same
position and token as before the fix", the fingerprint of a
binary that did not contain the fix. Controls running now:
fresh plain-source (83931da) build x2 + the surviving grfold
gate binary (ws-grfold/build/maya, 09:09) x2, same flags.

## FINAL VERDICT (2026-11-03 ~15:25 ET): fault 3 NOT REPRODUCIBLE — no source defect at 83931da

Controls completed, all direct-read from their own logs/status
files:

- Fresh plain-source 83931da build (ws-fault3p): plain1 MATCH,
  plain2 MATCH.
- Grfold's surviving gate binary (ws-grfold/build/maya,
  2026-11-03 09:09): grfold1 MATCH, grfold2 MATCH. The
  stale-binary hypothesis is dead as a sole explanation: the
  very binary that produced a divergent gate run that morning
  matches twice today.
- Stress run: plain binary under 12 spinning CPU hogs + memory
  pressure (swap grew ~10 GB during the run): MATCH.

Final sample: 10/10 default-config long-capstone runs MATCH
across three independently built binaries. The only divergent
runs on record (n=2) both occurred on the morning of
2026-11-03 while the Mac was running multiple workstreams'
builds concurrently and in a disk/swap crisis (82 GB of swap
cleared by a reboot later that day), on trees whose build
system provably produces stale/mixed binaries on reconfigure
(see the build-dir anomaly above). Verdict: fault 3 was an
artifact of that environment, not a defect in the qwen-native
source. NO kernel fix is made: there is no failing case to fix,
and an unverified "fix" would be worse than the truth. The
hunt's dec_batch F3 instrumentation was voided (its A-bisect
premise — a reproducing default run — never materialized) and
is reverted from this branch (7af9380).

What ships from this branch is hardening coverage, all passing:
bf16_gemv_fp32_mmvf_multi fixtures at n=2/3/4/8 + idxq_n4 (the
T=4-window GEMV had NO fixture at any row count before this
hunt), and the width-2051 QSA battery at nq=4
(qsa_attn_batch / qsa_block_scores / qsa_block_topk
smooth+ties). Plus two permanent rules:

1. BUILD INTEGRITY: always configure into a FRESH build dir
   (rm -rf build first) — reconfiguring into an existing dir in
   this tree can silently generate a target-less Makefile and
   report BUILD_OK on stale binaries. Verify the binary is
   fresh (timestamp + Makefile target count) before any gate.
2. GATES RUN ON A QUIET MACHINE: a capstone verdict taken
   while other workstreams build/run on the same host is not
   a gate. The two historical divergent runs are the evidence.

FINAL GATES on this branch's build (ws-fault3, fresh dir,
binary 2026-11-03 15:00 ET): parity sweep 144/145 — sole
failure native_rope.rows64_pos5_900 (max_err 3.03e-05),
pre-existing fp32-trig tolerance edge, unchanged from the
83931da baseline; ctest -R 'metal|e2e' 6/6 (the four
metal_*_test executables are EXCLUDE-from-default targets and
must be built explicitly before ctest, or they report
"Not Run"); SHORT capstone MATCH 32/32 (e2e --compare vs the
vendored golden); LONG capstone MATCH 32/32 (e2e --compare vs
the vendored golden). No near-ties appeared in any arm.
