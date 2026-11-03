# Capstone divergence at position 3 — investigation notes

Worktree `wt-diverge`, branch `qwen-native-diverge` off `qwen-native`
(1f13088). All runs on the Mac against staging build-v12 (the current
merged tree) and the vendored reference binary; logs and dumps in
`~/qwen-native-staging/ws-diverge-*`. **Step 3 (fix + rebuild) is BLOCKED:
the Data volume had 300 MiB free at start of work (209 MiB at end), below
the ~900 MiB floor — no build dir was created. This file is the evidence
chain and the localized fault; the tree is ready for the fix once space
exists.**

## Step 1 — control: the golden stands, the bug is native

The vendored CLI (`/Volumes/Sabrent/project-maya/build-metal/maya-metal-qwen`,
the same binary that serves on :8082, run standalone with the serve's own
flags, `--spec 2 --mtp /Users/duckets/Strata-data/mtp/rt --max-context 4096
--greedy --max-new 32`, short prompt) emitted, byte for byte, the golden:

```
17 11 220 18 11 220 20 11 220 22 11 220 16 16 11 220 16 18 11 220 16 22 11 220 16 24 11 220 17 18 11 220
```

So `goldens/q2_0_short.mqgt` (serve provenance) is **not** the artifact.
The divergence is in the native backend. (Log: `ws-diverge-control.log`.)

## The native fault, reproduced and narrowed

Native repro (build-v12/maya, coordinator's probe args, staging `mtp-rt`,
`STRATA_ROUTE_TAIL_SKIP=7`) reproduces the coordinator's 32 tokens exactly,
including `DIVERGE at position 3 (got 17, expected 18)`.

Arms run (all `--greedy --max-new 32`, short prompt, unless noted):

| Arm | Result at pos 3 | Reading |
|---|---|---|
| T=4 windows (default `--spec 2`) | **17 (wrong)** | the reported fault |
| `--mtp-max-t 1` (all windows T=1) | 18, golden through pos 13 | single-row windows + commit(1) are clean |
| `--spec-corrupt 1` (every draft wrong, T=4, all a=0) | 18, golden through pos 13, identical to T=1 arm | commit(1) after a multi-row window does **not** leak draft-row state |
| `--spec-oracle` (golden as drafts, T=4) | **17 (wrong)** | not a draft-quality or accept-loop fault |
| oracle + `--mtp-max-t 2` (`--max-new 8`) | **17 (wrong)**, tiling fully pinned (below) | commit(2) is the poisoned step |
| T=2 windows (MTP drafts) | **17 (wrong)** | one draft row suffices |
| `STRATA_DEC_BATCH=0` | identical 32 tokens to batched | batched router/GEMV/combine-multi paths exonerated |
| `STRATA_VERIFY_DEVICE_PLAN=1` | **17 (wrong)** | host pool plan-publish / flag-A gate path exonerated |

**Pinned tiling (oracle + T=2, no suffix windows, accepted 0:3 1:3):**
W1 T=1 @25 emits 17, commit(1). W2 [17, oracle 11] → outv = [11, 220],
a=1, **commit(2)**. W3 @28, x=220 → **outv[0] = 17 (wrong; the same state
reached by two commit(1)s in the T=1/corrupt arms gives 18)**. So the
state a commit(2) leaves behind is wrong, while two commit(1)s over the
same tokens are right, and commit(1) after a window whose draft rows were
garbage is also right. The corruption enters exactly when a window's
row-1 (draft) row is *kept* — i.e. through the row-≥1 intermediates the
commit replay consumes, or through state those rows wrote during the
window that commit does not roll back.

**The head is faithful; the logits themselves are wrong.**
`--dump-logits` on the T=4 run (file kept: `ws-diverge-logits.bin`):
row 28 (input token 220 at position 28) has argmax **17 @ 24.42 with
18 @ 23.47** — the sampler/`m_out_` readback reports the head's true
argmax, so task suspects (b)/(c) (stale/swapped out-buffers) and (a)
(the accept loop — also read-verified in `generate.cpp`: emitted tokens
are `outv[0..a]`, the target's picks) are **eliminated**. The error is
upstream of the head, in the window's computed state, and it is small
(~1 logit at the flip point), not garbage.

**Logit-vector diffs, T=4 dump vs T=1 dump (identical contexts):**
row 25 identical (max diff 0.000 — prefill + first window agree);
row 26 (first row computed inside a T=4 window, as row 0): max diff
0.366, mean 0.054; row 27: max 4.290; row 28: max 2.496 and the flip.
So multi-row window compute perturbs **even row 0**, and later rows
more. That is far above summation-order noise and it is *not* the
DEC_BATCH-gated kernels (arm above).

**Subsystem bisection (serve + `STRATA_STATE_HASH_GDN=1`):** two serve
runs of the existing binary (`GEN 3 <short prompt ids>`), one with
`--mtp-max-t 1`, one default T=4 — same emitted tokens [17, 11, 220],
same committed prefix. Per-GDN-layer state fingerprints: **layer 0
identical (8d28); every GDN layer 1..36 differs.** Layer 0's committed
state is replayed from its SSM intermediates (embeddings only) and
agrees; the divergence enters through the residual stream between
layer 0's state and layer 1's — i.e. via **layer 0's FFN (expert) output
for a window row**, the one stage whose work is grouped across the
window's rows (grouped expert compute + parts assembly), computed by
the pool/grouped path rather than per-row kernels. Everything after
inherits it, which is why the flip surfaces a window later at a
near-tie row.

Side observation: the serve `STRATA_STATE_HASH` block crashes after the
GDN per-layer print, before the summary line (both arms, same place —
the ple/kv hashing that follows). A debug-path defect, noted here, not
chased (out of scope, and see the build block).

## Localized fault (for the fix)

In scope per the evidence: the **multi-row MoE expert execution in the
verify window** — `Verifier::record_window`'s `post(l, grp)` assembly
in `src/core/verify.cpp` as served by the Metal backend layer:
`native_expert_grouped_based` over the group's whole entry set,
`moe_hit_add`, and the `parts_` assembly from the mapped miss rows
(`copy_rows_from_mapped` / `copy_from_mapped` in our stubs) — a
row-indexing/assembly slip that perturbs every row of a multi-row
group, growing with row index. Per-entry fixtures pass because they
test entries in isolation; the defect is in how a window's rows'
entries are composed. The commit replay machinery itself
(`gdn_conv_commit_k`, `gdn_step_norm_k` n_keep handling,
`copy_indexed_k`, the QSA indexer append) was read line-by-line
against the CUDA originals and matches; the accept loop, sampler,
head readback, drafters, PLE staging, and the flag/gate machinery are
each exonerated by an arm above or by the hash bisection.

**First fix step when a build exists:** a fixture that runs one
verify window of T=2 over a two-token prompt against the same two
tokens as two T=1 windows and compares the residual `final_R(t)` /
expert parts per row — it should fail on row 1 (and slightly on
row 0) at layer 0's FFN, and it localizes the assembly slip to one
of the three calls above. Then fix in the stub/runtime layer,
re-run this file's arms, and only then the capstone gates.

## Second, separate near-tie at position 14 (flagged, not conflated)

Every native arm — **including the all-T=1 chain** — emits 248046 at
output position 14 where the golden has 11. Native T=1 logits at that
row: 248046 @ 21.96 vs 11 @ 21.53 (margin 0.43). This is *not* the
position-3 fault (it needs no multi-row window) and may be an
independent near-tie against the vendored engine's rounding. The
coordinator should know: fixing the multi-row fault may still leave
`--compare q2_0_short` diverging at position 14; that row needs its
own verdict (vendored margin at that row could not be captured — the
vendored build's `--dump-logits` wrote an empty file under the CLI
flags used).

## Not done (build block)

No code fix, no ctest/sweep re-run, no long-capstone run: all need a
build dir (`~/qwen-native-staging/ws-diverge/`) and the Data volume is
at 209–300 MiB free. Nothing was deleted to make room (build-v12 and
mtp-rt untouched, per orders). Evidence kept in staging:
`ws-diverge-*.log`, `ws-diverge-logits.bin` (T=4 dump),
`ws-diverge-oracle.txt`, `ws-diverge-gen3.txt`.
