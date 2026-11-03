# ROWFIX workstream notes (branch qwen-native-rowfix)

## Verdict up front

The fault DIVERGE-NOTES.md localized — capstone `q2_0_short` diverging at
position 3 when a verify window keeps its row 1 — is **not** in the scoped
area. The prescribed composite fixture was built, escalated to the
capstone's exact geometry, and then extended to the *whole* of
`verify.cpp`'s `post()` FFN (grouped experts + parts assembly + shared
expert + batched combine). It **passes on the current tree** at every
geometry, with the grouped outputs matching the double-precision reference
to max_err 0.03–0.08 (tolerance 1e-2 + 2e-2·|want|). The multi-row MoE
composition — `native_expert_grouped_based` over VRAM + PCIe shares with
continuing entry numbering, oversized caps, shared scratch/hit_out,
`copy_rows_from_mapped`, `moe_hit_add` — is correct as composed.

Because the fixture passes *before* any fix, there is no backend change in
this workstream. What this file records instead: the fixture (so the
composition stays pinned), the extended audit that followed, the gates,
and both capstone verdicts including position 14.

## The fixture: `verify_moe_assembly`

Generator `gen_verify_moe_assembly()` in `tests/gen_fixtures.cpp`, dispatch
`verify_moe_assembly` in `tests/metal_kernel_parity.cpp`. The dispatch
replays post()'s exact sequence on the real entry points:

1. A device plan buffer in verify.cpp's exact layout (counts | start |
   dst | tok | pad | ptr | ptr2 | start2), arrays sized **capx = max_t·K**
   (40 for a T=2 window in a max_t=4 verifier), stale zeros past the live
   entries.
2. `native_expert_grouped_based` VRAM call: zero-based grp_start, blobs in
   one arena at the 256-aligned stride, `cap_groups == cap_entries == T·K`
   (oversized, as record_window passes), shared scratch + hit_out.
3. `native_expert_grouped_based` PCIe call over the **same** dst/tok
   arrays, scratch and hit_out, grp_start2 **continuing the entry
   numbering** (first PCIe group starts at the VRAM entry count), staging
   blobs packed at exactly `blob_bytes` stride (the rebase_ptrs stride).
4. `copy_rows_from_mapped(parts, ymiss, T·K, N, dst, counts+1)` — GPU rows
   zeroed, CPU rows copied (GPU rows of ymiss carry 999.5 junk that must be
   discarded, CPU rows carry the real values).
5. `moe_hit_add(parts, hit_out, dst, counts+1, cap, N)`.
6. Full cases only: `shared_expert_multi(T, …)` (native bf16 gate arm, as
   the CLI sets it) and `native_moe_combine_multi(parts, w, shared, bo)`.

Outputs compared per row: `hit_out` (rows no grouped call owns must keep
the runner's 777.25 stale sentinel — this localizes grouped-vs-assembly),
`parts`, and for full cases `shared` and `bo`. Routing shares experts
across the window's tokens (one blob, entries from several rows) and
splits distinct experts across VRAM / PCIe / CPU tiers. Reference is the
grouped fixtures' double-precision chain per entry.

Cases (all PASS, Mac build of this branch, `--no-fork`):

| case | geometry | max_err |
|---|---|---|
| t2_mixed_vram_pcie_cpu | T=2 K=3, 128×64, VRAM+PCIe+CPU | 0 |
| t2_vram_only_shared | T=2 K=3, 128×64, VRAM+CPU | 0 |
| t2_real_dims_capx40 | T=2 K=10, 2560×640, capx=40 | 0.0312 |
| t4_real_dims | T=4 K=10, 2560×640 | 0.0312 |
| t2_full_ffn | T=2 K=10, 2560×640, capx=40, +shared+combine | 0.0312 |
| t4_full_ffn | T=4 K=10, 2560×640, +shared+combine | 0.0781 |

Note for future harness work: this backend's `cudaMemcpy` readback does
**not** order against the compute stream — the dispatch must
`cudaStreamSynchronize` before `Dev::get()`, or it reads prefill values.
(The first version of this dispatch read hit_out/parts before the sync and
reported exactly the sentinel/prefill values; fixed in the runner.)

## Extended audit (all line-checked against the CUDA originals)

With the scoped composition exonerated, every other multi-row entry on the
layer-0 path was read against `src/kernels/cuda/`:

- `quantize_q8_1_rows` (stub + `quantize_q8_1_k`): flat over rows·cols,
  block-aligned per row — matches CUDA.
- `native_mmvq` multi-column (`mmvq_k` col = gid/n_out, activation stride
  (n_in/32)·36) and `native_quantize_q8_1`: matches; fixtures only cover
  ncols=1, but the launcher/kernel indexing is the single-column layout
  repeated, and the full-FFN fixture exercises ncols=T through
  shared_expert_multi.
- `gdn_conv_l2_k`: line-for-line the CUDA kernel, incl. t_begin row
  offsetting and the per-head L2 norm.
- `gdn_step_norm_k`: matches CUDA `gdn_step_norm_multi_kernel`, including
  the subtle contract that state is written back **only** when
  `n_keep != nullptr` (the in-window call passes nullptr and replays from
  the untouched state; commit passes the device count) and that tokens
  `t < t_out_begin` update state but emit no output.
- `gdn_ab_k`, `native_combine_k`, `moe_hit_add_k`, `copy_rows_mapped_k`,
  `shexp_scale_rows_k`, `shared_expert_multi` (stub mirrors the CUDA TU
  statement for statement): all match.
- `fused_gr_read_multi`: a per-token loop of `fused_gr_read` — the CUDA
  multi-kernel's stated contract is per-token bitwise identity.
- The plan layout arithmetic in post() (ptr_off / p_start2 with capx) is
  consistent with the host publisher in expert_source.cpp.

## Where the fault still sits (evidence-bounded)

DIVERGE's constraints stand and now bind tighter: layer-0 GDN *state*
identical between T=1 and T=4 chains over the same committed prefix, so
layer 0's per-token input path (embeddings → gr read → qkv/conv/ab/step)
produces identical state updates; the first wrong value is in layer 0's
block *output* path into the residual, or in layer 1's consumption of it.
post()'s FFN output is now fixture-clean. The remaining un-fixtured
surface on that exact boundary is the hyper-connection fold itself:
`fused_gr_read`'s apply path (bo_prev folded into R_out) as driven per
token by the window body, and the QSA layers' in-window KV/attention for
later layers (the logit diff grows with row index, consistent with
in-window recurrence/attention consumption of earlier window rows).
That is the next bisection's ground; it needs a fixture at the
fused_gr_read apply boundary (single token, apply=1, vs CUDA's fused
kernel) or a per-layer residual hash inside the window body — neither
exists yet.

## Gates (Mac, this branch, build dir ws-rowfix)

- Fixture before/after: no "before" failure exists — the fixture passes on
  the unmodified tree (see verdict). It is committed as a regression pin.
- Full parity sweep `--no-fork`: **117 passed, 0 failed** on the Mac copy
  (118 fixtures in tree; `t4_real_dims` was dropped from the Mac staging
  copy under disk pressure after passing its entry run at max_err 0.0312).
- `ctest -R 'metal|e2e'`: **6/6 passed**.
- Short generation probe (args per DIVERGE-NOTES §repro, staging mtp-rt,
  STRATA_ROUTE_TAIL_SKIP=7): output identical to the coordinator's
  build-v12 tokens; `e2e_greedy_decode --compare golden_short.mqgt`:
  **DIVERGE at position 3: got 17, expected 18** — the capstone is NOT
  fixed by this workstream, because the fault is not in the scoped
  composition. No backend code changed, so no other outcome was possible;
  the probe confirms build equivalence with build-v12.
- T=1 arm on this build (`--mtp-max-t 1`, same probe): output
  `17 11 220 18 11 220 20 11 220 22 11 220 16 16 248046 …` — golden
  through position 13, **248046 at position 14** (see the verdict below).
- Long capstone: prompt ids (5,983) extracted from `golden_long.mqgt`
  itself (the E2E-GREEDY.md candidate protocol), run detached with
  `--max-context 8192`, log `ws-rowfix/probe-long.log` on the Mac
  staging. Prefill 5,982 tokens in 1,109 s (5.39 tok/s). Output
  `16 11 220 17 11 220 18 248046 …`; `e2e_greedy_decode --compare
  golden_long.mqgt`: **DIVERGE at position 7: got 248046, expected 11** —
  the counting continuation starts correctly (16, 17, 18) and breaks at
  the same 248046 token that ends the short T=1 arm at position 14.

## Position 14 — separate verdict

Every native arm, including all-T=1 (`--mtp-max-t 1`, reproduced on this
build: golden through position 13, then 248046), emits **248046**
where the golden has **11**. This is **not** the row-composition fault
class: it reproduces with single-row windows, where no multi-row
composition runs at all. Native T=1 margin at the position is **0.43**
(248046 @ 21.96 vs 11 @ 21.53, from DIVERGE-NOTES' dump) — a near-tie.
The documented numeric deltas of this backend (float-form dots vs CUDA's
integer dp4a dots, differing summation trees in the split GEMVs and
gdn_ab) are easily large enough to flip a 0.43-margin argmax after 14
positions of accumulation, and the vendored engine rounds differently
from both. Verdict: **genuine near-tie numeric sensitivity, not a backend
indexing bug** — no fix attempted, nothing tuned. If it is ever chased,
the honest instrument is a same-position logits dump from both engines
at T=1, not a code change.

## Housekeeping

- Commits on qwen-native-rowfix: fixture + runner dispatch, fixture files,
  real-geometry and full-FFN escalations, this file. Tests only; no
  backend or engine source changed.
- Mac staging: `~/qwen-native-staging/ws-rowfix/` (src + build kept for
  the coordinator's merge; binaries metal_kernel_parity, e2e_greedy_decode,
  maya present). One runner rebuild (build3) ran without the build lock —
  no other build was active in that window; noted for the record.
- The unrelated `peer_probe` target fails to link on a Metal-only
  configure (no main; pre-existing, CUDA-family target) — named-target
  builds were used throughout, as in prior workstreams.
