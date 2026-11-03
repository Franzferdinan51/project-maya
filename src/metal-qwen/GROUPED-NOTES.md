# GROUPED NOTES — the grouped native experts on the Metal backend

Companion to `PREFILL-NOTES.md` / `GDN-NOTES.md`. This file records the
grouped-expert workstream's implementation of
`strata::kernels::native_expert_grouped`, its audit of the remaining stubbed
native expert surface, and its Mac verification.

## 1. What was implemented

`native_expert_grouped` (header `include/strata/kernels/iq_kernels.hpp`,
CUDA source `src/kernels/cuda/iq_kernels.cu`) was the one loud stub the
coordinator's end-to-end probe stopped at in decode: prefill completed, the
verify window reached the routed experts of a native pack, and the grouped
entry aborted by name. It is now implemented in the established pattern:

- Host launcher: `src/metal-qwen/stubs/iq_kernels.cpp` (replacing the stub),
  composing the same four stages as the CUDA TU, with the scratch carved
  byte-identically (`gate | up | h | hq`, 256-byte aligned regions,
  matching `native_expert_scratch_bytes`).
- MSL: `src/metal-qwen/kernels/native_expert.metal` —
  `native_gu_one_k` (one launch per group; `ceil(2*n_ff/8)` threadgroups of
  256, one simdgroup per row, the CUDA `GU_ROWS = 8` shape),
  `native_swiglu_k` (both arms of the CUDA `swiglu_entries_kernel`: plain
  silu*up at an infinite limit, GLM's clamped form at a finite one),
  the shared `quantize_q8_1_k` for the intermediate, and
  `native_down_one_k` (one simdgroup per output row).

### Why the host reads the group plan (measured, not assumed)

The first version of this port mirrored the grouped S2 kernels
(`s2_gu_grouped_k` in `kernels/s2_expert.metal`): counts read on the
device, each group's blob taken from the raw address `grp_ptr[g]` inside
MSL. On the Mac it produced **all-zero outputs for every fixture case**
(no fault, no abort — the kernels ran and wrote zeros). Raw contents
addresses are host values in this backend; that is precisely why
`verify_kernels.cpp` already executes the pointer-manipulating entries
(`fetch_blobs`, `resident_plan`) on the host. The launcher therefore
syncs the stream, reads `n_groups` / `grp_start` / `grp_ptr` from the
shared buffers on the host, and launches per group with the blob as a
genuine buffer argument (which the dispatch layer resolves through its
allocation map, interior pointers included). The cost is one sync plus
one launch per group per stage; the CUDA original is a single launch.
Correctness first, as with the cuBLAS compat surface.

**Consequence worth flagging:** the grouped S2 kernels
(`moe_grouped_s2`'s `s2_gu_grouped_k` / `s2_down_grouped_k`) carry the
same in-MSL address cast, have no fixture in the oracle, and — by this
measurement — would produce zeros the same way if a non-native (S2) pack
ever drove them on this backend. They are off the current probe's path
(the probe's pack is native), but they should not be trusted until they
get the same host-mediated treatment or a fixture of their own.
- The dots are the float form of the CUDA integer `vec_dot_*_q8_1`
  products: `dq_elem`'s dequantized weight times the dequantized q8_1
  activation, lane-strided in the CUDA `row_dot` order and reduced with
  the xor butterfly (`moe_warp_sum_xor`) — the same float-form contract
  `kernels/dense.metal`'s `mmvq_k` documents. For every admitted type the
  per-element products are identical to the integer dots; only the
  summation grouping differs.

Type coverage is the intersection of the CUDA launcher's switches with the
formats `dq_elem` decodes:

| Role | CUDA switch | Ported |
|---|---|---|
| gate/up | 16, 17, 18, 21, 22, 23, 29, 19, 42, 13 | **13 (Q5_K), 18 (IQ3_XXS), 23 (IQ4_XS), 42 (Q2_0)** |
| down | 18, 20, 23, 42 | **18, 20 (IQ4_NL), 23, 42 — the whole switch** |

That covers the packs in tree evidence: Q5_K routed gate/up (the CUDA
`vec_dot_q5_K_q8_1` comment names this model's file), Q2_0 experts, and
Q2_0 / IQ4_NL down. The remaining grid i-quants (IQ1_S/M, IQ2_XXS/XS/S,
IQ3_S gate/up) abort loudly by type in the host launcher, the same line
`iq_mmvq` draws — their integer dots carry per-block integer rescaling
(`sumi * ls / 8` and friends) that a float `dq_elem` transcription has not
been validated against, and no Qwen pack in the tree uses them.

### Prelude fix (dq_elem, type 18)

`dq_elem` in `kernels/common.metalh` carried its IQ3_XXS branch *after* the
`return 0.0f` for unsupported types, so type 18 — claimed as covered by
`iq_covered` and by `dq_geometry` — silently decoded to zero. The branch
is now reachable, and its superblock indexing is fixed to block-local
chunk indices (`(e % 256) / 32`, where the dead code used the row-global
`e / 32`, wrong for any row longer than one superblock). The branch's
arithmetic was checked element-for-element against the CUDA TU's
`dq_iq3_xxs`. This also repairs `iq_mmvq` / `iq_dequant_*` / `iq_embed_rows`
for type 18, which share `dq_elem`.

## 2. Audit — the native expert surface on the decode/verify path

Scope: every still-aborting entry in `src/metal-qwen/stubs/` whose family
touches experts, plus everything `layer.cpp` / `session.cpp` / `verify.cpp`
/ `mtp.cpp` (the Qwen engine TUs in the Metal build) call. Method: grep
the stub TUs for aborting entries, then grep each entry's callers across
the Metal-compiled TUs (`strata_engine`: layer, session, expert_source,
remote_experts, expert_cache, native_head, native_dense, verify, mtp;
`src/program/generate.cpp`; `src/prefill/prefill.cpp`).

| Entry | State | Evidence |
|---|---|---|
| `native_expert_grouped` | **implemented here** | callers `verify.cpp` (grouped lambda, native arm), `remote_experts.cpp` (`RemoteExperts::launch`), `glm_model.cu` (excluded TU) |
| `moe_grouped_s2`, `moe_hit_grouped_s2(_dev/_multi)`, `moe_group_resident`, `moe_hit_select(_multi)`, `moe_hit_add` | already implemented (MoE workstream) | callers `verify.cpp`, `session.cpp`, `mtp.cpp`, `remote_experts.cpp` — the non-native arm of the same paths |
| `moe_hit_grouped_s2_cpu_order` | off-path, stays loud by design | only caller `expert_source.cpp` under `d.hit_cpu_order`, an "experimental … opt-in only" flag (`expert_source.hpp`) set from `generate.cpp`'s `expert_cache_cpu_order` option, default off; its own stub file documents it as a bit-exactness parity diagnostic, not a production path |
| `iq_mmvq`, `iq_dequant_f16/f32`, `iq_embed_rows`, `iq_dequant_gu_f16` for grid types 16/17/19/21/22/29 | off-path for this pack; loud by type | these entries abort only for the grid i-quants; the Qwen native packs use Q5_K / Q2_0 / IQ4_XS / IQ3_XXS experts (all covered). No Metal-compiled TU passes a grid type on the generate path |
| `gemm_iq_f16_grouped(_supported)` | off-path by design | returns `false` (no abort); the prompt path's WMMA arm requires `STRATA_PF_WMMA` + a Volta device and falls back to `prefill::Gemm` — the arm the probe's successful prefill ran |
| `glmf::*` (glm_fast), `glmb::*`, `glm_dsa::*`, `glm_hc/kda/ffn/kv_stream`, `Glm5Model` methods, `exl3::*` | off-path (GLM/DS4 scope) | no caller in any Metal-compiled Qwen TU (grep over layer/session/verify/mtp/expert_source/native_head/native_dense/generate finds none); callers are the excluded `glm_*.cu`, `ds4_*`, `mm3_*` TUs. `remote_experts.cpp`'s `GlmExpertHelper` (which would call `glmf::moe_gate_up`) is never instantiated outside its own definitions |
| `prefill::mmq::q8_block_bytes / Context::run_glu / to_f16` | off-path | the stub TU's own header: no caller in the Metal build (the entries `prefill.cpp` uses are its own no-MMQ fallback block, `built() == false`) |
| `prefill::rdna4::*` work entries | off-path | only callers `glm_prefill.cu`, `ds4_iq.cu` (both excluded); probes answer false and callers fall back |
| `sample_draft`, `sample_verify` (sampled sampler chain) | off-path for generation | only callers are `glm_fast_path.cu` (excluded). The Qwen path samples with `sample_tokens` (`verify.cpp`, `mtp.cpp`, `generate.cpp`), whose greedy arm is implemented; the sampled (temperature) arm stays loud by design — its CUDA top-p/draw stage is FP64, which Metal has no equivalent for (see `stubs/sampler.cpp`) |
| `verify_kernels.cpp`, `kv_q4/kv_q8/kv_stream` aborts | not stubs | argument-validation aborts and host-executed entries (`fetch_blobs`, `resident_plan`) only; all work entries are implemented |

Net: after this workstream, no aborting entry remains on the Qwen
decode/verify generate path. The next thing the probe can stop on is a
wrong *number*, not a named stub.

## 3. Parity fixtures

Four fixtures for `native_expert_grouped` (generator
`gen_native_expert_grouped` in `tests/gen_fixtures.cpp`, runner dispatch in
`tests/metal_kernel_parity.cpp`; the runner fills `grp_ptr` with the blob
inputs' device addresses, as the engine does):

| Case | gu / down | Shape | Covers |
|---|---|---|---|
| `q2_0_two_groups_perm` | 42 / 42 | n_embd 128, n_ff 64, 2 groups, 3 entries | multi-group, `ent_dst` permutation |
| `q5k_iq4nl_one_group` | 13 / 20 | n_embd 256, n_ff 256, 1 group, 2 entries | the Q5_K gate/up + IQ4_NL down pack shape |
| `iq4xs_q2_0_clamped` | 23 / 42 | n_embd 256, n_ff 64, 2 groups, 3 entries | IQ4_XS gate/up, finite swiglu limit (5.0) |
| `q2_0_iq4xs_down` | 42 / 23 | n_embd 128, n_ff 256, 1 group, 2 entries | IQ4_XS down |

Reference: double-precision chain from the tree's scalar dequantizers
(`strata/artifact/dequant.hpp`) and `quantize_q8_1`'s exact contract.
Tolerance MaxRel abs 1e-2 / rel 2e-2 (two chained dots plus an intermediate
quantization). IQ3_XXS (18) has no fixture: `dequant.hpp` carries no scalar
IQ3_XXS dequantizer to chain a reference to; its kernel path is the same
`dq_elem` the fixtured types exercise, and its decode is now reachable
after the §1 prelude fix.

## 4. Mac verification (2026-11-02, branch qwen-native-grouped)

Staging `~/qwen-native-staging/ws-grouped/`, fresh build dir,
`-DSTRATA_ENABLE_METAL=ON`, targets `maya` + `metal_kernel_parity`:

- MSL pre-check (common.metalh + all `kernels/*.metal` concatenated,
  `xcrun -sdk macosx metal -c`): clean.
- `maya` and `metal_kernel_parity` build and link.
- `metal_kernel_parity --entry native_expert_grouped --no-fork`:
  **4/4 PASS** (max_err 0, 0.5, 0, 1.02e3 against a ~2.8e9-magnitude
  reference — relative ~4e-7).
- Full oracle `--no-fork`: **105/105 PASS** (the previous 101 plus the 4
  new). The forked mode reports 45 cases "by signal" in this SSH
  environment — the known post-fork MTLCompilerService XPC failure the
  runner's `--no-fork` flag exists for, not kernel faults; every one of
  those cases passes in `--no-fork` mode.
- `ctest -R 'metal|e2e'`: **6/6 PASS** (dispatch smoke, graph, session,
  gemm, parity list, e2e greedy selftest).

The end-to-end probe re-run (real pack, native `maya` binary) is the
coordinator's step: this workstream's claim is that decode no longer
stops at a named stub — §2's audit found no other aborting entry on the
generate path.
