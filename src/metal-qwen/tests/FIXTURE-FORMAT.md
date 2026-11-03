# The Qwen Metal parity oracle: fixture format, generator, runner

This directory is the parity oracle for Maya's native Qwen Metal backend
(`src/metal-qwen/`). Upstream gives GLM a model-level oracle
(`src/core/glm_model_test.cpp` against `ref/glm.py`) and gives Qwen nothing —
no `ref/qwen.py`, no `qwen_model_test.cpp`. This workstream builds the Qwen
substitute, in two layers:

1. **Per-entry fixtures** (this document): one binary fixture per
   (entry point, case), generated on the host from scalar references, run on
   the Mac against the real host entry points by `metal_kernel_parity`.
2. **Model-level greedy comparison** (`E2E-GREEDY.md`): a fixed prompt and an
   exact greedy token sequence — the capstone gate, the analogue of
   `glm_model_test.cpp`'s teacher-forced parity for a model with no Python
   reference.

Kernel-level precedent followed throughout: upstream's own parity programs
(`src/kernels/*_parity.cpp`, chained to ggml through
`include/strata/artifact/dequant.hpp`) and, for the QSA score family,
`tests/hip/native_qsa_score.cpp`'s exact approach (formula-generated inputs,
float FMA-order reference, bitwise verdict, observability guard). Nothing here
duplicates those programs: the fixtures *reuse their references* (the same
dequantizer header, the same header-contract formulas) and repackage the
verdicts into files a Metal build can consume, because the upstream programs
are CUDA/HIP executables that cannot run in a Metal build at all.
`bench/micro`'s `native_*_parity` programs named in CMakeLists are not in the
published tree (they exist only against a pinned llama.cpp oracle build), so
there is nothing there to reuse; the STRATA_ORACLE wiring stays CUDA-only.

## The .mqfx format (v1)

One file per (entry, case), named `<entry>.<case>.mqfx`. All integers
little-endian. The single definition of the layout is `fixture_format.hpp`,
shared by the generator and the runner so the two cannot drift.

```
u32  magic "MQFX"
u16  version = 1
u16  reserved = 0
str  family       (u32 length + bytes)   e.g. "gemv"
str  entry        exact host entry name  e.g. "s2_gemv"
str  case_name                           e.g. "s2_gate_2560x64"
u64  seed         the deterministic seed all inputs were drawn from
u8   tolerance kind: 0 bitwise | 1 max_abs | 2 max_rel
f64  tol_abs
f64  tol_rel      (max_rel verdict: |got-want| <= tol_abs + tol_rel*|want|)
u32  n_params_i, then per param: str name + i64 value   (dims, ggml type ids, flags)
u32  n_params_f, then per param: str name + f64 value   (eps, scales, out_fill)
u32  n_inputs,  then tensors
u32  n_outputs, then tensors             (the expected outputs)

tensor:
  str  name
  u8   dtype: 0 f32 | 1 f16 bits | 2 bf16 bits | 3 i32 | 4 i64 | 5 u8 raw | 6 i8 | 7 u16 raw
  u8   rank, then rank * i64 dims (row-major)
  u64  byte count, then the bytes
```

Rules:

- **Quantized data is `u8 raw`** (GGUF blocks, canonical code planes, Q8_1
  images) — a fixture never pretends a block format is a float tensor.
- **In-place entries** list the buffer twice under different names: the input
  tensor (`state`, `x`, `hist`) and the expected post-state (`state_out`,
  `out`, `hist_out`). The runner copies the input, calls, compares the copy.
- **`out_fill` param**: when present, the runner pre-fills f32 output buffers
  with that value before the call, so "the kernel must not write outside
  [0, n)" is itself checked (the QSA score fixture's sentinel does this,
  following the HIP precedent's -123.0f).
- **Deterministic seeds only.** Every input is splitmix64 over the fixture's
  seed or a closed-form formula (QSA score). Regenerating produces identical
  bytes; a fixture diff in review is a reference change, never noise.
- **Tolerance is per fixture, in the file**, chosen from the upstream parity
  program for the same family (dequant: bitwise; GEMV dots: max_rel in the
  1e-3 class the phase spec allows for fp16 paths; routers/sampler ids:
  bitwise/exact with weights at 1e-6).

## The generator (`gen_fixtures.cpp`, `build_gen.sh`)

Host-only (g++/clang++, no GPU, no backend), so fixtures can be produced and
reviewed anywhere:

```
src/metal-qwen/tests/build_gen.sh --run    # builds + regenerates fixtures/
```

Reference sources, per family — upstream-first, per the survey:

| family | reference | origin |
|---|---|---|
| dequant (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K) | `strata::dequantize_*` | `include/strata/artifact/dequant.hpp` — the validated scalar chain to ggml, same as `dequant_s2_parity.cpp` uses |
| dequant_s2 | hand decode `(code-1)*scale` | the canonical S2 contract in `dequant_s2.hpp` |
| s / s2 GEMV | double dot over canonical decode | the S-form contract (`s_gemv.hpp`); IQ4_NL table copied from dequant.hpp's ggml transcription |
| native_mmvq (+ typed, + _f32) | double dot of dequantized W × dequantized Q8_1 x | dequant.hpp + the Q8_1 layout in `native_mmvq.hpp` |
| router_top10 / native_router_top10 | double softmax, stable argsort, 2^-14 clamp | `router_top10_parity.cpp`'s host reference semantics |
| router_sigmoid | double sigmoid, bias in selection only | the formula stated in `router_sigmoid.hpp` |
| sampler (greedy) | argmax, lowest index on ties | `sampler.hpp` chain with temperature 0 |
| PLE postops + history | double transcription | the staged formula in `ple.hpp` + constants in `ngram.hpp` (eps 1e-6, kernel 4, dilation 3, row-fastest history) |
| GDN step / l2_norm | double transcription of `ref_step` | `gdn_parity.cpp`'s reference (modulo head pairing, decay before update, eps on the squared norm), restated in the kernel's (S, h_v, S) layout |
| native GDN step | same, readout scaled by 1/sqrt(S) | the pinned-operator note in `native_gdn.hpp` (q NOT pre-scaled for this entry; the plain `gdn_step` fixture pre-scales, per its own contract) |
| QSA gate / rms norm | double transcription | `qsa_parity.cpp`'s `ref_gate` (gate = second half, sigmoid) + ggml RMS norm |
| QSA score | float, FMA order | `tests/hip/native_qsa_score.cpp`, replicated formula-for-formula |

Dimensions are the Qwen model's real ones, read read-only from the pack on
the Mac (`~/Strata-data/packs/q2_0/index.txt`, `maya-qwen.json`):
n_embd 2560, vocab 248320, experts 512 with top-10, expert n_ff 640,
PLE/hc width 10240 (4 streams × 2560), QSA 24 heads × 256 (indexer 4 × 128,
block 4, top-k 2048), GDN S = 128. Fixtures subset the *output* width where a
full tensor would be wasteful (a GEMV's rows are independent; 64 of 640 rows
exercise the same code), and say so in the case name.

GDN and QSA statefulness, stated plainly: both families are stateful in the
engine, but their *steps* are pure functions of (state, inputs), so the
fixtureable unit is one step at small head counts with the full state in and
the full state out — including the deep-decay regime (`gate` in [-4,-1])
where the decay-order trap is actually visible, per `gdn_parity.cpp`'s own
lesson. QSA's selection (`topk_512`, `qsa_index`) is **not** fixtured yet:
its semantics ride on the device step buffer and the block/tail mapping, and
a honest fixture for it should be ported from `qsa_parity.cpp`'s oracle
vectors by the QSA workstream rather than invented here; the gate and score
ends of the family — where the arithmetic is — are fixtured now.

## The runner (`metal_kernel_parity.cpp`)

CMake target `metal_kernel_parity`, built only under `STRATA_ENABLE_METAL`
(wiring in `cmake/metal_qwen_backend.cmake`), linking `strata_kernels` +
`strata_metal_runtime`. It allocates through the runtime's CUDA-shaped
surface (`cudaMalloc`/`cudaMemcpy` over shared-memory Metal buffers), creates
one stream (the native entries require a non-null stream), and calls the
entry points with the exact signatures the engine uses.

```
metal_kernel_parity src/metal-qwen/tests/fixtures [--entry s2_gemv] [--list]
```

Prints one `PASS`/`FAIL <entry>.<case> max_err=...` line per fixture and a
summary; exit code = number of failed cases. **Each case runs in a forked
child** (120 s alarm): the Stage-1 stubs `abort()` by design, and the harness
must turn that into a per-case `FAIL (child killed by signal 6)` and keep
going. That is the stub smoke check: against an all-stub build the expected
result is **0 passed, every dispatched case failed by signal, harness exit
code = failures, runner itself completes** — anything else (a hang, a
runner crash, a PASS) means the harness is lying. As real kernels replace
stubs family by family, the same command turns green family by family with
no harness change.

## Verified state (2026-11-02, Mac staging ws-parity)

- Mac build (`STRATA_ENABLE_METAL=ON`, `-j 4`, under the build lock):
  configure + `metal_kernel_parity` + `e2e_greedy_decode` **build and link
  green**.
- Stub smoke run: `metal_kernel_parity fixtures` → **0 passed, 66 failed,
  all 66 by signal 6** (the stubs' `abort()`), harness completes, exit code
  66. Exactly the designed signature: the harness works, nothing passes
  against stubs, and no case can hang or take the runner down with it.
- `e2e_greedy_decode --selftest` passes on the Mac; `--run` exits 77 (skip)
  until real kernels + a CUDA-captured golden exist.

## Fixture inventory (this commit)

| entry | fixtures |
|---|---|
| `dequant_bf16` | 10 |
| `dequant_f16` | 10 |
| `dequant_f32` | 10 |
| `dequant_s2` | 1 |
| `gdn_l2_norm` | 1 |
| `gdn_step` | 1 |
| `native_gdn_step` | 1 |
| `native_iq4_nl_f32` | 1 |
| `native_iq4_nl_mmvq` | 1 |
| `native_iq4_xs_f32` | 1 |
| `native_iq4_xs_mmvq` | 1 |
| `native_mmvq` | 5 |
| `native_ple_postops` | 1 |
| `native_q2_0_f32` | 1 |
| `native_q2_0_mmvq` | 1 |
| `native_q4_k_f32` | 1 |
| `native_q4_k_mmvq` | 1 |
| `native_q8_0_f32` | 1 |
| `native_q8_0_mmvq` | 1 |
| `native_qsa_gate_apply` | 1 |
| `native_qsa_rms_norm_weighted` | 1 |
| `native_qsa_score` | 1 |
| `native_router_top10` | 1 |
| `native_router_top10_multi` | 1 |
| `ple_history_advance` | 1 |
| `qsa_gate_apply_f32` | 1 |
| `router_sigmoid` | 1 |
| `router_top10` | 1 |
| `s2_gemv` | 1 |
| `s2_gemv_q8` | 1 |
| `s_gemv` | 3 |
| `s_gemv_split` | 1 |
| `sample_tokens` | 1 |

**66 fixtures, 2.7 MB total**, committed under `src/metal-qwen/tests/fixtures/` (regenerate with `build_gen.sh --run`; bytes are deterministic).

## Coverage: every stub entry → fixture status

Status meanings: **fixture** = a committed fixture + runner dispatch gates
this entry today. **planned** = fixtureable, reference identified, lands with
its kernel family in Stage 2. **not-fixtureable** = no per-entry numeric
fixture is the right gate, with the reason (host config toggles and sync
primitives have no math to compare; other engines' families are gated by
their own upstream parity programs and are never called on the Qwen path).

**412 stub entries: 33 fixture-gated, 235 planned, 144 not-fixtureable.**

| stub file | entry | status | note |
|---|---|---|---|
| bf16_gemv.cpp | `strata::kernels::bf16_gemv\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| bf16_gemv.cpp | `strata::kernels::bf16_gemv_fp32_mmvf\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| bf16_gemv.cpp | `strata::kernels::bf16_gemv_fp32_mmvf_multi\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| bf16_gemv.cpp | `strata::kernels::bf16_gemv_split\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| cvec.cpp | `strata::kernels::cvec\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| cvec.cpp | `strata::kernels::cvec_apply\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| cvec.cpp | `strata::kernels::cvec_enabled\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| cvec.cpp | `strata::kernels::cvec_replicate\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| cvec.cpp | `strata::kernels::cvec_set_enabled\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| cvec.cpp | `strata::kernels::cvec_upload\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| dequant_bf16.cpp | `strata::kernels::dequant_bf16\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| dequant_bf16.cpp | `strata::kernels::dequant_bf16_supported\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| dequant_bf16.cpp | `strata::kernels::dequant_f16\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| dequant_bf16.cpp | `strata::kernels::dequant_f32\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| dequant_s2.cpp | `strata::kernels::dequant_s2\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| elementwise.cpp | `strata::kernels::add_inplace\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::copy_from_mapped\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::copy_i32_from_mapped\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::copy_rows_from_mapped\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::doorbell_publish\n` | not-fixtureable | device sync / bookkeeping primitive — no numeric output |
| elementwise.cpp | `strata::kernels::doorbell_ring\n` | not-fixtureable | device sync / bookkeeping primitive — no numeric output |
| elementwise.cpp | `strata::kernels::doorbell_wait\n` | not-fixtureable | device sync / bookkeeping primitive — no numeric output |
| elementwise.cpp | `strata::kernels::embedding_gather\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::f32_to_bf16_bulk\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::f32_to_f16_bulk\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::gdn_gate\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| elementwise.cpp | `strata::kernels::rms_norm_weighted\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::scale_inplace\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| elementwise.cpp | `strata::kernels::silu_inplace\n` | planned | refs: src/kernels/elementwise_parity.cpp |
| exl3.cpp | `strata::kernels::exl3::gemm_rows\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::k2_supported\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::launch_errors\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::mixed_supported\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::mma_supported\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::moe_down\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::moe_gate_up\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::mv\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::mv_cols\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::mv_splits\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::prepare\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::prompt_tc\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::prompt_wmma\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::reconstruct_inner\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::reconstruct_inner_rows\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::reconstruct_rows\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::rows_f16\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::rows_rot_f16\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::rows_rot_out\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::set_down\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::set_gate_up\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::set_skip\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::several_ok\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::supported\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::view\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| exl3.cpp | `strata::kernels::exl3::win_experts\n` | not-fixtureable | EXL3 format family (DS4/GLM packs); the Qwen Q2_0 pack never uses it — gated upstream by exl3_test |
| fused_gdn.cpp | `strata::kernels::fused_gdn_ab\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| fused_gdn.cpp | `strata::kernels::fused_gdn_conv_l2\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| fused_gdn.cpp | `strata::kernels::fused_gdn_step_norm\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| fused_gr.cpp | `strata::kernels::fused_gr_read\n` | fixture | GRFOLD correction: this entry IS on the Qwen path (layer.cpp fused halves, verify.cpp gr_read_group, mtp.cpp) — `fused_gr_read.*.mqfx`, double reference transcribed from src/kernels/cuda/fused_gr.cu, incl. apply=1 with nontrivial bo_prev and the in-place R_out == R shape |
| fused_gr.cpp | `strata::kernels::fused_gr_read_multi\n` | fixture | same `fused_gr_read.*.mqfx` cases with T = 2 / 4 (upstream gr_parity only ever compared the CUDA multi against the CUDA single, never against an independent reference) |
| fused_gr.cpp | `strata::kernels::fused_gr_supported\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| gdn.cpp | `strata::kernels::gdn_beta_gate\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| gdn.cpp | `strata::kernels::gdn_conv_step\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| gdn.cpp | `strata::kernels::gdn_l2_norm\n` | fixture | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| gdn.cpp | `strata::kernels::gdn_out_norm\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| gdn.cpp | `strata::kernels::gdn_step\n` | fixture | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| glm_batch.cpp | `strata::kernels::glmb::bf16_to_f16\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::bf16_to_f32\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::copy_i32\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::dsa_pool\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::dsa_prep\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::dsa_score\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::dsa_select\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::embed_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::expert_count\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::expert_scatter\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::f32_to_f16\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::hc_finish\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::hc_update\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::head_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::kda_conv\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::kda_conv_state\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::kda_rec\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::kda_rec_split\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::launch_errors\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::mla_attn\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::mla_decode_v70\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::moe_combine\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::moe_combine_add\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::mtp_in_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::rms_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::route\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::swiglu_f16\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::swiglu_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_batch.cpp | `strata::kernels::glmb::swiglu_rows_h16\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_dsa.cpp | `strata::kernels::glm_dsa_pool\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_dsa.cpp | `strata::kernels::glm_dsa_score\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_dsa.cpp | `strata::kernels::glm_dsa_select\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_dsa.cpp | `strata::kernels::glm_mla_attention\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::argmax\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::argmax_join\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::argmax_pair\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::dsa_prep\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::dsa_score\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::dsa_score_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::dsa_select\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::dsa_select_all\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::dsa_select_all_on\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::dsa_select_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::embed_streams\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::embed_tok\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::hc\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::hc_post\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::hc_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::head_prep\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::kda_gather\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::kda_layers\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::kda_prep\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::kda_rec\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::launch_errors\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla_attend\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla_attend_batch\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla_attend_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla_out\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla_q_abs\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla_rows_ok\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mla_rows_part_floats\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_cpu_wait\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_cpu_wait_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_cpu_wait_streams\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_down\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_experts_multi\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_fetch\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_gate_up\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_prefetch\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_route\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_supported\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_wait\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_wait_fetch\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_win_experts\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::moe_win_group\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mtp_in\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mv\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mv_nt\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::mv_supported\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::quantize_q8_1\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::rms_q8\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::row_bytes\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::rows_experts\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::set_skip\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::skip_gate\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::swiglu_q8\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_fast.cpp | `strata::kernels::glmf::tab_update\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_ffn.cpp | `strata::kernels::glm_swiglu_clamp\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_hc.cpp | `strata::kernels::glm_hc_post\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_hc.cpp | `strata::kernels::glm_hc_pre\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_hc.cpp | `strata::kernels::glm_hc_workspace_bytes\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_hc.cpp | `strata::kernels::glm_hc_workspace_init\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_kda.cpp | `strata::kernels::glm_kda_conv\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_kda.cpp | `strata::kernels::glm_kda_gate\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_kda.cpp | `strata::kernels::glm_kda_out_gate\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_kda.cpp | `strata::kernels::glm_kda_recurrence\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_kv_stream.cpp | `strata::kernels::glmf::glm_kv_resolve\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| glm_kv_stream.cpp | `strata::kernels::glmf::glm_kv_resolve_rows\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| gr.cpp | `strata::kernels::gr_read\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| gr.cpp | `strata::kernels::gr_set_fp32_activations\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| gr.cpp | `strata::kernels::gr_set_native_mmvf\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| gr.cpp | `strata::kernels::gr_workspace_init\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| gr.cpp | `strata::kernels::gr_write\n` | not-fixtureable | GLM engine family; never called on the Qwen path — gated by upstream glm_*_parity on CUDA/HIP |
| iq_kernels.cpp | `strata::kernels::gemm_iq_f16_grouped\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::gemm_iq_f16_grouped_supported\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::iq_dequant_f16\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::iq_dequant_f32\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::iq_dequant_gu_f16\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::iq_embed_rows\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::iq_mmvq\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::iq_row_bytes\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::iq_supported\n` | planned | refs: dequant.hpp + src/kernels/iq_parity.cpp fixture chain |
| iq_kernels.cpp | `strata::kernels::native_expert_grouped\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| iq_kernels.cpp | `strata::kernels::native_expert_layout\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| iq_kernels.cpp | `strata::kernels::native_expert_scratch_bytes\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| iq_kernels.cpp | `strata::kernels::quantize_q8_1_rows\n` | planned | needs the rounding-mode pin first (Q8_1/Q8_0/Q8_K byte exactness is a contract decision) |
| kv_q4.cpp | `strata::kernels::fwht256_cuda\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| kv_q4.cpp | `strata::kernels::kv_append_q4\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_q4.cpp | `strata::kernels::kv_append_q4_step\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_q4.cpp | `strata::kernels::kv_gather_q4_step\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_q8.cpp | `strata::kernels::kv_append_q8_step\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_q8.cpp | `strata::kernels::kv_gather_q8_step\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_stream.cpp | `strata::kernels::kv_block_bytes\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_stream.cpp | `strata::kernels::kv_ring_restore\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_stream.cpp | `strata::kernels::kv_ring_table\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_stream.cpp | `strata::kernels::kv_stage_from_host\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_stream.cpp | `strata::kernels::kv_stream_counters\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_stream.cpp | `strata::kernels::kv_stream_reset\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| kv_stream.cpp | `strata::kernels::kv_stream_resolve\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| mrope.cpp | `strata::kernels::mrope_table\n` | planned | refs: src/kernels/rope_parity.cpp |
| mrope.cpp | `strata::kernels::mrope_table_set\n` | planned | refs: src/kernels/rope_parity.cpp |
| native_flash_attn.cpp | `strata::kernels::native_flash_attn_short_step\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_gdn.cpp | `strata::kernels::native_gdn_enabled\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gdn.cpp | `strata::kernels::native_gdn_set_enabled\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gdn.cpp | `strata::kernels::native_gdn_step\n` | fixture | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gdn_preprocess.cpp | `strata::kernels::native_gdn_beta_gate\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gdn_preprocess.cpp | `strata::kernels::native_gdn_conv_silu\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gdn_preprocess.cpp | `strata::kernels::native_gdn_gate\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gdn_preprocess.cpp | `strata::kernels::native_gdn_l2_norm\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gdn_preprocess.cpp | `strata::kernels::native_gdn_out_norm\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| native_gr_norm.cpp | `strata::kernels::native_gr_rms_norm_weighted\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_gr_postops.cpp | `strata::kernels::native_gr_down_silu\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_gr_postops.cpp | `strata::kernels::native_gr_post\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_gr_postops.cpp | `strata::kernels::native_gr_pre_gated\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_iq4_nl_f32\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_iq4_nl_mmvq\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_iq4_xs_f32\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_iq4_xs_mmvq\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_mmvq\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_mmvq_multi_exact\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_mmvq_set_multi_exact\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_mmvq_supported\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_mmvq_weight_bytes\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q2_0_f32\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q2_0_mmvq\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q3_k_f32\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q3_k_mmvq\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q4_0_f32\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q4_0_mmvq\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q4_k_f32\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q4_k_mmvq\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q5_0_f32\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q5_0_mmvq\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q5_k_f32\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q5_k_mmvq\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q6_k_f32\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q6_k_mmvq\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q8_0_f32\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q8_0_mmvq\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_q8_1_bytes\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_mmvq.cpp | `strata::kernels::native_quantize_q8_1\n` | planned | needs the rounding-mode pin first (Q8_1/Q8_0/Q8_K byte exactness is a contract decision) |
| native_moe.cpp | `strata::kernels::native_moe_combine\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| native_moe.cpp | `strata::kernels::native_moe_combine_enabled\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| native_moe.cpp | `strata::kernels::native_moe_combine_multi\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| native_moe.cpp | `strata::kernels::native_moe_combine_set_enabled\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| native_ple_postops.cpp | `strata::kernels::native_ple_postops\n` | fixture | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| native_ple_postops.cpp | `strata::kernels::native_ple_postops_batch\n` | planned | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| native_qsa.cpp | `strata::kernels::native_qsa_enabled\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa.cpp | `strata::kernels::native_qsa_gate_apply\n` | fixture | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa.cpp | `strata::kernels::native_qsa_rms_norm_weighted\n` | fixture | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa.cpp | `strata::kernels::native_qsa_set_enabled\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa_indexer.cpp | `strata::kernels::native_qsa_indexer_append\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa_indexer.cpp | `strata::kernels::native_qsa_indexer_append_batch\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa_indexer.cpp | `strata::kernels::native_qsa_indexer_enabled\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa_indexer.cpp | `strata::kernels::native_qsa_indexer_set_enabled\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa_score.cpp | `strata::kernels::native_qsa_score\n` | fixture | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa_score.cpp | `strata::kernels::native_qsa_score_enabled\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_qsa_score.cpp | `strata::kernels::native_qsa_score_set_enabled\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| native_rope.cpp | `strata::kernels::native_rope_apply\n` | fixture | GRFOLD: double-precision transcription of src/kernels/cuda/native_rope.cu; in-place, rows 32/64/128 incl. two-position windows (the qb batched shape) |
| native_rope.cpp | `strata::kernels::native_rope_enabled\n` | planned | refs: src/kernels/rope_parity.cpp |
| native_rope.cpp | `strata::kernels::native_rope_set_enabled\n` | planned | refs: src/kernels/rope_parity.cpp |
| native_router.cpp | `strata::kernels::native_router_enabled\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_router.cpp | `strata::kernels::native_router_set_enabled\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_router.cpp | `strata::kernels::native_router_top10\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| native_router.cpp | `strata::kernels::native_router_top10_multi\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| ple.cpp | `strata::kernels::ple_block\n` | planned | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| ple.cpp | `strata::kernels::ple_block_available\n` | planned | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| ple.cpp | `strata::kernels::ple_block_scratch_bytes\n` | planned | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| ple.cpp | `strata::kernels::ple_history_advance\n` | fixture | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| ple.cpp | `strata::kernels::ple_native_postops_enabled\n` | planned | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| ple.cpp | `strata::kernels::ple_set_native_bf16\n` | planned | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| ple.cpp | `strata::kernels::ple_set_native_postops\n` | planned | ple_block needs the ggml-oracle capture (upstream ple_parity.cpp part C); postops + history already fixtured |
| prefill_gemm.cpp | `strata::prefill::Gemm::bf16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_gemm.cpp | `strata::prefill::Gemm::f16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_gemm.cpp | `strata::prefill::Gemm::init\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_gemm.cpp | `strata::prefill::Gemm::init_external\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_gemm.cpp | `strata::prefill::Gemm::native\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_gemm.cpp | `strata::prefill::Gemm::rebind\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_gemm.cpp | `strata::prefill::Gemm::~Gemm\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::blob_dequant\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::blob_dequant_f16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::copy_i32\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gate_attn\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gather_rows16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gdn_conv\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gdn_gates\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gdn_recurrence\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_broadcast\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_mix\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_mix_r\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_norm\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_norm_rs\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_silu\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_write\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::gr_write_norm_rs\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::kv_append\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::moe_combine\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::rms_rows\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::rope\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::round_f16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::route\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::split_q\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::swiglu_interleaved\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::swiglu_pair\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::swiglu_split\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::to_bf16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_kernels.cpp | `strata::prefill::to_f16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::act_bytes\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::available\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::experts\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::experts_down\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::experts_gate_up\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::quantize\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::quantize_rows\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_fused_rdna4.cpp | `strata::prefill::rdna4::supported\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::Context::Context\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::Context::run\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::Context::run_glu\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::Context::~Context\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::built\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::gather_native\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::gather_strata_q2\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::glu_supported\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::iota\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::matrix_bytes\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::pool_slab\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::q8_block_bytes\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::q8_bytes\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::quantize\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::supported\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::swiglu\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::to_f16\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| prefill_moe_mmq.cpp | `strata::prefill::mmq::to_f16_supported\n` | planned | prefill parity precedents: tests/hip/prefill_*_parity.cpp |
| qsa.cpp | `strata::kernels::indexer_key_append\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::kv_append\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| qsa.cpp | `strata::kernels::kv_append_step\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| qsa.cpp | `strata::kernels::kv_gather\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| qsa.cpp | `strata::kernels::kv_gather_step\n` | planned | refs: src/kernels/kv_q4_parity.cpp / kv_q8_parity.cpp (two-step append/gather fixtures) |
| qsa.cpp | `strata::kernels::qsa_attend\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::qsa_attend_step\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::qsa_gate_apply\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::qsa_gate_apply_f32\n` | fixture | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::qsa_index\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::qsa_index_step\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::qsa_step_fill\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::topk_512\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa.cpp | `strata::kernels::topk_512_step\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_decode_attn.cpp | `strata::kernels::qsa_decode_attn_batch\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_decode_attn.cpp | `strata::kernels::qsa_decode_attn_scratch_floats\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_decode_attn.cpp | `strata::kernels::qsa_decode_attn_step\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_prompt_attn.cpp | `strata::kernels::qsa_prompt_attn_batch\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_select.cpp | `strata::kernels::qsa_block_scores\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_select.cpp | `strata::kernels::qsa_block_scores_tc\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_select.cpp | `strata::kernels::qsa_block_topk\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| qsa_select.cpp | `strata::kernels::qsa_block_topk_ref\n` | planned | derive from src/kernels/qsa_parity.cpp oracle vectors (step-buffer state); gate/score ends already fixtured |
| quantize_act.cpp | `strata::kernels::dequant_q8_0\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| quantize_act.cpp | `strata::kernels::dequant_q8_K\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| quantize_act.cpp | `strata::kernels::quantize_q8_0\n` | planned | needs the rounding-mode pin first (Q8_1/Q8_0/Q8_K byte exactness is a contract decision) |
| quantize_act.cpp | `strata::kernels::quantize_q8_0_scaled\n` | planned | needs the rounding-mode pin first (Q8_1/Q8_0/Q8_K byte exactness is a contract decision) |
| quantize_act.cpp | `strata::kernels::quantize_q8_K\n` | planned | needs the rounding-mode pin first (Q8_1/Q8_0/Q8_K byte exactness is a contract decision) |
| rope.cpp | `strata::kernels::build_rope_table\n` | planned | refs: src/kernels/rope_parity.cpp |
| rope.cpp | `strata::kernels::rope_neox_apply\n` | planned | refs: src/kernels/rope_parity.cpp |
| router_sigmoid.cpp | `strata::kernels::router_sigmoid\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| router_top10.cpp | `strata::kernels::router_top10\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| s2_expert_grouped.cpp | `strata::kernels::moe_group_resident\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_grouped_s2\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_add\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_grouped_s2\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_grouped_s2_cpu_order\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_grouped_s2_dev\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_grouped_s2_multi\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_grouped_scratch_bytes\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_select\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_expert_grouped.cpp | `strata::kernels::moe_hit_select_multi\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| s2_gemv.cpp | `strata::kernels::s2_gemv\n` | fixture | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s2_gemv_q8.cpp | `strata::kernels::s2_gemv_q8\n` | fixture | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s2_gemv_fast\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s2_gemv_quads\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s_gemv\n` | fixture | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s_gemv_q8_0_split\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s_gemv_q8k\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s_gemv_q8k_split\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s_gemv_split\n` | fixture | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| s_gemv.cpp | `strata::kernels::s_gemv_split_async\n` | planned | same canonical planes as the gated GEMV fixtures (split/async/q8k variants) |
| sampler.cpp | `strata::kernels::sample_draft\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| sampler.cpp | `strata::kernels::sample_tokens\n` | fixture | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| sampler.cpp | `strata::kernels::sample_verify\n` | planned | Stage 2 follow-on: scalar ref derived from the CUDA source when its family lands |
| shared_expert.cpp | `strata::kernels::moe_combine\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| shared_expert.cpp | `strata::kernels::shared_expert\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| shared_expert.cpp | `strata::kernels::shared_expert_multi\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| shared_expert.cpp | `strata::kernels::shared_expert_scratch_bytes\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| shared_expert.cpp | `strata::kernels::shared_expert_set_native_bf16\n` | planned | refs: src/kernels/native_expert_parity.cpp / shared_expert_parity.cpp |
| verify_kernels.cpp | `strata::kernels::add_streams_broadcast\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::broadcast_streams\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::copy_i32_from_mapped_unless\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::copy_indexed\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::copy_or_zero_from_mapped\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::dense_steps\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::embedding_gather_dev\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::fetch_blobs\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::gather_rows\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::gdn_ab_multi\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| verify_kernels.cpp | `strata::kernels::gdn_conv_commit\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| verify_kernels.cpp | `strata::kernels::gdn_conv_l2_multi\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| verify_kernels.cpp | `strata::kernels::gdn_step_norm_multi\n` | planned | refs: src/kernels/gdn_parity.cpp (conv/beta-gate/out-norm transcriptions) |
| verify_kernels.cpp | `strata::kernels::gpu_stamp\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::ident_hits\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::map_ids\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::mtp_select\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::rebase_ptrs\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::resident_plan\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::row_top_prob\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
| verify_kernels.cpp | `strata::kernels::wait_flag_ge\n` | not-fixtureable | device sync / bookkeeping primitive — no numeric output |
| verify_kernels.cpp | `strata::kernels::wait_flag_ge_or\n` | not-fixtureable | device sync / bookkeeping primitive — no numeric output |
| verify_kernels.cpp | `strata::kernels::window_ids\n` | planned | spec-decode verify helpers; land with the MTP workstream fixtures |
