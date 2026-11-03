# Model-level parity: the greedy-decode capstone

## Why this layer exists

GLM's correctness oracle is model-level: `src/core/glm_model_test.cpp` runs a
whole model against `ref/glm.py`'s teacher-forced logits on the GPU. Qwen has
no Python reference and no model test (survey: `QWEN-UPSTREAM-GROUNDWORK.md`
§4 caveat, §7.2) — so the Qwen oracle is built from the two ends that do
exist: per-entry fixtures against scalar references (kernel level, see
`FIXTURE-FORMAT.md`), and this harness at the top: **the composed engine,
a fixed prompt, greedy sampling, an exact token-id sequence as the verdict.**

Greedy is what makes a model-level gate possible without a reference
implementation: at temperature 0 every stage is deterministic, the expected
output is a short list of integers, and the *first* diverging position is a
fault localiser (everything before it is exonerated). A logits-tolerance gate
would need the reference's logits; a token gate needs only its tokens, which
any working build can emit.

## The golden

`.mqgt` file (format implemented in `e2e_greedy_decode.cpp`, the single
definition): magic, version, pack id, model id, prompt token ids, expected
greedy token ids. One golden per (pack, prompt) — start with one short prompt
(8–16 tokens, 32 generated) at the Q2_0 pack, add a long-prompt golden
(≥ 2048 tokens, exercising QSA selection above its identity bound) when the
QSA family lands.

**Capture protocol** (on the reference build — CUDA, Batman or any CUDA host;
the HIP build is an equally valid reference and a good cross-check):

1. Build the tree with the CUDA backend; load the same pack the Metal build
   will run (`packs/q2_0`).
2. Feed the golden prompt; decode greedily (`SamplerParams{greedy=true}`,
   temperature 0, no penalties, no logit bias, spec/MTP off — the draft path
   must not be part of the reference) for N tokens.
3. Record the produced ids with the pack/model ids; commit the `.mqgt` under
   `src/metal-qwen/tests/goldens/`.

## The harness (`e2e_greedy_decode.cpp`)

- `--selftest` — golden format round-trip. Runs today, no model.
- `--compare <golden.mqgt> <candidate.txt>` — exact-compare any build's
  emitted ids (a whitespace-separated text file, so serve logs and future
  in-process runs both feed it) and report the first divergence. Runs today.
- `--run <pack_dir> <golden.mqgt>` — the Stage-2 final gate: drive the
  in-process session (same construction serve uses for the qwen family),
  greedy-decode, compare. **Skeleton**: prints the integration contract and
  exits 77 (the tree's `SKIP_RETURN_CODE` convention, cf. `pf_wmma_parity`)
  until real Metal kernels exist to generate with — a skip, never a fake
  pass.

## Captured goldens (vendored-serve source, 2026-11-02)

The first two goldens were captured from the **vendored Metal engine's serve**
(the behavioral reference for this port; the CUDA-host capture above remains
the canonical alternative):

- `goldens/q2_0_short.mqgt` — 26 prompt ids (chat-templated "list the first
  ten prime numbers"), 32 greedy ids.
- `goldens/q2_0_long.mqgt` — 5,983 prompt ids (a long document + a counting
  instruction), 32 greedy ids; exercises QSA selection above its identity
  bound.

Method, and why it is sound: serve returns text, not ids (the vendored engine
reports no logprobs), so the produced ids are recovered by retokenization —
faithful **only** for continuations with canonical tokenization. Both prompts
therefore continue as digit sequences, and every capture asserted:
`usage.prompt_tokens` == the /tokenize prompt length (identical conditioning),
`finish_reason == "length"` (no EOS inside the window),
`usage.completion_tokens == 32 == len(encode(content))`, a digit-pattern
match on the content, and a detokenize round-trip. A free-prose capture was
tried first and correctly rejected by these checks (32 produced tokens
retokenized to 18 — the model's own segmentation of prose is non-canonical).
Captures ran with the serve's standing config (spec on, per the reference
engine's normal operation) at temperature 0 with reasoning off; the native
candidate is produced by `maya generate --tokens <golden prompt ids>
--greedy --max-new 32` and fed to `--compare` as a whitespace-separated dump.

## Gate definition for Stage 2 (proposed to the coordinator)

Stage 2 is done when, on the Mac, in one run: `metal_kernel_parity` is green
on every fixture whose family has landed, **and** `e2e_greedy_decode --run`
reproduces the CUDA golden's 32 tokens exactly on the short prompt. The
long-prompt golden joins the gate when QSA selection lands.
