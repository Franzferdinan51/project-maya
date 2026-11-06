# DFlash2 for the MiniMax (mm3) Metal engine — implementation spec

Backend: `src/metal/mm3_dflash2.{hpp,cpp}`, class `maya::metal::Mm3DFlash2`, sitting behind
the same interface as `Mm3Draft` (load / context / propose). The engine dispatches on the
draft directory's `config.json`: a `dflash_config` object means DFlash2, anything else is
DSpark. Serve loop untouched.

Sources of truth, in order: the checkpoint itself (`incoai/MiniMax-M3.1-Flash-DFlash2`,
Apache-2.0, config.json + one BF16 `model.safetensors`, 4,247,776,488 bytes, downloaded to
`~/maya-models/dflash2/` on the Mac internal disk); SGLang's `dflash.py` /
`dflash_worker_v2.py` (the reference implementation); this repo's own GLM DFlash2 port
(`src/metal/glm_metal_dflash.cpp`, written against the CUDA engine and llama.cpp's
`src/models/dflash.cpp`) for the ggml idioms.

## Checkpoint facts (from its config.json — these win over any secondary source)

- architectures `DFlash2DraftModel`, model_type `qwen3`. hidden 6144, intermediate 12288,
  **5 layers**, heads 64, kv heads 8, head_dim 128, vocab 200064, eps 1e-6,
  rope_theta 5,000,000 (full rotary, NEOX), **sliding_window 2048**,
  max_position_embeddings 1,048,576.
- `dflash_config`: block_size **8**, conv_group_size 16, conv_kernel_size 2,
  mask_token_id 200061, selector_rank 256, selector_top_k 32,
  **target_layer_ids [5, 16, 26, 37, 47, 58] — SIX taps**, not DSpark's five
  [3, 17, 31, 45, 59]. fc.weight is [6144, 36864] = 6 x 6144, confirming six.
- No embed_tokens / lm_head in the checkpoint: it borrows the target's, like DSpark.
- All weights plain BF16 (no MXFP8 expansion, unlike DSpark). Norms are plain Qwen3
  RMSNorm (weight as stored — NO Gemma +1; the checkpoint config has no use_gemma_norm).

## Tensor map (safetensors, bare names -> engine storage)

All stored F32 (this ggml's Metal library has no BF16xBF16 mul_mat; same rule as DSpark).

- `fc.weight` [D, NT*D], `hidden_norm.weight`, `norm.weight` (final norm)
- per layer l: `layers.l.input_layernorm.weight`, `post_attention_layernorm.weight`,
  `self_attn.{q,k,v,o}_proj.weight`, `self_attn.{q,k}_norm.weight` [head_dim],
  `attention_conv.base_kernel` [2, conv_taps, D], `attention_conv.kernel_projection.weight`
  [2*conv_taps*G, D] with G = D / conv_group_size; same pair under `mlp_conv.`
  `mlp.{gate,up,down}_proj.weight`
- `candidate_selector.hidden_projection.weight` [rank, D],
  `candidate_selector.predecessor_codebook` / `successor_codebook` [V, rank] — stored
  transposed [rank, V] so ggml_get_rows gathers a token's codebook row as a column.

base_kernel layout note: stored as ggml [D, conv_taps, 2]; the linear byte layout is
identical to the safetensors [side][tap][channel] order, so it is a straight copy.

## Pass structure

context(taps, T, pos0) — per committed slice, exactly like DSpark's:
acc[t] = sum_j fc_j . taps_j[t]; g = rms(acc, hidden_norm); per layer
K = rope(rms_head(K_proj g, k_norm)), V = V_proj g written into BF16 caches at
pos0..pos0+T-1. No convolution, no attention — context positions only ever appear as
keys/values (SGLang's kv_proj_only path).

propose(anchor, pos, n) — ONE parallel pass over the block [anchor, mask x (B-1)] at
positions pos..pos+B-1 (B = block_size = 8, so at most B-1 = 7 drafts):
per layer: xn = rms(x, input_layernorm); dyn = kernel_projection . xn;
xc = dconv(xn, dyn, side 0); q/k/v from xc; q/k head-normed + rope; block K/V written to
the caches, then every row attends non-causally to the window [hi-W+1, hi],
hi = pos+B-1 (same shared-window convention as the DSpark port); o = o_proj . attn;
o = dconv(o, dyn, side 1); x += o; then the same wrap around the MLP
(xn2 = rms(x, post_attention_layernorm); dyn2; fc = dconv(xn2, dyn2, 0);
mlp = down(silu(gate fc) * up fc); mlp = dconv(mlp, dyn2, 1); x += mlp).
Final: hid = rms(x, norm); logits = head . bf16(hid) (target head, BF16 round-trip on
the hidden first — same as DSpark).

Grouped dynamic conv (SGLang `_grouped_conv`, ggml form from glm_metal_dflash.cpp):
coefficient for (row, tap, group g) = base_kernel[side][tap][channel] +
kernel_projection(x)[row, side, tap, g], shared by the group's 16 channels;
out[row] = sum_tap coef_tap * shift(x, tap), the shift zero-fills the first `tap` rows —
a block never reads the previous block.

Candidate selector (SGLang CandidateSelector + the GLM port's host walk):
draft slots are the block's mask rows 1..B-1. Per slot: cand = top-K (K = 32) of the
head logits, unary = those logits; gate = hidden_projection . hid[slot].
score(slot, prev_pick, c) = unary[c] + <A[prev] * gate, B[cand[c]]> with codebooks A/B.
Greedy lattice walk: slot 0's predecessor is the anchor token; each later slot's
predecessor is the previous slot's candidates, indexed by the walk's pick.
Confidence per slot = the pick's softmax share of that slot's lattice scores
(exp(0)/sum exp(s - smax) at the argmax) — the top-1 probability the serve loop's
confidence floor expects, over the selector's candidate distribution.

## Deliberate pass-1 scope

- Greedy lattice walk only (the serve loop is greedy/exact-match; SGLang's sampling
  walk with uniforms/temperatures is future work, as is D5 rejection sampling).
- Window convention shared across block rows, as in the DSpark port (SGLang's backend
  uses sliding_window - 1 per query; the <= B-1 key difference at the far tail is the
  same class of approximation the verified DSpark port carries).
- Verification gates: fixture agreement vs the torch reference in tools/mm3/
  (dflash2_ref.py) — loader, context, propose, selector walk, confidences — plus the
  ctest non-big mm3 subset. The end-to-end greedy byte-identity gate through the serve
  loop needs a live window and is NOT part of this pass.
