# Qwen 3.8 Flash Next — vision in Maya (promotion record)

Status: COMPLETE (overnight job maya-qwen-vision-overnight, 2026-11-02 23:20 ET).
Qwen left loaded with vision on as the standing engine.
Work done by AI (Muse / Juno) per Duckets' Maya contribution rules.

## What was already proven (evening staging, QWEN-VISION-STAGING.md)
- Encoder built at /Users/duckets/vision-staging/build-vision/bin/maya-vision
- Standalone GPU proof: test image → valid SVE1, 300 tokens (20×15), 106 ms;
  BF16 mmproj accepted; encoder footprint ~1.37 GiB.

## Promotion (this run)
- Backup: ~/maya-models/maya-qwen.json.bak-pre-vision
- Config ~/maya-models/maya-qwen.json: appended "--vision" to args; added
  top-level "vision" entry {exe: staging maya-vision, mmproj:
  /Users/duckets/Strata-data/mmproj-Qwen3.8-Flash-Next-BF16.gguf, model:
  shard-1 Q2_0 GGUF, gpu: true}. No serve code changes (per audits).
- Restart: ~/bin/maya-qwen.
- Verified after restart: engine process runs with --vision; maya-vision
  encoder resident; serve log "ready ... images on"; GET :8090/v1/models →
  input_modalities ["text","image","video"], n_ctx 262144 (standing settings
  preserved: context 262144, reasoning_effort high, idle_unload_s 0).

## Proof (2026-11-02 ~23:35 ET, through the front door :8090)
- Image end-to-end: staging test image (red circle upper-left, blue square
  lower-right, number 47 center) via POST /v1/chat/completions with a base64
  image_url part → answer: "A red circle – located on the left side… The
  number 47 – written in black, positioned in the middle… A blue square –
  located on the lower right side." EXACT MATCH.
- Usage: prompt_tokens 331 = 300 image tokens + ~31 text — the context
  guard's image-token estimate agrees with the expanded pad count (staging
  doc watch item, settled).
- Text control (same setup): "What is the capital of Australia?" →
  "The capital of Australia is Canberra." Correct.
- /v1/models: input_modalities ["text", "image", "video"], n_ctx 262144.
- Hermes smoke: `hermes chat -Q --provider maya-qwen -m
  qwen3.8-flash-next-q2_0` → exactly "MAYA-QWEN-VISION-OK".
- Note: request latency was high during the proof (~100 s/request) because
  the engine is single-request and the Mac was simultaneously running
  qwen-native port work; not a vision-path defect.

## Gates
- ctest (build-metal): 100% passed, 36/36 — mm3 baseline + restore test
  included, identical verdict class to the v1.1.50/Qwen-port baselines.
- Serve tests: 666 ran, 16 failures + 1 error = 17 — exactly the pristine
  v1.1.50 baseline count. Failing areas: test_responses_schema (structured
  outputs), test_sse_sets_tcp_nodelay, two test_models DeletePage cases.
  None are vision- or Qwen-related, and this change touched no serve code
  (config only), so: no regression.
- Standing settings after restart: context 262144, reasoning_effort high,
  idle_unload_s 0 — all preserved in the promoted config.

## Push
- No fork source files changed: the wiring is config-only
  (~/maya-models/maya-qwen.json, outside the repo) plus the pre-built
  staging encoder binary (binary — cannot go through the snapshot route;
  it lives at /Users/duckets/vision-staging/build-vision/bin/maya-vision
  on the Mac).
- This doc is pushed to branch qwen-metal as QWEN-VISION.md.
- Rollback: restore ~/maya-models/maya-qwen.json.bak-pre-vision and run
  ~/bin/maya-qwen (text-only service returns).
