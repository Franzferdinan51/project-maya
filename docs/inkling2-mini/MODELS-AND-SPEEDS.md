# Inkling 2 Mini — models and measured speeds

Machine for every number below: Apple M5 Max, 128 GB. Measured 2026-11-22
through Maya's serve unless a route says otherwise.

Two kinds of numbers appear here and they are **not** interchangeable:

- **Real-chat** — the serve log's own per-answer rate on real interactive use.
- **Synthetic bench** — scripted prompts. The long-prompt bench uses repeated
  filler text, which is the worst case for a speculative drafter (it predicts
  from context; filler gives it nothing to predict). Treat a synthetic decode
  figure as a floor, not the real-chat rate.

## Status (2026-11-22)

Production on `:8086` is **route 1** (the ik2 bridge + the netprojects MLX
build). The Strata route (route 3) ran in production for part of 2026-11-22
and was parked the same day: 37–39 tok/s on real chats against ~100 for the
bridge. It stays available as an alternate launcher; nothing was deleted.

## The builds

| build | format | size | draft / speculation | vision via Maya |
|---|---|---|---|---|
| `netprojects/Inkling-2-Mini-MLX-4bit` | MLX 4-bit, 4 shards + `mtp.safetensors` | 17.2 GB | MTP block draft (via the bridge) | yes |
| `Duckets/Inkling-2-Mini-MLX-4bit` | MLX 4-bit, 4 shards, **no MTP** | 17.3 GB | none possible (no MTP tensors) | yes (bridge, draft off) |
| `inkling-2-mini` SQ-20 pack | Strata squant | ~19 GB | DSpark (Strata engine) | no (transport) |
| `Inkling-2-Mini-UD-Q4_K_XL` | GGUF | 17.6 GB | DSpark via separate safetensors drafter | no |
| `Inkling-2-Mini-UD-Q4_K_M` | GGUF + dspark Q8_0 drafter GGUF | 16.6 + 0.4 GB | draft-dspark (llama-server) | no |

## Routes and speeds

### 1. Maya ik2 bridge + netprojects MLX — current production

`tools/ik2_bridge.py --backend mlx --mlx-draft-block 4`, single slot,
262144 context, config `~/maya-models/maya-inkling-2-mini.json`.

- Prefill ≈ **2,000 tok/s** at a 23k prompt (synthetic bench: 23,136 tokens
  read in ~11 s).
- Decode ≈ **100 tok/s** on real chats at ~20k context (owner-observed,
  serve monitor).
- Decode ≈ 40 tok/s on the synthetic filler bench — the drafter's worst
  case, see the note at the top.
- Vision works (the MLX backend reads images natively). Tool calls
  round-trip. This is the only route that has both the speed and images.

### 2. Maya ik2 bridge + the Duckets MLX port, draft off

Same bridge with `--mlx-draft-block 0`. The port carries no MTP tensors, and
the bridge refuses it with a draft block set ("No MTP tensors found").

- Prefill ≈ 1,500 tok/s, decode ≈ 35 tok/s at a 23k prompt (synthetic bench).
- Vision works. The same port also loads in LM Studio (oMLX runtime) with
  working structured tool calls.

### 3. Strata engine, SQ-20 pack

`strata-inkling` (engine 0.1.65, Strata 0.2.19), DSpark on, single slot,
262144 context.

- Standalone bench: decode **63.8 tok/s** median (long prose), prefill
  ≈ 2,200 tok/s.
- Through Maya the rate depends on the request shape. The engine's fast
  path needs DSpark + a single slot + a **greedy** request + no penalties:
  short greedy answers run 73–115 tok/s; sampled answers (temperature 1.0)
  and long-context answers run 23–55 tok/s. A serve whose default sampling
  is temperature 1.0 never touches the fast path — that, not the model, was
  the 38 tok/s report.
- Text only through Maya: the engine takes images as file keys while Maya
  sends embedding transport, so Maya refuses image requests rather than
  faking them.
- The SQ-20 requant is lossless (100% tensor match against the source) and
  beat every GGUF route tested on this machine.

### 4. Strata 0.2.20 (test copy) + Unsloth UD-Q4_K_XL GGUF

- Draft off: decode 39.8 tok/s, prefill ≈ 1,480 tok/s.
- DSpark on: decode 52.3 tok/s, prefill ≈ 1,807 tok/s. Only reachable via a
  separate `dspark.`-prefixed safetensors drafter — the GGUF drafter route
  itself is broken on Metal in 0.2.20 (the engine exits on a missing
  `dspark.markov_head` tensor; the split drafter GGUF's names are not read).
- Lossy requant (98.0% match). Slower than SQ-20 **and** lossy — not
  promoted, do not ship.

### 5. Maya ik2 bridge, llama variant (GGUF) — do not use

`tools/ik2_bridge.py` → `llama-server` (Unsloth prebuilt), UD-Q4_K_M with a
dspark Q8_0 drafter GGUF, draft n-max 3, fp8 KV.

- Prefill ≈ **370 tok/s** (46k prompt) — an order of magnitude under the
  MLX bridge on the same machine. Text only in this config. Kept here so
  nobody re-tests it by accident.

## Lessons baked into these numbers

- Same weights, different route = a 3x speed difference. The engine/bridge
  choice is the decision, not the quant.
- A speculative drafter only pays on real text. Repeated-filler benches
  understate it; short-answer benches overstate the long-answer rate.
- Sampling settings decide which engine path runs (see route 3's greedy
  fast path). A "slow model" report can be a sampling default.
- Context length is part of the rate. A speed quoted without its prompt
  size means nothing.
