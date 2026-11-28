# Speed pass — 2026-11-20 (Hy4 first, then MiniMax, then Qwen)

Method: one engine at a time; the same serve-path bench for every run (200-token
greedy decode on a fixed prompt, ~1900-token prefill, temperature 0, rates refined
by subtracting each run's own prefill share). Every Hy4 change passed the hy4
ctest suite (8/8, including the fixture e2e that compares Metal and CPU forwards
token for token) and a clean greedy prompt before it was kept.

## Hy4 Metal (Maya-S, 256K context) — the deep pass

| Stage | Decode tok/s | Prefill tok/s |
|---|---|---|
| baseline (external SSD pack) | 5.05 | 29.4 |
| pack moved to internal NVMe (old binary) | 4.98 | 29.3 |
| + residency truth, Engram RAM tier, weight-readback cache | 5.30 | 29.1 |
| + compact attention, batched per-head matmuls | 6.34 | 29.2 |
| + grouped expert matmuls (ggml_mul_mat_id) | 7.81 | 79.8 |
| + decode-window reuse | 7.88 | ~79 |
| + fused attention spans | 9.34 | 79.5 |
| + fused ffn spans | **11.53** | **79.9** |

Decode +128%, prefill +172%. Graph commits per decode token: ~348 -> ~150;
the profiled GPU span fell ~120 -> ~57 ms/token. Page faults per 100 decode
tokens fell ~9x; the Engram row cache serves ~91% of lookups from pinned RAM.

What the investigation found (the Monitor's "all 512 experts SSD only" was a
telemetry artifact, not the runtime state):

- The routed experts are copied into Metal buffers at load and are fully
  resident in unified memory. The STAT line carried no tier fields, so the
  Monitor's unified card read 0/0/512. The tiers are now reported truthfully
  (resident weight bytes, Engram lookup/hit/promotion counters, gather time).
- The real file-backed tenant is the Engram n-gram table (~63 GB, random row
  reads per token per Engram layer). It now has a bounded, CLOCK-evicted RAM
  row cache (the Strata ple_reader pattern), sized by --engram-cache-gb, with
  misses attributed to page-cache vs SSD by mincore.
- The load-copy left the mapped weight pages counted twice in RAM; drop_mapped
  now runs after the copy, with the hardware page size (Apple Silicon's 16K
  pages made the old 4096 constant a no-op).
- Attention converted the whole KV prefix F16->F32 every token and uploaded a
  prefix x heads mask (~25 GB/token of copies at 256K). It now attends over
  just the visible rows: a contiguous window view for SWA layers, a
  deduplicated gather for indexer layers; the sink row still appends. The
  indexer pool readback is sized to the live prefix, and group scoring runs
  in parallel with identical per-group arithmetic.
- The per-head q-absorption and v_b matmuls are batched over the head axis;
  the routed experts run as grouped mul_mat_id (one kernel each way instead
  of sixteen expert chains); the decode window keeps its tensors, buffer and
  gallocr across tokens; and each layer's host-free spans are fused into one
  graph commit (a dense layer is a single commit).

Output equivalence: the fixed coherence prompt's content is byte-identical
before/after; the reasoning trace differs in one tail token (a greedy flip
from softmax reduction-order drift in the compact attention), both fully
coherent. A clean arithmetic prompt answers correctly with finish=stop.

Known secondary issue (deferred, not a speed regression): on the /no_think
chat-template path the content field can loop the think-close token while
reasoning_content stays correct; reproduced on the pre-pass binary too. It
is a template/prefill interaction and needs its own parity pass against the
pack's chat template.

Next levers, in profile order: route in-graph (ggml_argsort_top_k, as the mm3
engine does) to remove the router readback and fuse a SWA MoE layer to one
commit — the pick tie-breaking must match the host sigmoid+bias+sort exactly;
then an F16 score path and persistent per-shape graphs.

## MiniMax M3.1 Flash (Maya-XS, 64K context) — baseline, no clear wins

Decode 13.17 tok/s, prefill 268.3 tok/s. The engine is already in good shape:
a persistent backend scheduler, two graphs per layer, routing in-graph
(sigmoid + correction bias + argsort_top_k, no host round-trip), grouped
expert matmuls with float-expert token grouping, compact kept-block sparse
attention, and every expert resident (the XS quant fits the machine whole).
A speculative-draft path exists but is not enabled by the XS serve config.
No change was taken: the remaining decode cost needs per-span instrumentation
to attribute before any edit would be a verified win.

## Qwen3.8 Flash Next (Strata q2_0 pack) — baseline, already fast

Decode 100.58 tok/s, prefill 1648.8 tok/s. The Strata tiering is doing its
job: experts loaded into RAM (~38 GB) with a 64-expert GPU cache, the n-gram
table streamed. No change was taken.
