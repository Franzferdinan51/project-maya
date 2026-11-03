# moe parity — VM harness + Mac mini-runner queue

`moe_ref.cpp` holds scalar references transcribed from the CUDA sources and
float simulations of the MSL kernels' arithmetic. VM results (2026-11-02,
seed 20261102, 64 tokens x 512 experts incl. an all-equal tie token and a
[-30,30] wide-range token):

```
router_top10   ids exact 64/64 tokens, max |w_ref - w_sim| 1.490e-08
native_router  ids exact 64/64 tokens (vs double ref), max |w delta| 1.490e-08
router_sigmoid ids exact 64/64 tokens, max |w_ref - w_sim| 2.980e-08
combine        max |double_ref - fma_sim| 1.192e-07 over 2560 elements
s2 expert      scalar pipeline ran (fixture dumped)
```

Run it with a directory argument to dump fixtures (router logits/ids/weights,
sigmoid bias/ids/weights, combine parts/w/shared/ref, s2 blob + x_q8_0 +
gate_up/out references). The s2 gate_up reference is in GATE-MAJOR layout
(first FF = gate, second FF = up), matching the kernels.

## MAC-QUEUE (for the coordinator)

1. DONE 2026-11-02 (ws-moe staging, lock held ~17 min in queue): archive of
   HEAD built with -DSTRATA_ENABLE_METAL=ON, target strata_kernels -j 4 —
   EXIT=0, all six moe host TUs compiled, no errors in build.log. STILL
   OPEN: (b) the MSL library (common.metalh + moe.metal +
   shared_expert.metal + s2_expert.metal) compiles — it is not wired into
   the build until rt's launch exists; a standalone `xcrun metal` syntax
   pass over the concatenated library would settle the open risks (the
   `grp_ptr` integer->device-pointer cast in s2_expert.metal's grouped
   kernels, `simdgroup_index_in_grid` in s2_gu_k/s2_down_k); (c)
   metal_kernel.hpp here vs rt's authoritative copy.
2. Mini-runner: load the fixtures above, run router_top10 / router_sigmoid /
   native_router_top10 / moe_combine / native_moe_combine /
   moe_hit_grouped_s2 (1 hit, slot 0, dst 0, x_scales null) through
   strata::metal::launch on a real stream, compare: router ids must match
   EXACTLY, weights <= 1e-6, combine <= 1e-5 abs, s2 out rel <= 1e-4
   (its own CUDA-vs-CPU contract is float-rounding level).
3. shared_expert needs dense-family entries live (s_gemv/s2_gemv_q8/
   quantize/native_mmvq) — run its parity after the dense merge, using
   upstream's shared_expert_parity selftest shape.
