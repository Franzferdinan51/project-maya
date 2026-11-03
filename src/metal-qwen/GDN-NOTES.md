# GDN/GR workstream notes (branch qwen-native-gdn)

## MAC-QUEUE (for coordinator / Mac runner)
MSL syntax check (no Maya build, no lock needed - standalone compiler):
  mkdir -p /tmp/gdn-msl && cd /tmp/gdn-msl
  # copy common.metalh, gdn.metal, gr.metal from ws-gdn staging src/metal-qwen/kernels/
  for f in gdn gr; do cat common.metalh $f.metal > $f.full.metal; \
    xcrun -sdk macosx metal -c $f.full.metal -o $f.air; done
Full build (through the lock, per QWEN-NATIVE-STATE.md):
  git archive HEAD | (on Mac) tar -x -C ~/qwen-native-staging/ws-gdn/src
  PATH=/opt/homebrew/bin:$PATH; mkdir /tmp/maya-build.lock  # retry 20s, <=20min
  cmake -S ~/qwen-native-staging/ws-gdn/src -B ~/qwen-native-staging/ws-gdn/build \
        -DSTRATA_ENABLE_METAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build ~/qwen-native-staging/ws-gdn/build -j 4; rmdir /tmp/maya-build.lock
  # blocked until workstream rt lands strata::metal::launch in the runtime.
MSL syntax check DONE 2026-11-02: both gdn.full.metal and gr.full.metal
compile clean with xcrun metal on the Mac (one fix needed: MSL has no
log1p; the prelude now uses the Kahan form log(u)*e/(u-1), u=1+e).
VM parity harness (already run, numbers in the commit message):
  g++ -O2 -std=c++17 tools/gdn_gr_scalar_ref.cpp -o /tmp/gdn_ref && /tmp/gdn_ref

## Design notes
- common.metalh extended/created by this workstream: sigmoid/silu/softplus
  helpers, bf16<->f32, tg_sum threadgroup reduction.
- gr_read native contract substitutes this family's generic BF16 GEMV
  (gr_gemv_kernel) for the CUDA path's bf16_gemv_fp32_mmvf (dense ws entry):
  same math (bf16 weights x fp32 acts, fp32 acc), different reduction tree.
- fused_gr: xn recomputed on the fly instead of a 40 KB threadgroup stage.
- fused_gr_read_multi launches fused_gr_read per token: bitwise-identical
  to single-token by construction (CUDA contract), not yet the P6 speedup.
- No prompt/chunk-specific entries exist in this family's headers: the
  layer calls the same per-token entries for prefill tokens (layer.cpp).
