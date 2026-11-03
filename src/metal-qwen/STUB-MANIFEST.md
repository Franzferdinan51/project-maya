# STUB MANIFEST — native Metal backend for upstream Maya's Qwen engine (stage 1 survey)

Generated 2026-11-02 by mechanical extraction from the tree (header parser + `.cu` definition
cross-check + caller grep), not by hand-copying. Companion files: `stubs/*.cpp` (one TU per
header group, 55 files) and `RISKS.md` (what will not compile as plain C++, and what to do
about each). The HIP backend is the template: `include/strata/hip_compat/` compiles these same
sources as HIP by shimming `cuda_runtime.h` / `cublas_v2.h`; the Metal backend is to compile
the host sources as C++ against a `metal_compat` shim and link the stubs below in place of
`src/kernels/cuda/*.cu` + `src/prefill/*.cu`.

## Counts

- **412 host-callable device entry points stubbed** in `stubs/` (252 QWEN-path, 41 SHARED Qwen+GLM, 111 GLM-only, 8 HIP/RDNA4-only prefill).
- 37 further declarations in the surveyed headers are **host-only** (implemented in `.cpp`, already plain C++ — §A2, no stub).
- 11 declarations are **HIP-only** (`#if defined(STRATA_USE_HIP)` — §A3, no stub, no Metal symbol).
- §B: the CUDA-runtime / cuBLAS surface the host sources call (the `metal_compat` shim surface).
- §C: headers carrying device code (the compat-approach risk list).

Tags: **QWEN** = reached by the Qwen family path (`src/core/layer.cpp`, `session.cpp`, `verify.cpp`,
`mtp.cpp`, `native_*.cpp`, `src/program/generate.cpp`, `src/prefill/prefill.cpp`); **GLM** = only GLM
callers/groups (`glm_*`); **SHARED** = caller evidence on both sides. Caller evidence is a name grep
over the engine sources, so for short common names (`mv`, `route`, `moe_combine`, `launch_errors`,
`supported`, `run`) the caller column can over-attribute across namespaces — the namespace in the
signature is authoritative. `Impl` was verified: every stubbed name occurs as a definition in its
mapped `.cu` (0 misses).

**Probe warning:** entries whose name ends in `_supported` / `_enabled` / `supported` / `built` /
`launch_errors` / `*_bytes` / `*_cols` / `*_splits` are *capability probes* the engine calls to decide
what to run. The stage-1 stubs abort on them exactly as specified, so a stage-1 binary dies at its
first probe; stage 1.5 should give probes honest `false`/`0` bodies before anything else (see RISKS.md §4).

## A. Device entry points (stubbed)

### `include/strata/kernels/bf16_gemv.hpp` → `stubs/bf16_gemv.cpp` (4 entries, impl: bf16_gemv.cu, native_bf16.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `bf16_gemv` | `void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream)` | bf16_gemv.cu, native_bf16.cu | layer.cpp, bf16_gemv_parity.cpp | QWEN |
| `bf16_gemv_split` | `void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int threads_per_row, void* stream)` | bf16_gemv.cu, native_bf16.cu | layer.cpp, bf16_gemv_parity.cpp | QWEN |
| `bf16_gemv_fp32_mmvf` | `void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream)` | bf16_gemv.cu, native_bf16.cu | layer.cpp, mtp.cpp, verify.cpp, gr.cu, ple.cu, shared_expert.cu | QWEN |
| `bf16_gemv_fp32_mmvf_multi` | `void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in, int64_t n_out, int n_tok, void* stream)` | bf16_gemv.cu, native_bf16.cu | verify.cpp, shared_expert.cu | QWEN |

### `include/strata/kernels/cvec.hpp` → `stubs/cvec.cpp` (6 entries, impl: cvec.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `cvec` | `const Cvec& cvec()` | cvec.cu | layer.cpp, verify.cpp, cvec_parity.cpp, prefill.cpp, generate.cpp | QWEN |
| `cvec_upload` | `bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last, int64_t n_embd, int64_t hc, std::string& err)` | cvec.cu | cvec_parity.cpp, generate.cpp | QWEN |
| `cvec_replicate` | `bool cvec_replicate(std::string& err)` | cvec.cu | generate.cpp | QWEN |
| `cvec_set_enabled` | `void cvec_set_enabled(bool on)` | cvec.cu | cvec_parity.cpp, generate.cpp | QWEN |
| `cvec_enabled` | `bool cvec_enabled()` | cvec.cu | cvec_parity.cpp | QWEN |
| `cvec_apply` | `void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj, int64_t inj_ld, bool write, void* stream)` | cvec.cu | layer.cpp, verify.cpp, cvec_parity.cpp, prefill.cpp | QWEN |

### `include/strata/kernels/dequant_bf16.hpp` → `stubs/dequant_bf16.cpp` (4 entries, impl: dequant_bf16.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `dequant_bf16_supported` | `bool dequant_bf16_supported(int ggml_type) noexcept` | dequant_bf16.cu | glm_prefill.cu, dequant_bf16_test.cpp | QWEN |
| `dequant_bf16` | `void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out, void* stream)` | dequant_bf16.cu | dequant_bf16_test.cpp | QWEN |
| `dequant_f16` | `void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out, void* stream)` | dequant_bf16.cu | glm_prefill.cu, gemm.cu | QWEN |
| `dequant_f32` | `void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream)` | dequant_bf16.cu | dequant_bf16_test.cpp | QWEN |

### `include/strata/kernels/dequant_s2.hpp` → `stubs/dequant_s2.cpp` (1 entries, impl: dequant_s2.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `dequant_s2` | `void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks)` | dequant_s2.cu | dequant_s2_parity.cpp | QWEN |

### `include/strata/kernels/elementwise.hpp` → `stubs/elementwise.cpp` (14 entries, impl: elementwise.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `embedding_gather` | `void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets, int64_t n, int code_bits, int code_bias, int group_elems, float* out, void* stream)` | elementwise.cu | layer.cpp, elementwise_parity.cpp | QWEN |
| `gdn_gate` | `void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens, int64_t h_v, void* stream)` | elementwise.cu | layer.cpp, elementwise_parity.cpp | QWEN |
| `scale_inplace` | `void scale_inplace(float* x, int64_t n, float s, void* stream)` | elementwise.cu | layer.cpp, elementwise_parity.cpp | QWEN |
| `add_inplace` | `void add_inplace(float* dst, const float* src, int64_t n, void* stream)` | elementwise.cu | expert_source.cpp | QWEN |
| `f32_to_f16_bulk` | `void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream)` | elementwise.cu | elementwise_parity.cpp | QWEN |
| `f32_to_bf16_bulk` | `void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream)` | elementwise.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |
| `silu_inplace` | `void silu_inplace(float* x, int64_t n, void* stream)` | elementwise.cu | layer.cpp, elementwise_parity.cpp | QWEN |
| `rms_norm_weighted` | `void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream)` | elementwise.cu | layer.cpp, verify.cpp, elementwise_parity.cpp | QWEN |
| `doorbell_ring` | `void doorbell_ring(uint32_t* d_seq, void* stream)` | elementwise.cu | layer.cpp | QWEN |
| `doorbell_wait` | `void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream)` | elementwise.cu | session.cpp | QWEN |
| `copy_from_mapped` | `void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream)` | elementwise.cu | mtp.cpp, session.cpp, verify.cpp | QWEN |
| `copy_rows_from_mapped` | `void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows, const int32_t* count, void* stream)` | elementwise.cu | verify.cpp | QWEN |
| `doorbell_publish` | `void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream)` | elementwise.cu | layer.cpp, verify.cpp | QWEN |
| `copy_i32_from_mapped` | `void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream)` | elementwise.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |

### `include/strata/kernels/exl3.hpp` → `stubs/exl3.cpp` (26 entries, impl: exl3.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `view` | `Mat view(const Mat& m, int c0, int nc)` | exl3.cu | glm_model.cu, glm_prefill.cu, ds4_gguf.cu, ds4_model.cu, mm3_exl3.cu | GLM |
| `k2_supported` | `bool k2_supported(int k2, int cb)` | exl3.cu | glm_model.cu, ds4_main.cpp, ds4_model.cu, exl3_legacy.cu, mm3_exl3.cu | GLM |
| `supported` | `bool supported(const Mat& m)` | exl3.cu | glm_model.cu, glm_prefill.cu, ds4_gguf.cu, ds4_iq.cu, ds4_model.cu, exl3_legacy.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu +2 more | SHARED |
| `mv` | `bool mv(const MvJob* jobs, int n, cudaStream_t s)` | exl3.cu | glm_dflash.cu, glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu, glm_multi.cu, glm_rows.cu, glm_window.cu, ds4_gguf.cu, ds4_iq.cu +6 more | GLM |
| `set_skip` | `void set_skip(const int* flag)` | exl3.cu | glm_fast_path.cu, glm_window.cu, mm3_exl3.cu | GLM |
| `mv_splits` | `int mv_splits(const Mat& m)` | exl3.cu | glm_model.cu, mm3_exl3.cu | GLM |
| `mv_cols` | `int mv_cols(const Mat& m)` | exl3.cu | ds4_model.cu, mm3_dspark.cu, mm3_exl3.cu, mm3_model.cu, mm3_tp.cu | GLM |
| `prepare` | `bool prepare(cudaStream_t s)` | exl3.cu | glm_fast_path.cu, ds4_gguf.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu, mm3_tp.cu | GLM |
| `mixed_supported` | `bool mixed_supported(const ExpertLayout& lay)` | exl3.cu | ds4_model.cu, mm3_exl3.cu | GLM |
| `moe_gate_up` | `void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s)` | exl3.cu | glm_fast_path.cu, glm_window.cu, remote_experts.cpp, ds4_gguf.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu | SHARED |
| `moe_down` | `void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s)` | exl3.cu | glm_fast_path.cu, glm_window.cu, remote_experts.cpp, ds4_gguf.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu | SHARED |
| `win_experts` | `bool win_experts(const WinArgs& a, cudaStream_t s)` | exl3.cu | glm_rows.cu, glm_window.cu, ds4_gguf.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu, mm3_tp.cu | GLM |
| `several_ok` | `bool several_ok()` | exl3.cu | glm_rows.cu, glm_window.cu, mm3_exl3.cu | GLM |
| `mma_supported` | `bool mma_supported()` | exl3.cu | mm3_exl3.cu | GLM |
| `prompt_tc` | `bool prompt_tc()` | exl3.cu | glm_prefill.cu | GLM |
| `prompt_wmma` | `bool prompt_wmma()` | exl3.cu | — (no engine caller found; tests/tools only) | GLM |
| `set_gate_up` | `void set_gate_up(const SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh, cudaStream_t s)` | exl3.cu | glm_prefill.cu, ds4_gguf.cu, ds4_iq.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu, mm3_tp.cu | GLM |
| `set_down` | `void set_down(const SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s)` | exl3.cu | glm_prefill.cu, ds4_gguf.cu, ds4_iq.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu, mm3_tp.cu | GLM |
| `gemm_rows` | `void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s)` | exl3.cu | mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu, mm3_tp.cu | GLM |
| `reconstruct_rows` | `void reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s)` | exl3.cu | glm_prefill.cu, ds4_model.cu, mm3_exl3.cu | GLM |
| `reconstruct_inner_rows` | `void reconstruct_inner_rows(const Mat& m, uint16_t* dst, cudaStream_t s)` | exl3.cu | glm_prefill.cu, mm3_exl3.cu | GLM |
| `rows_rot_f16` | `void rows_rot_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, const uint16_t* suh, uint16_t* out, cudaStream_t s)` | exl3.cu | glm_prefill.cu, mm3_exl3.cu | GLM |
| `rows_rot_out` | `void rows_rot_out(float* y, int ldy, int nrows, int ncols, const uint16_t* svh, cudaStream_t s)` | exl3.cu | glm_prefill.cu, mm3_exl3.cu | GLM |
| `reconstruct_inner` | `void reconstruct_inner(const Mat& m, uint16_t* dst, cudaStream_t s)` | exl3.cu | mm3_exl3.cu | GLM |
| `rows_f16` | `void rows_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, uint16_t* out, cudaStream_t s)` | exl3.cu | glm_prefill.cu, ds4_model.cu, mm3_exl3.cu | GLM |
| `launch_errors` | `int launch_errors()` | exl3.cu | glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu, glm_multi.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu, ds4_gguf.cu, ds4_model.cu +5 more | GLM |

### `include/strata/kernels/fused_gdn.hpp` → `stubs/fused_gdn.cpp` (3 entries, impl: fused_gdn.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `fused_gdn_conv_l2` | `void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads, float eps, void* stream)` | fused_gdn.cu | layer.cpp | QWEN |
| `fused_gdn_ab` | `void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd, int h_v, void* stream)` | fused_gdn.cu | layer.cpp | QWEN |
| `fused_gdn_step_norm` | `void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, void* stream)` | fused_gdn.cu | layer.cpp | QWEN |

### `include/strata/kernels/fused_gr.hpp` → `stubs/fused_gr.cpp` (3 entries, impl: fused_gr.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `fused_gr_supported` | `bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr)` | fused_gr.cu | layer.cpp, verify.cpp | QWEN |
| `fused_gr_read` | `void fused_gr_read(const FusedGrArgs& a, void* stream)` | fused_gr.cu | layer.cpp, cvec_parity.cpp, gr_parity.cpp | QWEN |
| `fused_gr_read_multi` | `void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf, int stamp_i0)` | fused_gr.cu | mtp.cpp, verify.cpp, gr_parity.cpp | QWEN |

### `include/strata/kernels/gdn.hpp` → `stubs/gdn.cpp` (5 entries, impl: gdn.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `gdn_step` | `void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* o, const GdnShapes& s, void* stream)` | gdn.cu | layer.cpp, gdn_parity.cpp | QWEN |
| `gdn_conv_step` | `void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels, int64_t d_conv, void* stream)` | gdn.cu | layer.cpp, gdn_parity.cpp | QWEN |
| `gdn_l2_norm` | `void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream)` | gdn.cu | layer.cpp, gdn_parity.cpp | QWEN |
| `gdn_beta_gate` | `void gdn_beta_gate(float* beta, int64_t h_v, void* stream)` | gdn.cu | layer.cpp | QWEN |
| `gdn_out_norm` | `void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S, float eps, void* stream)` | gdn.cu | layer.cpp, gdn_parity.cpp | QWEN |

### `include/strata/kernels/glm_batch.hpp` → `stubs/glm_batch.cpp` (29 entries, impl: glm_batch.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `launch_errors` | `int launch_errors()` | glm_batch.cu | glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu, glm_multi.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu, ds4_gguf.cu, ds4_model.cu +5 more | GLM |
| `bf16_to_f32` | `void bf16_to_f32(const uint16_t* src, float* dst, int64_t n, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu, ds4_model.cu, dequant_bf16_test.cpp | GLM |
| `f32_to_f16` | `void f32_to_f16(const float* src, void* dst, int64_t n, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu, s2_gemv_q8_parity.cpp, shared_expert_parity.cpp | GLM |
| `bf16_to_f16` | `void bf16_to_f16(const uint16_t* src, void* dst, int64_t n, cudaStream_t s)` | glm_batch.cu | mm3_dspark.cu, mm3_kernels.cu | GLM |
| `embed_rows` | `void embed_rows(const float* emb, float* R, int T, int n_embd, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `hc_update` | `void hc_update(const float* block_out, float* R, const float* post, const float* comb, float* ss, int T, int n_embd, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `hc_finish` | `void hc_finish(const HcFinishArgs& a, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `kda_conv` | `void kda_conv(const float* const proj[3], const float* const conv_w[3], const float* conv_state, float* const out[3], int T, int d_inner, int d_conv, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `kda_conv_state` | `void kda_conv_state(const float* const proj[3], float* conv_state, int T, int d_inner, int d_conv, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `kda_rec` | `void kda_rec(const float* q, const float* k, const float* v, const float* g1_raw, const float* dt_bias, const float* ssm_a, float lower_bound, const float* beta_raw, float* state, const float* g2_raw, const float* norm_w, float eps, int n_head, int T, void* out16, cudaStream_t s)` | glm_batch.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu | GLM |
| `kda_rec_split` | `void kda_rec_split(float* q, float* k, float* v, float* g1_raw, const float* dt_bias, const float* ssm_a, float lower_bound, const float* beta_raw, float* state, const float* g2_raw, const float* norm_w, float eps, int n_head, int T, void* out16, cudaStream_t s, int kv_split)` | glm_batch.cu | glm_prefill.cu | GLM |
| `dsa_prep` | `void dsa_prep(const DsaPrepArgs& a, cudaStream_t s)` | glm_batch.cu | glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu, glm_kv_stream_parity.cpp | GLM |
| `dsa_pool` | `void dsa_pool(const float* ik_cache, const float* ig_cache, const float* ape, float* pooled, int idx_key, int kpool, int pool0, int n, cudaStream_t s, int ring)` | glm_batch.cu | glm_prefill.cu | GLM |
| `dsa_score` | `void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int p0, int kpool, int T, int max_vis, float* score, int score_ld, cudaStream_t s)` | glm_batch.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu | GLM |
| `dsa_select` | `void dsa_select(const float* score, int score_ld, int p0, int kpool, int top_pools_max, int tail, int T, int n_sel_max, int* cells, int* n_sel, cudaStream_t s)` | glm_batch.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu | GLM |
| `mla_decode_v70` | `bool mla_decode_v70(const float* q_abs, const uint16_t* lat, const int* cells, int n_sel, int n_head, int kv_lora, float scale, float* ctx, float* part, int part_cap, cudaStream_t s, bool lat_q8)` | glm_batch.cu | glm_fast.cu | GLM |
| `mla_attn` | `void mla_attn(const float* q_abs, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max, int n_head, int kv_lora, float scale, int T, float* ctx, cudaStream_t s, bool lat_q8)` | glm_batch.cu | glm_prefill.cu | GLM |
| `head_rows` | `void head_rows(const float* R, const float* w, float eps, int T, int n_embd, float* h, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `mtp_in_rows` | `void mtp_in_rows(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int T, int n_embd, void* out16, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `rms_rows` | `void rms_rows(const float* in, const float* w, float eps, int T, int n_embd, float* x, void* x16, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu, mm3_host.cpp, mm3_tp.cu, prefill.cpp | SHARED |
| `route` | `void route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int T, int* ids, float* w, cudaStream_t s)` | glm_batch.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, ds4_kernels.cu, ds4_model.cu, mm3_kernels.cu, mm3_model.cu, mm3_tp.cu, prefill.cpp | SHARED |
| `expert_count` | `void expert_count(const int* ids, int n, int n_expert, int* counts, int* rank, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `copy_i32` | `void copy_i32(const int* src, int* dst, int n, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu, prefill.cpp | SHARED |
| `expert_scatter` | `void expert_scatter(const int* ids, const int* rank, const int* base, int n, int k, int* row_tok, int* pos, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `swiglu_rows` | `void swiglu_rows(const float* gu, float* h, int rows, int n_ff, float limit, cudaStream_t s, int ld_h)` | glm_batch.cu | glm_prefill.cu | GLM |
| `swiglu_rows_h16` | `void swiglu_rows_h16(const float* gu, void* h16, int rows, int n_ff, float limit, cudaStream_t s, int ld_h)` | glm_batch.cu | glm_prefill.cu | GLM |
| `swiglu_f16` | `void swiglu_f16(const float* gate, const float* up, void* h16, int64_t n, float limit, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu, ds4_iq.cu, ds4_model.cu | GLM |
| `moe_combine_add` | `void moe_combine_add(const float* out, int base, int lo, int hi, const int* pos, const float* w, int T, int k, int n_embd, float* ffn, cudaStream_t s)` | glm_batch.cu | glm_prefill.cu | GLM |
| `moe_combine` | `void moe_combine(const float* out, const int* pos, const float* w, const float* sh, int T, int k, int n_embd, float* ffn, cudaStream_t s)` | glm_batch.cu | layer.cpp, mtp.cpp, ds4_kernels.cu, ds4_model.cu, shared_expert_parity.cpp, mm3_kernels.cu, mm3_model.cu, mm3_tp.cu, prefill.cpp | SHARED |

### `include/strata/kernels/glm_dsa.hpp` → `stubs/glm_dsa.cpp` (4 entries, impl: glm_dsa.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `glm_dsa_pool` | `void glm_dsa_pool(const float* ik, const float* ig, const float* ape, int key_dim, int kpool, int n_pools, float* pooled, void* stream)` | glm_dsa.cu | glm_model.cu, glm_dsa_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_dsa_score` | `void glm_dsa_score(const float* iq, const float* pooled, const float* weights, int key_dim, int idx_heads, int n_tokens, int n_pools, float* score, void* stream)` | glm_dsa.cu | glm_model.cu, glm_dsa_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_dsa_select` | `void glm_dsa_select(const float* score, int n_pools, int kpool, int top_pools, int select_tail, int T, int n_sel, const int* pos, int* cells, void* stream)` | glm_dsa.cu | glm_model.cu, glm_dsa_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_mla_attention` | `void glm_mla_attention(const float* q_abs, const float* latents, const int* cells, const float* wv_b, int kv_lora, int n_head, int v_head, int qk_nope, int T, int n_sel, float* out, void* stream)` | glm_dsa.cu | glm_model.cu, glm_dsa_parity.cpp, glm_layer_parity.cpp | GLM |

### `include/strata/kernels/glm_fast.hpp` → `stubs/glm_fast.cpp` (55 entries, impl: glm_fast.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `mv` | `bool mv(const MvJob* jobs, int n, cudaStream_t s)` | glm_fast.cu | glm_dflash.cu, glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu, glm_multi.cu, glm_rows.cu, glm_window.cu, ds4_gguf.cu, ds4_iq.cu +6 more | GLM |
| `mv_nt` | `bool mv_nt(const MvJob* jobs, int n, int nt, cudaStream_t s)` | glm_fast.cu | glm_mtp_rows.cu, glm_rows.cu | GLM |
| `mv_supported` | `bool mv_supported(int type)` | glm_fast.cu | glm_dflash.cu, glm_fast_path.cu, ds4_gguf.cu, ds4_iq.cu, ds4_main.cpp, mm3_gguf.cu | GLM |
| `set_skip` | `void set_skip(const int* flag)` | glm_fast.cu | glm_fast_path.cu, glm_window.cu, mm3_exl3.cu | GLM |
| `skip_gate` | `void skip_gate(const volatile int* host_from, int t, int* flag, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_window.cu | GLM |
| `moe_supported` | `bool moe_supported(int type)` | glm_fast.cu | glm_fast_path.cu, remote_experts.cpp | SHARED |
| `launch_errors` | `int launch_errors()` | glm_fast.cu | glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu, glm_multi.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu, ds4_gguf.cu, ds4_model.cu +5 more | GLM |
| `row_bytes` | `size_t row_bytes(int type, int64_t n_in)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, mtp.cpp, ds4_gguf.cu, ds4_model.cu, mm3_gguf.cu, mm3_model.cu | SHARED |
| `quantize_q8_1` | `void quantize_q8_1(const float* x, void* xq, int n, cudaStream_t s)` | glm_fast.cu | glm_dflash.cu, glm_fast_path.cu, remote_experts.cpp | SHARED |
| `hc` | `void hc(const HcArgs& a, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu, layer.cpp, glm_hc_parity.cpp | SHARED |
| `hc_rows` | `void hc_rows(const HcArgs* a, int n, cudaStream_t s)` | glm_fast.cu | glm_rows.cu, glm_window.cu | GLM |
| `hc_post` | `void hc_post(const float* block_out, const float* R_old, const float* post, const float* comb, int n_embd, float* R_new, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu, ds4_kernels.cu, ds4_model.cu | GLM |
| `head_prep` | `void head_prep(const float* R, const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_mtp_rows.cu, glm_multi.cu, glm_rows.cu, glm_window.cu | GLM |
| `embed_streams` | `void embed_streams(const float* emb, float* R, int n_embd, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_mtp_rows.cu, glm_multi.cu, glm_rows.cu, glm_window.cu | GLM |
| `kda_prep` | `void kda_prep(const KdaPrepArgs& a, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu | GLM |
| `kda_rec` | `void kda_rec(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw, float* state, const float* g2, const float* norm_w, float eps, int n_head, int head_dim, void* out_q8_1, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu | GLM |
| `kda_layers` | `void kda_layers(const KdaPrepArgs* prep, const KdaRecArgs* rec, int n, float eps, cudaStream_t s)` | glm_fast.cu | glm_mtp_rows.cu, glm_rows.cu, glm_window.cu | GLM |
| `kda_gather` | `void kda_gather(const KdaGatherArgs& a, cudaStream_t s)` | glm_fast.cu | glm_rows.cu | GLM |
| `dsa_prep` | `void dsa_prep(const DsaPrepArgs& a, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu, glm_kv_stream_parity.cpp | GLM |
| `dsa_score` | `void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int n_pools, float* score, cudaStream_t s, bool token)` | glm_fast.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu | GLM |
| `dsa_select` | `void dsa_select(const float* score, int n_vis, int kpool, int top_pools, int n_sel, int pos, int* cells, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, glm_window.cu | GLM |
| `dsa_score_rows` | `void dsa_score_rows(const DsaScoreRows& a, int n, int key_dim, int idx_heads, cudaStream_t s)` | glm_fast.cu | glm_rows.cu | GLM |
| `dsa_select_rows` | `void dsa_select_rows(const DsaSelectRows& a, int n, int kpool, cudaStream_t s)` | glm_fast.cu | glm_rows.cu | GLM |
| `dsa_select_all_on` | `bool dsa_select_all_on()` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu | GLM |
| `dsa_select_all` | `void dsa_select_all(int n_vis, int kpool, int n_sel, int pos, int* cells, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu | GLM |
| `mla` | `void mla(const float* q, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat, const int* cells, int n_sel, int n_head, int qk_nope, int kv_lora, int v_head, void* out_q8_1, cudaStream_t s, bool lat_q8, const MlaMarks* marks, bool token)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu, glm_kv_stream_parity.cpp | GLM |
| `mla_q_abs` | `void mla_q_abs(const float* q, int64_t q_stride, const uint16_t* wk_b, int n_head, int qk_nope, int kv_lora, int ncol, float* q_abs, int64_t qa_stride, cudaStream_t s)` | glm_fast.cu | glm_rows.cu, glm_window.cu | GLM |
| `mla_attend` | `bool mla_attend(const float* q_abs, const uint16_t* lat, const int* cells, int n_sel, int n_head, int qk_nope, int kv_lora, float* ctx, cudaStream_t s, bool lat_q8)` | glm_fast.cu | glm_window.cu | GLM |
| `mla_attend_batch` | `bool mla_attend_batch(const float* q_abs, int64_t q_stride, const uint16_t* lat, const int* cells, int64_t c_stride, const int* n_sel, int T, int n_head, int qk_nope, int kv_lora, float* ctx, int64_t ctx_stride, cudaStream_t s, bool lat_q8)` | glm_fast.cu | glm_window.cu | GLM |
| `mla_rows_part_floats` | `size_t mla_rows_part_floats(int n_head, int kv_lora)` | glm_fast.cu | glm_rows.cu | GLM |
| `mla_rows_ok` | `bool mla_rows_ok(int kv_lora)` | glm_fast.cu | glm_rows.cu | GLM |
| `mla_attend_rows` | `bool mla_attend_rows(const float* q_abs, int64_t q_stride, const uint16_t* const* lat, const int* const* cells, const int* n_sel, int T, int n_head, int qk_nope, int kv_lora, float* ctx, int64_t ctx_stride, float* part, int part_tokens, cudaStream_t s, bool lat_q8)` | glm_fast.cu | glm_rows.cu | GLM |
| `mla_out` | `void mla_out(const float* ctx, int64_t ctx_stride, const uint16_t* wv_b, int n_head, int kv_lora, int v_head, int ncol, void* out_q8_1, int64_t out_stride_bytes, cudaStream_t s)` | glm_fast.cu | glm_rows.cu, glm_window.cu | GLM |
| `swiglu_q8` | `void swiglu_q8(const float* gate, const float* up, float limit, int n, void* hq, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu, ds4_kernels.cu, ds4_model.cu | GLM |
| `moe_route` | `void moe_route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int layer, const float* x, int n_embd, const MoeDev& d, const float* sh_gate, const float* sh_up, float sh_limit, int n_ff_sh, void* sh_hq, cudaStream_t s, const float* pred_logits, const float* pred_bias, int max_prefetch, const float* ahead_logits, const float* const* ahead_bias, int n_ahead, int skip_from, unsigned long long cpu_plan, int promote_min, int pf_rank, int near_n, int win, unsigned int* scr_used, int win_n, int disk_skip_from, float disk_skip_w, float miss_skip_w, int win_rows, bool resident_ok)` | glm_fast.cu | glm_fast_path.cu, glm_window.cu, layer.cpp, verify.cpp, generate.cpp | SHARED |
| `moe_win_experts` | `bool moe_win_experts(int gu_type, int d_type, const MoeWin& w, cudaStream_t s, bool par_group)` | glm_fast.cu | glm_rows.cu, glm_window.cu | GLM |
| `moe_win_group` | `void moe_win_group(const MoeWin& w, cudaStream_t s, bool par)` | glm_fast.cu | glm_rows.cu, glm_window.cu, ds4_model.cu | GLM |
| `moe_cpu_wait` | `void moe_cpu_wait(const MoeDev& d, int n_embd, float* out, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_window.cu | GLM |
| `moe_cpu_wait_streams` | `void moe_cpu_wait_streams(const MoeDev& d, int n_embd, const float* post, float* R, int hc, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu | GLM |
| `moe_prefetch` | `void moe_prefetch(const MoeDev& d, size_t blob_bytes, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu | GLM |
| `moe_wait` | `void moe_wait(const MoeDev& d, int n_embd, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu | GLM |
| `moe_fetch` | `void moe_fetch(const MoeDev& d, int k, size_t blob_bytes, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu, glm_window.cu | GLM |
| `moe_wait_fetch` | `void moe_wait_fetch(const MoeDev& d, int n_embd, int k, size_t blob_bytes, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_rows.cu | GLM |
| `moe_cpu_wait_rows` | `void moe_cpu_wait_rows(const CpuWaitRows& w, int n, int n_embd, cudaStream_t s)` | glm_fast.cu | glm_rows.cu | GLM |
| `moe_gate_up` | `void moe_gate_up(int gu_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_window.cu, remote_experts.cpp, ds4_gguf.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu | SHARED |
| `moe_down` | `void moe_down(int d_type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq, const float* sh_out, float* out, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_window.cu, remote_experts.cpp, ds4_gguf.cu, ds4_model.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu | SHARED |
| `moe_experts_multi` | `bool moe_experts_multi(int gu_type, int d_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, size_t down_off, const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, float* out, cudaStream_t s)` | glm_fast.cu | glm_window.cu | GLM |
| `rows_experts` | `bool rows_experts(int gu_type, int d_type, const uint8_t* base, size_t stride, size_t down_off, const int* light, int n, const int* row_tok, const float* x, int n_tok, int n_embd, int n_ff, float limit, int rlo, int rhi, void* xq_scratch, void* hq_scratch, float* out, int ld, cudaStream_t s)` | glm_fast.cu | glm_prefill.cu | GLM |
| `tab_update` | `void tab_update(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_prefill.cu | GLM |
| `mtp_in` | `void mtp_in(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int n, void* cat_q8_1, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu | GLM |
| `rms_q8` | `void rms_q8(float* h, const float* add, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_mtp.cu, glm_mtp_rows.cu | GLM |
| `embed_tok` | `void embed_tok(const void* table, int type, int n_rows, const int* tok, int n_embd, float* out, cudaStream_t s)` | glm_fast.cu | glm_fast_path.cu, glm_mtp.cu | GLM |
| `argmax` | `void argmax(const float* x, int n, int* out, cudaStream_t s, const int* map, int map_from)` | glm_fast.cu | glm_fast_path.cu, glm_model_big_test.cpp, glm_model_test.cpp, glm_mtp_rows.cu, glm_multi.cu, glm_rows.cu, glm_window.cu, ds4_kernels.cu, mm3_kernels.cu +2 more | SHARED |
| `argmax_pair` | `void argmax_pair(const float* x, int n, int* out, cudaStream_t s)` | glm_fast.cu | glm_window.cu | GLM |
| `argmax_join` | `void argmax_join(const int* a, const int* b, int c0, int* out, cudaStream_t s)` | glm_fast.cu | glm_window.cu | GLM |

### `include/strata/kernels/glm_ffn.hpp` → `stubs/glm_ffn.cpp` (1 entries, impl: glm_ffn.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `glm_swiglu_clamp` | `void glm_swiglu_clamp(const float* gate, const float* up, float limit, int64_t n, float* out, void* stream)` | glm_ffn.cu | glm_model.cu, glm_ffn_parity.cpp | GLM |

### `include/strata/kernels/glm_hc.hpp` → `stubs/glm_hc.cpp` (4 entries, impl: glm_hc.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `glm_hc_workspace_bytes` | `size_t glm_hc_workspace_bytes(const GlmHcShapes& s)` | glm_hc.cu | — (no engine caller found; tests/tools only) | GLM |
| `glm_hc_workspace_init` | `void glm_hc_workspace_init(const GlmHcShapes& s, void* base, GlmHcWorkspace& ws)` | glm_hc.cu | glm_model.cu, glm_hc_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_hc_pre` | `void glm_hc_pre(const float* R, const float* w_fn, const float* w_scale, const float* w_base, float norm_eps, float hc_eps, const GlmHcShapes& s, GlmHcWorkspace& ws, float* mixed, float* pre, float* post, float* comb, void* stream)` | glm_hc.cu | glm_model.cu, glm_hc_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_hc_post` | `void glm_hc_post(const float* block_out, const float* residual, const float* post, const float* comb, const GlmHcShapes& s, float* out, void* stream)` | glm_hc.cu | glm_model.cu, glm_hc_parity.cpp, glm_layer_parity.cpp | GLM |

### `include/strata/kernels/glm_kda.hpp` → `stubs/glm_kda.cpp` (4 entries, impl: glm_kda.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `glm_kda_conv` | `void glm_kda_conv(const float* x_proj, const float* conv_w, float* conv_state, int d_inner, int d_conv, int T, float* out, void* stream)` | glm_kda.cu | glm_model.cu, glm_kda_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_kda_gate` | `void glm_kda_gate(const float* g, const float* ssm_a, float lower_bound, int head_dim, int n_head, int T, float* g1, void* stream)` | glm_kda.cu | glm_model.cu, glm_kda_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_kda_recurrence` | `void glm_kda_recurrence(const float* q, const float* k, const float* v, const float* g1, const float* beta, float* state, int n_head, int head_dim, int T, float* out, void* stream)` | glm_kda.cu | glm_model.cu, glm_kda_parity.cpp, glm_layer_parity.cpp | GLM |
| `glm_kda_out_gate` | `void glm_kda_out_gate(const float* o, const float* norm_w, const float* g2, int head_dim, int n_head, int T, float eps, float* gated, void* stream)` | glm_kda.cu | glm_model.cu, glm_kda_parity.cpp, glm_layer_parity.cpp | GLM |

### `include/strata/kernels/glm_kv_stream.hpp` → `stubs/glm_kv_stream.cpp` (2 entries, impl: glm_kv_stream.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `glm_kv_resolve` | `void glm_kv_resolve(const KvStreamMap& m, const uint8_t* host, uint8_t* slots, int row_bytes, int page, const int* cells, int n_sel, int* slot_cells, cudaStream_t s)` | glm_kv_stream.cu | glm_kv_stream_parity.cpp | GLM |
| `glm_kv_resolve_rows` | `void glm_kv_resolve_rows(const KvResolveRows& a, int n, int row_bytes, int page, cudaStream_t s)` | glm_kv_stream.cu | glm_rows.cu | GLM |

### `include/strata/kernels/gr.hpp` → `stubs/gr.cpp` (5 entries, impl: gr.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `gr_set_fp32_activations` | `void gr_set_fp32_activations(bool enabled)` | gr.cu | gr_parity.cpp, generate.cpp | QWEN |
| `gr_set_native_mmvf` | `void gr_set_native_mmvf(bool enabled)` | gr.cu | gr_parity.cpp, generate.cpp | QWEN |
| `gr_workspace_init` | `size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out)` | gr.cu | layer.cpp, gr_parity.cpp | QWEN |
| `gr_read` | `void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up, const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed, float* inject, void* stream)` | gr.cu | layer.cpp, mtp.cpp, gr_parity.cpp, generate.cpp | QWEN |
| `gr_write` | `void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out, void* stream)` | gr.cu | layer.cpp, mtp.cpp, verify.cpp, gr_parity.cpp, prefill.cpp, generate.cpp | QWEN |

### `include/strata/kernels/iq_kernels.hpp` → `stubs/iq_kernels.cpp` (13 entries, impl: iq_kernels.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `iq_supported` | `bool iq_supported(int ggml_type) noexcept` | iq_kernels.cu | native_head.cpp, iq_parity.cpp | QWEN |
| `iq_row_bytes` | `size_t iq_row_bytes(int ggml_type, int64_t n) noexcept` | iq_kernels.cu | native_head.cpp, dequant_bf16.cu, native_mmvq.cu, pf_wmma_parity.cpp | QWEN |
| `quantize_q8_1_rows` | `void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream)` | iq_kernels.cu | glm_model.cu, remote_experts.cpp, verify.cpp, iq1_s_parity.cpp, iq_parity.cpp, native_expert_parity.cpp | SHARED |
| `iq_mmvq` | `void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | iq_kernels.cu | native_mmvq.cu | QWEN |
| `iq_dequant_f16` | `void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream)` | iq_kernels.cu | dequant_bf16.cu, pf_wmma_parity.cpp, prefill.cpp | QWEN |
| `iq_dequant_f32` | `void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream)` | iq_kernels.cu | native_head.cpp, dequant_bf16.cu, iq1_s_parity.cpp, iq_parity.cpp | QWEN |
| `iq_embed_rows` | `void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd, float* out, void* stream)` | iq_kernels.cu | native_head.cpp | QWEN |
| `iq_dequant_gu_f16` | `void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream)` | iq_kernels.cu | prefill.cpp | QWEN |
| `gemm_iq_f16_grouped` | `bool gemm_iq_f16_grouped(int ggml_type, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx, const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream, const int32_t* ids, const GroupedWeightTable* indirect, bool down)` | iq_kernels.cu | glm_prefill.cu, pf_wmma_parity.cpp, prefill.cpp | SHARED |
| `gemm_iq_f16_grouped_supported` | `bool gemm_iq_f16_grouped_supported(int dev)` | iq_kernels.cu | glm_prefill.cu | QWEN |
| `native_expert_layout` | `NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff)` | iq_kernels.cu | glm_model.cu, remote_experts.cpp, verify.cpp, native_expert_parity.cpp | SHARED |
| `native_expert_scratch_bytes` | `size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff)` | iq_kernels.cu | glm_model.cu, remote_experts.cpp, verify.cpp, native_expert_parity.cpp | SHARED |
| `native_expert_grouped` | `void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream, float swiglu_limit)` | iq_kernels.cu | glm_model.cu, remote_experts.cpp, verify.cpp, native_expert_parity.cpp | SHARED |

### `include/strata/kernels/kv_q4.hpp` → `stubs/kv_q4.cpp` (4 entries, impl: kv_q4.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `fwht256_cuda` | `void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream)` | kv_q4.cu | kv_q4_parity.cpp | QWEN |
| `kv_append_q4_step` | `void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host)` | kv_q4.cu | layer.cpp, mtp.cpp, verify.cpp, kv_hybrid_parity.cpp, kv_q4_parity.cpp, kv_stream_parity.cpp | QWEN |
| `kv_append_q4` | `void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage)` | kv_q4.cu | prefill.cpp | QWEN |
| `kv_gather_q4_step` | `void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream)` | kv_q4.cu | layer.cpp, kv_hybrid_parity.cpp, kv_q4_parity.cpp | QWEN |

### `include/strata/kernels/kv_q8.hpp` → `stubs/kv_q8.cpp` (2 entries, impl: kv_q8.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `kv_append_q8_step` | `void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host)` | kv_q8.cu | layer.cpp, mtp.cpp, verify.cpp, kv_hybrid_parity.cpp, kv_q8_parity.cpp, kv_stream_parity.cpp | QWEN |
| `kv_gather_q8_step` | `void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale, const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream)` | kv_q8.cu | layer.cpp, kv_hybrid_parity.cpp, kv_q8_parity.cpp | QWEN |

### `include/strata/kernels/kv_stream.hpp` → `stubs/kv_stream.cpp` (7 entries, impl: kv_stream.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `kv_stream_reset` | `void kv_stream_reset(const KvStreamMap& m, void* stream)` | kv_stream.cu | glm_model.cu, layer.cpp, glm_kv_stream_parity.cpp, kv_stream_parity.cpp | SHARED |
| `kv_stream_resolve` | `void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt, const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s, void* stream)` | kv_stream.cu | layer.cpp, kv_stream_parity.cpp | QWEN |
| `kv_ring_table` | `void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream)` | kv_stream.cu | layer.cpp, kv_stream_parity.cpp | QWEN |
| `kv_ring_restore` | `void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1, int64_t n_slots, const QsaShapes& s, void* stream)` | kv_stream.cu | mtp.cpp, kv_stream_parity.cpp | QWEN |
| `kv_stage_from_host` | `void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks, const QsaShapes& s, void* stream)` | kv_stream.cu | prefill.cpp | QWEN |
| `kv_stream_counters` | `KvStreamCounters kv_stream_counters(const KvStreamMap& m)` | kv_stream.cu | glm_model.cu, glm_kv_stream_parity.cpp, kv_stream_parity.cpp, generate.cpp | SHARED |
| `kv_block_bytes` | `uint64_t kv_block_bytes(const QsaShapes& s, int fmt)` | kv_stream.cu | layer.cpp | QWEN |

### `include/strata/kernels/mrope.hpp` → `stubs/mrope.cpp` (2 entries, impl: native_rope.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `mrope_table_set` | `void mrope_table_set(const int32_t* device_table)` | native_rope.cu | generate.cpp | QWEN |
| `mrope_table` | `const int32_t* mrope_table()` | native_rope.cu | native_qsa_indexer.cu, qsa.cu, rope.cu, kernels.cu | QWEN |

### `include/strata/kernels/native_flash_attn.hpp` → `stubs/native_flash_attn.cpp` (1 entries, impl: native_flash_attn.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_flash_attn_short_step` | `void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* step, int64_t capacity, int max_context, const QsaShapes& shapes, float* output, int32_t* status, const uint16_t* mask, void* stream)` | native_flash_attn.cu | layer.cpp | QWEN |

### `include/strata/kernels/native_gdn.hpp` → `stubs/native_gdn.cpp` (3 entries, impl: native_gdn.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_gdn_set_enabled` | `void native_gdn_set_enabled(bool enabled)` | native_gdn.cu | generate.cpp | QWEN |
| `native_gdn_enabled` | `bool native_gdn_enabled()` | native_gdn.cu | layer.cpp | QWEN |
| `native_gdn_step` | `void native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* output, const GdnShapes& shapes, void* stream)` | native_gdn.cu | layer.cpp | QWEN |

### `include/strata/kernels/native_gdn_preprocess.hpp` → `stubs/native_gdn_preprocess.cpp` (5 entries, impl: native_gdn_preprocess.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_gdn_conv_silu` | `void native_gdn_conv_silu(float* history, const float* input, const float* weights, float* raw_output, float* silu_output, int64_t channels, int64_t d_conv, void* stream)` | native_gdn_preprocess.cu | layer.cpp | QWEN |
| `native_gdn_l2_norm` | `void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream)` | native_gdn_preprocess.cu | layer.cpp | QWEN |
| `native_gdn_beta_gate` | `void native_gdn_beta_gate(float* beta, int64_t heads, void* stream)` | native_gdn_preprocess.cu | layer.cpp | QWEN |
| `native_gdn_gate` | `void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t heads, void* stream)` | native_gdn_preprocess.cu | layer.cpp | QWEN |
| `native_gdn_out_norm` | `void native_gdn_out_norm(const float* output, const float* z, const float* gamma, float* destination, int64_t heads, int64_t cols, float epsilon, void* stream)` | native_gdn_preprocess.cu | layer.cpp | QWEN |

### `include/strata/kernels/native_gr_norm.hpp` → `stubs/native_gr_norm.cpp` (1 entries, impl: native_gr_norm.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_gr_rms_norm_weighted` | `void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols, int n_rows, float epsilon, void* stream)` | native_gr_norm.cu | gr.cu, native_ple_postops.cu | QWEN |

### `include/strata/kernels/native_gr_postops.hpp` → `stubs/native_gr_postops.cpp` (3 entries, impl: native_gr_postops.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_gr_down_silu` | `void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream)` | native_gr_postops.cu | gr.cu | QWEN |
| `native_gr_pre_gated` | `void native_gr_pre_gated(const float* xn, float* gate, float* mixed, int n_embd, int hc, bool fused_layer, void* stream)` | native_gr_postops.cu | gr.cu | QWEN |
| `native_gr_post` | `void native_gr_post(const float* residual, const float* block_out, const float* inject, float* output, int n_embd, int hc, void* stream)` | native_gr_postops.cu | gr.cu | QWEN |

### `include/strata/kernels/native_mmvq.hpp` → `stubs/native_mmvq.cpp` (27 entries, impl: native_mmvq.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_q8_1_bytes` | `std::size_t native_q8_1_bytes(int n_in, int ncols)` | native_mmvq.cu | mtp.cpp, native_dense.cpp, native_head.cpp, verify.cpp, ple.cu, ple_parity.cpp | QWEN |
| `native_mmvq_set_multi_exact` | `void native_mmvq_set_multi_exact(bool exact)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_mmvq_multi_exact` | `bool native_mmvq_multi_exact()` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_quantize_q8_1` | `void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream)` | native_mmvq.cu | layer.cpp, mtp.cpp, native_head.cpp, verify.cpp, ple.cu, shared_expert.cu | QWEN |
| `native_q5_k_mmvq` | `void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q5_k_f32` | `void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | native_head.cpp | QWEN |
| `native_q2_0_mmvq` | `void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q2_0_f32` | `void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | ple_parity.cpp | QWEN |
| `native_q3_k_mmvq` | `void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q3_k_f32` | `void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_iq4_xs_mmvq` | `void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_iq4_xs_f32` | `void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q4_k_mmvq` | `void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q4_k_f32` | `void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q6_k_mmvq` | `void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q6_k_f32` | `void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q4_0_mmvq` | `void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q4_0_f32` | `void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q5_0_mmvq` | `void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q5_0_f32` | `void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q8_0_mmvq` | `void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_q8_0_f32` | `void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_iq4_nl_mmvq` | `void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_iq4_nl_f32` | `void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_mmvq_supported` | `bool native_mmvq_supported(int ggml_type) noexcept` | native_mmvq.cu | native_dense.cpp, native_head.cpp, shared_expert.cu | QWEN |
| `native_mmvq_weight_bytes` | `std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out)` | native_mmvq.cu | glm_model.cu, native_dense.cpp, native_head.cpp, ple.cu, shared_expert.cu, iq1_s_parity.cpp | SHARED |
| `native_mmvq` | `void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream)` | native_mmvq.cu | glm_model.cu, layer.cpp, mtp.cpp, native_head.cpp, verify.cpp, ple.cu, shared_expert.cu, iq1_s_parity.cpp, iq_parity.cpp | SHARED |

### `include/strata/kernels/native_moe.hpp` → `stubs/native_moe.cpp` (4 entries, impl: native_moe.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_moe_combine_set_enabled` | `void native_moe_combine_set_enabled(bool enabled)` | native_moe.cu | generate.cpp | QWEN |
| `native_moe_combine_enabled` | `bool native_moe_combine_enabled()` | native_moe.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |
| `native_moe_combine` | `void native_moe_combine(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k, void* stream)` | native_moe.cu | layer.cpp, mtp.cpp | QWEN |
| `native_moe_combine_multi` | `void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k, int n_tok, void* stream)` | native_moe.cu | verify.cpp | QWEN |

### `include/strata/kernels/native_ple_postops.hpp` → `stubs/native_ple_postops.cpp` (2 entries, impl: native_ple_postops.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_ple_postops` | `void native_ple_postops(const float* projected_key, const float* hidden, const float* value, const float* history, const PleWeights& weights, const NativePlePostopsBuffers& buffers, void* stream)` | native_ple_postops.cu | ple.cu | QWEN |
| `native_ple_postops_batch` | `void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w, float* query_norm, float* gated, float* gate, int T, void* stream)` | native_ple_postops.cu | prefill.cpp | QWEN |

### `include/strata/kernels/native_qsa.hpp` → `stubs/native_qsa.cpp` (4 entries, impl: native_qsa.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_qsa_set_enabled` | `void native_qsa_set_enabled(bool enabled)` | native_qsa.cu | generate.cpp | QWEN |
| `native_qsa_enabled` | `bool native_qsa_enabled()` | native_qsa.cu | layer.cpp, verify.cpp | QWEN |
| `native_qsa_rms_norm_weighted` | `void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols, int n_rows, float epsilon, void* stream)` | native_qsa.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |
| `native_qsa_gate_apply` | `void native_qsa_gate_apply(const float* attn, const float* q_full, float* output, int n_head, int head_dim, void* stream)` | native_qsa.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |

### `include/strata/kernels/native_qsa_indexer.hpp` → `stubs/native_qsa_indexer.cpp` (4 entries, impl: native_qsa_indexer.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_qsa_indexer_set_enabled` | `void native_qsa_indexer_set_enabled(bool enabled)` | native_qsa_indexer.cu | generate.cpp | QWEN |
| `native_qsa_indexer_enabled` | `bool native_qsa_indexer_enabled()` | native_qsa_indexer.cu | layer.cpp | QWEN |
| `native_qsa_indexer_append_batch` | `void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma, float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells, float freq_base, void* stream)` | native_qsa_indexer.cu | prefill.cpp | QWEN |
| `native_qsa_indexer_append` | `void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base, const float* gamma, float epsilon, const QsaIndexerBuffers& buffers, const QsaShapes& shapes, int64_t max_cells, float freq_base, void* stream)` | native_qsa_indexer.cu | layer.cpp, verify.cpp, prefill.cpp | QWEN |

### `include/strata/kernels/native_qsa_score.hpp` → `stubs/native_qsa_score.cpp` (3 entries, impl: native_qsa_score.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_qsa_score_set_enabled` | `void native_qsa_score_set_enabled(bool enabled)` | native_qsa_score.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_qsa_score_enabled` | `bool native_qsa_score_enabled()` | native_qsa_score.cu | — (no engine caller found; tests/tools only) | QWEN |
| `native_qsa_score` | `void native_qsa_score(const float* pooled, const float* query, const float* block_bias, const QsaShapes& shapes, const int32_t* step, int64_t max_blocks, int64_t max_cells, float* cell_scores, void* stream)` | native_qsa_score.cu | — (no engine caller found; tests/tools only) | QWEN |

### `include/strata/kernels/native_rope.hpp` → `stubs/native_rope.cpp` (3 entries, impl: native_rope.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_rope_set_enabled` | `void native_rope_set_enabled(bool enabled)` | native_rope.cu | generate.cpp | QWEN |
| `native_rope_enabled` | `bool native_rope_enabled()` | native_rope.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |
| `native_rope_apply` | `void native_rope_apply(const float* x, float* out, int rows, int head_dim, int n_rot, float freq_base, const int* positions, void* stream)` | native_rope.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |

### `include/strata/kernels/native_router.hpp` → `stubs/native_router.cpp` (4 entries, impl: native_router.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `native_router_set_enabled` | `void native_router_set_enabled(bool enabled)` | native_router.cu | generate.cpp | QWEN |
| `native_router_enabled` | `bool native_router_enabled()` | native_router.cu | layer.cpp, mtp.cpp, verify.cpp | QWEN |
| `native_router_top10` | `void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream)` | native_router.cu | layer.cpp, mtp.cpp | QWEN |
| `native_router_top10_multi` | `void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream)` | native_router.cu | verify.cpp | QWEN |

### `include/strata/kernels/ple.hpp` → `stubs/ple.cpp` (7 entries, impl: ple.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `ple_set_native_bf16` | `void ple_set_native_bf16(bool enabled)` | ple.cu | ple_parity.cpp, generate.cpp | QWEN |
| `ple_set_native_postops` | `void ple_set_native_postops(bool enabled)` | ple.cu | generate.cpp | QWEN |
| `ple_native_postops_enabled` | `bool ple_native_postops_enabled()` | ple.cu | prefill.cpp | QWEN |
| `ple_block_scratch_bytes` | `uint64_t ple_block_scratch_bytes()` | ple.cu | layer.cpp, verify.cpp, ple_parity.cpp | QWEN |
| `ple_block` | `void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w, PleOut& out, void* scratch, void* stream)` | ple.cu | layer.cpp, verify.cpp, ple_parity.cpp, prefill.cpp | QWEN |
| `ple_history_advance` | `void ple_history_advance(float* hist, const float* normalized, void* stream)` | ple.cu | layer.cpp, verify.cpp, ple_parity.cpp, prefill.cpp | QWEN |
| `ple_block_available` | `bool ple_block_available()` | ple.cu | ple_parity.cpp | QWEN |

### `include/strata/kernels/qsa.hpp` → `stubs/qsa.cpp` (14 entries, impl: qsa.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `qsa_step_fill` | `void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s)` | qsa.cu | layer.cpp, session.cpp, verify.cpp, kv_hybrid_parity.cpp, qsa_parity.cpp, prefill.cpp | QWEN |
| `kv_append` | `void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos, const float* kcur, const float* vcur, const QsaShapes& s, void* stream)` | qsa.cu | qsa_parity.cpp, prefill.cpp | QWEN |
| `kv_gather` | `void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids, int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream)` | qsa.cu | qsa_parity.cpp | QWEN |
| `indexer_key_append` | `void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm, float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab, const float* sin_tab, void* stream)` | qsa.cu | layer.cpp, qsa_parity.cpp | QWEN |
| `qsa_index` | `void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s, int64_t n_kv, float* cell_scores, void* stream)` | qsa.cu | qsa_parity.cpp | QWEN |
| `topk_512` | `void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids, void* stream)` | qsa.cu | qsa_parity.cpp | QWEN |
| `qsa_attend` | `void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids, const QsaShapes& s, float* attn, float* weights, void* stream)` | qsa.cu | qsa_parity.cpp | QWEN |
| `qsa_gate_apply` | `void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream)` | qsa.cu | qsa_parity.cpp | QWEN |
| `qsa_gate_apply_f32` | `void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream)` | qsa.cu | layer.cpp, verify.cpp | QWEN |
| `qsa_index_step` | `void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s, const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream)` | qsa.cu | layer.cpp, qsa_parity.cpp | QWEN |
| `topk_512_step` | `void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step, int32_t* ids, void* stream)` | qsa.cu | layer.cpp, qsa_parity.cpp | QWEN |
| `kv_gather_step` | `void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream)` | qsa.cu | layer.cpp, kv_q8_parity.cpp | QWEN |
| `qsa_attend_step` | `void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, const int32_t* step, int64_t max_ids, const QsaShapes& s, float* attn, float* weights, void* stream)` | qsa.cu | layer.cpp | QWEN |
| `kv_append_step` | `void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host)` | qsa.cu | layer.cpp, mtp.cpp, verify.cpp, kv_q8_parity.cpp, kv_stream_parity.cpp | QWEN |

### `include/strata/kernels/qsa_decode_attn.hpp` → `stubs/qsa_decode_attn.cpp` (3 entries, impl: qsa_decode_attn.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `qsa_decode_attn_scratch_floats` | `uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s)` | qsa_decode_attn.cu | layer.cpp, mtp.cpp, verify.cpp, kv_hybrid_parity.cpp, kv_stream_parity.cpp, qsa_prompt_attn_parity.cpp, prefill.cpp | QWEN |
| `qsa_decode_attn_step` | `void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step, int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream)` | qsa_decode_attn.cu | layer.cpp, kv_hybrid_parity.cpp | QWEN |
| `qsa_decode_attn_batch` | `void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream)` | qsa_decode_attn.cu | mtp.cpp, verify.cpp, kv_hybrid_parity.cpp, kv_stream_parity.cpp, qsa_prompt_attn_parity.cpp, prefill.cpp | QWEN |

### `include/strata/kernels/qsa_prompt_attn.hpp` → `stubs/qsa_prompt_attn.cpp` (1 entries, impl: qsa_prompt_attn.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `qsa_prompt_attn_batch` | `bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream)` | qsa_prompt_attn.cu | kv_hybrid_parity.cpp, qsa_prompt_attn_parity.cpp, prefill.cpp | QWEN |

### `include/strata/kernels/qsa_select.hpp` → `stubs/qsa_select.cpp` (4 entries, impl: qsa_select.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `qsa_block_scores` | `void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks)` | qsa_select.cu | layer.cpp, verify.cpp, qsa_select_bench.cpp, prefill.cpp | QWEN |
| `qsa_block_scores_tc` | `bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks)` | qsa_select.cu | qsa_select_bench.cpp, prefill.cpp | QWEN |
| `qsa_block_topk` | `void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap, const QsaShapes& s, int32_t* ids, void* stream)` | qsa_select.cu | layer.cpp, verify.cpp, qsa_select_bench.cpp, prefill.cpp | QWEN |
| `qsa_block_topk_ref` | `void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap, const QsaShapes& s, int32_t* ids, void* stream)` | qsa_select.cu | qsa_select_bench.cpp | QWEN |

### `include/strata/kernels/quantize_act.hpp` → `stubs/quantize_act.cpp` (5 entries, impl: quantize_act.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `quantize_q8_0` | `void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream)` | quantize_act.cu | layer.cpp, ple.cu, s2_expert_grouped.cu, shared_expert.cu, quantize_act_parity.cpp, s2_gemv_q8_parity.cpp, s_gemv_q8k_parity.cpp, shared_expert_parity.cpp | QWEN |
| `quantize_q8_0_scaled` | `void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream)` | quantize_act.cu | expert_source.cpp, mtp.cpp, remote_experts.cpp, session.cpp, verify.cpp, s2_expert_grouped.cu | QWEN |
| `dequant_q8_0` | `void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream)` | quantize_act.cu | quantize_act_parity.cpp | QWEN |
| `quantize_q8_K` | `void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream)` | quantize_act.cu | layer.cpp, shared_expert.cu, quantize_act_parity.cpp, s_gemv_q8k_parity.cpp | QWEN |
| `dequant_q8_K` | `void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream)` | quantize_act.cu | quantize_act_parity.cpp | QWEN |

### `include/strata/kernels/rope.hpp` → `stubs/rope.cpp` (2 entries, impl: rope.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `build_rope_table` | `void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab)` | rope.cu | layer.cpp, qsa_parity.cpp, rope_parity.cpp | QWEN |
| `rope_neox_apply` | `void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab, const float* sin_tab, const int* pos, void* stream)` | rope.cu | layer.cpp, mtp.cpp, verify.cpp, rope_parity.cpp | QWEN |

### `include/strata/kernels/router_sigmoid.hpp` → `stubs/router_sigmoid.cpp` (1 entries, impl: router_sigmoid.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `router_sigmoid` | `void router_sigmoid(const float* logits, const float* bias, int n_tokens, int n_expert, int k, float w_scale, bool norm_w, int* ids, float* weights, void* stream)` | router_sigmoid.cu | glm_model.cu, glm_layer_parity.cpp, router_sigmoid_parity.cpp | GLM |

### `include/strata/kernels/router_top10.hpp` → `stubs/router_top10.cpp` (1 entries, impl: router_top10.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `router_top10` | `void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights, void* stream)` | router_top10.cu | layer.cpp, mtp.cpp, router_top10_parity.cpp, kernels.cu | QWEN |

### `include/strata/kernels/s2_expert_grouped.hpp` → `stubs/s2_expert_grouped.cpp` (10 entries, impl: s2_expert_grouped.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `moe_hit_grouped_scratch_bytes` | `uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff)` | s2_expert_grouped.cu | mtp.cpp, remote_experts.cpp, verify.cpp, generate.cpp | QWEN |
| `moe_hit_grouped_s2` | `void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out, void* stream, const float* x_scales)` | s2_expert_grouped.cu | expert_source.cpp | QWEN |
| `moe_hit_select` | `void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst, int32_t* count, void* stream)` | s2_expert_grouped.cu | session.cpp | QWEN |
| `moe_hit_grouped_s2_dev` | `void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out, void* stream, const float* x_scales)` | s2_expert_grouped.cu | session.cpp | QWEN |
| `moe_hit_add` | `void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap, int64_t n_embd, void* stream)` | s2_expert_grouped.cu | session.cpp, verify.cpp | QWEN |
| `moe_grouped_s2` | `void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream)` | s2_expert_grouped.cu | mtp.cpp, remote_experts.cpp, verify.cpp | QWEN |
| `moe_group_resident` | `void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst, int32_t* ent_tok, void* stream)` | s2_expert_grouped.cu | mtp.cpp | QWEN |
| `moe_hit_select_multi` | `void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst, int32_t* count, void* stream)` | s2_expert_grouped.cu | — (no engine caller found; tests/tools only) | QWEN |
| `moe_hit_grouped_s2_multi` | `void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0, const float* x_scales, int k_per_token, void* scratch, float* out, void* stream)` | s2_expert_grouped.cu | — (no engine caller found; tests/tools only) | QWEN |
| `moe_hit_grouped_s2_cpu_order` | `void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out, void* stream, const float* x_scales, float* gate_up_trace)` | s2_expert_grouped.cu | expert_source.cpp | QWEN |

### `include/strata/kernels/s2_gemv.hpp` → `stubs/s2_gemv.cpp` (1 entries, impl: s2_gemv.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `s2_gemv` | `void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out)` | s2_gemv.cu | s2_gemv_parity.cpp | QWEN |

### `include/strata/kernels/s2_gemv_q8.hpp` → `stubs/s2_gemv_q8.cpp` (1 entries, impl: s2_gemv_q8.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `s2_gemv_q8` | `void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out, int threads_per_row, void* stream)` | s2_gemv_q8.cu | layer.cpp, ple.cu, shared_expert.cu, s2_gemv_q8_parity.cpp | QWEN |

### `include/strata/kernels/s_gemv.hpp` → `stubs/s_gemv.cpp` (8 entries, impl: s2_gemv_fast.cu, s2_gemv_quads.cu, s_gemv.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `s_gemv` | `void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | s_gemv_parity.cpp, s_gemv_q8k_parity.cpp | QWEN |
| `s_gemv_split` | `void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | s2_gemv_q8_parity.cpp, s_gemv_parity.cpp | QWEN |
| `s2_gemv_quads` | `void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out, int threads_per_row)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | s_gemv_parity.cpp | QWEN |
| `s2_gemv_fast` | `void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out, int threads_per_row, bool stage_x)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | s_gemv_parity.cpp | QWEN |
| `s_gemv_split_async` | `void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row, void* stream)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | concurrent_main.cpp, overlap_main.cpp | QWEN |
| `s_gemv_q8k` | `void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | s_gemv_q8k_parity.cpp | QWEN |
| `s_gemv_q8k_split` | `void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | layer.cpp, shared_expert.cu, s_gemv_q8k_parity.cpp | QWEN |
| `s_gemv_q8_0_split` | `void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream)` | s_gemv.cu, s2_gemv_fast.cu, s2_gemv_quads.cu | layer.cpp, shared_expert.cu, s_gemv_q8k_parity.cpp | QWEN |

### `include/strata/kernels/sampler.hpp` → `stubs/sampler.cpp` (3 entries, impl: sampler.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `sample_tokens` | `void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len, const SamplerParams& p, int* out, void* stream)` | sampler.cu | glm_model.cu, mtp.cpp, verify.cpp, ds4_model.cu, sampler_parity.cpp, generate.cpp | SHARED |
| `sample_draft` | `void sample_draft(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, SpecList* q, int* out, void* stream)` | sampler.cu | glm_fast_path.cu, ds4_model.cu | QWEN |
| `sample_verify` | `void sample_verify(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, const SpecList* q, const int* draft, int* out, void* stream, const int* history, int history_len)` | sampler.cu | glm_fast_path.cu, ds4_model.cu | QWEN |

### `include/strata/kernels/shared_expert.hpp` → `stubs/shared_expert.cpp` (5 entries, impl: shared_expert.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `shared_expert_set_native_bf16` | `void shared_expert_set_native_bf16(bool enabled)` | shared_expert.cu | shared_expert_parity.cpp, generate.cpp | QWEN |
| `shared_expert_scratch_bytes` | `uint64_t shared_expert_scratch_bytes(int64_t n_ff)` | shared_expert.cu | layer.cpp, mtp.cpp, shared_expert_parity.cpp | QWEN |
| `shared_expert` | `void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form, const uint8_t* gate_codes, const float* gate_scales, const float* gate_off, const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off, const SForm& down_form, const uint8_t* down_codes, const float* down_scales, const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out, int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32, const NativeSharedWeights* native)` | shared_expert.cu | layer.cpp, mtp.cpp, shared_expert_parity.cpp | QWEN |
| `shared_expert_multi` | `void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw, const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd, int64_t n_ff, void* stream)` | shared_expert.cu | verify.cpp | QWEN |
| `moe_combine` | `void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd, int64_t k, void* stream)` | shared_expert.cu | layer.cpp, mtp.cpp, ds4_kernels.cu, ds4_model.cu, shared_expert_parity.cpp, mm3_kernels.cu, mm3_model.cu, mm3_tp.cu, prefill.cpp | QWEN |

### `include/strata/kernels/verify_kernels.hpp` → `stubs/verify_kernels.cpp` (23 entries, impl: verify_kernels.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `gdn_conv_l2_multi` | `void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads, float eps, int n_tok, void* stream, int t_begin)` | verify_kernels.cu | verify.cpp | QWEN |
| `gdn_conv_commit` | `void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `gdn_ab_multi` | `void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `gdn_step_norm_multi` | `void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok, const int32_t* n_keep, void* stream, int t_out_begin)` | verify_kernels.cu | verify.cpp | QWEN |
| `wait_flag_ge` | `void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `gpu_stamp` | `void gpu_stamp(unsigned long long* buf, int i, void* stream)` | verify_kernels.cu | verify.cpp, fused_gr.cu | QWEN |
| `resident_plan` | `void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan, long long capx, uint32_t* skip, uint32_t ring, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `wait_flag_ge_or` | `void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `copy_i32_from_mapped_unless` | `void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `copy_or_zero_from_mapped` | `void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `embedding_gather_dev` | `void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens, int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes, uint64_t row_groups, float* out, void* stream)` | verify_kernels.cu | mtp.cpp, verify.cpp, prefill.cpp | QWEN |
| `broadcast_streams` | `void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `copy_indexed` | `void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `fetch_blobs` | `void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `rebase_ptrs` | `void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream)` | verify_kernels.cu | verify.cpp | QWEN |
| `add_streams_broadcast` | `void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream)` | verify_kernels.cu | mtp.cpp, prefill.cpp | QWEN |
| `ident_hits` | `void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream)` | verify_kernels.cu | — (no engine caller found; tests/tools only) | QWEN |
| `mtp_select` | `void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst, int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p)` | verify_kernels.cu | mtp.cpp | QWEN |
| `gather_rows` | `void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream)` | verify_kernels.cu | mtp.cpp, ds4_kernels.cu, ds4_model.cu | QWEN |
| `map_ids` | `void map_ids(int32_t* ids, const int32_t* table, int n, void* stream)` | verify_kernels.cu | mtp.cpp | QWEN |
| `row_top_prob` | `void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream)` | verify_kernels.cu | mtp.cpp | QWEN |
| `dense_steps` | `void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream)` | verify_kernels.cu | — (no engine caller found; tests/tools only) | QWEN |
| `window_ids` | `void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream)` | verify_kernels.cu | mtp.cpp | QWEN |

### `include/strata/prefill/gemm.hpp` → `stubs/prefill_gemm.cpp` (7 entries, impl: gemm.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `~Gemm` | `~Gemm()` | gemm.cu | — (no engine caller found; tests/tools only) | QWEN |
| `init` | `bool init(void* stream, int64_t scratch_elems, std::string& err)` | gemm.cu | session.cpp, verify.cpp, ds4_lt.cu, ds4_model.cu, ple_reader.cpp, prefill.cpp, generate.cpp | QWEN |
| `init_external` | `bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes, std::string& err)` | gemm.cu | glm_prefill.cu, prefill.cpp | SHARED |
| `bf16` | `void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta)` | gemm.cu | layer.cpp, mtp.cpp, ds4_vision.cu, gr.cu, ple.cu, glm_kv_stream_parity.cpp, mm3_dspark.cu, mm3_vision.cu, prefill.cpp | SHARED |
| `f16` | `void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta)` | gemm.cu | glm_prefill.cu, moe_mmq.cu, prefill.cpp | SHARED |
| `native` | `void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta)` | gemm.cu | expert_source.cpp, ds4_iq_cpu.cpp, shared_expert_parity.cpp, prefill.cpp, generate.cpp | QWEN |
| `rebind` | `void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes)` | gemm.cu | glm_prefill.cu, prefill.cpp | SHARED |

### `include/strata/prefill/kernels.hpp` → `stubs/prefill_kernels.cpp` (28 entries, impl: kernels.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `gr_norm` | `void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gr_norm_rs` | `void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gr_mix_r` | `void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream, uint16_t* mixed_h)` | kernels.cu | prefill.cpp | QWEN |
| `gr_write_norm_rs` | `void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gr_silu` | `void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gr_mix` | `void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream, uint16_t* mixed_h)` | kernels.cu | prefill.cpp | QWEN |
| `gr_write` | `void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream)` | kernels.cu | layer.cpp, mtp.cpp, verify.cpp, gr_parity.cpp, prefill.cpp, generate.cpp | QWEN |
| `gr_broadcast` | `void gr_broadcast(const float* e, float* R, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gdn_gates` | `void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gdn_conv` | `void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gdn_recurrence` | `void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `route` | `void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream)` | kernels.cu | glm_fast_path.cu, glm_prefill.cu, glm_rows.cu, ds4_kernels.cu, ds4_model.cu, mm3_kernels.cu, mm3_model.cu, mm3_tp.cu, prefill.cpp | SHARED |
| `blob_dequant` | `void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream)` | kernels.cu | — (no engine caller found; tests/tools only) | QWEN |
| `swiglu_interleaved` | `void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `swiglu_pair` | `void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `swiglu_split` | `void swiglu_split(const float* gu, uint16_t* h16, int64_t n, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `copy_i32` | `void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream)` | kernels.cu | glm_prefill.cu, prefill.cpp | SHARED |
| `gather_rows16` | `void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `moe_combine` | `void moe_combine(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo, int64_t T, void* stream)` | kernels.cu | layer.cpp, mtp.cpp, ds4_kernels.cu, ds4_model.cu, shared_expert_parity.cpp, mm3_kernels.cu, mm3_model.cu, mm3_tp.cu, prefill.cpp | QWEN |
| `rms_rows` | `void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream)` | kernels.cu | glm_prefill.cu, mm3_host.cpp, mm3_tp.cu, prefill.cpp | SHARED |
| `rope` | `void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream)` | kernels.cu | ds4_vision.cu, prefill.cpp | QWEN |
| `split_q` | `void split_q(const float* q_full, float* q, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `gate_attn` | `void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `kv_append` | `void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size, uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage)` | kernels.cu | qsa_parity.cpp, prefill.cpp | QWEN |
| `to_f16` | `void to_f16(const float* x, uint16_t* y, int64_t n, void* stream)` | kernels.cu | glm_prefill.cu, mm3_dspark.cu, mm3_kernels.cu, mm3_model.cu, mm3_tp.cu, prefill.cpp | SHARED |
| `to_bf16` | `void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream)` | kernels.cu | gr_parity.cpp, mm3_dspark.cu, mm3_kernels.cu, prefill.cpp | QWEN |
| `round_f16` | `void round_f16(const float* x, float* y, int64_t n, void* stream)` | kernels.cu | prefill.cpp | QWEN |
| `blob_dequant_f16` | `void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream)` | kernels.cu | prefill.cpp | QWEN |

### `include/strata/prefill/moe_fused_rdna4.hpp` → `stubs/prefill_moe_fused_rdna4.cpp` (8 entries, impl: moe_fused_rdna4.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `available` | `bool available()` | moe_fused_rdna4.cu | glm_fast_path.cu, glm_prefill.cu, remote_experts.cpp, ds4_lt.cu, mm3_gguf.cu | HIP/RDNA4 |
| `supported` | `bool supported(int gu_type, int down_type)` | moe_fused_rdna4.cu | glm_model.cu, glm_prefill.cu, ds4_gguf.cu, ds4_iq.cu, ds4_model.cu, exl3_legacy.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu +2 more | HIP/RDNA4 |
| `act_bytes` | `size_t act_bytes(int64_t rows, int64_t cols)` | moe_fused_rdna4.cu | glm_prefill.cu, ds4_iq.cu | HIP/RDNA4 |
| `quantize` | `void quantize(const float* x, int64_t rows, int64_t cols, void* xa, void* stream)` | moe_fused_rdna4.cu | glm_prefill.cu, ds4_iq.cu, prefill.cpp | HIP/RDNA4 |
| `experts` | `void experts(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds, const int* host_bounds, void* tiles, const void* xa, const int* src, void* ha, float* out, void* stream)` | moe_fused_rdna4.cu | expert_source.cpp, glm_mtp.cu, glm_rows.cu, glm_window.cu, overlap_main.cpp, ds4_gguf.cu, ds4_iq_cpu.cpp, ds4_model.cu, ds4_packinfo.cpp +8 more | HIP/RDNA4 |
| `experts_gate_up` | `void experts_gate_up(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds, const int* host_bounds, void* tiles, const void* xa, const int* src, void* ha, void* stream)` | moe_fused_rdna4.cu | ds4_iq.cu | HIP/RDNA4 |
| `experts_down` | `void experts_down(const uint8_t* wbase, size_t stride, const Geometry& geometry, int n_expert, const int* bounds, const int* host_bounds, void* tiles, const void* ha, float* out, void* stream)` | moe_fused_rdna4.cu | ds4_iq.cu | HIP/RDNA4 |
| `quantize_rows` | `void quantize_rows(const float* x, int64_t ldx, const int* idx, int64_t rows, int64_t cols, void* xa, void* stream)` | moe_fused_rdna4.cu | ds4_iq.cu | HIP/RDNA4 |

### `include/strata/prefill/moe_mmq.hpp` → `stubs/prefill_moe_mmq.cpp` (18 entries, impl: moe_mmq.cu)

| Entry | Signature (as declared, defaults in header) | Impl | Called from | Tag |
|---|---|---|---|---|
| `built` | `bool built()` | moe_mmq.cu | glm_prefill.cu, ds4_iq.cu, prefill.cpp | SHARED |
| `supported` | `bool supported(int ggml_type)` | moe_mmq.cu | glm_model.cu, glm_prefill.cu, ds4_gguf.cu, ds4_iq.cu, ds4_model.cu, exl3_legacy.cu, mm3_exl3.cu, mm3_gguf.cu, mm3_model.cu +2 more | SHARED |
| `matrix_bytes` | `size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols)` | moe_mmq.cu | prefill.cpp | QWEN |
| `q8_bytes` | `size_t q8_bytes(int64_t rows, int64_t cols)` | moe_mmq.cu | glm_prefill.cu, ds4_iq.cu, prefill.cpp | SHARED |
| `pool_slab` | `void pool_slab(int dev, void* base, size_t bytes)` | moe_mmq.cu | glm_prefill.cu | QWEN |
| `q8_block_bytes` | `size_t q8_block_bytes()` | moe_mmq.cu | glm_prefill.cu | QWEN |
| `quantize` | `void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows, void* stream)` | moe_mmq.cu | glm_prefill.cu, ds4_iq.cu, prefill.cpp | SHARED |
| `glu_supported` | `bool glu_supported(int ggml_type, int64_t n_ff)` | moe_mmq.cu | glm_prefill.cu | QWEN |
| `Context` | `Context()` | moe_mmq.cu | ds4_iq.cu, prefill.cpp | QWEN |
| `~Context` | `~Context()` | moe_mmq.cu | ds4_iq.cu, prefill.cpp | QWEN |
| `run` | `void run(const Product& p, void* stream)` | moe_mmq.cu | concurrent_main.cpp, expert_source.cpp, glm_cpu_bench.cpp, glm_fast_path.cu, glm_pack_test.cpp, glm_prefill.cu, native_head.cpp, verify.cpp, ds4_gguf.cu +22 more | SHARED |
| `run_glu` | `void run_glu(const Product& p, float limit, void* stream)` | moe_mmq.cu | glm_prefill.cu | QWEN |
| `gather_native` | `void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes, void* gu_dst, void* d_dst, void* stream)` | moe_mmq.cu | prefill.cpp | QWEN |
| `gather_strata_q2` | `void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream)` | moe_mmq.cu | prefill.cpp | QWEN |
| `swiglu` | `void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream)` | moe_mmq.cu | ds4_gguf.cu, ds4_iq.cu, ds4_kernels.cu, ds4_model.cu, glm_fast.cu, mm3_exl3.cu, prefill.cpp | SHARED |
| `iota` | `void iota(int32_t* dst, int64_t n, void* stream)` | moe_mmq.cu | glm_fast_path.cu, glm_prefill.cu, ds4_model.cu, router_top10_parity.cpp, mm3_dspark.cu, mm3_kernels.cu, mm3_tp.cu, prefill.cpp | SHARED |
| `to_f16_supported` | `bool to_f16_supported(int ggml_type)` | moe_mmq.cu | glm_prefill.cu | QWEN |
| `to_f16` | `bool to_f16(int ggml_type, const void* w, uint16_t* dst, int64_t n, void* stream)` | moe_mmq.cu | glm_prefill.cu, mm3_dspark.cu, mm3_kernels.cu, mm3_model.cu, mm3_tp.cu, prefill.cpp | SHARED |

## A2. Host-only declarations in the surveyed headers (no stub — plain C++ already)

These are declared in kernel/prefill headers but implemented in host `.cpp` files; a Metal build
compiles those `.cpp` files as-is (subject to §B). Listed so the surface is complete and nobody
goes looking for their `.cu`:

- `include/strata/kernels/ngram.hpp` (24 decls, impl `src/kernels/ngram.cpp`): PleTable, bytes_read, close, collect, gather, gather_batch, io_report, iq4nl_code, iq4nl_dequant_row, is_open, issue, mode, ngram_mixed, ngram_rows, open, ple_artifact_consts, ple_prefetch_enable, ple_prefetch_enabled, read_row, rows, set_injected_delay_us, ~PleTable
- `include/strata/prefill/prefill.hpp` (13 decls, impl `src/prefill/prefill.cpp`): Prefill, bytes_needed, carve, chunk, draft_kv, drain_pipe, init, pinned_share, relayout, run, set_pinned_share, ~Prefill

## A3. HIP-only declarations (no stub — not compiled unless `STRATA_USE_HIP`)

- `include/strata/kernels/glm_batch.hpp` — `void mla_attn_f16q(const uint16_t* q16, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max, int n_head, int kv_lora, float scale, int T, float* ctx, cudaStream_t s, bool lat_q8)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_batch.hpp` — `void mla_attn_wmma2(const float* q_abs, const uint16_t* lat, const int* cells, const int* n_sel, int n_sel_max, int n_head, int kv_lora, float scale, int T, float* ctx, cudaStream_t s, bool lat_q8)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_batch.hpp` — `bool mla_wmma2_supported()` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_batch.hpp` — `bool mla_wmma2_default()` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_expert_bench.hpp` — `void gate_up(Variant variant, int type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t stream)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_expert_bench.hpp` — `void down(Variant variant, int type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq, const float* sh_out, float* out, cudaStream_t stream)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_expert_bench.hpp` — `void mv_config(int r, int u, int wpb, int lds)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_expert_bench.hpp` — `bool dsa_score_config(const char* label)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_expert_bench.hpp` — `bool mla_attn_config(const char* label)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_expert_bench.hpp` — `int lanes_kind(cudaStream_t s)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)
- `include/strata/kernels/glm_expert_bench.hpp` — `void warp_reduce_check(const float* in, float* sum_shfl, float* sum_lanes, float* max_shfl, float* max_lanes, int n_warps, cudaStream_t s)` (declared only under `#if defined(STRATA_USE_HIP)`; any implementation is under the same guard in the GLM .cu files)

## B. CUDA runtime / cuBLAS surface the host sources call (the `metal_compat` shim surface)

Scope: 53 host-side files — `src/core/*.cpp` + `src/core/*.cu` (device.cu, pinned.cu,
glm_*.cu), `src/program/generate.cpp`, `src/prefill/*.cpp` + `*.cu`, `src/spec/*.cpp`,
`src/ngram/*.cpp`, `src/kernels/ngram.cpp`, `src/platform/*.cpp`. Counts are call sites in those
files (comment lines excluded). One enum constant matched the pattern and is not a call:
`cudaDevAttrHostNumaId` (an attribute id passed to `cudaDeviceGetAttribute`).

| API | Calls | Example sites |
|---|---:|---|
| `cudaSetDevice` | 277 | src/core/device.cu:108, src/core/device.cu:148, src/core/glm_dflash.cu:704 |
| `cudaMemcpyAsync` | 225 | src/core/concurrent_main.cpp:130, src/core/concurrent_main.cpp:135, src/core/expert_cache.cpp:589 |
| `cudaGetLastError` | 176 | src/core/device.cu:167, src/core/expert_cache.cpp:206, src/core/expert_cache.cpp:229 |
| `cudaMalloc` | 173 | src/core/concurrent_main.cpp:97, src/core/concurrent_main.cpp:100, src/core/concurrent_main.cpp:101 |
| `cudaStreamSynchronize` | 145 | src/core/concurrent_main.cpp:144, src/core/concurrent_main.cpp:145, src/core/concurrent_main.cpp:153 |
| `cudaFree` | 145 | src/core/concurrent_main.cpp:190, src/core/concurrent_main.cpp:191, src/core/concurrent_main.cpp:192 |
| `cudaMemcpy` | 131 | src/core/concurrent_main.cpp:105, src/core/concurrent_main.cpp:106, src/core/expert_cache.cpp:618 |
| `cudaEventRecord` | 129 | src/core/concurrent_main.cpp:131, src/core/concurrent_main.cpp:138, src/core/expert_source.cpp:1174 |
| `cudaGetErrorString` | 121 | src/core/concurrent_main.cpp:35, src/core/device.cu:15, src/core/expert_cache.cpp:387 |
| `cudaFreeHost` | 110 | src/core/expert_cache.cpp:390, src/core/expert_cache.cpp:506, src/core/expert_source.cpp:448 |
| `cudaHostAlloc` | 81 | src/core/expert_cache.cpp:385, src/core/expert_source.cpp:615, src/core/glm_dflash.cu:1013 |
| `cudaEventDestroy` | 78 | src/core/concurrent_main.cpp:146, src/core/glm_dflash.cu:714, src/core/glm_dflash.cu:716 |
| `cudaStreamWaitEvent` | 49 | src/core/concurrent_main.cpp:140, src/core/glm_dflash.cu:1142, src/core/glm_dflash.cu:1289 |
| `cudaEventCreateWithFlags` | 43 | src/core/concurrent_main.cpp:128, src/core/glm_dflash.cu:1131, src/core/glm_dflash.cu:1133 |
| `cudaMemGetInfo` | 35 | src/core/device.cu:118, src/core/expert_cache.cpp:418, src/core/glm_fast_path.cu:1008 |
| `cudaMemsetAsync` | 34 | src/core/expert_source.cpp:1143, src/core/glm_fast_path.cu:3474, src/core/glm_fast_path.cu:3541 |
| `cudaEventElapsedTime` | 31 | src/core/glm_dflash.cu:1437, src/core/glm_fast_path.cu:3374, src/core/glm_fast_path.cu:3387 |
| `cudaEventCreate` | 28 | src/core/glm_dflash.cu:1361, src/core/glm_dflash.cu:1362, src/core/glm_fast_path.cu:3359 |
| `cudaDeviceSynchronize` | 27 | src/core/device.cu:169, src/core/expert_cache.cpp:309, src/core/expert_cache.cpp:325 |
| `cudaGetDevice` | 26 | src/core/expert_cache.cpp:268, src/core/glm_dflash.cu:703, src/core/glm_dflash.cu:730 |
| `cudaMemset` | 25 | src/core/expert_cache.cpp:448, src/core/glm_dflash.cu:1021, src/core/glm_fast_path.cu:891 |
| `cudaHostGetDevicePointer` | 25 | src/core/expert_source.cpp:627, src/core/expert_source.cpp:1337, src/core/expert_source.cpp:1925 |
| `cudaStreamDestroy` | 24 | src/core/concurrent_main.cpp:194, src/core/concurrent_main.cpp:195, src/core/glm_dflash.cu:717 |
| `cudaEventSynchronize` | 21 | src/core/glm_dflash.cu:1225, src/core/glm_fast_path.cu:3372, src/core/glm_fast_path.cu:3385 |
| `cudaStreamCreateWithFlags` | 19 | src/core/glm_fast_path.cu:713, src/core/glm_fast_path.cu:714, src/core/glm_fast_path.cu:715 |
| `cudaGraphLaunch` | 19 | src/core/graph.cpp:77, src/core/mtp.cpp:684, src/core/mtp.cpp:711 |
| `cudaDeviceGetAttribute` | 17 | src/core/glm_fast_path.cu:354, src/core/glm_fast_path.cu:683, src/core/glm_fast_path.cu:1958 |
| `cudaEventQuery` | 13 | src/core/expert_source.cpp:1177, src/core/expert_source.cpp:1188, src/core/glm_fast_path.cu:5135 |
| `cudaGraphExecDestroy` | 13 | src/core/graph.cpp:41, src/core/mtp.cpp:103, src/core/mtp.cpp:104 |
| `cudaGraphDestroy` | 13 | src/core/graph.cpp:42, src/core/graph.cpp:118, src/core/mtp.cpp:547 |
| `cublasGemmEx` | 10 | src/core/glm_prefill.cu:471, src/core/glm_prefill.cu:485, src/core/glm_prefill.cu:499 |
| `cudaStreamBeginCapture` | 9 | src/core/graph.cpp:49, src/core/mtp.cpp:566, src/core/mtp.cpp:576 |
| `cudaGetDeviceCount` | 8 | src/core/device.cu:93, src/core/expert_cache.cpp:216, src/core/glm_model.cu:1990 |
| `cudaGetDeviceProperties` | 8 | src/core/device.cu:111, src/core/glm_fast_path.cu:83, src/core/glm_fast_path.cu:3296 |
| `cudaStreamCreate` | 7 | src/core/concurrent_main.cpp:108, src/core/concurrent_main.cpp:109, src/core/overlap_main.cpp:101 |
| `cudaMemcpy2DAsync` | 7 | src/core/glm_dflash.cu:1409, src/core/glm_dflash.cu:1411, src/core/glm_dflash.cu:1681 |
| `cudaStreamEndCapture` | 7 | src/core/graph.cpp:55, src/core/graph.cpp:117, src/core/mtp.cpp:545 |
| `cudaStreamQuery` | 6 | src/core/glm_fast_path.cu:4388, src/core/glm_window.cu:1422, src/core/session.cpp:876 |
| `cudaGraphInstantiate` | 6 | src/core/graph.cpp:65, src/core/mtp.cpp:550, src/core/session.cpp:215 |
| `cublasSetWorkspace` | 5 | src/core/glm_prefill.cu:1344, src/core/glm_prefill.cu:1569, src/prefill/gemm.cu:342 |
| `cudaStreamCreateWithPriority` | 4 | src/core/glm_dflash.cu:741, src/core/glm_dflash.cu:1130, src/core/glm_model.cu:3324 |
| `cudaDeviceCanAccessPeer` | 3 | src/core/expert_cache.cpp:219, src/core/glm_prefill.cu:4334, src/core/glm_prefill.cu:4335 |
| `cudaDeviceGetStreamPriorityRange` | 3 | src/core/glm_dflash.cu:737, src/core/glm_dflash.cu:1129, src/core/glm_model.cu:3323 |
| `cudaDeviceGetPCIBusId` | 3 | src/core/glm_fast_path.cu:269, src/core/glm_fast_path.cu:3294, src/core/glm_fast_path.cu:4288 |
| `cudaHostRegister` | 3 | src/core/glm_fast_path.cu:398, src/core/pinned.cu:150, src/core/pinned.cu:161 |
| `cudaHostUnregister` | 3 | src/core/glm_fast_path.cu:494, src/core/pinned.cu:217, src/core/pinned.cu:219 |
| `cublasCreate` | 3 | src/core/glm_prefill.cu:1327, src/prefill/gemm.cu:336, src/prefill/gemm.cu:368 |
| `cublasSetStream` | 3 | src/core/glm_prefill.cu:1332, src/prefill/gemm.cu:340, src/prefill/gemm.cu:371 |
| `cublasSetMathMode` | 3 | src/core/glm_prefill.cu:1333, src/prefill/gemm.cu:343, src/prefill/gemm.cu:376 |
| `cudaGraphGetNodes` | 3 | src/core/graph.cpp:59, src/core/verify.cpp:858, src/core/verify.cpp:860 |
| `cublasSgemm` | 2 | src/core/glm_prefill.cu:1345, src/core/glm_prefill.cu:2045 |
| `cublasDestroy` | 2 | src/core/glm_prefill.cu:1946, src/prefill/gemm.cu:326 |
| `cublasSgemmStridedBatched` | 2 | src/core/glm_prefill.cu:2663, src/core/glm_prefill.cu:2699 |
| `cudaLaunchHostFunc` | 2 | src/core/glm_window.cu:1394, src/core/verify.cpp:1198 |
| `cudaPeekAtLastError` | 2 | src/core/layer.cpp:1185, src/core/native_head.cpp:87 |
| `cudaGraphUpload` | 2 | src/core/mtp.cpp:557, src/core/verify.cpp:895 |
| `cudaDriverGetVersion` | 1 | src/core/device.cu:122 |
| `cudaRuntimeGetVersion` | 1 | src/core/device.cu:123 |
| `cudaGetDriverEntryPointByVersion` | 1 | src/core/expert_cache.cpp:187 |
| `cudaGetDriverEntryPoint` | 1 | src/core/expert_cache.cpp:191 |
| `cudaFuncSetAttribute` | 1 | src/core/glm_dflash.cu:1261 |
| `cudaStreamIsCapturing` | 1 | src/core/glm_fast_path.cu:4387 |
| `cudaProfilerStart` | 1 | src/core/glm_fast_path.cu:6974 |
| `cudaProfilerStop` | 1 | src/core/glm_fast_path.cu:6975 |
| `cudaMemcpy2D` | 1 | src/core/glm_model.cu:3285 |
| `cudaMemcpyPeerAsync` | 1 | src/core/glm_prefill.cu:3612 |
| `cudaDeviceEnablePeerAccess` | 1 | src/core/glm_prefill.cu:4340 |
| `cudaMemcpyPeer` | 1 | src/core/glm_prefill.cu:4366 |
| `cudaInitDevice` | 1 | src/core/remote_experts.cpp:360 |
| `cudaGraphNodeGetType` | 1 | src/core/verify.cpp:864 |
| `cudaGraphKernelNodeGetParams` | 1 | src/core/verify.cpp:868 |
| `cudaFuncGetName` | 1 | src/core/verify.cpp:872 |
| `cudaMallocHost` | 1 | src/program/generate.cpp:976 |

Additional APIs used **inside the kernel `.cu` TUs themselves** (host side of those files; a Metal
build does not compile those TUs, listed for completeness): `cudaMemcpyToSymbol`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

Types / enums / constants the shim must also define (grep over host sources + headers):
`cudaStream_t` (dominant — 137 mentions in headers alone), `cudaEvent_t`, `cudaGraph_t`,
`cudaGraphExec_t`, `cudaGraphNode_t`, `cudaError_t`, `cudaDeviceProp`/`cudaDeviceProp_t`,
`cudaDeviceAttr_t`, `cudaFuncAttribute_t`, `cudaMemcpyKind_t` (+ `cudaMemcpyHostToDevice` /
`DeviceToHost` / `DeviceToDevice` / `Default` / `HostToHost`), `cudaStreamCaptureMode_t` /
`cudaStreamCaptureStatus_t`, `cudaHostAllocDefault/Mapped/Portable`, `cudaHostRegisterMapped/Portable`,
`cudaStreamNonBlocking`, `cudaEventDisableTiming`, `cudaStreamCaptureModeThreadLocal`,
`cudaEnableDefault`, `cudaErrorNotReady`, `cudaErrorMemoryAllocation`, `cudaErrorStreamCaptureUnsupported`,
`CUdeviceptr`, `CUresult`, `CUmemGenericAllocationHandle`, `CUmemAccessDesc`, `CUDA_SUCCESS`,
`CUDA_ERROR_OUT_OF_MEMORY` (the last six only for `src/core/expert_cache.cpp`'s VMM path — see RISKS.md §2),
and the cuBLAS side: `cublasHandle_t`, `cublasStatus_t`, `cublasOperation_t` (`CUBLAS_OP_N/T`),
`cudaDataType` (`CUDA_R_16F/16BF/32F`), `CUBLAS_COMPUTE_32F`, `CUBLAS_DEFAULT_MATH`,
`CUBLAS_PEDANTIC_MATH`, `CUBLAS_GEMM_DEFAULT`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

Hard cases inside §B (details in RISKS.md): the **CUDA-graph** family (`cudaStreamBeginCapture` /
`EndCapture`, `cudaGraphLaunch/Instantiate/Destroy`, `cudaGraphExecDestroy`, `cudaGraphGetNodes`,
`cudaGraphUpload`, `cudaStreamIsCapturing`, `cudaLaunchHostFunc`) has no Metal analogue; the
**VMM driver API** reached through `cudaGetDriverEntryPoint(ByVersion)` in `expert_cache.cpp`
(`cuMemCreate/Map/Unmap/SetAccess/AddressReserve/AddressFree/GetAllocationGranularity/Release`) has
no Metal analogue either; **peer** calls (`cudaDeviceCanAccessPeer`, `cudaMemcpyPeer*`,
`cudaDeviceEnablePeerAccess`) are multi-GPU only and can fail-closed on a one-GPU Mac.

## C. Headers carrying device code (risk list for the compat approach)

Counts are literal occurrences in the header text (comments included, and flagged where a count is
comment-only). Verdicts: **CLEAN** = declarations/plain structs only, compiles as C++ once
`cudaStream_t` exists; **SHIM-TYPES** = includes `<cuda_runtime.h>` but only needs its types;
**HD-GUARD** = host/device-annotated inline code already guarded by `#if defined(__CUDACC__) ||
defined(__HIPCC__)` (or `__CUDA_ARCH__`), so it compiles away *provided metal_compat does not
define those macros*; **DEVICE** = real device code a host C++ compiler cannot take.

| Header | __global__ | __device__ | <<< | thread/block idx | includes cuda_runtime.h | Verdict |
|---|---:|---:|---:|---:|---|---|
| `include/strata/core/glm_dflash.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/glm_model.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/graph.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/mtp.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/on_device.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/remote_experts.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/session.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/verify.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/core/vram_mark.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES |
| `include/strata/hip_compat/intrinsics.hpp` | 0 | 14 | 0 | 0 | no | DEVICE — HIP-only (included solely from hip_compat's cuda_runtime.h). Metal compat must NOT include it; it needs its own intrinsic story for the real kernels, not for host code |
| `include/strata/kernels/bf16_bits.hpp` | 0 | 1 | 0 | 0 | no | HD-GUARD — `STRATA_BF16_HD` inline converters; safe if `__CUDACC__`/`__HIPCC__` stay undefined |
| `include/strata/kernels/dp4a.hpp` | 0 | 2 | 0 | 0 | no | HD-GUARD — PTX-asm `__dp4a` only under `__CUDA_ARCH__ < 610`; empty on host compiles. Device-side only; no host entries |
| `include/strata/kernels/elementwise.hpp` | 1 | 0 | 1 | 0 | no | CLEAN — the `__global__`/`<<<` counts are a comment explaining why doorbell kernels live in the .cu |
| `include/strata/kernels/exl3.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES — includes cuda_runtime.h, uses only cudaStream_t |
| `include/strata/kernels/f16_bits.hpp` | 0 | 1 | 0 | 0 | no | HD-GUARD — same pattern (`STRATA_HD`) |
| `include/strata/kernels/glm_batch.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES — same; plus 4 HIP-only decls (§A3) |
| `include/strata/kernels/glm_fast.hpp` | 0 | 1 | 0 | 0 | yes | SHIM-TYPES + HD — includes cuda_runtime.h (cudaStream_t) and has one `__host__ __device__ constexpr` helper (`lat8_rec_bytes`): compat's cuda_runtime.h must define `__host__`/`__device__`/`__forceinline__` as empty macros |
| `include/strata/kernels/glm_kv_stream.hpp` | 0 | 0 | 0 | 0 | yes | SHIM-TYPES — same |
| `include/strata/kernels/gr.hpp` | 0 | 0 | 1 | 0 | no | CLEAN — the `<<<` count is a comment |
| `include/strata/kernels/mrope.hpp` | 0 | 1 | 0 | 0 | no | HD-GUARD — `mrope_pos` device inline is `#if __CUDACC__||__HIPCC__`-guarded; its 2 host entries are stubbed (native_rope.cu) |
| `include/strata/kernels/rope.hpp` | 0 | 2 | 0 | 0 | no | HD-GUARD — `STRATA_ROPE_HD` inline `rope_neox_pair` |
| `include/strata/hip_compat/cuda_runtime_api.h` | 0 | 0 | 0 | 0 | yes | HIP shim itself — the template for metal_compat, not an input to it |

All other headers under `include/strata/kernels/`, `include/strata/core/` and `include/strata/prefill/`
not listed above are CLEAN by this audit (declarations, plain structs, host inline helpers in plain C++).
Core headers that include `<cuda_runtime.h>` and therefore also need the shim's types:
`core/glm_dflash.hpp`, `core/glm_model.hpp`, `core/graph.hpp`,
`core/mtp.hpp`, `core/on_device.hpp` (calls `cudaGetDevice`/`cudaSetDevice` inline in a RAII guard),
`core/remote_experts.hpp`, `core/session.hpp`, `core/verify.hpp`, `core/vram_mark.hpp` (calls
`cudaMemGetInfo` etc. inline). `.cuh` files (`src/kernels/cuda/dsa_topk.cuh`, `glm_lat8.cuh`,
`src/ds4/*.cuh`, `src/mm3/*.cuh`) are device-only and are never included by host TUs — out of scope
for the compat compile, in scope when real kernels are ported.

