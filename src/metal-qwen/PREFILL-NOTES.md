# PREFILL NOTES — the Qwen prompt path on the Metal backend

Companion to `RISKS.md` §5/§7 and `STUB-MANIFEST.md`. This file records what
the prefill workstream found in the sources, what it implemented, the
convention decisions, and what is still stubbed (and why that is safe).

## 1. The cuBLAS surface inventory (from the sources, not guessed)

grep over every host translation unit for `cublas` (2026-11-02, this tree):

| Caller | In the Metal build? | Calls |
|---|---|---|
| `src/prefill/gemm.cu` (`prefill::Gemm`) | **yes** — now compiled as C++ (see §3) | `cublasCreate`, `cublasDestroy`, `cublasSetStream`, `cublasSetWorkspace`, `cublasSetMathMode`, `cublasGemmEx` |
| `src/prefill/prefill.cpp` | yes | **none** — it reaches BLAS only through `prefill::Gemm` |
| `src/core/glm_prefill.cu` | **no** — excluded with the GLM engine (RISKS.md §3) | `cublasCreate/Destroy/SetStream/SetWorkspace/SetMathMode`, `cublasGemmEx`, `cublasSgemm`, `cublasSgemmStridedBatched` |
| everything else (`src/core/*.cpp`, `src/program/`, `src/kernels/`) | — | none |

`Gemm`'s `GemmEx` call is always the same shape: `CUBLAS_OP_T x CUBLAS_OP_N`,
m = N, n = T, k = K, A = W (`CUDA_R_16BF` or `CUDA_R_16F`, lda = K), B = X
(same type, ldb = K), C `CUDA_R_32F` (ldc = ldy), alpha = 1, beta = the
caller's, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT`. The HIP-only
hipBLASLt fast path inside `gemm.cu` is preprocessor-gated on `__HIPCC__`
and compiles out.

The compat header `include/strata/metal_compat/cublas_v2.h` is shaped like
`include/strata/hip_compat/cublas_v2.h` (same names mapped onto the backend,
real cuBLAS enum values) and declares exactly: the handle lifecycle above
plus `cublasGetStream`, and the three GEMM entry points — `cublasGemmEx`,
`cublasSgemm`, `cublasSgemmStridedBatched` (the minimum RISKS.md §5 names;
the latter two are implemented for real even though their only in-tree
caller is currently the excluded GLM TU, and the GEMM test exercises them).

Implementation: `src/metal-qwen/runtime/metal_blas.cpp`, part of
`strata_metal_runtime`, host-side over Apple Accelerate (`cblas_sgemm`).

### Convention reasoning (where GEMM ports silently go wrong)

cuBLAS is **column-major**: `C(m×n) = alpha · op(A)(m×k) · op(B)(k×n) +
beta · C`, A/B/C column-major at lda/ldb/ldc, `op` = identity or transpose
per `CUBLAS_OP_N/T` (C is the conjugate transpose; for real types it is the
transpose).

What `Gemm` intends, in its own row-major terms (`gemm.hpp`): `Y[T, N] =
X[T, K] · W[N, K]^T`, X and W row-major, Y row-major at stride ldy. The
column-major call achieves it by reinterpretation, as `gemm.cu`'s own
comment states: the W pointer (row-major [N][K] bytes) read column-major is
a [K][N] matrix whose transpose (OP_T) is W; the X pointer read
column-major is [K][T] = X^T (OP_N); the product is the column-major [N][T]
matrix `W · X^T`, and those bytes read row-major are [T][N] = `X · W^T` =
Y. lda = ldb = K and ldc = ldy are the row lengths of the row-major
arrays, which are exactly the column-major leading dimensions of the
reinterpreted matrices. Nothing is copied or transposed by the call itself.

The implementation does **not** rely on the transpose trick. It
materializes `op(A)` and `op(B)` as FP32 row-major matrices element by
element from the column-major definitions above (for OP_T, element
(i, p) of op(A) is `A[p + i·lda]`; for OP_N it is `A[i + p·lda]`), reads C
into a row-major [m][n] buffer through ldc, calls one
`cblas_sgemm(CblasRowMajor, NoTrans, NoTrans, …)`, and scatters the result
back through ldc. Every transpose combination and every leading dimension
therefore goes through the single definition of the column-major layout —
there is no second convention to get wrong. The cost is the materializing
copies; see §5.

FP32 accumulate: FP16 and BF16 both widen to FP32 **exactly** (every value
of either format is an FP32 value), so converting inputs to FP32 and
accumulating in FP32 through `cblas_sgemm` is precisely the
`CUBLAS_COMPUTE_32F` contract the CUDA call makes (narrow inputs, FP32
accumulate, FP32 output). `CUBLAS_COMPUTE_32F_PEDANTIC` is accepted and is
the same computation (there is no reduced-precision path to disallow).
Other compute types and C types return `CUBLAS_STATUS_NOT_SUPPORTED`,
which `gemm.cu`'s `ck()` turns into a request failure — loud, not silent.

Stream semantics: Metal-runtime buffers are host-visible shared memory and
stream work is one command buffer per launch, in commit order on the
stream's queue; a host GEMM cannot be encoded into that queue. The
implementation therefore synchronizes the handle's stream
(`cudaStreamSynchronize`) before computing and computes synchronously on
the calling thread: all prior work on the stream has completed before the
first input byte is read, and this thread issues no later work until the
results are written. For the way the prompt path drives its stream
(single-threaded), that is stricter ordering than the asynchronous CUDA
call provides, never looser.

`cublasSetWorkspace` is accepted and recorded but unused: the CUDA caller
sets a workspace so the handle never allocates mid-call; the host
implementation allocates its materialization scratch with `malloc` per
call instead. That is a performance note (§5), not a semantic difference.

## 2. `prefill::Gemm` wiring

`cmake/metal_qwen_backend.cmake` sets `src/prefill/gemm.cu`'s language to
CXX (the same treatment `device.cu`/`pinned.cu` already get), and the
shared `CMakeLists.txt` Metal branch builds `strata_prefill` from
`prefill.cpp + gemm.cu`. The stage-1 stub TU
`src/metal-qwen/stubs/prefill_gemm.cpp` is deleted: its symbols are the
real ones now. `strata_metal_blas` remains as an interface alias of the
runtime target for the shared CMake. The GLM TU stays excluded, so this
changes nothing for GLM.

## 3. Prefill kernel entries (`include/strata/prefill/kernels.hpp`)

All 28 entries are implemented: host launchers in
`src/metal-qwen/stubs/prefill_kernels.cpp`, kernels in
`src/metal-qwen/kernels/prefill.metal`, transcribed from
`src/prefill/kernels.cu` (constants N 2560, HC 4, LR 320, S 128, HK 16,
HV 48, C 10240 are the kernel TU's own). Entry-by-entry:

| Entry | Form | Notes |
|---|---|---|
| `gr_norm`, `gr_norm_rs` | tg/row | CUDA `block_sum` reduction order preserved (`pf_block_sum`) |
| `gr_mix_r`, `gr_mix` | elementwise | optional bf16/f16 images gated by flags + dummy binding |
| `gr_write`, `gr_write_norm_rs` | elementwise / tg-row | fused F-2 form included |
| `gr_silu`, `gr_broadcast` | elementwise | |
| `gdn_gates` | elementwise | softplus via the prelude's log1p form, as the decode GDN port does |
| `gdn_conv` | tiled + hist + L2 | both CUDA arms: serial walk (T ≤ 64 or `STRATA_GDN_CONV_SERIAL`) and the tiled walk |
| `gdn_recurrence` | cols + out-norm | both CUDA arms: the split form and the serial `STRATA_GDN_REC_HEADS` form |
| `route` | simdgroup/token | 512/256 experts in MSL (the CUDA `route_kernel<16>/<8>`); other counts fall back to `kernels::router_top10`, exactly as `kernels.cu` does |
| `blob_dequant`, `blob_dequant_f16` | elementwise | one kernel, half/bf16 flag |
| `swiglu_interleaved/split/pair` | elementwise | |
| `copy_i32`, `gather_rows16` | elementwise | gather is per-element (the CUDA uint4 chunking is a bandwidth variant) |
| `moe_combine` | elementwise | |
| `rms_rows` | tg/row | `block_sum` order preserved |
| `rope` | tg/(token, head) | NEOX on the first 64 dims, as the CUDA kernel; mrope table honoured (see §4) |
| `split_q`, `gate_attn` | elementwise | |
| `kv_append` | tg/(token, kv head, group, is_v) | fp16 and int8 modes; device pool + host-mirror + staging destinations gated by flags, as the CUDA kernel branches |
| `to_f16`, `to_bf16`, `round_f16` | elementwise | |

Parity: fixtures in `src/metal-qwen/tests/fixtures/` (family `prefill`,
generator `gen_prefill()` in `gen_fixtures.cpp`, runner dispatch in
`metal_kernel_parity.cpp`) cover every entry above **except**
`blob_dequant`/`blob_dequant_f16`: a fixture for the blob geometry is
~11 MB (1.4 MB blob in, 9.8 MB of u16 matrices out) against the whole
existing fixture set's 2.8 MB, so those two are verified by transcription
and MSL compilation only — the one fixture gap, stated plainly. Their
arithmetic is one dequantize-by-4-elements expression per code byte,
shared with the `dq_elem` Q2_0 path the dense family already fixtures.

## 4. Adjacent entries the prompt path reaches

- **`strata::kernels::mrope_table[_set]`** (`stubs/mrope.cpp`): was a
  stage-1 abort stub; `prefill::rope`'s CUDA TU reads the table pointer on
  the host. Now the host-state pair, ported from
  `src/kernels/cuda/native_rope.cu` (one atomic pointer; the CUDA version
  keeps one per device and this backend has one device).
- **MMQ** (`stubs/prefill_moe_mmq.cpp`): the Metal build does not define
  `STRATA_PREFILL_MMQ` (the `strata_mmq` library is CUDA/HIP-only), so
  `prefill.cpp` itself compiles its no-MMQ fallback block — `built()` is
  honestly false, `mmq_plan()` disables the path, and the experts go
  through `Gemm` (the FP16 path). The stub TU now defines only the
  complement of the header that `prefill.cpp` does not (defining the same
  symbols twice would collide the day the TU is pulled in): `pool_slab`
  is a no-op (no MMQ scratch pool exists to lend), `glu_supported` /
  `to_f16_supported` answer false, and `run_glu` / `to_f16` /
  `q8_block_bytes` stay loud stubs. Their only in-tree callers are
  `moe_mmq.cu` (not compiled) and the excluded GLM/DS4 TUs.
- **RDNA4 fused experts** (`stubs/prefill_moe_fused_rdna4.cpp`): probes
  now answer honestly (`available`/`supported` false, `act_bytes` 0, per
  RISKS.md §4/§7); the work entries stay loud stubs. Unreachable in this
  build by caller evidence: the only callers are
  `src/core/glm_prefill.cu` and `src/ds4/ds4_iq.cu`, both excluded, and
  `prefill.cpp` never calls the namespace. The kernels are WMMA/RDNA-only
  upstream; there is nothing to port.

## 5. Remaining gaps / performance notes

- The cuBLAS implementation is correctness-first: FP32 materialization of
  both operands plus one `cblas_sgemm`. A future performance pass would
  encode an `MPSMatrixMultiplication` on the stream's command buffer
  instead (true async, no copies) — it must reproduce the §1 conventions
  and the FP32-accumulate contract, and re-run `metal_gemm_test`.
- `blob_dequant[_f16]` have no fixture (see §3).
- `gdn_recurrence`'s fixture tolerance (2e-4 abs / 2e-3 rel) is shared
  with its f16 output image, so its f32 state check is looser than the
  other families'; the state arithmetic itself is the same code path the
  f16 image verifies at format precision.
- Nothing else on the Qwen prompt path remains stubbed: with this
  workstream, `prefill.cpp`'s full call surface (Gemm, the 28 kernel
  entries, MMQ-off, mrope) resolves to real implementations.

## Mac verification (2026-11-02, branch qwen-native-prefill @ 5a0f970)

Staging `~/qwen-native-staging/ws-prefill/`, fresh build dir, `-DSTRATA_ENABLE_METAL=ON`:

- MSL pre-check (common.metalh + all kernels concatenated, `xcrun -sdk
  macosx metal -c`): clean. (First attempt caught three kernels mixing
  scalar and vector built-in position attributes — illegal in MSL —
  fixed by taking `uint3` position attributes and projecting.)
- Full build of `maya` + test targets: clean.
- `ctest -R metal`: **5/5 passed** — metal_dispatch_smoke,
  metal_graph_test, metal_session_test, metal_kernel_parity_list, and
  metal_gemm_test (31 checks: cublasSgemm in all four transpose
  combinations, cublasSgemmStridedBatched, GemmEx in the exact prefill
  call shape, prefill::Gemm bf16/f16 at {1,7}x2560x2560, expert
  gate/up 64x1280x2560, expert down 64x2560x640, qkv 3x10240x2560 and
  an odd 33x640x320 shape, beta 0 and 1, strided ldy, and Gemm::native
  Q8_0 — all against the double-precision reference).
- `metal_kernel_parity` over the full fixture set: **87/101 passed;
  all 35 prefill fixtures pass** (worst max_err: gdn_recurrence
  9.77e-04 within its stated bound, gate_attn 4.88e-04, the f16-image
  cases at format precision, everything else <= 5e-07 or exact). The
  14 failures are outside this workstream's surface and pre-date it on
  this branch base: 4 QSA entries whose kernels are unimplemented
  stubs on this base (implemented on the QSA workstream's branch) and
  10 numeric failures in dequant_f16 / native_*_f32 / native_ple
  entries (other workstreams' files; no file in this workstream
  touches them).
- Environment note: during verification, MTLCompilerService was
  intermittently unreachable from the SSH bootstrap domain (children
  aborting with "Unable to reach MTLCompilerService" on pipeline
  creation, including for pre-existing kernels). A standalone probe
  creating all 32 pf_* pipelines from the identical source succeeded,
  and affected entries pass on re-run once the service recovers; the
  signal deaths above are that instability plus the genuine QSA
  stubs, not kernel defects.

## Coordinator probe facts (2026-11-02, merged tree, real pack)

- The end-to-end probe that reached `Gemm::init_external` as its stop
  point used `--prefill auto` with 256-token chunks; this workstream's
  Gemm (real `gemm.cu` over the cuBLAS compat above) is the
  implementation that stop point was waiting for.
- The MTP rt dir builds cleanly with `tools/mtp_rt.py` from the
  shipped mtp GGUF.
- The pack runs WITHOUT `--mmap-experts` (it has no experts.bin;
  expert-cache streaming works).
