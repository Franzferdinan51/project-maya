# RISKS — compiling upstream Maya's Qwen engine as C++ against a Metal compat layer

Companion to `STUB-MANIFEST.md` (§C is the per-header audit; this file is the judgement:
what actually breaks, and the recommended handling for each — **compat define**,
**small patch**, or **exclusion + stub**). Surveyed 2026-11-02 on the `qwen-native` tree.
One in-flight fact to know first: the tree already carries the beginnings of this backend
(`include/strata/metal_compat/`, `cmake/metal_qwen_backend.cmake`, `STRATA_USE_METAL`
guards in `src/core/device.cu` / `expert_cache.cpp` / `remote_experts.cpp`). Where a risk
below is already mitigated in-tree, that is stated — the rest are open.

## 0. What is NOT a risk

- **The kernel headers are almost all clean.** Of ~60 headers under
  `include/strata/kernels/`, only 9 carry any device tokens at all, and every one of those
  is comment text or preprocessor-guarded (§C). The 412 stubbed entry points were compiled
  on this VM with `g++ -std=c++17` against a minimal fake `cuda_runtime.h`
  (`cudaStream_t = void*`, `__host__/__device__/__global__/__forceinline__` defined empty):
  **all 55 stub TUs compile, and every stubbed qualified name is present in the objects.**
- **Unified memory is the friendly case.** `cudaMalloc`, `cudaHostAlloc`
  (81 call sites), `cudaHostGetDevicePointer`, mapped-pinned doorbells
  (`doorbell_ring/wait/publish`, `copy_*_from_mapped`) all assume host and device share an
  address space or can cheaply fake it. On Apple Silicon a "device pointer" can simply be
  a host pointer into a shared `MTLBuffer`, and the doorbell spin protocol works as
  written — this is strictly easier than the HIP wave64 coherent-memory workaround in
  `hip_compat/cuda_runtime.h`.

## 1. CUDA graphs — the one structural blocker · SMALL PATCH (not a shim)

`cudaStreamBeginCapture/EndCapture`, `cudaGraphInstantiate/Launch/Destroy`,
`cudaGraphExecDestroy`, `cudaGraphGetNodes`, `cudaGraphUpload`, `cudaStreamIsCapturing`,
`cudaLaunchHostFunc` — 60+ call sites across `src/core/graph.cpp`, `mtp.cpp`,
`session.cpp`, `verify.cpp`, `src/program/generate.cpp`. Metal has no stream capture and
no graph object; a shim can only fake the types, and any fake that "succeeds" will replay
nothing — the worst kind of wrong (silent zero work).

**Recommended:** patch the capture layer, not the call sites. `CapturedGraph::record`
(`include/strata/core/graph.hpp`) already stores the layer body as a `std::function`, so
under `STRATA_USE_METAL` a `CapturedGraph` can keep the recorded bodies and simply
*invoke the body on every `launch()`* instead of capturing it — correct, slower, and
confined to `graph.cpp` plus the `MtpDrafter` capture wrappers in `mtp.cpp`. Shim-only
alternative (BeginCapture returns an error) is **not** safe: several callers treat capture
failure as fatal rather than falling back.

## 2. CUDA VMM driver API in `expert_cache.cpp` — already gated · COMPAT DEFINE

`src/core/expert_cache.cpp` dlopens the driver VMM API (`cuMemCreate/Map/Unmap/
SetAccess/AddressReserve/AddressFree/GetAllocationGranularity/Release`) through
`cudaGetDriverEntryPoint(ByVersion)`. There is no Metal analogue — but the whole path is
behind `#if CUDART_VERSION >= 12050` (expert_cache.cpp:186), with a plain-allocation path
below it. **Handling:** the compat `cuda_runtime.h` defines `CUDART_VERSION 11080`
(the in-tree `metal_compat` header already does exactly this) — no patch, no stub. The
same define also keeps `verify.cpp:871`'s `cudaFuncGetName` diagnostic (gated ≥ 12030)
out. Do not "helpfully" bump the reported version.

## 3. GLM engine TUs contain real `__global__` kernels · EXCLUSION + STUB (Qwen scope)

`src/core/glm_model.cu` (10 `__global__`), `glm_dflash.cu` (14), `glm_prefill.cu` (3),
`glm_fast_path.cu` (1) are host TUs with embedded device code — they will never compile
as C++. So do the DS4/MM3 trees (`src/ds4/*.cuh`, `src/mm3/*.cuh` are device-only).
**Handling for a Qwen-only Metal build:** exclude `glm_*.cu`, `src/ds4/`, `src/mm3/`
from the target (their kernel entry points are stubbed and GLM-tagged in the manifest
for the later GLM port). The catch: `src/program/generate.cpp` includes
`core/glm_model.hpp` + `kernels/glm_fast.hpp` and carries the `--glm-pack` program path,
so exclusion needs one small patch — compile the GLM program entry out under
`STRATA_USE_METAL` (or stub the `Glm5Model` class surface, which is much larger).
Same shape, smaller: `src/prefill/ggml_cuda_host.cu` `#include`s ggml-cuda's
`common.cuh` — exclude; Metal prefill goes through `prefill::Gemm` (§5).

## 4. Capability probes must not abort in stage 1.5 · STUB EXCEPTION

Stage 1 follows the brief literally: every stub aborts. But the engine *probes before it
runs* — `*_supported`, `*_enabled`, `built`, `launch_errors`, `*_bytes`, `mv_splits`,
`mv_cols`, `row_bytes`, `native_moe_combine_enabled`, … An aborting probe kills the
process during setup, before any real kernel is reached, and makes even `--help`-level
bring-up impossible. **Handling:** as the first stage-1.5 step, replace probe stubs with
honest `false` / `0` / empty bodies (they are enumerable by the name patterns in the
manifest's probe warning), and port the pure host-state flag pairs for real
(`native_moe_combine_set_enabled/enabled`, `native_qsa_score_set_enabled/enabled`,
`native_mmvq_set_multi_exact/multi_exact`, `cvec_set_enabled/enabled` — trivial globals
currently trapped inside `.cu` files). Everything else keeps aborting until implemented.

## 5. cuBLAS — small, but real · RUNTIME IMPLEMENTATION

Only two consumers matter: `prefill::Gemm` (`src/prefill/gemm.cu`: `cublasGemmEx`,
`cublasSgemm(StridedBatched)`, workspace/stream plumbing) for the Qwen prompt path, and
`glm_prefill.cu` (excluded with GLM, §3). **Handling:** declare the cuBLAS surface in
`metal_compat/cublas_v2.h` (as `hip_compat` does) and implement the minimum —
`GemmEx` for FP16/BF16→FP32 and `Sgemm` — over Metal Performance Shaders / Accelerate in
the runtime TU. Declaring without implementing is fine only while `gemm.cu` is also
excluded, which would remove the prompt path's GEMMs entirely.

## 6. x86 CPU kernel path · EXCLUSION (ARM)

`include/strata/kernels/cpu/*.hpp` and `src/kernels/cpu/*.cpp` are AVX2/AVX-512
intrinsics code (the CPU expert lane). No CUDA in them, but they cannot compile on Apple
Silicon. **Handling:** exclude from the Metal target; the tree already carries
`src/kernels/cpu/portable.cpp` as the non-intrinsics route — verify it covers the
`expert`/`native_expert` entry points the Qwen path links, and treat a scalar CPU expert
lane as a later performance item, not a bring-up blocker.

## 7. HIP/RDNA-only prefill experts · EXCLUSION + STUB (done)

`src/prefill/moe_fused_rdna4.cu` is WMMA/RDNA-specific and `src/prefill/moe_mmq.cu`
wraps llama.cpp's CUDA MMQ kernels (`ggml-cuda/mmq.cuh`). Both are stubbed
(`prefill_moe_fused_rdna4.cpp`, `prefill_moe_mmq.cpp`); `rdna4::available()` and
`mmq::built()/supported()` are probes under §4 and must answer `false`, after which the
prompt path falls back to the `Gemm` route by design.

## 8. Device intrinsics headers — never on the Metal include path · COMPAT HYGIENE

- `include/strata/hip_compat/intrinsics.hpp` (device `__dp4a`, `byte_perm`, wave
  shuffles/fences) is included only from the HIP shim. Metal compat must not include it.
- `include/strata/kernels/dp4a.hpp` contains PTX asm, but only under
  `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610` — empty in a host compile.
- `bf16_bits.hpp`, `f16_bits.hpp`, `rope.hpp`, `mrope.hpp` put `__host__ __device__`
  inline helpers behind `#if defined(__CUDACC__) || defined(__HIPCC__)`.
- **Rule that keeps all four safe:** metal_compat defines `__host__`, `__device__`,
  `__global__`, `__forceinline__`, `__shared__`, `__grid_constant__` as *empty macros*,
  and must **never** define `__CUDACC__`, `__HIPCC__` or `__CUDA_ARCH__`. (The in-tree
  `metal_compat/cuda_runtime.h` already follows this.) Real-kernel ports will need Metal
  equivalents of `__dp4a`/shuffles/`__ldg` — that is kernel-port work, not host compile.

## 9. Smaller shim-semantics risks (runtime phase, not compile phase)

- **Events are timing instruments here** (`cudaEventElapsedTime` feeds stage timing and
  the spec/draft cost models). A shim whose events always report 0 ms will silently
  corrupt adaptive draft-length decisions. Implement events as command-buffer
  completion timestamps (or a host clock at sync points) from the start.
- **`cudaLaunchHostFunc`** (`verify.cpp`, `glm_window.cu`): map to a command-buffer
  completion handler in the runtime; a shim that runs the callback immediately breaks
  ordering the verify window relies on.
- **Peer APIs** (`cudaDeviceCanAccessPeer`, `cudaMemcpyPeer*`,
  `cudaDeviceEnablePeerAccess`, `cudaInitDevice`): fail closed (0 / error). The
  layer-split and remote-expert paths probe them and disable themselves; returning
  success would be a lie with consequences.
- **`cudaProfilerStart/Stop`, `cudaFuncSetAttribute`, `cudaOccupancy*`**: profiler
  calls are GLM-TU-only or no-op-able; `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  appears only inside excluded kernel TUs. Safe to define as success/no-op in compat.
- **`src/core/pinned.cu`** is pure runtime calls (no kernels) — compile it as C++ under
  the shim (rename or force language in CMake). `src/core/device.cu` already has its
  `STRATA_USE_METAL` host-loop poison fill in-tree.
- **`exl3_legacy.cu` is `strata::kernels::exl3::legacy`**, not the `exl3.hpp` API —
  don't let a glob of the `.cu` files suggest the legacy file implements the header.

## 10. Not surveyed / not verified here

- Compile-testing the *host* sources themselves (`generate.cpp` is 9,255 lines) against
  the real `metal_compat` headers — that is the coordinator's build step; this survey
  compile-tested only the stub TUs, against a *fake* shim, without `STRATA_USE_HIP`.
- macOS/POSIX gaps in `src/platform/*.cpp` (upstream's non-Windows path is Linux-tested;
  e.g. `direct_file.cpp`, `memory.cpp`'s resident locking) — expect small patches, none
  device-related.
- DS4/MM3 header surfaces: excluded by scope (§3); their `.cuh` device headers were
  counted but their host entry points were not extracted.
