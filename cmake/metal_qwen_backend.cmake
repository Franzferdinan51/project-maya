# Opt-in Metal configuration for the Qwen engine: Maya's own strata core (src/core, src/program)
# compiled as C++ against the compat layer in include/strata/metal_compat, with the device
# kernels provided by src/metal-qwen (Stage 1: named stubs that stop loudly; the MSL kernels
# replace them family by family). Wired the way cmake/hip_backend.cmake wires HIP: this file
# only sets the backend variables and the runtime target; the targets themselves are the
# shared ones in CMakeLists.txt.
if(NOT APPLE)
  message(FATAL_ERROR "STRATA_ENABLE_METAL builds for macOS on Apple Silicon")
endif()
enable_language(OBJCXX)

set(STRATA_METAL_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/metal_compat")

# The MSL kernel library source, embedded at build time: common.metalh + every
# src/metal-qwen/kernels/*.metal concatenated (cmake/embed_metal_kernels.cmake).
set(_strata_metal_kernels_h "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_kernels_source.h")
file(GLOB _strata_metal_kernel_files CONFIGURE_DEPENDS src/metal-qwen/kernels/*.metal)
add_custom_command(OUTPUT "${_strata_metal_kernels_h}"
  COMMAND ${CMAKE_COMMAND} -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src/metal-qwen/kernels
          -DOUT=${_strata_metal_kernels_h} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal_kernels.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal_kernels.cmake
          ${CMAKE_CURRENT_SOURCE_DIR}/src/metal-qwen/kernels/common.metalh
          ${_strata_metal_kernel_files}
  COMMENT "Embedding the Metal kernel library source")

# The runtime: the CUDA-runtime-shaped surface over Metal buffers in shared memory
# (src/metal-qwen/runtime/metal_runtime.mm) plus kernel dispatch
# (src/metal-qwen/runtime/metal_dispatch.mm, strata::metal::launch). PUBLIC like the
# CUDA/HIP runtime targets, because host translation units (src/core/graph.cpp is the
# called-out case) use the runtime's types and entry points without naming the header
# themselves.
add_library(strata_metal_runtime STATIC src/metal-qwen/runtime/metal_runtime.mm
                                        src/metal-qwen/runtime/metal_dispatch.mm
                                        src/metal-qwen/runtime/metal_blas.cpp
                                        "${_strata_metal_kernels_h}")
target_include_directories(strata_metal_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
                                                        src/metal-qwen/runtime)
target_include_directories(strata_metal_runtime PUBLIC
  "${STRATA_METAL_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_metal_runtime PUBLIC STRATA_USE_METAL=1)
target_link_libraries(strata_metal_runtime PUBLIC "-framework Metal" "-framework Foundation"
                                                  "-framework Accelerate")
foreach(_language IN ITEMS CXX)
  target_compile_options(strata_metal_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:${_language}>:-include>"
    "$<$<COMPILE_LANGUAGE:${_language}>:${STRATA_METAL_COMPAT_INCLUDE_DIR}/cuda_runtime.h>")
endforeach()

# cuBLAS on Metal: include/strata/metal_compat/cublas_v2.h declares the surface
# (shaped like the HIP compat header) and src/metal-qwen/runtime/metal_blas.cpp
# implements it host-side over Accelerate, as part of strata_metal_runtime
# above. The prompt path's Gemm TU (src/prefill/gemm.cu) is compiled as C++
# against it, like the CUDA/HIP builds compile it against real cuBLAS /
# hipBLAS (src/core/glm_prefill.cu also calls cuBLAS but is excluded with the
# GLM engine). The blas target name stays an interface alias of the runtime
# for the shared CMakeLists.
add_library(strata_metal_blas INTERFACE)
target_link_libraries(strata_metal_blas INTERFACE strata_metal_runtime)

set(_strata_gpu_runtime_target strata_metal_runtime)
set(_strata_gpu_blas_target strata_metal_blas)
set(_strata_gpu_include_directories ${STRATA_METAL_COMPAT_INCLUDE_DIR})

# CMake does not infer C++ from the core's two .cu suffixes; the HIP file sets the mirror
# property for its language. device.cu / pinned.cu are the runtime glue (host code; the one
# __global__ in device.cu has a STRATA_USE_METAL host path).
set_source_files_properties(src/core/device.cu src/core/pinned.cu PROPERTIES LANGUAGE CXX)
# gemm.cu is host code too (the cuBLAS calls go to the compat implementation);
# its HIP-only hipBLASLt section is preprocessor-gated out without __HIPCC__.
set_source_files_properties(src/prefill/gemm.cu PROPERTIES LANGUAGE CXX)

# The dispatch smoke executable: launches kernels/smoke.metal's smoke_add through
# strata::metal::launch on known inputs and verifies the outputs (Mac-only by
# construction - this file is only included for the Metal backend).
add_executable(metal_dispatch_smoke src/metal-qwen/runtime/dispatch_smoke_main.cpp)
target_include_directories(metal_dispatch_smoke PRIVATE src/metal-qwen/runtime)
target_link_libraries(metal_dispatch_smoke PRIVATE strata_metal_runtime)

# The capture-layer test: GraphRegistry record/launch under the body-invoke design.
add_executable(metal_graph_test src/metal-qwen/runtime/graph_body_test.cpp)
target_link_libraries(metal_graph_test PRIVATE strata_core)

# The session capture test: session_capture / session_capture_token store their
# layer sequences as bodies (and mint the exec slots) without running them.
add_executable(metal_session_test src/metal-qwen/runtime/session_body_test.cpp)
target_link_libraries(metal_session_test PRIVATE strata_engine)

# The prefill GEMM test: the cuBLAS compat surface and prefill::Gemm against
# a double-precision host reference (src/metal-qwen/PREFILL-NOTES.md).
add_executable(metal_gemm_test src/metal-qwen/tests/metal_gemm_test.cpp)
target_link_libraries(metal_gemm_test PRIVATE strata_prefill strata_metal_runtime)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME metal_dispatch_smoke COMMAND metal_dispatch_smoke)
  add_test(NAME metal_graph_test COMMAND metal_graph_test)
  add_test(NAME metal_session_test COMMAND metal_session_test)
  add_test(NAME metal_gemm_test COMMAND metal_gemm_test)
endif()

message(STATUS "Strata: Metal enabled (Qwen engine, Apple Silicon)")
