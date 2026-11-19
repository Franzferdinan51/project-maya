# Maya's Metal build (Apple Silicon): maya-metal runs GLM-5.3-Flash through the pinned llama.cpp's ggml with its Metal
# backend.  It needs none of the CUDA/HIP engine and none of the x86 CPU kernels (AVX-512/AVX2, cpuid), so this build
# is its own short list of targets; the top-level CMakeLists.txt returns right after including it.
#
#   -DMAYA_ENABLE_METAL=ON            this build
#   -DMAYA_GGML_DIR=<llama.cpp>       a checkout at the pinned commit (default: fetched)
#   -DMAYA_METAL_TESTS=OFF            leave out the parity tests (the setup does)
#   -DMAYA_METAL_PORTABLE=ON          no -mcpu=native (a build for other Macs)
if(NOT APPLE)
  message(FATAL_ERROR "MAYA_ENABLE_METAL builds for macOS on Apple Silicon")
endif()
enable_language(C OBJC)

option(MAYA_METAL_TESTS "Build maya-metal's parity tests" ON)
option(MAYA_METAL_PORTABLE "Build maya-metal for any Apple Silicon Mac" OFF)
set(MAYA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)   # the shaders live inside the binary: nothing to install
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(MAYA_METAL_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()

if(MAYA_GGML_DIR)
  set(_maya_llamacpp_dir ${MAYA_GGML_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(maya_llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
    GIT_SHALLOW FALSE)
  FetchContent_GetProperties(maya_llamacpp)
  if(NOT maya_llamacpp_POPULATED)
    FetchContent_Populate(maya_llamacpp)
  endif()
  set(_maya_llamacpp_dir ${maya_llamacpp_SOURCE_DIR})
endif()
# ggml is built from a copy in the build tree with third_party/ggml/metal-sparse-mla.patch applied (ggml_sparse_mla:
# the DSA layers' sparse attention as one Metal kernel; the checkout stays untouched) and, on top of it,
# third_party/ggml/metal-tensor-fa.patch (the tensor-core flash-attention kernels: upstream llama.cpp PR #29570 +
# #30491 ported to the pinned ggml), and third_party/ggml/metal-allocator-free-blocks.patch
# (upstream v1.1.50: raises the ggml allocator free-block cap so long Metal prefill no longer aborts
# at ~92K input tokens; applied last, fails closed).  The copy is redone when the checkout, its ggml.h
# or any patch changes.
# A checkout the sparse patch does not apply to (or no git) builds as it is, and the engine keeps its gather +
# mul_mm attention; a tensor patch that does not apply on top of the sparse one leaves the sparse-patched copy
# building without the tensor kernels, with the same kind of warning.
set(_maya_gg ${_maya_llamacpp_dir}/ggml)
set(_maya_mla_patch ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/metal-sparse-mla.patch)
set(_maya_fa_patch ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/metal-tensor-fa.patch)
set(_maya_alloc_patch ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/metal-allocator-free-blocks.patch)
set(_maya_mla_copy ${CMAKE_BINARY_DIR}/maya-ggml)
file(SHA256 ${_maya_mla_patch} _maya_mla_patch_sha)
file(SHA256 ${_maya_fa_patch} _maya_fa_patch_sha)
file(SHA256 ${_maya_alloc_patch} _maya_alloc_patch_sha)
file(SHA256 ${_maya_gg}/include/ggml.h _maya_mla_src_sha)
set(_maya_mla_stamp "${_maya_gg}|${_maya_mla_src_sha}|${_maya_mla_patch_sha}|${_maya_fa_patch_sha}|${_maya_alloc_patch_sha}")
set(_maya_mla_done "")
if(EXISTS ${_maya_mla_copy}/.maya-stamp)
  file(READ ${_maya_mla_copy}/.maya-stamp _maya_mla_done)
endif()
set(_maya_mla ON)
if(NOT _maya_mla_done STREQUAL _maya_mla_stamp)
  find_package(Git)
  if(NOT GIT_FOUND)
    message(WARNING "git was not found to apply third_party/ggml/metal-sparse-mla.patch: ggml is built as it is, and the "
                    "DSA layers keep the gather + mul_mm attention")
    set(_maya_mla OFF)
  else()
    file(REMOVE_RECURSE ${_maya_mla_copy})
    file(COPY ${_maya_gg}/ DESTINATION ${_maya_mla_copy})
    # (GIT_CEILING_DIRECTORIES: git applies the patch as a plain patch tool, not inside the repository holding the
    # build tree)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                            ${GIT_EXECUTABLE} apply --ignore-whitespace ${_maya_mla_patch}
                    WORKING_DIRECTORY ${_maya_mla_copy} RESULT_VARIABLE _maya_mla_rc ERROR_VARIABLE _maya_mla_err)
    if(_maya_mla_rc EQUAL 0)
      execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                              ${GIT_EXECUTABLE} apply --ignore-whitespace ${_maya_fa_patch}
                      WORKING_DIRECTORY ${_maya_mla_copy} RESULT_VARIABLE _maya_fa_rc ERROR_VARIABLE _maya_fa_err)
      if(NOT _maya_fa_rc EQUAL 0)
        message(WARNING "third_party/ggml/metal-tensor-fa.patch (made for the pinned ggml) does not apply on top of "
                        "the sparse-mla patch in ${_maya_mla_copy} - ggml is built without the tensor flash-attention "
                        "kernels:\n${_maya_fa_err}")
      endif()
      # The pinned ggml allocator's 256 free ranges can fragment during long
      # Metal prefill and abort at ~92K input tokens. Patch ONLY the copied ggml
      # tree, never the vendored llama.cpp checkout. Fail closed on patch drift.
      execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                              ${GIT_EXECUTABLE} apply --ignore-whitespace ${_maya_alloc_patch}
                      WORKING_DIRECTORY ${_maya_mla_copy} RESULT_VARIABLE _maya_alloc_rc
                      ERROR_VARIABLE _maya_alloc_err)
      if(NOT _maya_alloc_rc EQUAL 0)
        message(FATAL_ERROR "Metal ggml allocator patch failed against the pinned ggml copy:\n${_maya_alloc_err}")
      endif()
      file(WRITE ${_maya_mla_copy}/.maya-stamp "${_maya_mla_stamp}")
    else()
      message(WARNING "third_party/ggml/metal-sparse-mla.patch (made for the pinned ggml) does not apply to ${_maya_gg} - "
                      "it is built as it is, and the DSA layers keep the gather + mul_mm attention:\n${_maya_mla_err}")
      file(REMOVE_RECURSE ${_maya_mla_copy})
      set(_maya_mla OFF)
    endif()
  endif()
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_maya_mla_patch} ${_maya_fa_patch} ${_maya_alloc_patch} ${_maya_gg}/include/ggml.h)
if(_maya_mla)
  add_subdirectory(${_maya_mla_copy} ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
else()
  add_subdirectory(${_maya_gg} ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endif()

# the engine reports Maya's release (VERSION) as its version
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _maya_version LIMIT_COUNT 1)
add_library(maya_metal STATIC src/metal/glm_metal.cpp src/metal/glm_metal_load.cpp src/metal/expert_pool.cpp
                              src/metal/glm_metal_dflash.cpp src/metal/spec.cpp src/metal/sampler.cpp
                              src/metal/mm3_metal.cpp src/metal/mm3_metal_load.cpp src/metal/mm3_serve.cpp
                              src/metal/mm3_dspark.cpp)
target_include_directories(maya_metal PUBLIC src/metal)
target_compile_definitions(maya_metal PUBLIC MAYA_VERSION="${_maya_version}")
target_link_libraries(maya_metal PUBLIC ggml ggml-base ggml-cpu ggml-metal)
if(_maya_mla)
  target_compile_definitions(maya_metal PRIVATE MAYA_METAL_SPARSE_MLA=1)
endif()

add_executable(maya-metal src/metal/main.cpp src/metal/serve.cpp)
target_link_libraries(maya-metal PRIVATE maya_metal)

add_executable(maya-metal-mm3 src/metal/mm3_main.cpp)
target_link_libraries(maya-metal-mm3 PRIVATE maya_metal)

if(MAYA_METAL_TESTS)
  enable_testing()
  if(_maya_mla)
    # Pure allocator regression: no model weights or Metal device required.
    add_executable(metal_allocator_free_blocks_test tools/test_metal_alloc_free_blocks.c)
    target_include_directories(metal_allocator_free_blocks_test PRIVATE
      ${_maya_mla_copy}/include ${_maya_mla_copy}/src)
    target_compile_options(metal_allocator_free_blocks_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(metal_allocator_free_blocks_test PRIVATE -Wl,-dead_strip)
    add_test(NAME metal_allocator_free_blocks COMMAND metal_allocator_free_blocks_test)
  endif()
  # the CUDA engine's GLM gates (src/core/glm_model_test.cpp, glm_model_big_test.cpp) on maya-metal
  add_executable(metal_model_test src/metal/metal_model_test.cpp)
  target_link_libraries(metal_model_test PRIVATE maya_metal)
  add_executable(metal_glm_backoff_test src/metal/metal_glm_backoff_test.cpp)
  target_link_libraries(metal_glm_backoff_test PRIVATE maya_metal)
  add_executable(metal_sampler_test src/metal/metal_sampler_test.cpp)
  target_link_libraries(metal_sampler_test PRIVATE maya_metal)
  add_test(NAME metal_sampler_test COMMAND metal_sampler_test)
  # the --serve protocol as serve/server.py drives it: PAUSE keeps the state, seeded sampling repeats, spec=1 changes
  # nothing (tools/test_metal_serve.py; Python without numpy)
  find_program(MAYA_SERVE_PYTHON3 python3)
  if(MAYA_SERVE_PYTHON3)
    add_test(NAME metal_serve_test
             COMMAND ${MAYA_SERVE_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_metal_serve.py $<TARGET_FILE:maya-metal>
                     ${CMAKE_CURRENT_SOURCE_DIR}/data/glm5-synth.gguf)
    add_test(NAME metal_serve_restore_test
             COMMAND ${MAYA_SERVE_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_metal_serve_restore.py $<TARGET_FILE:maya-metal>
                     ${CMAKE_CURRENT_SOURCE_DIR}/data/glm5-synth.gguf)
  endif()
  add_test(NAME metal_model_test COMMAND metal_model_test small)
  # MiniMax-M3.1 (src/metal/mm3_metal.cpp): the synthetic fixture is generated into the build tree
  # (tools/mm3_metal_fixture.py needs numpy, like the GLM fixture generators)
  add_executable(metal_mm3_test src/metal/metal_mm3_test.cpp)
  target_link_libraries(metal_mm3_test PRIVATE maya_metal)
  find_program(MAYA_MM3_PYTHON3 python3)
  if(MAYA_MM3_PYTHON3)
    add_test(NAME metal_mm3_gen
             COMMAND ${MAYA_MM3_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/mm3_metal_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf)
    set_tests_properties(metal_mm3_gen PROPERTIES FIXTURES_SETUP metal_mm3_gguf)
    add_test(NAME metal_mm3_test
             COMMAND metal_mm3_test ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf
                     ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf.expected.bin)
    add_test(NAME metal_mm3_test_cpu
             COMMAND metal_mm3_test ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf
                     ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf.expected.bin)
    set_tests_properties(metal_mm3_test_cpu PROPERTIES ENVIRONMENT MAYA_MM3_CPU=1)
    set_tests_properties(metal_mm3_test metal_mm3_test_cpu PROPERTIES FIXTURES_REQUIRED metal_mm3_gguf)
    add_test(NAME metal_mm3_serve_test
             COMMAND ${MAYA_MM3_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_mm3_serve.py $<TARGET_FILE:maya-metal-mm3>
                     ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf)
    set_tests_properties(metal_mm3_serve_test PROPERTIES FIXTURES_REQUIRED metal_mm3_gguf)
    # adaptive draft backoff (src/metal/spec_backoff.hpp): the policy and a serve-loop replica
    # on the synth fixture; the replica fabricates its drafts, so it needs no DSpark fixture
    # and registers here, outside the torch-conditional DSpark block below
    add_executable(metal_mm3_backoff_test src/metal/metal_mm3_backoff_test.cpp)
    target_link_libraries(metal_mm3_backoff_test PRIVATE maya_metal)
    add_test(NAME metal_mm3_backoff_test
             COMMAND metal_mm3_backoff_test ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf)
    set_tests_properties(metal_mm3_backoff_test PROPERTIES FIXTURES_REQUIRED metal_mm3_gguf)
    # DSpark (src/metal/mm3_dspark.cpp): the fixture generator is a torch reference
    # (tools/mm3/dspark_ref.py), so its tests register only when a python with torch is found
    add_executable(metal_mm3_dspark_test src/metal/metal_mm3_dspark_test.cpp)
    target_link_libraries(metal_mm3_dspark_test PRIVATE maya_metal)
    foreach(_cand "$ENV{HOME}/ComfyUI/.venv/bin/python" ${MAYA_MM3_PYTHON3})
      if(NOT MAYA_DSPARK_PYTHON3)
        execute_process(COMMAND ${_cand} -c "import torch, safetensors"
                        RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        if(_rc EQUAL 0)
          set(MAYA_DSPARK_PYTHON3 ${_cand})
        endif()
      endif()
    endforeach()
    if(MAYA_DSPARK_PYTHON3)
      add_test(NAME metal_mm3_dspark_gen
               COMMAND ${MAYA_DSPARK_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/mm3/dspark_ref.py fixture
                       ${CMAKE_CURRENT_BINARY_DIR}/dspark-fixture)
      set_tests_properties(metal_mm3_dspark_gen PROPERTIES FIXTURES_SETUP metal_mm3_dspark)
      add_test(NAME metal_mm3_dspark_test
               COMMAND metal_mm3_dspark_test ${CMAKE_CURRENT_BINARY_DIR}/dspark-fixture)
      set_tests_properties(metal_mm3_dspark_test PROPERTIES FIXTURES_REQUIRED metal_mm3_dspark)
    add_executable(metal_mm3_divergence_test src/metal/metal_mm3_divergence_test.cpp)
    target_link_libraries(metal_mm3_divergence_test PRIVATE maya_metal)
    add_test(NAME metal_mm3_divergence_test
             COMMAND metal_mm3_divergence_test ${CMAKE_CURRENT_BINARY_DIR}/minimax-m3-synth.gguf
                     ${CMAKE_CURRENT_BINARY_DIR}/dspark-fixture/dspark)
    set_tests_properties(metal_mm3_divergence_test PROPERTIES FIXTURES_REQUIRED "metal_mm3_gguf;metal_mm3_dspark")
    endif()
  endif()
  # the same gates on ggml-cpu (the oracle backend), and with the expert pool at its floor and small slices
  add_test(NAME metal_model_test_cpu COMMAND metal_model_test small)
  add_test(NAME metal_model_test_evict COMMAND metal_model_test small)
  set_tests_properties(metal_model_test_cpu PROPERTIES ENVIRONMENT MAYA_METAL_CPU=1)
  set_tests_properties(metal_model_test_evict PROPERTIES
                       ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1;MAYA_METAL_UBATCH=3;MAYA_METAL_ATTN_CHUNK=2")
  # verify windows (speculative decoding): partly wrong drafts must decode exactly as plain decoding does
  add_test(NAME metal_glm_backoff_test COMMAND metal_glm_backoff_test data/glm5-synth.gguf)
  add_test(NAME metal_model_window COMMAND metal_model_test window data/glm5-synth.gguf 40)
  # a 2048-token slice gathered 4 tokens a chunk: the graph outgrows any fixed budget (--ubatch 2048 aborted)
  add_test(NAME metal_model_big_slice COMMAND metal_model_test prefill data/glm5-synth.gguf 2048)
  set_tests_properties(metal_model_big_slice PROPERTIES ENVIRONMENT "MAYA_METAL_UBATCH=2048;MAYA_METAL_ATTN_CHUNK=4")
  set_tests_properties(metal_model_test metal_model_test_cpu metal_model_test_evict metal_model_window
                       metal_model_big_slice metal_glm_backoff_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # the 45-layer fixture: its 1.2 GB GGUF is generated into the build tree (tools/glm_big_fixture.py needs numpy)
  find_program(MAYA_PYTHON3 python3)
  if(MAYA_PYTHON3)
    add_test(NAME metal_model_big_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_big_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    set_tests_properties(metal_model_big_gen PROPERTIES FIXTURES_SETUP metal_big_gguf)
    add_test(NAME metal_model_big_test COMMAND metal_model_test big ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    add_test(NAME metal_model_big_test_evict
             COMMAND metal_model_test big ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    set_tests_properties(metal_model_big_test_evict PROPERTIES
                         ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1;MAYA_METAL_UBATCH=64")
    add_test(NAME metal_model_big_window COMMAND metal_model_test window ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf 48)
    add_test(NAME metal_model_big_window_evict
             COMMAND metal_model_test window ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf 48)
    set_tests_properties(metal_model_big_window_evict PROPERTIES ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1")
    # MTP drafting (the NextN block, tools/glm_mtp_fixture.py adds one to the fixtures) decodes as plain decoding does
    add_test(NAME metal_mtp_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-small.gguf)
    add_test(NAME metal_mtp_big_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py --big
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-big.gguf)
    set_tests_properties(metal_mtp_gen metal_mtp_big_gen PROPERTIES FIXTURES_SETUP metal_mtp_gguf)
    add_test(NAME metal_model_mtp COMMAND metal_model_test mtp ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-small.gguf 40)
    add_test(NAME metal_model_big_mtp COMMAND metal_model_test mtp ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-big.gguf 48)
    add_test(NAME metal_model_big_mtp_evict COMMAND metal_model_test mtp ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-big.gguf 48)
    set_tests_properties(metal_model_big_mtp_evict PROPERTIES ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1")
    # DFlash2 drafting (tools/glm_dflash_fixture.py: a drafter for the 45-layer fixture, its KV ring wrapping)
    add_test(NAME metal_dflash_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_dflash_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-dflash-drafter.gguf)
    set_tests_properties(metal_dflash_gen PROPERTIES FIXTURES_SETUP metal_mtp_gguf)
    add_test(NAME metal_model_big_dflash COMMAND metal_model_test dflash ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf
                                                 ${CMAKE_CURRENT_BINARY_DIR}/glm5-dflash-drafter.gguf 120)
    # quantized experts (Q8_0): a 512-token prompt slice takes ggml-metal's matrix-matrix expert kernel (at most 1024
    # experts a tensor), all resident and with a small pool
    add_test(NAME metal_q8_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py --big --no-nextn --q8-experts
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-q8.gguf)
    set_tests_properties(metal_q8_gen PROPERTIES FIXTURES_SETUP metal_mtp_gguf)
    add_test(NAME metal_model_big_prefill COMMAND metal_model_test prefill ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-q8.gguf 512)
    add_test(NAME metal_model_big_prefill_evict
             COMMAND metal_model_test prefill ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-q8.gguf 512)
    set_tests_properties(metal_model_big_prefill_evict PROPERTIES ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1")
    # 16 heads: the DSA products take ggml-metal's mul_mm (dsa_mm), the gathers padded and not, tails of 1-4 tokens
    add_test(NAME metal_wide_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py --wide --no-nextn
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-wide.gguf)
    add_test(NAME metal_wide512_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py --wide 512 --no-nextn
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-wide512.gguf)
    set_tests_properties(metal_wide_gen metal_wide512_gen PROPERTIES FIXTURES_SETUP metal_mtp_gguf)
    add_test(NAME metal_model_wide_prefill COMMAND metal_model_test prefill ${CMAKE_CURRENT_BINARY_DIR}/glm5-wide.gguf 300)
    # (the 512-wide latent: ggml_sparse_mla's D=512 kernel, the engine's own latent width)
    add_test(NAME metal_model_wide512_prefill
             COMMAND metal_model_test prefill ${CMAKE_CURRENT_BINARY_DIR}/glm5-wide512.gguf 300)
    # the same with the latent kept F32 (--kv-f32): ggml_sparse_mla's F32 kernel
    add_test(NAME metal_model_wide512_prefill_f32
             COMMAND metal_model_test prefill ${CMAKE_CURRENT_BINARY_DIR}/glm5-wide512.gguf 300)
    set_tests_properties(metal_model_wide512_prefill_f32 PROPERTIES ENVIRONMENT "MAYA_METAL_ATTN_CHUNK=5;MAYA_METAL_KV_F32=1"
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_mtp_gguf)
    set_tests_properties(metal_model_wide_prefill metal_model_wide512_prefill PROPERTIES ENVIRONMENT "MAYA_METAL_ATTN_CHUNK=5"
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_mtp_gguf)
    set_tests_properties(metal_model_big_prefill metal_model_big_prefill_evict PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_mtp_gguf TIMEOUT 3600)
    set_tests_properties(metal_model_mtp metal_model_big_mtp metal_model_big_mtp_evict PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_mtp_gguf TIMEOUT 3600)
    set_tests_properties(metal_model_big_dflash PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                         FIXTURES_REQUIRED "metal_big_gguf;metal_mtp_gguf" TIMEOUT 3600)
    set_tests_properties(metal_model_big_test metal_model_big_test_evict metal_model_big_window metal_model_big_window_evict PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_big_gguf TIMEOUT 3600)
  endif()
endif()

# ---- Qwen 3.8 Flash Next (CYBER-FROST) on Metal: maya-metal-qwen -------------------------------
# A third Metal engine, vendored whole under src/qwen-metal/ from Niko1221/Strata at c62d5e3 (MIT;
# src/qwen-metal/LICENSE) - the engine measured standalone on this Mac class - in the same style as
# src/ds4: a self-contained engine directory with its own host copy, headers and data, so neither
# tree's strata core shadows the other's.  Unlike GLM (ggml graph ops), this engine is raw Metal:
# its .metal kernels compile at run time over metal/src/mtl_rt.* + metal/src/gpu/runtime.cpp
# (metal-cpp), and ggml enters only as ggml-cpu/ggml-base (expert CPU lanes, quant type traits) -
# the fork build's own targets from the pinned checkout above, so there is no second ggml here.
# Its serve protocol is the shared one: serve/server.py's StrataEngine drives it from a config
# (~/maya-models/maya-qwen.json) exactly as it drives the CUDA build of the same generate.cpp.
option(MAYA_QWEN_METAL "Build the Qwen3.8-Flash-Next Metal engine (maya-metal-qwen)" ON)
set(_maya_qwen_root ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen-metal)
if(MAYA_QWEN_METAL AND EXISTS ${_maya_qwen_root}/metal-cpp/Metal/Metal.hpp)
  enable_language(OBJCXX)
  set(_qw_host ${_maya_qwen_root}/src)
  set(_qw_metal ${_maya_qwen_root}/metal)
  set(_qw_ggml_common ${_maya_qwen_root}/ggml-common/ggml-common.h)

  # the kernel tree's runtime face (his mtl_rt): the Metal runtime + GGUF reader + CPU references
  add_library(qwen_mtl_rt STATIC ${_qw_metal}/src/mtl_rt.mm ${_qw_metal}/src/ref_quant.cpp
                                 ${_qw_metal}/src/gguf_lite.cpp)
  target_include_directories(qwen_mtl_rt PUBLIC ${_qw_metal}/src ${_maya_qwen_root}/ggml-common
                                                ${_maya_qwen_root}/include)
  target_compile_options(qwen_mtl_rt PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_compile_definitions(qwen_mtl_rt PRIVATE
    STRATA_GGML_COMMON_DEFAULT="${_qw_ggml_common}"
    STRATA_METAL_KERNELS_DEFAULT="${_qw_metal}/kernels")
  target_link_libraries(qwen_mtl_rt PUBLIC "-framework Metal" "-framework Foundation")

  # the device runtime: the shared host code's CUDA names over Metal (metal_compat first, as HIP does)
  add_library(qwen_gpu_metal STATIC ${_qw_metal}/src/gpu/runtime.cpp)
  target_include_directories(qwen_gpu_metal BEFORE PUBLIC ${_qw_metal}/include/metal_compat
                             ${_qw_metal}/include ${_maya_qwen_root}/include)
  target_include_directories(qwen_gpu_metal SYSTEM PUBLIC ${_maya_qwen_root}/metal-cpp)
  target_compile_definitions(qwen_gpu_metal PUBLIC STRATA_GPU_METAL=1)
  target_link_libraries(qwen_gpu_metal PUBLIC "-framework Metal" "-framework Foundation"
                                              "-framework QuartzCore")

  # the engine's host: his STRATA_METAL_HOST_SOURCES list, from the vendored host copy
  set(_qw_host_sources
    ${_qw_host}/core/device.cu ${_qw_host}/core/pinned.cu ${_qw_host}/platform/memory.cpp
    ${_qw_host}/core/graph.cpp ${_qw_host}/core/weights.cpp ${_qw_host}/core/layout.cpp
    ${_qw_host}/core/layer.cpp ${_qw_host}/core/session.cpp ${_qw_host}/core/expert_source.cpp
    ${_qw_host}/core/remote_experts.cpp ${_qw_host}/core/expert_cache.cpp ${_qw_host}/core/native_head.cpp
    ${_qw_host}/core/native_dense.cpp ${_qw_host}/core/verify.cpp ${_qw_host}/core/mtp.cpp
    ${_qw_host}/core/conversation_snapshot.cpp ${_qw_host}/core/conversation_state.cpp
    ${_qw_host}/core/conversation_memory.cpp ${_qw_host}/core/peer_experts.cpp
    ${_qw_host}/core/arch_defaults.cpp ${_qw_host}/core/vmm.cpp ${_qw_host}/core/conversation_file.cpp
    ${_qw_host}/kernels/ngram.cpp ${_qw_host}/ngram/ple_reader.cpp ${_qw_host}/platform/direct_file.cpp
    ${_qw_host}/prefill/prefill.cpp
    ${_qw_host}/spec/suffix_drafter.cpp ${_qw_host}/spec/controller.cpp ${_qw_host}/spec/draft_policy.cpp
    ${_qw_host}/spec/draft_source.cpp
    ${_qw_host}/kernels/cpu/pool.cpp ${_qw_host}/kernels/cpu/expert_layout.cpp
    ${_qw_host}/kernels/cpu/native_expert.cpp ${_qw_host}/kernels/cpu/hadamard.cpp
    ${_qw_host}/kernels/cpu/tq_cpu.cpp)
  set_source_files_properties(${_qw_host}/core/device.cu ${_qw_host}/core/pinned.cu
                              PROPERTIES LANGUAGE CXX)
  add_library(qwen_host STATIC ${_qw_host_sources})
  target_include_directories(qwen_host PUBLIC ${_maya_qwen_root}/include ${_qw_host}/core
                             ${_maya_qwen_root}/ggml-common
                             ${_maya_llamacpp_dir}/ggml/include ${_maya_llamacpp_dir}/ggml/src
                             ${_maya_llamacpp_dir}/ggml/src/ggml-cpu)
  target_compile_definitions(qwen_host PUBLIC STRATA_NATIVE_EXPERTS=1 STRATA_VERSION="0.1.55-metal")
  target_compile_options(qwen_host PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
  target_link_libraries(qwen_host PUBLIC qwen_gpu_metal ggml-cpu ggml-base)

  # the kernel library: Metal implementations of the entry points, plus his committed stubs for
  # the ones not ported (a stub stops the engine with the entry point's name, never silently)
  file(GLOB _qw_engine_sources CONFIGURE_DEPENDS ${_qw_metal}/src/engine/*.cpp)
  add_library(qwen_kernels STATIC ${_qw_engine_sources})
  target_include_directories(qwen_kernels PRIVATE ${_qw_metal}/src ${_maya_qwen_root}/ggml-common)
  target_compile_definitions(qwen_kernels PRIVATE
    STRATA_GGML_COMMON_DEFAULT="${_qw_ggml_common}"
    STRATA_METAL_KERNELS_DEFAULT="${_qw_metal}/kernels"
    STRATA_METAL_ENGINE_KERNELS_DEFAULT="${_qw_metal}/src/engine/msl"
    STRATA_METAL_DATA_DEFAULT="${_maya_qwen_root}/data")
  target_link_libraries(qwen_kernels PUBLIC qwen_host qwen_mtl_rt)

  add_executable(maya-metal-qwen ${_qw_host}/program/generate.cpp)
  target_link_libraries(maya-metal-qwen PRIVATE qwen_kernels qwen_host qwen_kernels)

  # ---- Qwen unit-test gate (audit AUDIT-QWEN F2, 2026-11-13): the vendored Strata tree
  # ships a real unit-test suite (src/qwen-metal/src/{core,program}/*_test.cpp) that no
  # build compiled, so the daily-driver engine had zero ctest coverage while GLM and
  # MiniMax have fixture gates. The tests are host-side by construction (header logic
  # plus the qwen_host sources); each is a standalone main() linked against qwen_host,
  # whose PUBLIC includes carry the strata headers, src/core and the metal_compat
  # shims. Named metal_qwen_* like the metal_mm3_* set.
  # Excluded a priori, with reason:
  #   pinned_shared_win_test - the Windows variant of pinned_shared_test (<windows.h>
  #     included unguarded); the POSIX pinned_shared_test covers the same header.
  #   conversation_snapshot_test - EXCLUDED AS A FINDING (2026-11-13): wired, it
  #     builds and runs, and fails in 0.03s on the engine's own stub -
  #     strata::kernels::kv_stream_reset is a committed no-implementation stub
  #     (metal/src/engine/stubs.cpp:238 -> missing.cpp), and the snapshot path
  #     calls it. The test is upstream's and the failure is real engine surface:
  #     that entry point has no Metal implementation. Porting it is engine work,
  #     out of scope for the test gate; production checkpoint reuse is measured
  #     working (98%+ in Hermes sessions), so the production snapshot path does
  #     not reach this stub - the test exercises a reset path serve does not.
  #     Re-wire this test the day kv_stream_reset gets a Metal implementation.
  foreach(_qw_test
          adapt_async_default arch_defaults auto_cache batch_telemetry
          conversation_cache conversation_copy_streams conversation_file
          conversation_memory conversation_split_failure
          conversation_validation coupled_draft draft_cap gated_proposal
          logit_bias nv_defaults pin_budget pinned_shared ple_residency
          session_save_reclaim spec_admission spec_prob uma_cache window_evict
          window_limits)
    add_executable(metal_qwen_${_qw_test}_test ${_qw_host}/core/${_qw_test}_test.cpp)
    target_link_libraries(metal_qwen_${_qw_test}_test PRIVATE qwen_host)
    add_test(NAME metal_qwen_${_qw_test}_test COMMAND metal_qwen_${_qw_test}_test)
  endforeach()
  foreach(_qw_test adapt_decay batch_groups conv_cache draft_kv_plan
                   message_boundary split_ring)
    add_executable(metal_qwen_${_qw_test}_test ${_qw_host}/program/${_qw_test}_test.cpp)
    target_link_libraries(metal_qwen_${_qw_test}_test PRIVATE qwen_host)
    add_test(NAME metal_qwen_${_qw_test}_test COMMAND metal_qwen_${_qw_test}_test)
  endforeach()
  # Link fixes found by the first build: a conversation test pulls
  # conversation_state/layer objects whose kernel and window-checkpoint symbols
  # live in qwen_kernels (the engine links both libraries; the tests must too),
  # and nv_defaults_test's subject (nv_default_env) is implemented in
  # core/nv_defaults.cpp, which the Metal host source list omits (NVIDIA
  # platform defaults) - compile the vendored unit into the test target
  # rather than into the engine library.
  target_link_libraries(metal_qwen_conversation_validation_test PRIVATE qwen_kernels)
  target_sources(metal_qwen_nv_defaults_test PRIVATE ${_qw_host}/core/nv_defaults.cpp)
  unset(_qw_test)
  # Serve-level smoke: there is no synthetic Strata pack fixture (the GLM/MiniMax
  # fixtures are GGUFs for the ggml engines; a Strata pack is a directory of native
  # shards plus tokenizer, not generatable by a small script), so the gate here is the
  # binary itself: --help must print the usage banner and exit 0. A restore-point
  # fixture is deliberately NOT registered: the engine's READY line advertises no
  # restore and its GEN parser takes no marks/shared keys (audit E2) - it relies on
  # engine-automatic checkpoints, which a fixture must not fake.
  add_test(NAME metal_qwen_smoke_test COMMAND maya-metal-qwen --help)
else()
  message(STATUS "maya-metal-qwen skipped: vendored metal-cpp headers not found at "
                 "${_maya_qwen_root}/metal-cpp (MAYA_QWEN_METAL=${MAYA_QWEN_METAL})")
endif()

# ---- Hy4-Flash on Metal: maya-metal-hy4 ------------------------------------------------------
# The Metal port of upstream's maya-hy4 (src/hy4, CUDA), staged per the port plan
# (HY4-METAL-PORT-PLAN.md in the Maya goal workspace): upstream's host loader
# (src/hy4/hy4_pack.hpp) and plan (src/hy4/hy4_plan.hpp) are reused unchanged; the GPU
# forward is reimplemented under src/hy4-metal/. Until a stage lands, the binary loads
# a pack and stops loudly rather than serving anything (the fork's stub rule).
option(MAYA_HY4_METAL "Build the Hy4-Flash Metal engine (maya-metal-hy4)" ON)
if(MAYA_HY4_METAL)
  add_executable(maya-metal-hy4 src/hy4-metal/hy4_metal_main.cpp)
  target_include_directories(maya-metal-hy4 PRIVATE src/hy4 src/mm3)
  target_compile_features(maya-metal-hy4 PRIVATE cxx_std_17)

  # Loader fixture gate: a synthetic mini-pack written by the test itself (no model
  # files), loaded through upstream's own hy4::Pack - geometry, tensor table and the
  # Engram hash parse asserted, plus a wrong-arch pack refused.
  add_executable(metal_hy4_pack_test src/hy4-metal/metal_hy4_pack_test.cpp)
  target_include_directories(metal_hy4_pack_test PRIVATE src/hy4 src/mm3)
  target_compile_features(metal_hy4_pack_test PRIVATE cxx_std_17)
  add_test(NAME metal_hy4_pack_test COMMAND metal_hy4_pack_test)
endif()
