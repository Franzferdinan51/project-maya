#!/bin/sh
# Build and run the qsa parity runner on the Mac. Run from the repo root of
# a qwen-native-qsa checkout (the staging copy). Not part of the engine
# build: it links the family's real host launcher TUs against a test-only
# dispatch shim (tools/qsa_parity/launch_shim.mm).
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/qsa_parity_runner"
CXX="${CXX:-clang++}"
FLAGS="-std=c++17 -O1 -g -I $ROOT/include -I $ROOT/include/strata/metal_compat -include $ROOT/include/strata/metal_compat/cuda_runtime.h"
FAMILY="rope mrope native_rope native_qsa native_qsa_indexer native_qsa_score qsa qsa_decode_attn qsa_prompt_attn qsa_select native_flash_attn kv_q4 kv_q8 kv_stream"
OBJS=""
mkdir -p "${TMPDIR:-/tmp}/qsa_parity_obj"
for f in $FAMILY; do
    $CXX $FLAGS -c "$ROOT/src/metal-qwen/stubs/$f.cpp" -o "${TMPDIR:-/tmp}/qsa_parity_obj/$f.o"
    OBJS="$OBJS ${TMPDIR:-/tmp}/qsa_parity_obj/$f.o"
done
$CXX $FLAGS -c "$ROOT/tools/qsa_parity/runner.cpp" -o "${TMPDIR:-/tmp}/qsa_parity_obj/runner.o"
$CXX -std=c++17 -O1 -g -I "$ROOT/include" -c "$ROOT/tools/qsa_parity/launch_shim.mm" \
    -o "${TMPDIR:-/tmp}/qsa_parity_obj/shim.o" -fobjc-arc
$CXX "${TMPDIR:-/tmp}/qsa_parity_obj/runner.o" "${TMPDIR:-/tmp}/qsa_parity_obj/shim.o" $OBJS \
    -framework Metal -framework Foundation -o "$OUT"
"$OUT" "$ROOT/src/metal-qwen/kernels"
