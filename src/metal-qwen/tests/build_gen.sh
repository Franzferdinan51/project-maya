#!/bin/sh
# Build the host-only fixture generator (no GPU, no backend) and, with
# --run, regenerate the committed fixture set.
set -e
cd "$(dirname "$0")/../../.."   # repo root
CXX="${CXX:-c++}"
"$CXX" -std=c++17 -O2 -Wall -I include -I src/metal-qwen/tests \
    src/metal-qwen/tests/gen_fixtures.cpp -o /tmp/gen_fixtures
echo "built /tmp/gen_fixtures"
if [ "$1" = "--run" ]; then
    mkdir -p src/metal-qwen/tests/fixtures
    /tmp/gen_fixtures src/metal-qwen/tests/fixtures
fi
