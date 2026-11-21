#!/bin/zsh
# Inkling 2 Mini on the Mac, Maya slot 8085 (Hermes provider maya-ik2).
# ggml/Metal runtime (llama.cpp build in ~/llama-rpc-mac) serving the
# lmstudio-community Q4_K_M GGUF. The UD-Q4_K_XL does not load in this
# llama.cpp build (looped-arch tensor layout) - use Q4_K_M.
# NOTE: the native maya-ik2 engine is CUDA-only; this is the Mac route.
pkill -f 'llama-server.*8085' 2>/dev/null; sleep 1
nohup ~/llama-rpc-mac/build/bin/llama-server   --model ~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf   --alias inkling-2-mini --host 0.0.0.0 --port 8085   --ctx-size 32768 --n-gpu-layers 999 --jinja   > ~/maya-ik2-mac.log 2>&1 &
echo started
