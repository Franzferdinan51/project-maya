<h1 align="center">Inkling 2 Mini on ONE RTX 3090: 128 tok/s, up to 210 with the drafter</h1>

<p align="center"><b>Thinking Machines' 26B looped MoE, all of it in 24 GB of VRAM. 128 tok/s plain decode, 150 to 210 tok/s on code and edits with the official DSpark drafter, 2,700 tok/s prefill, up to 344K context, images and tools. llama.cpp gets 57 tok/s on the same card.</b></p>

<p align="center">
<a href="docs/INKLING2-MINI.md"><b>Read the guide</b></a> ·
<a href="#quick-start"><b>Quick start</b></a> ·
<a href="https://huggingface.co/unsloth/Inkling-2-Mini-GGUF"><b>Weights on Hugging Face</b></a>
</p>

| Text, greedy | Plain decode, tok/s | With the drafter, tok/s | llama.cpp b12590 |
|---|---:|---:|---:|
| Prose | 126 | 131 | 57.0 |
| New code | 125 | 149 (through the server 182 to 207) | |
| Editing a document | 124 | 212 | |
| Copying code | 124 | 188 | |
| A 64K token document | 104 | 125 | 42.7 (65,536 deep) |

One RTX 3090 at a 300 W limit, Unsloth `UD-Q4_K_XL`, FP8 KV cache, a cold prompt every time. Prefill runs at 2,742 tok/s on 4K tokens, 2,361 on 64K and 2,079 on 128K. Every launch and log is in the [guide](docs/INKLING2-MINI.md#numbers).

## What you get

* **`maya-ik2`**, an engine for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): the 24 layers, the loop (layers 6 to 17 run 3 times, 48 layer visits per token), the relative bias, the short convs and the attention residuals.
* **The model stays in VRAM.** All 3,072 experts (14.7 GiB), the dense weights and an FP8 KV cache fit in 24 GB: 344,064 tokens without the drafter, 262,144 with it. Nothing is read from RAM or disk while it answers.
* **Speculative decoding with the official drafter.** Thinking Machines ships a 3-layer DSpark drafter with the model. The engine checks its guesses 2 to 8 at a time in one pass, and drafts only when that pays: code and edits get a lot faster, prose stays as fast as plain decode. Greedy and sampled output are the model's own.
* **The original weights.** The experts of layers 0 to 19 are Thinking Machines' own NVFP4 bits, the way Unsloth's GGUF carries them. A 3090 has no FP4 hardware, so the engine decodes the 4-bit codes in registers and feeds the tensor cores.
* **Images, tools, reasoning control, streaming** through OpenAI and Anthropic compatible APIs on `/v1`.

## Quick start

Linux, an RTX 3090 (or another 24 GB `sm_86` card), CUDA 12.8, Python 3.10 or newer.

```bash
git clone https://github.com/alesha-pro/maya-inkling-mini && cd maya-inkling-mini
pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub

# the engine
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-ik2

# the image encoder (skip it for text only)
git clone --depth 1 -b b12590 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-b12590
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp-b12590
cmake --build build-vision -j 24 --target maya-vision

# the weights (17.6 GB), the drafter (0.6 GB) and the tokenizer files
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir models
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
  --chat-template tools/ik2/chat_template.jinja --check

python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini.json --host 127.0.0.1 --port 18000
```

Then talk to `http://127.0.0.1:18000/v1` with any OpenAI client, model `inkling-2-mini`. There are 4 configs in [`docs/inkling2-mini/`](docs/inkling2-mini/): the default (the drafter, 262K context, image encoder on the CPU), the longest context (344K, no drafter), text only, and the image encoder on the GPU.

## Why it's faster than llama.cpp

llama.cpp runs about 7,350 kernel launches per token for this model. A token here is 757 launches replayed as one CUDA graph, and the GPU never waits for the host between the embedding and the logits.

After that a token costs what its weights weigh: 3.7 GB read from VRAM, half of it experts. So the kernels are built around memory speed. Attention reads the KV cache at up to 846 GB/s, the NVFP4 experts go through the tensor cores at 745 GB/s, and an FP8 cache beats FP16 deep in a context because it is half the bytes.

The drafter's window has its own path: the tokens' expert picks are grouped on the GPU so an expert is read once for all the tokens that picked it, and the window runs as one graph too. 5 tokens cost 15.6 ms against 7.8 ms for one.

The [guide](docs/INKLING2-MINI.md#how-it-works) has the details, and what I tried that didn't work.

## Quality

The engine computes the model the GGUF holds, no pruning and no requantizing. On 96 GSM8K questions, greedy: 74 solved with plain decode and 77 with the drafter (the drafter only changes which pass computes a token, so the difference is noise). Against the float64 reference in `ref/inkling2.py` the engine picks the same top token on 45 to 46 of 47 positions, and the reference agrees with llama.cpp on 47 of 47.

## Limits

* One request at a time. Two GPUs run as expert parallel (`IK2_EP=1`, tested on Windows with an RTX 5080 and an RTX 4060 Ti by @cobra91), the drafter is one GPU's.
* It reads `UD-Q4_K_XL` only. `UD-Q8_K_XL` keeps the last 4 layers' experts in BF16 and the engine doesn't load them.
* Decode slows down with depth, because 8 of the 48 layer visits attend over the full prefix: 16 KB of cache per token of context is read for every new token.
* The drafter helps code, edits and structured text. On free prose it accepts about 2 tokens per window and the engine mostly decodes plain.
* The image encoder runs on the CPU in the default configs. A new image takes about 4.5 s on a 48-core EPYC, the same image again 0.5 s.
* Tested on RTX 3090s with the `gpu-unlock` kernel module, Ubuntu 22.04, driver 610.43. Windows builds since @cobra91's port; other cards are untested.

## About this repo

This is [Project Maya](https://github.com/mw00/project-maya) v1.1.70, the engine family by Peasant Smith ([@mw00](https://github.com/mw00)) and the Maya contributors, which grew out of [Strata](https://github.com/Niko1221/Strata) by Niko1221. `maya-ik2` started as the Inkling 2 Small engine from Maya's pull requests [#981](https://github.com/mw00/project-maya/pull/981) and [#1079](https://github.com/mw00/project-maya/pull/1079). This branch adds the loop, the path that keeps every expert in VRAM, the tensor core kernels for attention and experts, FP8 KV for prefill, GPU sampling candidates, speculative decoding with the official drafter, and the Mini profile of the server. The Windows / CUDA 13 build and two-GPU expert parallel are by [@cobra91](https://github.com/cobra91). Most of it is offered upstream in [mw00/project-maya#1095](https://github.com/mw00/project-maya/pull/1095). The rest of Maya (GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Hy4-flash and the app around them) is still in the tree but isn't maintained here: use [upstream Maya](https://github.com/mw00/project-maya) for those.

Weights: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its drafter by Thinking Machines, GGUF by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).

## License

The code is MIT, as upstream ([LICENSE](LICENSE)). The model weights are Apache 2.0.
