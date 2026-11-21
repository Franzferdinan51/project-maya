# Inkling 2 Mini on one RTX 3090: the guide

`maya-ik2` runs [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) with every weight in the VRAM of one 24 GB card. This page is how to build and run it, what speed to expect, and how it works. The engine's design notes, with the model's exact arithmetic, are in [INKLING2.md](INKLING2.md) (section 8 is Mini).

## Quick start

You need Linux, an RTX 3090 or another 24 GB `sm_86` card, CUDA 12.8 with `nvcc` on the `PATH`, CMake 3.21 or newer, Ninja, and Python 3.10 or newer. The model takes 17.6 GB on disk and 23.6 GiB of VRAM at the full context.

### 1. Build the engine

```bash
git clone https://github.com/alesha-pro/maya-inkling-mini && cd maya-inkling-mini
pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-ik2
```

CMake downloads the pinned llama.cpp commit for ggml on its own. If the system has more than one CUDA, point CMake at the right one: `-DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`.

### 2. Build the image encoder (optional)

Images go through `maya-vision`, a small program built on llama.cpp's `mtmd` library. Upstream llama.cpp has the Inkling 2 image tower since tag `b12590`.

```bash
git clone --depth 1 -b b12590 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-b12590
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp-b12590
cmake --build build-vision -j 24 --target maya-vision
```

This build runs the encoder on the CPU. For the GPU add `-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` and use the `ik2-mini-gpu-vision.json` config: the encoder then holds 1.3 GiB of VRAM and the context drops to 262,144.

### 3. Get the weights and the tokenizer

```bash
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir models
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
  --chat-template tools/ik2/chat_template.jinja --check
```

The second command fetches the drafter, `mtp.safetensors` (0.6 GB) from Thinking Machines' own repository; skip it if you run the configs without the drafter. The third writes the tokenizer files the server reads to `ik2/tokenizer`. Only `UD-Q4_K_XL` works: `UD-Q8_K_XL` keeps the experts of the last 4 layers in BF16, which the engine doesn't load. Both files carry the same NVFP4 experts for layers 0 to 19, so `UD-Q8_K_XL` wouldn't be faster or much more exact anyway.

### 4. Run

```bash
python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini.json --host 127.0.0.1 --port 18000
```

The paths in the config are relative to the checkout, so start the server from there. It is ready in about 10 s.

```bash
curl -s http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini", "max_tokens": 300, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "Write a haiku about a lighthouse."}]}'
```

The server speaks the OpenAI chat and responses APIs and the Anthropic messages API, with streaming, tool calls and images (`image_url` with a URL or base64 data). The model has no switch for its reasoning, only an effort level. `"reasoning_effort": "none"` makes the answer start at once: the server opens the answer's text segment for the model. Without it the model thinks first, and the thinking counts against `max_tokens`.

To pick the card, set `CUDA_VISIBLE_DEVICES` before the command.

## Configs

All 4 are in [`docs/inkling2-mini/`](inkling2-mini/).

| Config | Context | Drafter | Image encoder | VRAM |
|---|---:|---|---|---:|
| `ik2-mini.json` | 262,144 | yes | CPU | 23.8 GiB |
| `ik2-mini-long.json` | 344,064 | no | CPU | 23.3 GiB at start, 23.6 at 340K tokens |
| `ik2-mini-text.json` | 344,064 | no | none | the same |
| `ik2-mini-gpu-vision.json` | 262,144 | no | GPU, 1.3 GiB | 23.7 GiB |

The context is the prompt, the image tokens and the answer together. A smaller `--max-context` frees VRAM: the FP8 KV cache of the 8 layer visits without a window takes 16 KB per token, 5.5 GiB at 344,064. The drafter holds 0.7 GiB of weights plus 4 KB per position of the context its global layer sees (`IK2_DRAFT_CTX`, 65,536 in the config).

The `env` block in the configs:

| Setting | What it does |
|---|---|
| `IK2_KV=fp8` | The KV cache in FP8 (E4M3). Half the bytes of FP16, and the faster one deep in a context |
| `IK2_GATE_BF16=1`, `IK2_GATE_TILE=64` | The loop's gate weights kept as the 16 bits they really are, a fixed block shape for their kernel |
| `IK2_Q8_UNROLL=1` | Dense Q8 kernels compiled for this model's widths |
| `IK2_BATCH_PREFILL=1` | Experts of a prompt chunk handled in batches: about 6% faster prefill |
| `IK2_DRAFT_CTX=65536` | The drafter's global layer sees the last 65,536 positions (default: the whole context, which needs 1 GiB more at 262K) |

With `--draft` the engine decides every step whether to draft. A window of B tokens is worth (1 - a^B) / (1 - a) tokens, where a is how often recent drafts were accepted; it drafts when that, over the window's measured time, beats a plain token's time, and after 8 plain tokens it drafts once to look again. `IK2_DRAFT_WIDTH` (default 5) sets the widest window, `IK2_DRAFT_ADAPT=0` always drafts at that width.

The tensor core expert kernels are on by default; `IK2_EXPERT_MMA=0` in `env` switches back to the scalar ones, which give the same logits bit for bit as the older builds. Every setting is listed in [SETTINGS.md](SETTINGS.md).

## Numbers

One RTX 3090 (GPU 3 of 4, the others idle) at a 300 W power limit, driver 610.43, the `gpu-unlock` kernel module loaded, AMD EPYC 7642, Ubuntu 22.04. Unsloth `UD-Q4_K_XL`, FP8 KV, context 344,064, greedy. Each row is a cold start: the prompt is read from scratch, then the engine generates.

| Prompt, tokens | Prefill, tok/s | Decode after it, tok/s |
|---:|---:|---:|
| 47 | 187 | 128.7 |
| 4,096 | 2,742 | 124.2 |
| 63,981 (a real document) | 2,361 | 105.6 |
| 127,910 (a real document) | 2,079 | 90.6 |
| 340,000 | 1,473 | 62.3 |

The command behind every row:

```bash
IK2_KV=fp8 IK2_GATE_TILE=64 IK2_GATE_BF16=1 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1 CUDA_VISIBLE_DEVICES=3 \
  build/maya-ik2 --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf --ctx 344064 --chunk 2048 --tokens @IDS.txt --gen 400
```

`IDS.txt` is a comma separated list of token ids. Run the CLI with `--ctx`: without it the engine doesn't keep every expert in VRAM and takes its slower general path.

Through the HTTP server, one client, no drafter: a new story 129.5 tok/s, new Python code at temperature 0.7 127.2 tok/s. A chat turn of 12 new tokens is read in 56 ms.

With the drafter (context 262,144, `IK2_DRAFT_CTX=65536`, the same card), against plain decode on the same build:

| Text | Prompt, tokens | Plain, tok/s | Drafter, tok/s | Tokens per window | Plain tokens left |
|---|---:|---:|---:|---:|---:|
| Prose | 42 | 126.0 | 131.1 | 2.63 | 241 of 383 |
| New code | 51 | 125.5 | 148.7 | 2.93 | 99 |
| Editing a document | 512 | 123.7 | 211.6 | 4.36 | 17 |
| Copying code | 512 | 124.0 | 188.1 | 3.71 | 1 |
| A real document | 63,981 | 103.6 | 124.8 | 3.27 | 49 |

```bash
IK2_KV=fp8 IK2_GATE_TILE=64 IK2_GATE_BF16=1 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1 IK2_DRAFT_CTX=65536 \
  build/maya-ik2 --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf --ctx 262144 --chunk 2048 --tokens @IDS.txt --gen 384 \
  --draft models/mtp.safetensors
```

Through the server new code runs at 182 to 207 tok/s greedy and about 205 at temperature 0.7. A coding agent (DeepSeek Harness) writing a 3-file landing page wrote its files at 150 to 167 tok/s.

llama.cpp `b12590` with the Inkling 2 pull requests, same card and file: 57.0 tok/s decode and 1,052 tok/s prefill of 4,096 tokens; 65,536 tokens deep 42.7 and 630.

Decode slows down with depth for a physical reason. 8 of the 48 layer visits attend over the full prefix, so every new token reads 16 KB of cache per token of context: 5.4 GB at 340K, on top of the 3.7 GB of weights.

## How it works

Inkling 2 Mini is a 26B MoE: 24 layers of weights, 128 experts each (8 per token), and layers 6 to 17 run 3 times, which makes 48 layer visits per token. 40 visits use a sliding window of 1,024 keys, 8 see everything.

**Everything in VRAM.** The experts are 14.7 GiB and the dense weights 1.1 GiB as the GGUF's Q8_0 blocks. On a 24 GB card that leaves room for the KV cache, so a token needs nothing from the host between its embedding and its logits. The router's picks are resolved on the device through a table of pointers, and the 781 kernel launches of a token replay as one CUDA graph.

**NVFP4 without FP4 hardware.** The experts of layers 0 to 19 are 4-bit codes with one scale per 16 values. The kernel loads the codes straight from VRAM, turns them into FP16 in registers (a code times its scale is exact in FP16) and hands them to the tensor cores, which sum in FP32. No decoded copy is ever stored. The input goes in as 2 FP16 parts in 2 of the 8 tensor core columns, which costs nothing and gives back the precision of an FP32 input.

**Weights in flight.** The first tensor core kernel was slower than the scalar one: 100 tok/s against 118. A warp waits for a block's bytes before it asks for the next ones, so the speed of a kernel here is set by how many bytes are requested at once. With 8 warps on the same 16 rows, each taking its own stride of the row, the same code runs at 745 GB/s.

**Attention at memory speed.** A token's attention runs on the tensor cores, a warp at a time straight from the cache into registers: 846 GB/s over an FP16 cache, 776 GB/s over FP8.

**Fewer launches.** Every kernel launch costs 2 µs or more whatever it does. Convs ride the products before them, a looped layer's q, k and v products are one launch, the loop's messages for all 12 core layers are one launch per pass: 1,404 launches per token became 781.

**The drafter's window.** Thinking Machines' drafter (3 layers, fed by the model's stream at visits 5, 23 and 41 and its final hidden) guesses the next tokens, and the model checks the anchor and the guesses in one pass. That pass has its own path: positions read from the device, the window's expert picks grouped on the GPU so an expert is read once for all the tokens that picked it (5 tokens read 2.7 times one token's experts, not 5), the attention reading each key once for the window's queries, the short convs' histories recorded and rolled back in one launch for the rejected tokens, one CUDA graph per window size. 5 tokens cost 15.6 ms against 7.8 ms for one, and 40.8 ms through the general prompt path. Sampled output stays exact: a guess is accepted with the model's probability of it, otherwise the model's distribution without it is sampled.

**Prefill** decodes the experts a chunk needs to FP16 and runs cuBLAS GEMMs, with a flash attention kernel that reads the FP8 cache. Short prompts up to 96 tokens take a path that reads each expert once for up to 8 of its tokens.

What I tried that didn't help:

* **The drafter through the general prompt path.** A window of 5 cost 41 ms there (the expert groups built on the host, a sync per layer): code 66 tok/s. It needed the window path above.
* **A drafter context cut to the last 16K positions.** Faster drafting on long contexts, but on a 64K document 2.24 tokens per window against 3.07 with the whole context.
* **A vocabulary shortlist in the drafter's Markov head.** The same time, fewer tokens accepted.
* **Skipping V reads for small attention weights.** The model's global attention is spread out: 0.0% of the steps could be skipped on a real 64K text.
* **Asking for more bytes ahead in the dense Q8 kernels.** They already run at 715 GB/s, the memory's limit.
* **Approximating the expert input with 2 int8 parts and DP4A.** 125 tok/s, but 71 of 96 on GSM8K against 73.

## Quality

* 96 GSM8K questions, greedy, reasoning off: 73 solved with the scalar expert kernels, 74 with the tensor core ones (3 lost, 4 gained).
* 399 forced tokens of real text with an FP16 cache, tensor core experts against scalar: median KL 5.7e-5, the same top token on 394. The model is sensitive on this text: the tensor core attention against its scalar kernel gives 1.7e-4 and 393.
* The drafter's window against the general prompt path on the same tokens, FP16 cache: KL about 5e-8 per row (up to 5e-4 at near router ties). 96 GSM8K questions, greedy: 77 solved with the drafter, 74 without.
* An FP8 cache against FP16: median KL 1.1e-3. Any 2 FP8 runs drift apart within a few tokens, because a difference of 1e-5 flips codes of a 3-bit mantissa. That is the cache format, not a bug.
* The float64 reference `ref/inkling2.py` agrees with llama.cpp on 47 of 47 top tokens, and the engine with the reference on 45 to 46 of 47.

## What else is in the tree

The branch also carries experimental code that the configs above leave off: a 1,048,576 token context with experts swapped between RAM and VRAM (about 9 tok/s at that depth), a CPU lane for expert products on AVX2 hosts and verified copying of text from the prompt for edits (`IK2_SUFFIX`, which the drafter replaces). They are described in [INKLING2.md](INKLING2.md), section 8.

## Credits

[Project Maya](https://github.com/mw00/project-maya) by Peasant Smith and the Maya contributors is the base, and `maya-ik2` began as the Inkling 2 Small engine of Maya's pull requests #981 and #1079. [Strata](https://github.com/Niko1221/Strata) by Niko1221 is where Maya came from. The model is by [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini), the GGUF by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), ggml and `mtmd` by the llama.cpp authors. The Windows / CUDA 13 build and two-GPU expert parallel are by [@cobra91](https://github.com/cobra91).
