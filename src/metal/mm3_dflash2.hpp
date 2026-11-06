// src/metal/mm3_dflash2.hpp - the DFlash2 draft head behind the Mm3Draft interface
// (incoai/MiniMax-M3.1-Flash-DFlash2, Apache-2.0): a 5-layer GQA block drafter that reads
// SIX target taps [5, 16, 26, 37, 47, 58], drafts a whole block in ONE parallel pass
// (no Markov chain) with DFlash2's dynamic grouped convolutions wrapped around every
// attention and MLP sublayer, and picks its tokens with the learned candidate selector
// (top-32 head candidates per slot, a rank-256 codebook lattice walk).  The engine holds
// one of these next to the DSpark draft and dispatches on the draft dir's config.
// Spec: DFLASH2-SPEC.md in the repo root.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mm3_dspark.hpp"   // Mm3DraftGeo: the draft geometry the engine dispatches on

struct ggml_tensor;

namespace maya::metal {

class Mm3DFlash2 {
public:
    Mm3DFlash2();
    ~Mm3DFlash2();
    Mm3DFlash2(const Mm3DFlash2&) = delete;
    Mm3DFlash2& operator=(const Mm3DFlash2&) = delete;

    // dir holds config.json + model.safetensors (bare tensor names, all BF16);
    // emb/head are the target's tensors (read-only, borrowed).  max_ctx sizes the
    // draft's own KV caches.
    bool load(const std::string& dir, ggml_tensor* emb, ggml_tensor* head, int64_t max_ctx, std::string& err);
    bool loaded() const { return p != nullptr; }
    const Mm3DraftGeo& geo() const;
    int64_t max_ctx() const;
    int top_k() const;        // selector candidates per slot (32 on the real checkpoint)
    int conv_group() const;   // grouped-conv channels per coefficient (16)
    int conv_taps() const;    // grouped-conv kernel size (2)

    // feed the context with the taps of T committed rows at positions pos0..pos0+T-1.
    // taps: [n_taps][T][dim] F32, tap-major.
    bool context(const float* taps, int T, int64_t pos0, std::string& err);
    void reset_context();
    int64_t context_hi() const;   // one past the highest position the context has been fed

    // up to n drafts after `anchor` (at position pos) with per-token confidences: the
    // pick's softmax share of its slot's lattice scores.  A DFlash2 block yields at
    // most geo().block - 1 drafts (the anchor row drafts nothing; the mask rows do).
    // False only on a hard error; empty drafts with no error means "no room".
    bool propose(int32_t anchor, int64_t pos, int n, std::vector<int32_t>& drafts, std::vector<float>& confs,
                 std::string& err);

    // the last proposal's final hidden states [block][dim] and logits [block][vocab] (tests)
    const std::vector<float>& last_hid() const;
    const std::vector<float>& last_logits() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}  // namespace maya::metal
