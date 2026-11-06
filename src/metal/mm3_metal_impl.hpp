// src/metal/mm3_metal_impl.hpp - Mm3Metal's private state, shared by the loader (mm3_metal_load.cpp) and
// the graph (mm3_metal.cpp).  The GGUF file set, the weight arena and the backends are the GLM engine's
// (glm_metal_impl.hpp): the two engines differ in geometry, state and graphs, not in how a GGUF is read.
#pragma once

#include "glm_metal_impl.hpp"   // TensorLoc, Weights, free_memory (maya::metal)
#include "mm3_dflash2.hpp"
#include "mm3_dspark.hpp"
#include "mm3_metal.hpp"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace maya::metal {

struct Mm3LayerState {          // what a layer keeps between tokens, all BF16 as the CUDA engine's caches
    ggml_tensor* kc = nullptr;  // [head_dim, n_kv, n_ctx]
    ggml_tensor* vc = nullptr;  // [head_dim, n_kv, n_ctx]
    ggml_tensor* ikc = nullptr; // [ix_dim, n_ctx] the indexer's one shared key per token
};

struct Mm3Metal::Impl {
    Mm3Options o;
    Mm3Geometry g;
    Mm3Stats st;

    // files: the main shards plus the indexers' side file when the set has one
    std::vector<gguf_context*> ggufs;
    std::vector<ggml_context*> metas;
    std::vector<int> fds;
    std::unordered_map<std::string, TensorLoc> loc;

    // backends
    ggml_backend_dev_t gpu_dev = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    Weights wts;   // every tensor resident, the routed experts included

    // state + carries between the two graphs of a layer
    ggml_context* sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    std::vector<Mm3LayerState> ls;
    ggml_tensor* c_h = nullptr;    // [n_embd, n_ubatch] F32 the slice's residual stream
    ggml_tensor* c_q = nullptr;    // [head_dim, n_head, n_ubatch] F32
    ggml_tensor* c_iq = nullptr;   // [ix_dim, ix_heads, n_ubatch] BF16 (the values the scores are made of)
    std::vector<uint8_t> arena;    // the graphs' ggml context (metadata only), kept between forwards
    int64_t n_ctx = 0;
    int64_t pos = 0;
    int64_t snap_pos = -1;

    Mm3Draft draft;          // the DSpark draft (empty until load_draft)
    Mm3DFlash2 draft2;       // the DFlash2 draft (ditto; a draft dir holds one or the other)
    int draft_kind = 0;      // 0 = none, 1 = DSpark, 2 = DFlash2 (load_draft sniffs the config)
    bool has_draft = false;

    // the active draft, whichever kind is loaded (both expose the same interface shape)
    const std::vector<int>& draft_taps() const { return draft_kind == 2 ? draft2.geo().taps : draft.geo().taps; }
    int active_draft_block() const { return draft_kind == 2 ? draft2.geo().block : draft.geo().block; }
    void draft_reset() {
        if (draft_kind == 2) draft2.reset_context();
        else draft.reset_context();
    }
    bool draft_context(const float* taps, int T, int64_t pos0, std::string& err) {
        return draft_kind == 2 ? draft2.context(taps, T, pos0, err) : draft.context(taps, T, pos0, err);
    }
    bool draft_propose(int32_t anchor, int64_t pos, int n, std::vector<int32_t>& drafts,
                       std::vector<float>& confs, std::string& err) {
        return draft_kind == 2 ? draft2.propose(anchor, pos, n, drafts, confs, err)
                               : draft.propose(anchor, pos, n, drafts, confs, err);
    }

    ~Impl();
    bool open_files(const std::string& path, std::string& err);
    bool read_geometry(std::string& err);
    bool init_backends(std::string& err);
    bool load_weights(std::string& err);
    bool init_state(std::string& err);
    ggml_tensor* w(const std::string& name) const;      // nullptr if absent
    ggml_tensor* wl(int il, const char* suffix) const;  // blk.<il>.<suffix>
    bool compute(ggml_cgraph* gf, const std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>>& up,
                 std::string& err);
    // logits_rows > 1 (with rows_out): the last layer's graph also writes that many rows of
    // logits (the slice's rows, row-major) - a verify window reads them all
    bool run_slice(const int32_t* tok, int T, std::vector<float>& logits, std::string& err,
                   int logits_rows = 1, std::vector<float>* rows_out = nullptr);
};

}  // namespace maya::metal
