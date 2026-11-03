// include/strata/core/verify.hpp - plan v0.3 P6: the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU (measured on decode traces: 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer) and a profile-filled VRAM expert tier with its residency table on the device.
#pragma once

#include <cstdio>

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer);

struct VerifyHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert] slot or -1
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;   ///< E-6: host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;                  ///< E-6: how many (for the device copy)
    int64_t blob = 0;
};

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;

    /// The watchdog's view of the window in flight (issue #31): the layer, the GPU's sequence, the flags.
    void diag(std::FILE* f) const;

    /// `max_t` <= kVerifyMaxT.  `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err);

    /// One window: `tokens[0..T)` at positions pos0.., the pool served per layer; `out[t]` = argmax after token t.
    /// The PLE rows are gathered here from `ss.ple_prev` and the tokens.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// The sampling the verify window's head applies (temperature / top_p / top_k / seed).  Set per
    /// request; greedy by default.  The sampling itself runs OUTSIDE the captured graph - its
    /// parameters would otherwise be baked forever - so this can change between requests freely.
    void set_sampling(const strata::kernels::SamplerParams& sp) {
        sampling_ = sp;   // row t of a window at pos0 draws Philox(seed, pos0 + t): see run()
        if (next_) next_->set_sampling(sp);
    }

    /// The penalty histories for `sampling_.penalty_last_n`: ONE ROW PER WINDOW ROW, T rows of `history_len`
    /// int32 slots at that stride (`strata::kernels::penalty_rows` builds them), most recent token LAST, unused
    /// front slots -1 (the kernel reads only the tail window).  Row t follows the window's drafts 1..t - staging
    /// row 0 alone (before 0.1.19) left the drafted rows with unwritten histories.  Null disables the penalties
    /// entirely - the neutral run's sampling call is byte-for-byte what it was.  The engine re-uploads the rows
    /// before every window; the buffer must hold kVerifyMaxT rows and stay alive across the request.
    void set_history(const int32_t* history, int history_len) {
        hist_d_ = history;
        hist_len_ = history_len;
        if (next_) next_->set_history(history, history_len);
    }
    /// Off: `run` skips the request's head sampling and `out` is the recorded greedy pick.  For windows whose
    /// picks are discarded - a prompt read through windows commits every token - so they cost no sampler launch
    /// or sync and never read a history staged for another position.
    void set_head_sampling(bool on) { head_sampling_ = on; if (next_) next_->set_head_sampling(on); }

    /// LAYER SPLIT (multi-GPU): this verifier runs layers [layer_begin, layer_end) of every window.  A stage that
    /// does not start at layer 0 takes its residual from `handoff_in` instead of embedding the tokens; a stage that
    /// does not end at the last layer writes its residual to `handoff_out` and has no head.  The hand-off holds,
    /// per token, the residual R (hc x n_embd), the last layer's pending write bo (n_embd) and inject (hc): the next
    /// stage folds that write into its first read exactly as the unsplit window does, so the split is bit-exact.
    /// Both pointers must be device-visible (mapped pinned memory, portable when the stages are on two devices).
    /// Set before `init`.  Default: the whole model, no hand-off.
    void set_stage(int64_t layer_begin, int64_t layer_end, const float* handoff_in, float* handoff_out) {
        lb_ = layer_begin; le_ = layer_end; hand_in_ = handoff_in; hand_out_ = handoff_out;
    }
    /// The next stage: `run` and `commit` continue into it (its pool calls get `next_user`); sampling settings
    /// and `final_R` are the last stage's.
    void set_next(Verifier* next, void* next_user) { next_ = next; next_user_ = next_user; }
    /// floats per token in a hand-off buffer
    static int64_t handoff_floats(const ModelGeometry& g) { return (int64_t) g.hc * g.n_embd + g.n_embd + g.hc; }

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.
    bool commit(int n_keep, std::string& err);

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return next_ ? next_->final_R_all() : R_; }
    /// The last window's head logits, rows [0, rows) (n_vocab floats each, raw: before any sampling), copied to
    /// `host`; valid until the next `run` (--dump-logits, issue #658).
    bool read_head_logits(int rows, float* host, std::string& err);

    /// The GPU plan the pool writes each layer (VRAM hits + the PCIe share of the misses); give it to the
    /// dispatch (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return &sink_; }
    /// Plan v0.3 P6: split the window into two token groups and pipeline the CPU experts of one with the GPU work
    /// of the other (default on).  Set before the first `run`.
    void set_split(bool on) { split_ = on; }
    /// Plan v0.3 P6: how the PCIe share of the misses reaches the GPU: 0 = DMA into staging (the copy engine works
    /// beside the CPU; best when the CPU is compute-bound, the i-quants), 1 = the grouped kernel reads the mapped
    /// arena directly, 2 = a copy kernel stages it inside the graph (no API calls on the pool's thread; best when
    /// the CPU is RAM-bound, Q2_0).  Set before the first `run`.
    void set_pcie_mode(int mode) { sink_.pcie_mode = mode; }
    /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    int64_t windows = 0;
    /// STRATA_VERIFY_PROFILE=1 - GPU stage times of the windows since the last call (ms per
    /// window), as one line; empty when off.
    std::string profile_report();

private:
    bool capture(int T, std::string& err);
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();   ///< greedy by default; per-request via set_sampling
    const int32_t* hist_d_ = nullptr;   ///< penalty-history row (set_history); null = no penalties apply
    int hist_len_ = 0;
    bool head_sampling_ = true;          ///< set_head_sampling
    int device_ = -1;                    ///< the device `init` ran on: run/commit switch to it (layer split)
    bool device_plan_ = false;            ///< E-6: resident-only layers planned on the device (STRATA_VERIFY_DEVICE_PLAN)
    uint32_t* skip_ = nullptr;            ///< E-6: per group, the ring whose plan the device built (0: the host's)
    unsigned long long* slot_off_d_ = nullptr;   ///< E-6: the slot offsets on the device
    int64_t lb_ = 0, le_ = -1;           ///< set_stage: the layers this verifier runs (-1: to the last)
    const float* hand_in_ = nullptr;
    float* hand_out_ = nullptr;
    Verifier* next_ = nullptr;
    void* next_user_ = nullptr;
    bool ple_stage() const { return lb_ <= 1 && 1 < le_; }   ///< holds layer 1, where the PLE block runs
    bool capture_commit(std::string& err);
    bool record_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);
    static constexpr int kProfPer = 32;              // stamps per layer
    bool prof_on_ = false;
    unsigned long long* prof_ = nullptr;              // device: n_layers * kProfPer + 4 stamps
    std::vector<unsigned long long> prof_h_;
    double prof_sum_[2][kProfPer] = {};   // [GDN / QSA layers][stage]
    int64_t prof_windows_ = 0;
    // GRFOLD debug (STRATA_METAL_HASH_RESID, off by default): per-layer
    // residual fingerprints. Stages: 0 R after gr_read(0), 1 mixed after
    // gr_read(0), 2 bo after the mixer, 3 R after gr_read(1), 4 bo after
    // post()'s combine, 5 mixed after gr_read(1) (the FFN input), 6 router
    // ids+weights, 7 shared expert output, 8 parts after moe_hit_add,
    // 9 nat_xq (quantize_q8_1_rows output), 10 plan dst, 11 plan tok,
    // 12 plan counts+start, 13 plan start2, 14 hit_out before moe_hit_add.
    // Slot = ((l * kResidStages) + stage) * max_t + t.
    static constexpr int kResidStages = 25;
    bool hash_on_ = false;
    unsigned long long* hash_d_ = nullptr;
    std::vector<unsigned long long> hash_h_;
    int64_t hash_windows_ = 0;
    // Per-entry-row parts fingerprints: [layer][t][entry] for stage 8 (final
    // parts) and stage 13 (parts right after the mapped copy).
    unsigned long long *hash8_d_ = nullptr, *hash13_d_ = nullptr;
    std::vector<unsigned long long> hash8_h_, hash13_h_;
    // Layer-0 value snapshots (debug): hit_out before moe_hit_add and parts
    // after it, copied device-to-device in stream order inside post().
    float *snap_hit_ = nullptr, *snap_parts_ = nullptr;
    int64_t snap_win_ = -1;
    // GRFOLD debug: strided residual value snapshots [layer][t][sample],
    // filled where stage 0 is hashed and printed with the RESID block, to
    // measure per-layer deviation growth between window tilings in VALUES.
    float* val_d_ = nullptr;
    std::vector<float> val_h_;
    static constexpr int kValSamples = 64;
    bool val_on_ = false;
    // GRFOLD debug: QSA chain value snapshots [layer][t][which][sample],
    // which: 0 = attn_ (post attention), 1 = attn32_ (post gate), 2 = bo_
    // (post o-projection mmvq).
    float* val2_d_ = nullptr;
    std::vector<float> val2_h_;
    // GRFOLD debug: selection snapshots [layer][t][2][64]: block 0 = the
    // first 64 selected cell ids (bit-cast), block 1 = the 4 step fields.
    float* val3_d_ = nullptr;
    std::vector<float> val3_h_;
    // GRFOLD debug: single-writer probe [4][64]: 0 = attn_ L3 pos27,
    // 1 = attn32_ L3 pos27, 2 = qcur_ L3 pos27, 3 = attn_ L3 pos26.
    // Written by exactly one window, so no cross-window slot reuse.
    float* val4_d_ = nullptr;
    std::vector<float> val4_h_;

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int64_t next_pos_ = -1;   ///< where a window that continues the last committed one starts (-1: none yet)
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t exec_[9] = {};
    cudaGraphExec_t commit_exec_ = nullptr;
#ifdef STRATA_USE_METAL
    // Metal has no stream capture (src/metal-qwen/RISKS.md SS1): capture() and
    // capture_commit() store their launch sequence as a body, and the launch
    // sites invoke the body instead of cudaGraphLaunch
    // (src/metal-qwen/runtime/CAPTURE-NOTES.md). The exec slots still hold
    // minted (empty) graph execs so the "already captured" checks and the
    // destructor's destroys keep the CUDA shape. The bodies are plain void():
    // a Verifier only ever launches on its own stream (cs_), unlike the
    // session graphs, whose launch stream is the caller's.
    std::function<void()> window_body_[9];
    std::function<void()> commit_body_;
    // STRATA_VERIFY_NODES accounting on Metal: the graph-node introspection
    // (cudaGraphGetNodes / cudaGraphKernelNodeGetParams) has no analogue, so
    // the kernel-name list is taken from the stored sequence itself - the
    // runtime records the body's own launch names around its first launch.
    // A dry recording pass at capture time is not an option: record_window
    // mutates the very state (KV appends, GDN recurrence) the window exists
    // to advance, so the names can only be read as the sequence really runs.
    bool window_names_pending_[9] = {};
#endif

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    float* h_x_ = nullptr;       float* m_x_ = nullptr;         // doorbell payload: T * n_embd
    int32_t* h_ids_ = nullptr;   int32_t* m_ids_ = nullptr;     // T * k
    float* h_w_ = nullptr;       float* m_w_ = nullptr;         // T * k
    uint32_t* h_seq_ = nullptr;  uint32_t* m_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr;
    uint32_t* h_flagA_ = nullptr; uint32_t* m_flagA_ = nullptr;  // the GPU plan is in place
    uint32_t* h_flagB_ = nullptr; uint32_t* m_flagB_ = nullptr;  // the PCIe share's DMA copies have landed
    cudaStream_t copy_ = nullptr;                                 // the copy engine's stream (DMA of missed experts)
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[2 * 64 * 2] = {};                          // host-function arguments, one per (layer, group)
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    int32_t* h_plan_ = nullptr;  int32_t* m_plan_ = nullptr;     // counts | start | dst | tok | ptr (as int32 pairs)
    int64_t plan_i32_ = 0;                                        // int32 words in the plan block
    GpuPlanSink sink_;
    uint32_t cur_layer_ = 0;
    static void publish_plan(void* ctx);
    void set_plan_slot(int grp);
    bool split_ = false;   // opt-in (--spec-split): exact but slower, see the overlap study
    int groups_[9] = {};
    float* h_ymiss_ = nullptr;   float* m_ymiss_ = nullptr;     // T * k * n_embd

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *xn_ = nullptr;
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* sel_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *hit_out_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr;
    int32_t* plan_ = nullptr;                                     // device copy of the plan block
    uint8_t* staging_ = nullptr;                                  // VRAM slots for the PCIe share of the misses
    static constexpr int64_t kStagingBlobs = 16;
    uint8_t* hit_xq_ = nullptr;
    uint8_t* nat_xq_ = nullptr;   // plan v0.3 P6: q8_1 activations for a native pack's grouped experts
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    uint16_t* sh_bf16_ = nullptr;
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
