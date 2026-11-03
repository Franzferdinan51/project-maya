// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#if defined(_WIN32)
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include "strata/platform/cpu_relax.hpp"

#ifdef STRATA_USE_METAL
// Declared in src/metal-qwen/runtime/metal_internal.hpp, which only the
// runtime's own translation units include; declared again here for the one
// capture-layer caller (the node-introspection recording in capture()).
namespace strata::metal_internal {
void launch_name_recording(std::vector<std::string>* sink);
}  // namespace strata::metal_internal
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
}  // namespace

#ifdef STRATA_USE_METAL
namespace {
// ---- capture on Metal: stored bodies instead of graphs (src/metal-qwen/runtime/CAPTURE-NOTES.md) ----
//
// Mint a real (empty) graph exec through the runtime's graph state machine, so
// the exec slots are genuine handles the destructor can destroy in the usual
// way. The work itself lives in the stored body; launching the exec is never
// how Metal runs it. (The same arrangement as mtp.cpp's capture wrappers.)
cudaGraphExec_t verify_mint_empty_exec(cudaStream_t cs) {
    if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return nullptr;
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(cs, &graph) != cudaSuccess || !graph) return nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    return exec;
}

// A body that fails at launch has no caller left to return an error to, and a
// silently half-run window would corrupt the draft chain it verifies, so it
// aborts loudly (mtp.cpp's mtp_body_fail rule).
[[noreturn]] void verify_body_fail(const char* what, const std::string& err) {
    std::fprintf(stderr, "strata metal: verify %s body failed at launch: %s\n", what, err.c_str());
    std::abort();
}

// Run a captured sequence: the stored body on Metal, the graph everywhere
// else. With `names` set, the runtime records the body's own launch names
// around the invocation - the node-introspection substitute (verify.hpp).
bool verify_graph_launch(cudaGraphExec_t exec, const std::function<void()>* body, cudaStream_t cs,
                         std::vector<std::string>* names) {
    if (body && *body) {
        if (names) strata::metal_internal::launch_name_recording(names);
        (void) cudaGetLastError();   // clear first: a stale error is not the body's
        (*body)();
        if (names) strata::metal_internal::launch_name_recording(nullptr);
        return cudaGetLastError() == cudaSuccess;
    }
    return cudaGraphLaunch(exec, cs) == cudaSuccess;
}

// The STRATA_VERIFY_NODES accounting from the stored sequence's own recorded
// launch names: the same report shape the CUDA path prints from the graph's
// nodes (kinds sorted by count, top 40).
void verify_print_node_accounting(const std::vector<std::string>& names, int T) {
    std::map<std::string, int> kinds;
    for (const std::string& n : names) ++kinds[n];
    std::vector<std::pair<int, std::string>> v;
    for (auto& [k2, c] : kinds) v.push_back({c, k2});
    std::sort(v.rbegin(), v.rend());
    std::fprintf(stderr, "maya verify: the %d-token window body launches %zu nodes:", T, names.size());
    for (size_t i = 0; i < v.size() && i < 40; ++i)
        std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
    std::fprintf(stderr, "\n");
}
}  // namespace
#define VERIFY_LAUNCH(exec, body, cs, names) (verify_graph_launch((exec), (body), (cs), (names)))
#else
#define VERIFY_LAUNCH(exec, body, cs, names) (cudaGraphLaunch((exec), (cs)) == cudaSuccess)
#endif

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    std::fprintf(f, "  verify window: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", last_t_, (long long) last_pos0_, cur_layer_ + 1,
                 rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
}

Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    if (cs_) cudaStreamDestroy(cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (arena_) cudaFree(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[0].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        hist_snap_ = b.take<float>(T * HS);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    hash_on_ = std::getenv("STRATA_METAL_HASH_RESID") != nullptr;
    if (hash_on_) {
        const size_t nh = (size_t) g.n_layers * kResidStages * (size_t) max_t;
        if (cudaMalloc((void**) &hash_d_, nh * 8) != cudaSuccess) { hash_on_ = false; hash_d_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(hash_d_, 0, nh * 8); hash_h_.assign(nh, 0); }
        const size_t nr = (size_t) g.n_layers * (size_t) max_t * (size_t) ss.k;
        if (cudaMalloc((void**) &hash8_d_, nr * 8) == cudaSuccess && cudaMalloc((void**) &hash13_d_, nr * 8) == cudaSuccess) {
            cudaMemset(hash8_d_, 0, nr * 8); cudaMemset(hash13_d_, 0, nr * 8);
            hash8_h_.assign(nr, 0); hash13_h_.assign(nr, 0);
        } else { cudaGetLastError(); hash8_d_ = hash13_d_ = nullptr; }
        const size_t ns = (size_t) max_t * (size_t) ss.k * (size_t) g.n_embd;
        if (cudaMalloc((void**) &snap_hit_, ns * 4) != cudaSuccess || cudaMalloc((void**) &snap_parts_, ns * 4) != cudaSuccess) {
            cudaGetLastError(); snap_hit_ = snap_parts_ = nullptr;
        }
        const size_t nv = (size_t) g.n_layers * (size_t) max_t * kValSamples;
        if (cudaMalloc((void**) &val_d_, nv * 4) == cudaSuccess) {
            cudaMemset(val_d_, 0, nv * 4); val_h_.assign(nv, 0); val_on_ = true;
        } else { cudaGetLastError(); val_d_ = nullptr; }
        if (val_on_) {
            const size_t nv2 = (size_t) g.n_layers * (size_t) max_t * 8 * kValSamples;
            if (cudaMalloc((void**) &val2_d_, nv2 * 4) == cudaSuccess) {
                cudaMemset(val2_d_, 0, nv2 * 4); val2_h_.assign(nv2, 0);
            } else { cudaGetLastError(); val2_d_ = nullptr; }
            const size_t nv3 = (size_t) g.n_layers * (size_t) max_t * 2 * kValSamples;
            if (cudaMalloc((void**) &val3_d_, nv3 * 4) == cudaSuccess) {
                cudaMemset(val3_d_, 0, nv3 * 4); val3_h_.assign(nv3, 0);
            } else { cudaGetLastError(); val3_d_ = nullptr; }
            const size_t nv4 = (size_t) 4 * kValSamples;
            if (cudaMalloc((void**) &val4_d_, nv4 * 4) == cudaSuccess) {
                cudaMemset(val4_d_, 0, nv4 * 4); val4_h_.assign(nv4, 0);
            } else { cudaGetLastError(); val4_d_ = nullptr; }
        }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        device_plan_ = v != nullptr && std::atoi(v) != 0;
    }
    if (device_plan_) {
        bool ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); device_plan_ = false; }
    }
    std::fprintf(stderr, "maya verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) count.used / 1048576.0);
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

bool Verifier::read_head_logits(int rows, float* host, std::string& err) {
    if (next_ != nullptr) return next_->read_head_logits(rows, host, err);
    if (rows < 0 || rows > max_t_ || head_logits_ == nullptr) {
        err = "verify: no head logits for that many rows";
        return false;
    }
    if (rows == 0) return true;
    if (cudaSetDevice(device_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess ||
        cudaMemcpy(host, head_logits_, (size_t) rows * (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        err = std::string("verify: reading the head logits failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2) ? 2 : 1;
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    auto stamp = [&](int64_t l, int i, int grp) { if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs); };
    auto rhash = [&](int64_t l, int stage, int t, const float* p, int64_t n) {
        if (hash_on_) resid_hash(p, n, hash_d_, (int) (((l * kResidStages) + stage) * MT + t), cs);
    };
    auto rhash_row = [&](unsigned long long* buf, int64_t l, int t, int e, const float* p, int64_t n) {
        if (hash_on_ && buf) resid_hash(p, n, buf, (int) ((l * MT + t) * K + e), cs);
    };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    groups_[T] = G;

    // ---- the window's inputs, from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);
    if (ple_on) copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    if (lb_ > 0) {
        for (int t = 0; t < T; ++t) {
            copy_from_mapped(Rt(t), hand_in_ + (size_t) t * HB, HC * N, cs);
            copy_from_mapped(bo_ + (size_t) t * N, hand_in_ + (size_t) t * HB + HC * N, N, cs);
            copy_from_mapped(inj2_ + (size_t) t * HC, hand_in_ + (size_t) t * HB + HC * N + N, HC, cs);
        }
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            for (int t = tb; t < te; ++t) {
                gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                try {
                    ple_block(ple_ + t * N, Rt(t), ss.ple.hist, ss.ple.w, po, ss.ple.scratch, cs);
                    ple_history_advance(ss.ple.hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, ss.ple.hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        gr_read_group(0, pending, inj2_, inj_);
        for (int t = tb; t < te; ++t) { rhash(l, 0, t, Rt(t), HC * N); rhash(l, 1, t, mixed_ + t * N, N); }
        if (val_on_)   // GRFOLD debug: strided residual values entering this layer (kernel: stream-ordered)
            for (int t = tb; t < te; ++t)
                resid_sample(Rt(t), (int64_t) HC * N, val_d_, (int) (l * max_t_ + t), kValSamples, cs_);
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) gi * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                native_mmvq(wqkv->native_type, wqkv->native_data, xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                stamp(l, 2, grp);
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb);
                stamp(l, 3, grp);
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4, grp);
                native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, cs);
                stamp(l, 5, grp);
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, nullptr, cs, tb);
                stamp(l, 6, grp);
                native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);
                native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, cs);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, (float) qsa_freq_base(), pos, cs);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && !st.kv_q4;
                native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                else for (int t = tb; t < te; ++t)
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                stamp(l, 7, grp);
                native_mmvq(wk->native_type, wk->native_data, xq_, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                native_mmvq(wv->native_type, wv->native_data, xq_, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_q4) {   // Q4_0 KV (kv_q4.hpp): K and V rotated before they are stored
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                for (int t = tb; t < te; ++t) {
                    const int32_t* step_t = step_ + t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                    if (hash_on_ && st.k_pool && !st.kv_q4 && !st.kv_int8 && !st.kv_hybrid) {
                        // GRFOLD per-row pool fingerprint: pages through THIS row.
                        const int64_t pages_used = (last_pos0_ + t + s.page_size) / s.page_size;
                        const int64_t nf = pages_used * (int64_t) NKV * s.page_size * HD / 2;
                        rhash(l, 23, t, (const float*) st.k_pool, nf);
                        rhash(l, 24, t, (const float*) st.v_pool, nf);
                    }
                }
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int t = tb; t < te; ++t)
                    native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              (float) qsa_freq_base(), cs);
                for (int t = tb; t < te; ++t) {
                    rhash(l, 18, t, kcur_ + (size_t) t * NKV * HD, NKV * HD);
                    rhash(l, 19, t, vcur_ + (size_t) t * NKV * HD, NKV * HD);
                }
                if (val2_d_)
                    for (int t = tb; t < te; ++t) {
                        resid_sample(kcur_ + (size_t) t * NKV * HD, (int64_t) NKV * HD, val2_d_,
                                     (int) ((l * max_t_ + t) * 8 + 4), kValSamples, cs_);
                        resid_sample(vcur_ + (size_t) t * NKV * HD, (int64_t) NKV * HD, val2_d_,
                                     (int) ((l * max_t_ + t) * 8 + 5), kValSamples, cs_);
                    }
                stamp(l, 9, grp);
                native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD),
                            n, cs);
                for (int t = tb; t < te; ++t) rhash(l, 20, t, qfull_ + (size_t) t * NH * 2 * HD, NH * 2 * HD);
                if (val2_d_)
                    for (int t = tb; t < te; ++t)
                        resid_sample(qfull_ + (size_t) t * NH * 2 * HD, (int64_t) NH * 2 * HD, val2_d_,
                                     (int) ((l * max_t_ + t) * 8 + 7), kValSamples, cs_);
                if (qb) {
                    if (cudaMemcpy2DAsync(qcur_ + tb * NH * HD, (size_t) HD * 4, qfull_ + tb * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    for (int t = tb; t < te; ++t) rhash(l, 21, t, qcur_ + (size_t) t * NH * HD, NH * HD);
                    if (val2_d_)
                        for (int t = tb; t < te; ++t)
                            resid_sample(qcur_ + (size_t) t * NH * HD, (int64_t) NH * HD, val2_d_,
                                         (int) ((l * max_t_ + t) * 8 + 6), kValSamples, cs_);
                    // GRFOLD decontamination: production math ONLY (norm_rope), the
                    // hash-gated rms/rope variant from 76c8a05 is gone; stage 22
                    // observes qcur_ after the production norm_rope.
                    norm_rope(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH);
                    for (int t = tb; t < te; ++t) rhash(l, 22, t, qcur_ + (size_t) t * NH * HD, NH * HD);
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, cs);
                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
                } else {
                for (int t = tb; t < te; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    rhash(l, 21, t, qc, NH * HD);
                    if (val2_d_)
                        resid_sample(qc, (int64_t) NH * HD, val2_d_, (int) ((l * max_t_ + t) * 8 + 6), kValSamples, cs_);
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);   // production math only (see qb branch note)
                    rhash(l, 22, t, qc, NH * HD);
                    if (st.kv_q4) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = tb; t < te; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                stamp(l, 10, grp);
                for (int t = tb; t < te; ++t) rhash(l, 15, t, qcur_ + (size_t) t * NH * HD, NH * HD);
                if (val2_d_)
                    for (int t = tb; t < te; ++t)
                        resid_sample(qcur_ + (size_t) t * NH * HD, (int64_t) NH * HD, val2_d_,
                                     (int) ((l * max_t_ + t) * 8 + 3), kValSamples, cs_);
                if (val4_d_ && l == 3)
                    for (int t = tb; t < te; ++t)
                        if (last_pos0_ + t == 27)
                            resid_sample(qcur_ + (size_t) t * NH * HD, (int64_t) NH * HD, val4_d_, 2, kValSamples, cs_);
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                 s, scores_ + (size_t) tb * max_blocks_, cs);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, cs);
                for (int t = tb; t < te; ++t) rhash(l, 16, t, (const float*) (sel_ + (size_t) t * cap_), cap_);
                if (val3_d_) {
                    for (int t = tb; t < te; ++t) {
                        resid_sample((const float*) (sel_ + (size_t) t * cap_), 64, val3_d_,
                                     (int) ((l * max_t_ + t) * 2 + 0), kValSamples, cs_);
                        resid_sample((const float*) (step_ + t * kStepCount), 4,
                                     val3_d_ + ((size_t) (l * max_t_ + t) * 2 + 1) * kValSamples, 0, 4, cs_);
                    }
                }
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                const QsaAttnPools pools = qsa_attn_pools(st);
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                for (int t = tb; t < te; ++t) rhash(l, 17, t, attn_ + (size_t) t * NH * HD, NH * HD);
                if (val2_d_)
                    for (int t = tb; t < te; ++t)
                        resid_sample(attn_ + (size_t) t * NH * HD, (int64_t) NH * HD, val2_d_,
                                     (int) ((l * max_t_ + t) * 8 + 0), kValSamples, cs_);
                if (val4_d_ && l == 3)
                    for (int t = tb; t < te; ++t) {
                        if (last_pos0_ + t == 27)
                            resid_sample(attn_ + (size_t) t * NH * HD, (int64_t) NH * HD, val4_d_, 0, kValSamples, cs_);
                        if (last_pos0_ + t == 26)
                            resid_sample(attn_ + (size_t) t * NH * HD, (int64_t) NH * HD, val4_d_, 3, kValSamples, cs_);
                    }
                stamp(l, 13, grp);
                if (st.kv_q4 || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t) {
                    if (native_qsa_enabled())
                        native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                              (int) NH, (int) HD, cs);
                    else
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                }
                stamp(l, 14, grp);
                for (int t = tb; t < te; ++t) rhash(l, 20, t, attn32_ + (size_t) t * NH * HD, NH * HD);
                if (val2_d_)
                    for (int t = tb; t < te; ++t)
                        resid_sample(attn32_ + (size_t) t * NH * HD, (int64_t) NH * HD, val2_d_,
                                     (int) ((l * max_t_ + t) * 8 + 1), kValSamples, cs_);
                if (val4_d_ && l == 3)
                    for (int t = tb; t < te; ++t)
                        if (last_pos0_ + t == 27)
                            resid_sample(attn32_ + (size_t) t * NH * HD, (int64_t) NH * HD, val4_d_, 1, kValSamples, cs_);
                native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs);
                if (hash_on_) {   // xq_ rows are group-relative (the quantize call writes at the base)
                    const size_t rowb = native_q8_1_bytes((int) (NH * HD), n) / (size_t) n;
                    for (int t = tb; t < te; ++t)
                        rhash(l, 21, t, (const float*) (xq_ + (size_t) (t - tb) * rowb), (int64_t) (rowb / 4));
                }
                native_mmvq(wo->native_type, wo->native_data, xq_, bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);
                if (val2_d_)
                    for (int t = tb; t < te; ++t)
                        resid_sample(bo_ + (size_t) t * N, (int64_t) N, val2_d_,
                                     (int) ((l * max_t_ + t) * 8 + 2), kValSamples, cs_);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        for (int t = tb; t < te; ++t) rhash(l, 2, t, bo_ + t * N, N);
        gr_read_group(1, true, inj_, inj2_);
        for (int t = tb; t < te; ++t) { rhash(l, 3, t, Rt(t), HC * N); rhash(l, 5, t, mixed_ + t * N, N); }
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        if (dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && NE == 512 && K == 10) {
            try {
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        for (int t = tb; t < te; ++t) rhash(l, 6, t, (const float*) (ids_ + t * K), K);
        if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                          (uint32_t) ((l - lb_) * G + grp + 1), cs);
        doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                         m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
        stamp(l, 17, grp);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = xq_;
            if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, cs);   // contiguous rows
            else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, cs);
            try {
                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                    sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, cs);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
        }
        for (int t = tb; t < te; ++t) rhash(l, 7, t, shared_ + t * N, N);
        if (strata::kernels::cpu::expert_layout().native) {
            quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            for (int t = tb; t < te; ++t)
                rhash(l, 9, t, (const float*) (nat_xq_ + (size_t) t * (N / 32) * 36), (N / 32) * 9);
        }
        else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
            wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
            copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
        } else {
            wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's GPU plan
            copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
        }
        stamp(l, 19, grp);
        if (hash_on_) {
            rhash(l, 10, tb, (const float*) (pl + 4 + capx + 1), cap);
            rhash(l, 11, tb, (const float*) (pl + 4 + capx + 1 + capx), cap);
            rhash(l, 12, tb, (const float*) pl, 4);
        }
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.
        // On Metal the native arm calls the based form: this lambda runs INSIDE the captured window body,
        // where a host sync or readback of the plan would deadlock the body against its own flag spins -
        // the based kernels read the plan on the device and address each blob as an offset inside the one
        // allocation that holds them all (the expert-cache arena for the VRAM groups; this group's staging
        // region for the PCIe groups). The S2 arm's grouped kernels already read their plan on the device.
        const int64_t stage_per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
        uint8_t* const stage_base = staging_ + (size_t) (grp * stage_per) * lay.max_blob;
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn,
                           const void* blob_base, uint64_t blob_span) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
#ifdef STRATA_USE_METAL
                native_expert_grouped_based(L, blob_base, blob_span, gp, gs, gn, p_dst, p_tok, cap, cap,
                                            nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, hit_out, cs);
#else
                (void) blob_base;
                (void) blob_span;
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, hit_out, cs);
#endif
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
            }
        };
        const uint64_t cache_span =
            hits_.n_slots > 0 && hits_.blob > 0
                ? (hits_.slot_off ? (uint64_t) hits_.slot_off[hits_.n_slots - 1] + (uint64_t) hits_.blob
                                  : (uint64_t) hits_.n_slots * (uint64_t) hits_.blob)
                : ~0ull;
        grouped(p_ptr, p_start, p_counts, hits_.cache_base, cache_span);
        stamp(l, 20, grp);
        if (device_plan_) wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs);
        else wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe share is in staging (DMA) or mapped
        // Stage the PCIe share, then point at staging. Mode 2 stages with a copy kernel (fetch_blobs);
        // mode 1 (direct) reads the mapped aliases in place on CUDA. On Metal the aliases live in
        // unregistered host memory a kernel can neither bind nor dereference, so mode 1 takes the staged
        // route too - the pool-side DMA has already landed the blobs (fetch_blobs is a no-op there, its
        // staging owned by Verifier::fetch_dma; rebase_ptrs does the pointing, in stream order).
        bool stage_pcie = sink_.pcie_mode == 2;
#ifdef STRATA_USE_METAL
        if (sink_.pcie_mode == 1) stage_pcie = true;
#endif
        if (stage_pcie) {
            fetch_blobs(p_ptr2, p_counts + 2, stage_base, (int64_t) lay.blob_bytes(l), (int) stage_per, cs);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage_base, (int64_t) lay.blob_bytes(l), cs);
        }
        stamp(l, 21, grp);
        grouped(p_ptr2, p_start2, p_counts + 2, stage_base, (uint64_t) stage_per * (uint64_t) lay.max_blob);
        stamp(l, 22, grp);
        if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
            wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
            copy_or_zero_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                     skip_ + grp, ring, cs);
        } else {
            wait_flag_ge(m_flag_, ring, cs);               // the CPU's share is in the mapped rows
            stamp(l, 23, grp);
            if (dec_batch)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                copy_rows_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                      p_dst, p_counts + 1, cs);
            else
                copy_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
        }
        for (int t = tb; t < te; ++t) rhash(l, 13, t, parts_ + (size_t) t * K * N, K * N);
        for (int t = tb; t < te; ++t)
            for (int e = 0; e < K; ++e) rhash_row(hash13_d_, l, t, e, parts_ + (size_t) (t * K + e) * N, N);
        if (hash_on_ && snap_hit_ && l == 0 && snap_win_ != hash_windows_) {
            snap_win_ = hash_windows_;
            cudaMemcpyAsync(snap_hit_, hit_out, (size_t) (te - tb) * K * N * 4, cudaMemcpyDeviceToDevice, cs);
        }
        for (int t = tb; t < te; ++t) rhash(l, 14, t, hit_out + (size_t) (t - tb) * K * N, K * N);
        moe_hit_add(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, cap, N, cs);
        for (int t = tb; t < te; ++t) rhash(l, 8, t, parts_ + (size_t) t * K * N, K * N);
        for (int t = tb; t < te; ++t)
            for (int e = 0; e < K; ++e) rhash_row(hash8_d_, l, t, e, parts_ + (size_t) (t * K + e) * N, N);
        if (hash_on_ && snap_parts_ && l == 0 && snap_win_ == hash_windows_)
            cudaMemcpyAsync(snap_parts_, parts_ + (size_t) tb * K * N, (size_t) (te - tb) * K * N * 4, cudaMemcpyDeviceToDevice, cs);
        if (dec_batch && n > 1 && native_moe_combine_enabled()) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        stamp(l, 24, grp);
        for (int t = tb; t < te; ++t) rhash(l, 4, t, bo_ + t * N, N);
        if (l == g.n_layers - 1) {
            for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
            if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };

    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        for (int t = 0; t < T; ++t) {
            copy_from_mapped(hand_out_ + (size_t) t * HB, Rt(t), HC * N, cs);
            copy_from_mapped(hand_out_ + (size_t) t * HB + HC * N, bo_ + (size_t) t * N, N, cs);
            copy_from_mapped(hand_out_ + (size_t) t * HB + HC * N + N, inj2_ + (size_t) t * HC, HC, cs);
        }
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        for (int t = 0; t < T; ++t) {
            BlockBuffers bb = ss.block;
            bb.R = Rt(t);
            bb.mixed = head_mixed_ + t * N;
            if (head_ != nullptr && head_->loaded()) {
                if (!lm_head_mix(wt, g, bb, cs, err)) return false;
            } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                return false;
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

bool Verifier::capture(int T, std::string& err) {
    if (exec_[T] != nullptr) return true;
#ifdef STRATA_USE_METAL
    {
        // No stream capture on Metal: store the window as a body that re-runs
        // record_window on this verifier's stream at launch, and mint a real
        // (empty) exec for the slot. record_window reads its per-window values
        // from the mapped staging the host refills before every launch
        // (run()), never from capture-time arguments, so invoking the body
        // computes exactly what a replay would.
        window_body_[T] = [this, T] {
            std::string e;
            if (!record_window(T, cs_, e)) verify_body_fail("window", e);
        };
        exec_[T] = verify_mint_empty_exec(cs_);
        if (!exec_[T]) { err = "verify: window capture (Metal)"; return false; }
        window_names_pending_[T] = std::getenv("STRATA_VERIFY_NODES") != nullptr;
        return true;
    }
#endif
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
#if !defined(STRATA_USE_HIP)   // a CUDA debug listing (node types, kernel names)
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            cudaGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) == cudaSuccess) {
                    const char* fn = nullptr;
                    // cudaFuncGetName arrived in CUDA 12.3; 12.0/12.2 still list the node type.
#if CUDART_VERSION >= 12030
                    if (cudaFuncGetName(&fn, kp.func) == cudaSuccess && fn) name = fn;
#else
                    (void) fn;
#endif
                }
            } else if (ty == cudaGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == cudaGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "maya verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
    }
#endif
    const cudaError_t ie = cudaGraphInstantiate(&exec_[T], graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_[T], cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "maya verify: captured the %d-token window (upload %s, sync %s)\n", T,
                 cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

// The commit sequence itself, separate from its capture for the same reason
// record_window is: on Metal there is no capture, so capture_commit stores
// THIS as a body and the graph path records it.
bool Verifier::record_commit(std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) return false;
                float* state = ss.gdn_state + (size_t) gdn_index * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_);
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) return false;
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int64_t t = 0; t < MT; ++t)
                    native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              (float) qsa_freq_base(), cs_);
                ++qsa_index;
            }
        }
        if (ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        return false;
    }
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
#ifdef STRATA_USE_METAL
    {
        // No stream capture on Metal: store the commit sequence as a body and
        // mint a real (empty) exec for the slot (the capture() arrangement).
        // The body reads its inputs from the mapped commit staging the host
        // refills before every launch (commit()), so invoking it is the replay.
        commit_body_ = [this] {
            std::string e;
            if (!record_commit(e)) verify_body_fail("commit", e);
        };
        commit_exec_ = verify_mint_empty_exec(cs_);
        if (!commit_exec_) { err = "verify: commit capture (Metal)"; return false; }
        return true;
    }
#endif
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    const bool ok = record_commit(err);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[0].max_cells) { err = "verify: the window runs past the context"; return false; }
    // (v1.1.32) Windows' HIP runtime submits queued GPU work only at a host wait (#133's lesson): a window that does
    // not continue the last committed one - the first after a prompt was read, a conversation taken back - waits for
    // the device first, so it reads what the prompt path and the expert-cache refills wrote.  MAYA_QWEN_SYNC: 0 off,
    // 1 that (the default on Windows + HIP), 2 before every window (a diagnosis arm)
    static const int sync_mode = [] {
        if (const char* v = std::getenv("MAYA_QWEN_SYNC")) return std::atoi(v);
#if defined(_WIN32) && defined(STRATA_USE_HIP)
        return 1;
#else
        return 0;
#endif
    }();
    if (sync_mode >= 2 || (sync_mode == 1 && pos0 != next_pos_)) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) { err = std::string("verify: device sync: ") + cudaGetErrorString(se); return false; }
    }
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    if (ss.ple.ready() && ple_stage()) {
        uint32_t rows[kVerifyMaxT * PLE_N_HEADS];
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows, (size_t) T, h_ple_, err)) return false;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
#ifdef STRATA_USE_METAL
    std::vector<std::string> launch_names;
    std::vector<std::string>* names_p = nullptr;
    if (window_names_pending_[T]) {
        window_names_pending_[T] = false;
        names_p = &launch_names;
    }
#else
    std::vector<std::string>* names_p = nullptr;
    (void) names_p;
#endif
    if (!VERIFY_LAUNCH(exec_[T], &window_body_[T], cs_, names_p)) {
        err = std::string("verify: launch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
#ifdef STRATA_USE_METAL
    if (names_p != nullptr) verify_print_node_accounting(launch_names, T);
#endif
    (void) cudaStreamQuery(cs_);
    VDBG("launched\n");
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    for (int64_t k = 0; k < (le_ - lb_) * G; ++k) {
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            strata_cpu_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) { err = "verify: timed out at layer " + std::to_string(l); return false; }
        }
        const Clock::time_point b = Clock::now();
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (hash_on_ && l == lb_ && grp == 0 && windows < 6) {   // GRFOLD debug: the real layer-0 plan
            std::fprintf(stderr, "PLAN win=%lld pos0=%lld grp0 n=%d ids:", (long long) windows, (long long) last_pos0_, n);
            for (int i = 0; i < n * (int) ss.k; ++i) std::fprintf(stderr, " %d", h_ids_[(size_t) tb * ss.k + i]);
            std::fprintf(stderr, " counts: %d %d %d %d start:", sink_.counts[0], sink_.counts[1], sink_.counts[2], sink_.counts[3]);
            for (int i = 0; i <= sink_.counts[0]; ++i) std::fprintf(stderr, " %d", sink_.start[i]);
            std::fprintf(stderr, " start2:");
            for (int i = 0; i <= sink_.counts[2]; ++i) std::fprintf(stderr, " %d", sink_.start2[i]);
            std::fprintf(stderr, " dst:");
            for (int i = 0; i < sink_.counts[1]; ++i) std::fprintf(stderr, " %d", sink_.dst[i]);
            std::fprintf(stderr, " tok:");
            for (int i = 0; i < sink_.counts[1]; ++i) std::fprintf(stderr, " %d", sink_.tok[i]);
            std::fprintf(stderr, " ptr_off:");
            for (int i = 0; i < sink_.counts[0]; ++i)
                std::fprintf(stderr, " %lld", (long long) (sink_.ptr[i] - (unsigned long long) (uintptr_t) hits_.cache_base));
            std::fprintf(stderr, " ptr2_raw:");
            for (int i = 0; i < sink_.counts[2]; ++i) std::fprintf(stderr, " %llx", (unsigned long long) sink_.ptr2[i]);
            std::fprintf(stderr, "\n");
            if (last_pos0_ == 26 || last_pos0_ == 27) {
                const float* ym = h_ymiss_ + (size_t) tb * ss.k * g.n_embd;
                for (int i = 0; i < n * (int) ss.k; ++i) {
                    const float* r = ym + (size_t) i * g.n_embd;
                    double s = 0;
                    for (int64_t j = 0; j < g.n_embd; ++j) s += r[j];
                    std::fprintf(stderr, "YMISS win=%lld row%d [%g %g %g %g] sum=%g\n", (long long) windows, i, r[0], r[1], r[2], r[3], s);
                }
            }
        }
        VDBG("layer %lld served\n", (long long) l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        strata_store_fence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
    if (prof_on_ && G == 1) {       // the window's GPU stage stamps
        cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
        const int64_t L = g.n_layers;
        auto at = [&](int64_t l, int i) { return prof_h_[(size_t) (l * kProfPer + i)]; };
        for (int64_t l = 0; l < L; ++l) {
            const int kind = is_qsa_layer(g, l) ? 1 : 0;
            unsigned long long prev = at(l, 0);
            for (int i = 1; i <= 24; ++i) {
                const unsigned long long x = at(l, i);
                if (x == 0 || x < prev) continue;
                prof_sum_[kind][i] += (double) (x - prev);
                prev = x;
            }
            if (l + 1 < L) prof_sum_[kind][25] += (double) (at(l + 1, 0) - at(l, 24));
            prof_sum_[kind][27] += (double) (at(l, 27) - at(l, 0));    // hc-read0: norm
            prof_sum_[kind][28] += (double) (at(l, 28) - at(l, 27));   //           down
            prof_sum_[kind][29] += (double) (at(l, 1) - at(l, 28));    //           up
            prof_sum_[kind][1] -= (double) (at(l, 1) - at(l, 0));      // (hc-read0 shown split)
        }
        prof_sum_[0][26] += (double) (at(L, 1) - at(L, 0));
        ++prof_windows_;
    }
    if (hash_on_ && hash_windows_ < 40) {   // GRFOLD debug: this window's residual fingerprints
        ++hash_windows_;
        if (cudaMemcpy(hash_h_.data(), hash_d_, hash_h_.size() * 8, cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::fprintf(stderr, "RESID window %lld pos0=%lld T=%d layers[%lld,%lld)\n", (long long) hash_windows_,
                         (long long) pos0, T, (long long) lb_, (long long) le_);
            for (int64_t l = lb_; l < le_; ++l) {
                char line[2048];
                int off = std::snprintf(line, sizeof line, "RESID L%lld", (long long) l);
                for (int st = 0; st < kResidStages; ++st)
                    for (int t = 0; t < T; ++t)
                        off += std::snprintf(line + off, sizeof line - (size_t) off, " s%dt%d=%016llx", st, t,
                                             (unsigned long long) hash_h_[((l * kResidStages) + st) * max_t_ + t]);
                std::fprintf(stderr, "%s\n", line);
            }
            if (val_on_ && cudaMemcpy(val_h_.data(), val_d_, val_h_.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
                for (int64_t l = lb_; l < le_; ++l)
                    for (int t = 0; t < T; ++t) {
                        std::fprintf(stderr, "RESIDVAL pos=%lld L%lld t%d", (long long) (pos0 + t), (long long) l, t);
                        for (int k = 0; k < kValSamples; ++k)
                            std::fprintf(stderr, " %.9g", (double) val_h_[((size_t) (l * max_t_) + t) * kValSamples + k]);
                        std::fprintf(stderr, "\n");
                    }
            if (val2_d_ && cudaMemcpy(val2_h_.data(), val2_d_, val2_h_.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
                for (int64_t l = lb_; l < le_; ++l)
                    for (int t = 0; t < T; ++t)
                        for (int w = 0; w < 8; ++w) {
                            std::fprintf(stderr, "RESIDVAL2 pos=%lld L%lld t%d w%d", (long long) (pos0 + t), (long long) l, t, w);
                            for (int k = 0; k < kValSamples; ++k)
                                std::fprintf(stderr, " %.9g",
                                             (double) val2_h_[(((size_t) (l * max_t_) + t) * 8 + w) * kValSamples + k]);
                            std::fprintf(stderr, "\n");
                        }
            if (val3_d_ && cudaMemcpy(val3_h_.data(), val3_d_, val3_h_.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
                for (int64_t l = lb_; l < le_; ++l)
                    for (int t = 0; t < T; ++t) {
                        const float* selb = &val3_h_[((size_t) (l * max_t_) + t) * 2 * kValSamples];
                        const float* stpb = selb + kValSamples;
                        std::fprintf(stderr, "RESIDVAL3 pos=%lld L%lld t%d step=%d,%d,%d,%d sel=", (long long) (pos0 + t),
                                     (long long) l, t, ((const int32_t*) stpb)[0], ((const int32_t*) stpb)[1],
                                     ((const int32_t*) stpb)[2], ((const int32_t*) stpb)[3]);
                        for (int k = 0; k < 32; ++k) std::fprintf(stderr, " %d", ((const int32_t*) selb)[k]);
                        std::fprintf(stderr, "\n");
                    }
            if (val4_d_ && pos0 <= 27 && 27 < pos0 + T &&
                cudaMemcpy(val4_h_.data(), val4_d_, val4_h_.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                static const char* nm[4] = {"attn27", "attn32_27", "qcur27", "attn26"};
                for (int w = 0; w < 4; ++w) {
                    std::fprintf(stderr, "RESIDVAL4 %s", nm[w]);
                    for (int k = 0; k < kValSamples; ++k)
                        std::fprintf(stderr, " %.9g", (double) val4_h_[(size_t) w * kValSamples + k]);
                    std::fprintf(stderr, "\n");
                }
            }
            if (ss.qsa_states) {
                // GRFOLD: raw pool bytes, QSA layer 0 (L3), logical page 6.
                const QsaState& st0 = ss.qsa_states[0];
                const int64_t PS = shapes_of(g).page_size, NKVv = g.n_head_kv, HDv = g.head_dim;
                int32_t pt[16];
                if (st0.page_table && st0.k_pool && st0.v_pool &&
                    cudaMemcpy(pt, st0.page_table, sizeof pt, cudaMemcpyDeviceToHost) == cudaSuccess && pt[6] >= 0) {
                    const int64_t phys = pt[6];
                    std::vector<uint16_t> pk((size_t) NKVv * PS * HDv), pv((size_t) NKVv * PS * HDv);
                    if (cudaMemcpy(pk.data(), st0.k_pool + (size_t) phys * NKVv * PS * HDv, pk.size() * 2,
                                   cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(pv.data(), st0.v_pool + (size_t) phys * NKVv * PS * HDv, pv.size() * 2,
                                   cudaMemcpyDeviceToHost) == cudaSuccess) {
                        auto f16 = [](uint16_t x) -> float {
                            const float mag = std::ldexp((float) (1024 + (x & 0x3ff)), (int) ((x >> 10) & 0x1f) - 25);
                            const float sub = (float) (x & 0x3ff) * 5.96046448e-08f;
                            const float v = ((x >> 10) & 0x1f) == 0 ? sub : mag;
                            return (x >> 15) ? -v : v;
                        };
                        for (int64_t cell = 0; cell < PS; ++cell)
                            for (int64_t h = 0; h < NKVv; ++h) {
                                const uint16_t* kr = &pk[((size_t) h * PS + cell) * HDv];
                                const uint16_t* vr = &pv[((size_t) h * PS + cell) * HDv];
                                double ks = 0, vs = 0;
                                for (int64_t d = 0; d < HDv; ++d) { ks += f16(kr[d]); vs += f16(vr[d]); }
                                std::fprintf(stderr,
                                             "RESIDPOOL pos0=%lld cell=%lld h=%lld K sum=%.9g f4=%.9g,%.9g,%.9g,%.9g V "
                                             "sum=%.9g f4=%.9g,%.9g,%.9g,%.9g\n",
                                             (long long) pos0, (long long) (6 * PS + cell), (long long) h, ks,
                                             (double) f16(kr[0]), (double) f16(kr[1]), (double) f16(kr[2]),
                                             (double) f16(kr[3]), vs, (double) f16(vr[0]), (double) f16(vr[1]),
                                             (double) f16(vr[2]), (double) f16(vr[3]));
                            }
                    }
                }
            }
            if (snap_hit_ && last_pos0_ == 26) {
                const int64_t kk = ss.k, nn = g.n_embd;
                std::vector<float> sh((size_t) max_t_ * kk * nn), sp((size_t) max_t_ * kk * nn);
                if (cudaMemcpy(sh.data(), snap_hit_, sh.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    cudaMemcpy(sp.data(), snap_parts_, sp.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    for (int t = 0; t < T; ++t)
                        for (int e = 0; e < kk; ++e) {
                            const float* hr = &sh[(size_t) (t * kk + e) * nn];
                            const float* pr = &sp[(size_t) (t * kk + e) * nn];
                            double hs = 0, ps = 0;
                            for (int64_t i = 0; i < nn; ++i) { hs += hr[i]; ps += pr[i]; }
                            std::fprintf(stderr, "VAL t%d e%d hit[%g %g %g %g] sum=%g parts[%g %g %g %g] sum=%g\n", t, e,
                                         hr[0], hr[1], hr[2], hr[3], hs, pr[0], pr[1], pr[2], pr[3], ps);
                        }
                }
            }
            if (hash8_d_ && hash_windows_ <= 6 &&
                cudaMemcpy(hash8_h_.data(), hash8_d_, hash8_h_.size() * 8, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(hash13_h_.data(), hash13_d_, hash13_h_.size() * 8, cudaMemcpyDeviceToHost) == cudaSuccess) {
                const int64_t kk = ss.k;
                for (int64_t l = lb_; l < le_ && l < lb_ + 1; ++l)
                    for (int t = 0; t < T; ++t)
                        for (int e = 0; e < kk; ++e) {
                            const size_t idx = (size_t) ((l * max_t_ + t) * kk + e);
                            std::fprintf(stderr, "RP L%lld t%d e%d final=%016llx copied=%016llx\n", (long long) l, t, e,
                                         (unsigned long long) hash8_h_[idx], (unsigned long long) hash13_h_[idx]);
                        }
            }
        }
    }
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            cudaMemcpy(h.data(), head_logits_, h.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "maya dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(v->copy_, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    strata_store_fence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

bool Verifier::commit(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!VERIFY_LAUNCH(commit_exec_, &commit_body_, cs_, nullptr)) {
        err = std::string("verify: commit launch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    next_pos_ = last_pos0_ + n_keep;
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err);
}

}  // namespace strata::core
