// src/metal-qwen/stubs/glm_model.cpp - STAGE 1 STUBS for strata::core::Glm5Model
// (include/strata/core/glm_model.hpp).
//
// The GLM runner's real definitions live in the glm_*.cu translation units, which a Metal
// build does not compile (device code; RISKS.md SS3). src/program/generate.cpp still links
// the class through its --glm-pack path, so the Metal build needs definitions for every
// out-of-line method. Work-performing methods print their name and abort, exactly like the
// kernel stubs: a GLM pack on this build stops loudly instead of silently doing nothing.
// The exceptions are honest by construction and marked SAFE below: the destructor and
// cleanup no-ops (a stub model never allocated anything), capability probes (false / 0 /
// empty is their truthful answer for an unimplemented engine, cf. RISKS.md SS4), trivial
// accessors over members that stay at their defaults, and tuning setters for machinery
// that does not exist here. Glm5Geometry::from_gguf and Glm5Model::set_routing_dump fail
// through their error channel instead: their callers check the return before anything
// runs, so err + false IS the loud stop there.
//
// Work done with Muse AI assistance.

#include "strata/core/glm_model.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {
namespace {
[[noreturn]] void glm_stub(const char* name) {
    std::fprintf(stderr, "strata metal: unimplemented Glm5Model entry: %s\n", name);
    std::abort();
}
}  // namespace

Glm5Model::~Glm5Model() {}
bool Glm5Model::load(const std::string& gguf_path, int64_t max_ctx, std::string& err) { glm_stub("bool Glm5Model::load"); }
bool Glm5Model::load_pack(const std::string& pack_dir, int64_t max_ctx, std::string& err, int dev, int l0, int l1) { glm_stub("bool Glm5Model::load_pack"); }
bool Glm5Model::load_pack_env(const std::string& pack_dir, int64_t max_ctx, std::string& err, const std::string& layer_split) { glm_stub("bool Glm5Model::load_pack_env"); }
bool Glm5Model::load_pack_place(const std::string& pack_dir, int64_t max_ctx, std::string& err, const std::string& layer_split) { glm_stub("bool Glm5Model::load_pack_place"); }
bool Glm5Model::load_pack_split(const std::string& pack_dir, int64_t max_ctx, const std::vector<int>& bounds, const std::vector<int>& devs, std::string& err) { glm_stub("bool Glm5Model::load_pack_split"); }
Glm5Model::FastStats Glm5Model::fast_stats() const { return FastStats{}; }
std::vector<Glm5Model::VramReport> Glm5Model::vram_report() const { return {}; }
std::vector<Glm5Model::PoolVram> Glm5Model::pool_vram() const { return {}; }
std::pair<size_t, size_t> Glm5Model::commit_capped() const { return {0, 0}; }
double Glm5Model::pcie_share() const { return 0.0; }
void Glm5Model::lane_drift_check() {}
void Glm5Model::lane_retime() {}
bool Glm5Model::set_pcie_share(double share) { (void)share; return false; }
int Glm5Model::cpu_lane_threads() const { return 0; }
int Glm5Model::cpu_lane_threads_max() const { return 0; }
bool Glm5Model::set_cpu_lane_threads(int n) { (void)n; return false; }
void Glm5Model::set_read_chunks(int n) { (void)n; }
void Glm5Model::set_miss_skip(double w, int from) { (void)w; (void)from; }
bool Glm5Model::snapshot_save() { glm_stub("bool Glm5Model::snapshot_save"); }
bool Glm5Model::save_usage() { glm_stub("bool Glm5Model::save_usage"); }
std::string Glm5Model::usage_path() const { return {}; }
bool Glm5Model::save_expert_profile(const std::string& path, std::string& err) { glm_stub("bool Glm5Model::save_expert_profile"); }
bool Glm5Model::snapshot_restore() { glm_stub("bool Glm5Model::snapshot_restore"); }
bool Glm5Model::ckpt_save(int slot, std::string& err) { glm_stub("bool Glm5Model::ckpt_save"); }
bool Glm5Model::ckpt_restore(int slot, std::string& err) { glm_stub("bool Glm5Model::ckpt_restore"); }
bool Glm5Model::ckpt_save_snapshot(int slot, std::string& err) { glm_stub("bool Glm5Model::ckpt_save_snapshot"); }
uint64_t Glm5Model::slot_save(const std::string& path, std::string& err) { glm_stub("uint64_t Glm5Model::slot_save"); }
bool Glm5Model::slot_load(const std::string& path, int64_t n_pos, std::string& err) { glm_stub("bool Glm5Model::slot_load"); }
uint64_t Glm5Model::slot_save(std::vector<uint8_t>& data, std::string& err) { glm_stub("uint64_t Glm5Model::slot_save"); }
bool Glm5Model::slot_load(const std::vector<uint8_t>& data, int64_t n_pos, std::string& err) { glm_stub("bool Glm5Model::slot_load"); }
int Glm5Model::sample_token(strata::kernels::SamplerParams& sp, std::string& err) { glm_stub("int Glm5Model::sample_token"); }
const float* Glm5Model::logit_bias_table(int slot, const std::vector<std::pair<int32_t, float>>& bias, std::string& err) { glm_stub("const float* Glm5Model::logit_bias_table"); }
void Glm5Model::pick_tokens(const float* logits, int T, const strata::kernels::SamplerParams& sp, const int32_t* win, int n0, int* out, cudaStream_t s) { glm_stub("void Glm5Model::pick_tokens"); }
void Glm5Model::reset() { glm_stub("void Glm5Model::reset"); }
bool Glm5Model::forward(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err) { glm_stub("bool Glm5Model::forward"); }
bool Glm5Model::vision_lend(size_t& bytes, std::string& err) { glm_stub("bool Glm5Model::vision_lend"); }
bool Glm5Model::vision_reclaim(std::string& err) { glm_stub("bool Glm5Model::vision_reclaim"); }
size_t Glm5Model::vision_lend_bytes() const { return 0; }
bool Glm5Model::step(int32_t token, std::string& err) { glm_stub("bool Glm5Model::step"); }
bool Glm5Model::head_split_link(Glm5Model* helper, std::string& err) { glm_stub("bool Glm5Model::head_split_link"); }
bool Glm5Model::win_head_split(int t, int sl, const float* hx, float* lg, std::string& err) { glm_stub("bool Glm5Model::win_head_split"); }
bool Glm5Model::step_layers(int64_t p, std::string& err) { glm_stub("bool Glm5Model::step_layers"); }
bool Glm5Model::step_issue(int32_t token, int64_t& p, std::string& err) { glm_stub("bool Glm5Model::step_issue"); }
bool Glm5Model::forward_pipeline(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err) { glm_stub("bool Glm5Model::forward_pipeline"); }
cudaError_t Glm5Model::pack_moe_tail(std::string& err, const std::vector<int>& ids, const std::vector<float>& rw, float* A, int layer) { glm_stub("cudaError_t Glm5Model::pack_moe_tail"); }
void Glm5Model::pack_release_views() {}
void Glm5Model::stage_worker() { glm_stub("void Glm5Model::stage_worker"); }
void Glm5Model::stage_join() {}
void Glm5Model::pool_ensure(int layer, const int* ids, int k) { glm_stub("void Glm5Model::pool_ensure"); }
int32_t Glm5Model::pool_evict(int cls) { glm_stub("int32_t Glm5Model::pool_evict"); }
bool Glm5Model::pack_moe_tail_device(std::string& err, const std::vector<int>& ids, const std::vector<float>& rw, float* A, int layer) { glm_stub("bool Glm5Model::pack_moe_tail_device"); }
bool Glm5Model::pen_alloc(int w, std::string& err) { glm_stub("bool Glm5Model::pen_alloc"); }
void Glm5Model::pen_free() {}
bool Glm5Model::fast_setup(std::string& err) { glm_stub("bool Glm5Model::fast_setup"); }
bool Glm5Model::fast_setup_finish(std::string& err) { glm_stub("bool Glm5Model::fast_setup_finish"); }
bool Glm5Model::dual_prefill_ready() { return false; }
bool Glm5Model::calc_prefill_ready() { return false; }
void Glm5Model::ram_left_check() const { glm_stub("void Glm5Model::ram_left_check"); }
void Glm5Model::fast_destroy() {}
bool Glm5Model::forward_fast(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err) { glm_stub("bool Glm5Model::forward_fast"); }
bool Glm5Model::fast_token(int32_t token, std::string& err) { glm_stub("bool Glm5Model::fast_token"); }
bool Glm5Model::fast_layers(int64_t p, bool hop_in, std::string& err) { glm_stub("bool Glm5Model::fast_layers"); }
bool Glm5Model::fast_dsa(int il, int64_t p, std::string& err) { glm_stub("bool Glm5Model::fast_dsa"); }
bool Glm5Model::fast_moe(int il, bool& pf_pending, std::string& err, bool cpu_wait) { glm_stub("bool Glm5Model::fast_moe"); }
bool Glm5Model::fast_moe_experts(int il, const MoeIo& io, bool pred, int n_ah, const float* const* ah_bias, bool& pf_pending, std::string& err, bool rows) { glm_stub("bool Glm5Model::fast_moe_experts"); }
void Glm5Model::fast_moe_route(int il, const MoeIo& io, const strata::kernels::glmf::MoeDev& md, void* sh_hq, bool pred, int n_ah, const float* const* ah_bias, bool lane, int win, int win_n, int win_rows) { glm_stub("void Glm5Model::fast_moe_route"); }
void Glm5Model::fast_service() {}
bool Glm5Model::rows_moe_grouped(int il, int n, std::string& err) { glm_stub("bool Glm5Model::rows_moe_grouped"); }
void Glm5Model::fast_read_part(int il, int e, int role, uint8_t* blob, int chunk, int n_chunks) { glm_stub("void Glm5Model::fast_read_part"); }
void Glm5Model::fast_read_expert_x3(int il, int e, uint8_t* blob) { glm_stub("void Glm5Model::fast_read_expert_x3"); }
void Glm5Model::fast_ahead_route(int il, const int* ids, unsigned int miss, const short (*ahead)[8]) { glm_stub("void Glm5Model::fast_ahead_route"); }
void Glm5Model::fast_ahead_reader() { glm_stub("void Glm5Model::fast_ahead_reader"); }
bool Glm5Model::fast_boundary(std::string& err) { glm_stub("bool Glm5Model::fast_boundary"); }
int Glm5Model::fast_sample(strata::kernels::SamplerParams& sp, std::string& err) { glm_stub("int Glm5Model::fast_sample"); }
int Glm5Model::fast_sample_verify(strata::kernels::SamplerParams& sp, std::string& err) { glm_stub("int Glm5Model::fast_sample_verify"); }
bool Glm5Model::kv_stream_setup(int64_t cells, std::string& err) { glm_stub("bool Glm5Model::kv_stream_setup"); }
Glm5Model::KvStream* Glm5Model::kv_stream_make(int64_t cells, int64_t ctx, std::string& err) { glm_stub("Glm5Model::KvStream* Glm5Model::kv_stream_make"); }
void Glm5Model::kv_stream_free(KvStream* K) { (void)K; }
void Glm5Model::kv_stream_reset_maps() {}
std::vector<int> Glm5Model::dsa_layers() const { return {}; }
void Glm5Model::snap_pool_copy(bool restore) { glm_stub("void Glm5Model::snap_pool_copy"); }
void Glm5Model::ckpt_pool_copy(int64_t pos, float* host, bool restore) { glm_stub("void Glm5Model::ckpt_pool_copy"); }
bool Glm5Model::fast_warm(std::string& err) { glm_stub("bool Glm5Model::fast_warm"); }
bool Glm5Model::fast_cpu_lane_setup(std::string& err) { glm_stub("bool Glm5Model::fast_cpu_lane_setup"); }
bool Glm5Model::fast_helper_setup(std::string& err) { glm_stub("bool Glm5Model::fast_helper_setup"); }
bool Glm5Model::fast_helper_adapt(std::string& err) { glm_stub("bool Glm5Model::fast_helper_adapt"); }
bool Glm5Model::fast_helper_check(std::string& err) { glm_stub("bool Glm5Model::fast_helper_check"); }
void Glm5Model::fast_cpu_experts(int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out) { glm_stub("void Glm5Model::fast_cpu_experts"); }
void Glm5Model::lane_experts(glmfast::Workers& pool, uint8_t* act, uint8_t* hq, float* ff, float* dn, float* x3, int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out) { glm_stub("void Glm5Model::lane_experts"); }
void Glm5Model::win_lanes_stop() {}
void Glm5Model::win_lanes_make(std::shared_ptr<glmfast::Workers> p0, std::shared_ptr<glmfast::Workers> p1, const std::vector<int>& c0, const std::vector<int>& c1) { glm_stub("void Glm5Model::win_lanes_make"); }
void Glm5Model::win_cpu_merged(int il, int nb, const strata::kernels::glmf::MoeRequest* const* rq, strata::kernels::glmf::CpuAnswer* const* ans, const unsigned int* seq) { glm_stub("void Glm5Model::win_cpu_merged"); }
bool Glm5Model::load_mtp(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs, std::string& err) { glm_stub("bool Glm5Model::load_mtp"); }
bool Glm5Model::fast_mtp(int64_t p, int32_t next_tok, std::string& err, int step, bool sub, bool read_back, const float* emb_staged, bool head) { glm_stub("bool Glm5Model::fast_mtp"); }
void Glm5Model::mtp_draft_setup(bool chain) { glm_stub("void Glm5Model::mtp_draft_setup"); }
void Glm5Model::mtp_vocab_scan(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs) { glm_stub("void Glm5Model::mtp_vocab_scan"); }
int Glm5Model::mtp_draft(int32_t next_tok, std::string& err) { glm_stub("int Glm5Model::mtp_draft"); }
bool Glm5Model::has_mtp() const { return false; }
int64_t Glm5Model::kv_resident() const { return 0; }
int64_t Glm5Model::kv_resident_cells(int kpool, int64_t max_ctx) { (void)kpool; (void)max_ctx; return 0; }
Glm5Model::KvStreamStats Glm5Model::kv_stream_stats() const { return KvStreamStats{}; }
bool Glm5Model::spec_ready() const { return false; }
bool Glm5Model::decode_spec(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit, int64_t& produced, std::string& err) { glm_stub("bool Glm5Model::decode_spec"); }
bool Glm5Model::spec_kda_copy(bool restore) { glm_stub("bool Glm5Model::spec_kda_copy"); }
size_t Glm5Model::spec_kda_bytes() const { return 0; }
bool Glm5Model::spec_kda_reserve(std::string& err) { glm_stub("bool Glm5Model::spec_kda_reserve"); }
Glm5Model* Glm5Model::spec_head_last() { return nullptr; }
Glm5Model* Glm5Model::spec_tail_last() { return nullptr; }
bool Glm5Model::hop_to_next(Glm5Model* B, int64_t p, std::string& err) { glm_stub("bool Glm5Model::hop_to_next"); }
bool Glm5Model::spec_head(int64_t p, int32_t token, std::string& err) { glm_stub("bool Glm5Model::spec_head"); }
bool Glm5Model::spec_tail(Glm5Model* head, int64_t p, std::string& err) { glm_stub("bool Glm5Model::spec_tail"); }
bool Glm5Model::mtp_decode_wanted(int dev, int n_parts) { (void)dev; (void)n_parts; return false; }
bool Glm5Model::mtp_ready() const { return false; }
bool Glm5Model::decode_mtp(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit, int64_t& produced, std::string& err) { glm_stub("bool Glm5Model::decode_mtp"); }
bool Glm5Model::decode_spec_pipe(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit, int64_t& produced, std::string& err) { glm_stub("bool Glm5Model::decode_spec_pipe"); }
bool Glm5Model::mtp_fill_prev(int32_t token, int64_t p, std::string& err) { glm_stub("bool Glm5Model::mtp_fill_prev"); }
int Glm5Model::mtp_choose(int n_cap) { glm_stub("int Glm5Model::mtp_choose"); }
bool Glm5Model::mtp_cache_setup(std::string& err) { glm_stub("bool Glm5Model::mtp_cache_setup"); }
void Glm5Model::mtp_cache_free() {}
bool Glm5Model::mtp_cache_rows(int64_t p0, int n, const float* h, const int32_t* next, std::string& err, bool sync, float* emb_rows_h) { glm_stub("bool Glm5Model::mtp_cache_rows"); }
void Glm5Model::seq_bind(int s) { cur_seq_ = s; }
void Glm5Model::seq_bind_all(int s) { glm_stub("void Glm5Model::seq_bind_all"); }
bool Glm5Model::seq_alloc_arenas(std::string& err) { glm_stub("bool Glm5Model::seq_alloc_arenas"); }
uint64_t Glm5Model::seq_layout(int64_t ctx, SeqCtx& c) const { glm_stub("uint64_t Glm5Model::seq_layout"); }
bool Glm5Model::seq_pipe_ready(std::string& err) { glm_stub("bool Glm5Model::seq_pipe_ready"); }
bool Glm5Model::multi_issue(int s, int32_t token, const strata::kernels::SamplerParams& sp, std::string& err) { glm_stub("bool Glm5Model::multi_issue"); }
int Glm5Model::multi_take(int s, std::string& err) { glm_stub("int Glm5Model::multi_take"); }
bool Glm5Model::multi_boundary(std::string& err) { glm_stub("bool Glm5Model::multi_boundary"); }
int Glm5Model::rows_max() { return 0; }
int Glm5Model::split_rows_setting() { return 0; }
bool Glm5Model::multi_rows(const int* seqs, const int32_t* tokens, const strata::kernels::SamplerParams* sps, int n, int* tokens_out, std::string& err) { glm_stub("bool Glm5Model::multi_rows"); }
bool Glm5Model::split_rows_ok() const { return false; }
bool Glm5Model::multi_issue_rows(const int* seqs, const int32_t* tokens, const strata::kernels::SamplerParams* sps, int n, std::string& err) { glm_stub("bool Glm5Model::multi_issue_rows"); }
int Glm5Model::multi_mtp_setting() { return 0; }
int Glm5Model::multi_mtp_cap() { return 0; }
bool Glm5Model::mtpw_alloc(std::string& err) { glm_stub("bool Glm5Model::mtpw_alloc"); }
bool Glm5Model::mtpw_setup(std::string& err) { glm_stub("bool Glm5Model::mtpw_setup"); }
void Glm5Model::mtpw_destroy() {}
bool Glm5Model::mtpw_ok() const { return false; }
bool Glm5Model::mtpw_front(const int* seqs, const int* counts, int n, const int32_t* tokens, std::string& err) { glm_stub("bool Glm5Model::mtpw_front"); }
bool Glm5Model::mtpw_back(const int* seqs, const int* counts, int n, const int32_t* tokens, const strata::kernels::SamplerParams* sps, std::string& err) { glm_stub("bool Glm5Model::mtpw_back"); }
bool Glm5Model::mtpw_picks(const int* seqs, const int* counts, int n, int32_t* picks, std::string& err) { glm_stub("bool Glm5Model::mtpw_picks"); }
bool Glm5Model::mtpw_commit(const int* seqs, const int* counts, const int* keep, int n, std::string& err) { glm_stub("bool Glm5Model::mtpw_commit"); }
bool Glm5Model::mtpw_draft_queue(const int* seqs, const int* keep, int n, const int32_t* written, const int32_t* last, const int* drafts, std::string& err) { glm_stub("bool Glm5Model::mtpw_draft_queue"); }
bool Glm5Model::mtpw_draft_take(int32_t* out, std::string& err) { glm_stub("bool Glm5Model::mtpw_draft_take"); }
bool Glm5Model::rows_bind_win(const int* seqs, const int* counts, int n, RowSeq* rw, std::string& err) { glm_stub("bool Glm5Model::rows_bind_win"); }
bool Glm5Model::mtpw_part(const int* seqs, const int* counts, int n, const int32_t* tokens, const strata::kernels::SamplerParams* sps, Glm5Model* prev, bool last, std::string& err) { glm_stub("bool Glm5Model::mtpw_part"); }
int Glm5Model::mtpw_window() const { return 1; }
void Glm5Model::mtpw_bind(int sq, RowSeq& r) const { glm_stub("void Glm5Model::mtpw_bind"); }
bool Glm5Model::mtpw_cache_rows(const RowSeq& rb, int64_t p0, int n, const float* h, const int32_t* next, float* emb_h, std::string& err) { glm_stub("bool Glm5Model::mtpw_cache_rows"); }
bool Glm5Model::mtpw_draft_rows(RowSeq* rr, const int* slot, int nr, std::string& err) { glm_stub("bool Glm5Model::mtpw_draft_rows"); }
size_t Glm5Model::mtpw_kin_offset(int il) const { glm_stub("size_t Glm5Model::mtpw_kin_offset"); }
bool Glm5Model::mtpw_commit_part(const int* seqs, const int* counts, const int* keep, int n, std::string& err) { glm_stub("bool Glm5Model::mtpw_commit_part"); }
bool Glm5Model::rows_bind(const int* seqs, int n, RowSeq* rw, std::string& err) { glm_stub("bool Glm5Model::rows_bind"); }
bool Glm5Model::rows_embed(const RowSeq* rw, const int32_t* tokens, int n, std::string& err) { glm_stub("bool Glm5Model::rows_embed"); }
bool Glm5Model::rows_layers(RowSeq* rw, int n, std::string& err) { glm_stub("bool Glm5Model::rows_layers"); }
bool Glm5Model::rows_dsa(int il, RowSeq* rw, int n, std::string& err) { glm_stub("bool Glm5Model::rows_dsa"); }
bool Glm5Model::rows_moe(int il, RowSeq* rw, int n, std::string& err) { glm_stub("bool Glm5Model::rows_moe"); }
bool Glm5Model::rows_head(const RowSeq* rw, const strata::kernels::SamplerParams* sps, int n, std::string& err) { glm_stub("bool Glm5Model::rows_head"); }
bool Glm5Model::rows_alloc(std::string& err) { glm_stub("bool Glm5Model::rows_alloc"); }
void Glm5Model::rows_destroy() {}
bool Glm5Model::win_alloc(int tmax, std::string& err) { glm_stub("bool Glm5Model::win_alloc"); }
void Glm5Model::win_free() {}
bool Glm5Model::fast_window_layers(int64_t p0, int T, std::string& err) { glm_stub("bool Glm5Model::fast_window_layers"); }
bool Glm5Model::fast_dsa_window(int il, int64_t p0, int t0, int T, std::string& err) { glm_stub("bool Glm5Model::fast_dsa_window"); }
bool Glm5Model::fast_moe_window(int il, int t0, int T, int part, int gid, std::string& err) { glm_stub("bool Glm5Model::fast_moe_window"); }
bool Glm5Model::window(const int32_t* tokens, int T, int32_t* out, std::string& err, const strata::kernels::SamplerParams* sp) { glm_stub("bool Glm5Model::window"); }
bool Glm5Model::window_pipe(const int32_t* tokens, int T, int32_t* out, std::string& err, const strata::kernels::SamplerParams* sp) { glm_stub("bool Glm5Model::window_pipe"); }
bool Glm5Model::win_pipe_begin(int32_t first, std::string& err, const strata::kernels::SamplerParams* sp, bool fresh) { glm_stub("bool Glm5Model::win_pipe_begin"); }
bool Glm5Model::win_pipe_begin_part(bool fresh, std::string& err) { glm_stub("bool Glm5Model::win_pipe_begin_part"); }
void Glm5Model::win_commit_part(int n_keep) { (void)n_keep; }
std::chrono::steady_clock::time_point Glm5Model::win_pipe_begun_at(int k) const { (void)k; return {}; }
bool Glm5Model::win_pipe_push(int32_t token, std::string& err) { glm_stub("bool Glm5Model::win_pipe_push"); }
bool Glm5Model::win_pipe_pick(int t, int32_t& pick, std::string& err) { glm_stub("bool Glm5Model::win_pipe_pick"); }
bool Glm5Model::win_pipe_settle(std::string& err) { glm_stub("bool Glm5Model::win_pipe_settle"); }
void Glm5Model::win_pipe_cancel_from(int t) { (void)t; }
void Glm5Model::win_pipe_drain() {}
bool Glm5Model::win_pipe_end(int32_t* out, std::string& err) { glm_stub("bool Glm5Model::win_pipe_end"); }
bool Glm5Model::win_pipe_part(int k, int t, std::string& err) { glm_stub("bool Glm5Model::win_pipe_part"); }
void Glm5Model::win_crew_stop() {}
void Glm5Model::win_kda_rows(int il, int t, float** proj, float** fa, float** ga, float** beta) { glm_stub("void Glm5Model::win_kda_rows"); }
bool Glm5Model::window_commit(int n_keep, std::string& err, bool wait) { glm_stub("bool Glm5Model::window_commit"); }
void Glm5Model::win_commit_wait() {}
bool Glm5Model::win_reserve(std::string& err) { glm_stub("bool Glm5Model::win_reserve"); }
void Glm5Model::win_prof_report() {}
bool Glm5Model::dflash_load(const std::string& path, int dev, std::string& err) { glm_stub("bool Glm5Model::dflash_load"); }
bool Glm5Model::dflash_bind(std::string& err) { glm_stub("bool Glm5Model::dflash_bind"); }
void Glm5Model::dflash_capture(int il, const float* R, int n, int64_t p0, float* scratch, int rows, cudaStream_t s) { glm_stub("void Glm5Model::dflash_capture"); }
bool Glm5Model::dflash_ready() const { return false; }
void Glm5Model::dflash_reset() {}
bool Glm5Model::decode_dflash(strata::kernels::SamplerParams& sp, int64_t max_new, int n_max, float p_min, const std::function<bool(int)>& emit, int64_t& produced, std::string& err, int n_min, float p_cum) { glm_stub("bool Glm5Model::decode_dflash"); }
bool Glm5Model::prefill_calc(const CalcJob& job, std::string& err) { glm_stub("bool Glm5Model::prefill_calc"); }
bool Glm5Model::prefill_setup(std::string& err) { glm_stub("bool Glm5Model::prefill_setup"); }
size_t Glm5Model::prefill_borrow_bytes() const { return 0; }
size_t Glm5Model::prefill_trim_prestage(size_t max_borrow) { glm_stub("size_t Glm5Model::prefill_trim_prestage"); }
bool Glm5Model::prefill_bind(uint8_t* region, size_t bytes, std::string& err) { glm_stub("bool Glm5Model::prefill_bind"); }
bool Glm5Model::prefill_carve(int T, int64_t end) { glm_stub("bool Glm5Model::prefill_carve"); }
bool Glm5Model::prefill_lend(std::string& err) { glm_stub("bool Glm5Model::prefill_lend"); }
bool Glm5Model::prefill_return(std::string& err) { glm_stub("bool Glm5Model::prefill_return"); }
void Glm5Model::prefill_destroy() {}
void Glm5Model::prefill_cap(int T, const char* why) { (void)T; (void)why; }
void Glm5Model::prefill_settle(double pinned_share) { (void)pinned_share; }
std::pair<size_t, size_t> Glm5Model::prefill_bytes_for(size_t T, size_t kv_stage) const { (void)T; (void)kv_stage; return {0, 0}; }
Glm5Model::PinnedPrompt Glm5Model::prefill_pinned() const { return PinnedPrompt{}; }
Glm5Model::PoolBudget Glm5Model::pool_budget(size_t free_b, size_t total_b, int part, size_t floor_b, size_t slot_b, bool unified) { (void)free_b; (void)total_b; (void)part; (void)floor_b; (void)slot_b; (void)unified; return PoolBudget{}; }
Glm5Model::PoolBudget Glm5Model::part_budget(size_t free_b, size_t total_b, size_t floor_b, size_t slot_b) const { glm_stub("Glm5Model::PoolBudget Glm5Model::part_budget"); }
void Glm5Model::pool_check(size_t floor_b, size_t slot_b, size_t all_b) { (void)floor_b; (void)slot_b; (void)all_b; }
size_t Glm5Model::mmq_scratch_reserve() { return 0; }
int Glm5Model::pool_keep(int n_exp_used) { glm_stub("int Glm5Model::pool_keep"); }
bool Glm5Model::lend_tail(size_t limit, uint64_t& moved, uint64_t& dropped, std::string& err) { glm_stub("bool Glm5Model::lend_tail"); }
bool Glm5Model::prefill_half(int64_t p0, int T, std::string& err, const int32_t* next_ids, const float* const* next_img) { glm_stub("bool Glm5Model::prefill_half"); }
bool Glm5Model::prefill(const std::vector<int32_t>& tokens, std::string& err, int32_t next_token) { glm_stub("bool Glm5Model::prefill"); }
bool Glm5Model::prefill_run(const std::vector<int32_t>& tokens, std::string& err) { glm_stub("bool Glm5Model::prefill_run"); }
int Glm5Model::prefill_chunk() const { return 0; }
int Glm5Model::prefill_chunk_for(int64_t n) const { (void)n; return 0; }
int64_t Glm5Model::prefill_floor(int64_t n, int call_min) const { (void)n; return call_min; }
bool Glm5Model::dump_state(const std::string& path) { glm_stub("bool Glm5Model::dump_state"); }

bool Glm5Model::set_routing_dump(const std::string& path, std::string& err) {
    (void)path;
    err = "Glm5Model::set_routing_dump: the GLM runner is not implemented in the Metal build";
    return false;
}

bool Glm5Geometry::from_gguf(const strata::GgufFile& g, Glm5Geometry& out, std::string& err) {
    (void)g; (void)out;
    err = "Glm5Geometry::from_gguf: the GLM runner is not implemented in the Metal build";
    return false;
}

// ---- GlmDflash (include/strata/core/glm_dflash.hpp): Glm5Model holds one by
// unique_ptr, so the Metal link needs this class too. Its real definitions live in
// glm_dflash.cu. The pimpl Impl is completed empty here, in the only TU that defines
// the constructor/destructor in this build.
struct GlmDflash::Impl {};
GlmDflash::GlmDflash() = default;
GlmDflash::~GlmDflash() = default;
bool GlmDflash::load(const std::string& path, int dev, std::string& err) { glm_stub("strata::core::GlmDflash::load"); }
bool GlmDflash::bind(int target_embd, int target_vocab, int target_layers, std::string& err) { glm_stub("strata::core::GlmDflash::bind"); }
int GlmDflash::dev() const { return -1; }
cudaStream_t GlmDflash::stream() { return nullptr; }
size_t GlmDflash::bytes() const { return 0; }
int GlmDflash::block() const { return 0; }
int GlmDflash::ctx() const { return 0; }
const std::vector<int>& GlmDflash::target_layers() const { static const std::vector<int> none; return none; }
std::string GlmDflash::describe() const { return {}; }
float* GlmDflash::stage_row(int64_t p, int k) const { glm_stub("strata::core::GlmDflash::stage_row"); }
size_t GlmDflash::stage_pitch() const { return 0; }
int GlmDflash::ring_size() const { return 0; }
void GlmDflash::staged(int64_t p, int k) { (void)p; (void)k; }
bool GlmDflash::inject(int64_t lo, int64_t hi, cudaStream_t s, std::string& err) { glm_stub("strata::core::GlmDflash::inject"); }
bool GlmDflash::draft(int32_t anchor, int64_t n_past, int n_max, float p_min, const WSlot& head, const uint8_t* emb, int emb_type, cudaStream_t s, std::vector<int32_t>& out, std::string& err, std::vector<float>* conf) { glm_stub("strata::core::GlmDflash::draft"); }
bool GlmDflash::draft_start(int32_t anchor, int64_t n_past, int n_max, float p_min, const WSlot& head, const uint8_t* emb, int emb_type, cudaStream_t s, std::string& err) { glm_stub("strata::core::GlmDflash::draft_start"); }
bool GlmDflash::draft_finish(std::vector<int32_t>& out, std::string& err, std::vector<float>* conf, int n_min, float p_cum) { glm_stub("strata::core::GlmDflash::draft_finish"); }
bool GlmDflash::split_head(const WSlot* lo, const WSlot* hi, int c0, int hdev, cudaStream_t hs, float* hx, float* hlg, void* hcand, size_t cand_bytes, int rows, std::string& err) { glm_stub("strata::core::GlmDflash::split_head"); }
void GlmDflash::reset(int64_t from) { (void)from; }
bool GlmDflash::last_rows(int n, std::vector<float>& hidden, std::vector<int>& cand, std::vector<float>& logit, std::vector<int>& pick, std::vector<float>& conf) const { glm_stub("strata::core::GlmDflash::last_rows"); }
int GlmDflash::top_k() const { return 0; }

size_t glm_prefill_lend_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa, bool has_dense,
                              bool has_moe, bool mtp, int64_t max_ctx, size_t gstride, size_t kv_stage) {
    (void)g; (void)T; (void)has_kda; (void)has_dsa; (void)has_dense; (void)has_moe; (void)mtp;
    (void)max_ctx; (void)gstride; (void)kv_stage;
    return 0;
}

}  // namespace strata::core

