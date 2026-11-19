// src/hy4-metal/metal_hy4_pack_real_test.cpp - the Hy4 Metal port's stage 1 gate:
// upstream's loader (src/hy4/hy4_pack.hpp) against a pack written by the REAL packer
// (tools/hy4/pack_gguf.py) from a synthetic GGUF set (tools/hy4_metal_fixture.py -
// the packer test's own set, its expert latent width raised 128 -> 256 for the
// loader's whole-256s rule). Every expected value below is the fixture's known
// geometry, read off the GGUF the fixture writes - not the packer's say-so alone:
// the point is that the two agree, through the files on disk.
//
// Usage: metal_hy4_pack_real_test <pack-dir>   (the fixture's <model>/maya-hy4)

#include "hy4_pack.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

bool feq(float a, float b) { return std::fabs(a - b) <= 1e-6f * (1.0f + std::fabs(b)); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: metal_hy4_pack_real_test <pack-dir>\n");
        return 2;
    }
    hy4::Pack pack;
    std::string err;
    if (!pack.load(argv[1], err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const hy4::Geo& g = pack.g;

    // the geometry, as the fixture's GGUF states it
    CHECK(g.n_layer == 3 && g.n_mtp == 1 && g.n_embd == 256 && g.n_vocab == 269);
    CHECK(g.context == 4096 && g.n_ff == 512 && g.n_dense_lead == 1);
    CHECK(g.n_expert == 4 && g.n_used == 2 && g.n_ff_exp == 256 && g.n_lat == 256);
    CHECK(g.n_shared == 1 && feq(g.w_scale, 4.0f) && g.w_norm);
    CHECK(g.limit.size() == 4 && feq(g.limit[0], 0.0f) && feq(g.limit[1], 10.0f) &&
          feq(g.limit[2], 10.0f) && feq(g.limit[3], 10.0f));
    CHECK(feq(g.eps, 1e-5f));
    CHECK(g.q_lora == 128 && g.kv_lora == 64 && g.k_mla == 96 && g.v_mla == 64);
    CHECK(g.rope == 32 && feq(g.rope_base, 1e7f) && feq(g.rope_base_swa, 1e4f) && g.n_swa == 512);
    CHECK(g.heads == std::vector<int>({64, 32, 32, 64}));
    CHECK(g.swa == std::vector<int>({0, 1, 1, 0}));
    CHECK(g.indexer == std::vector<int>({1, 0, 0, 1}));
    CHECK(g.ix_heads == 4 && g.ix_dim == 128 && g.ix_top_k == 2048 && g.ix_kpool == 4 && g.ix_tail);
    CHECK(g.hc == 4 && feq(g.hc_eps, 1e-6f) && feq(g.hc_mag, 2.0f));
    CHECK(g.bos == 256 && g.eos == 257);
    CHECK(g.kv_row() == 96 && g.nope() == 64);   // kv_lora + rope; k_mla - rope

    // the tensor table: two shards, eight tensors, the packer's types and byte counts
    CHECK(pack.files.size() == 2 && pack.t.size() == 8);
    const hy4::Tensor* emb = pack.get("token_embd.weight");
    CHECK(emb != nullptr && emb->type == "F32" && emb->file == 0);
    CHECK(emb->n(0) == 256 && emb->n(1) == 269 && emb->bytes == 256ull * 269 * 4);
    CHECK(pack.data(*emb) == pack.map[0] + emb->off);
    CHECK(pack.data(*emb)[0] == 0 && pack.data(*emb)[12345] == 0);   // the writer's pattern: tensor 0 of its file
    const hy4::Tensor* out = pack.get("output.weight");
    CHECK(out != nullptr && out->type == "Q8_0" && out->bytes == 269ull * 272);
    const hy4::Tensor* hcfn = pack.get("output_hc_fn.weight");
    CHECK(hcfn != nullptr && hcfn->type == "F32" && hcfn->n(0) == 1024 && hcfn->n(1) == 4);
    const hy4::Tensor* qa0 = pack.get("blk.0.attn_q_a.weight");
    CHECK(qa0 != nullptr && qa0->type == "Q4_K" && qa0->file == 0 && qa0->bytes == 128ull * 144);
    const hy4::Tensor* qa2 = pack.get("blk.2.attn_q_a.weight");
    CHECK(qa2 != nullptr && qa2->type == "Q4_K" && qa2->file == 1 && qa2->bytes == 128ull * 144);
    const hy4::Tensor* ple = pack.get("per_layer_token_embd.weight");
    CHECK(ple != nullptr && ple->type == "F8_E4M3" && ple->file == 1);
    CHECK(ple->n(0) == 32 && ple->n(1) == 64 && ple->bytes == 2048);
    CHECK(pack.data(*ple)[0] == 1 && pack.data(*ple)[2047] == 1);   // tensor 1 of its file
    CHECK(pack.get("blk.3.attn_q_a.weight") == nullptr);            // the MTP block's tensors are the set's own
                                                                    // only where the GGUF names them - this set
                                                                    // names none past blk.2's

    // the Engram hash, as engram.bin carries it
    const hy4::Engram& eg = pack.eg;
    CHECK(eg.vocab == 269 && eg.modules == 1 && eg.ngram == 3 && eg.heads == 2);
    CHECK(eg.head_dim == 32 && eg.rows == 64 && eg.cols() == 4);
    CHECK(eg.pad == 3);   // the token map's value at the pad token (id 3 maps to itself)
    CHECK(eg.token_map[0] == 0 && eg.token_map[263] == 263);
    CHECK(eg.token_map[264] == -1 && eg.token_map[268] == -1);   // the map's missing tail
    CHECK(eg.module_of(1) == 0 && eg.module_of(0) == -1 && eg.module_of(2) == -1);
    CHECK(eg.mult[0] == 7 && eg.mult[1] == ~0ull - 2 && eg.mult[2] == (1ull << 40));
    CHECK(eg.primes[0] == 11 && eg.primes[1] == 13 && eg.primes[2] == 17 && eg.primes[3] == 19);
    CHECK(eg.offsets[0] == 0 && eg.offsets[1] == 11 && eg.offsets[2] == 24 && eg.offsets[3] == 41);

    CHECK(!pack.has("engram_eps") && !pack.has("engram_scales"));

    std::fprintf(stderr, "metal_hy4_pack_real_test: PASS\n");
    return 0;
}
