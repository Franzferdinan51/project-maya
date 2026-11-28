// src/hy4-metal/metal_hy4_pack_test.cpp - the Hy4 Metal port's loader fixture gate.
//
// Writes a synthetic mini-pack (hy4.txt + tensors.txt + one tiny shard + a minimal
// HYE1 engram.bin) into a temp dir - no model files - and loads it through upstream's
// own hy4::Pack (src/hy4/hy4_pack.hpp), asserting the geometry, the tensor table and
// the Engram hash parse. A second pack with the wrong arch must be refused.
// The synthetic geometry satisfies hy4_pack.hpp's own validation (list lengths,
// divisibility and ordering rules), at toy scale.

#include "hy4_pack.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {

bool write_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char*) data, (std::streamsize) bytes);
    return (bool) f;
}

bool write_text(const std::string& path, const std::string& s) {
    return write_file(path, s.data(), s.size());
}

std::string make_pack(const char* arch) {
    char tmpl[] = "/tmp/metal_hy4_pack_test_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    if (!dir) return "";
    const std::string d = dir;

    std::vector<uint8_t> shard(4096);
    for (size_t i = 0; i < shard.size(); ++i) shard[i] = (uint8_t) (i * 31 + 7);
    if (!write_file(d + "/shard0.bin", shard.data(), shard.size())) return "";

    std::string hy4 = "arch " + std::string(arch) + "\n";
    hy4 += "model_dir " + d + "\n";
    hy4 += "n_layer 2\nn_mtp 1\nn_embd 256\nn_vocab 1000\ncontext_length 4096\n";
    hy4 += "n_ff 512\nn_dense_lead 1\nn_expert 8\nn_expert_used 2\nn_ff_exp 256\n";
    hy4 += "n_expert_latent 256\nn_expert_shared 1\nexpert_weights_scale 1.0\n";
    hy4 += "expert_weights_norm 1\nswiglu_limit 0,0,0\nrms_eps 0.00001\n";
    hy4 += "q_lora 128\nkv_lora 64\nk_mla 128\nv_mla 128\nrope_dim 32\n";
    hy4 += "rope_base 10000\nrope_base_swa 10000\nn_swa 128\n";
    hy4 += "heads 4,4,4\nswa 0,0,0\nindexer 0,0,0\n";
    hy4 += "ix_heads 4\nix_dim 64\nix_top_k 4\nix_kpool 2\nix_select_tail 1\n";
    hy4 += "hc 1\nhc_eps 0.000001\nhc_magnitude 1.0\nbos 1\neos 2\n";
    if (!write_text(d + "/hy4.txt", hy4)) return "";

    if (!write_text(d + "/tensors.txt",
                    "file 0 shard0.bin\n"
                    "t token_embd.weight F32 0 0 1024 256 4\n"
                    "t blk.0.attn_q.weight Q6_K 0 1024 512 256 2\n")) return "";

    // minimal HYE1: vocab 1000, 1 module, ngram 2, 2 heads, head_dim 32, rows 16
    std::vector<uint8_t> eg;
    auto put = [&](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*) p;
        eg.insert(eg.end(), b, b + n);
    };
    const int32_t h[7] = {1000, 1, 2, 2, 32, 0, 16};
    put("HYE1", 4);
    put(h, sizeof h);
    std::vector<int32_t> token_map(1000, 3), layers(1, 5);
    std::vector<uint64_t> mult(2, 97), primes(2, 1000003), offsets(2, 0);
    put(token_map.data(), token_map.size() * 4);
    put(layers.data(), layers.size() * 4);
    put(mult.data(), mult.size() * 8);
    put(primes.data(), primes.size() * 8);
    put(offsets.data(), offsets.size() * 8);
    if (!write_file(d + "/engram.bin", eg.data(), eg.size())) return "";

    return d;
}

}  // namespace

int main() {
    const std::string d = make_pack("hy_v4_flash");
    CHECK(!d.empty());

    hy4::Pack pack;
    std::string err;
    if (!pack.load(d, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const hy4::Geo& g = pack.g;
    CHECK(g.n_layer == 2 && g.n_mtp == 1 && g.n_embd == 256 && g.n_vocab == 1000);
    CHECK(g.n_expert == 8 && g.n_used == 2 && g.n_dense_lead == 1);
    CHECK(g.kv_row() == 64 + 32);          // kv_lora + rope
    CHECK(g.nope() == 128 - 32);           // k_mla - rope
    CHECK(pack.files.size() == 1 && pack.map_bytes[0] == 4096);

    const hy4::Tensor* emb = pack.get("token_embd.weight");
    CHECK(emb != nullptr && emb->type == "F32" && emb->n(0) == 256 && emb->n(1) == 4);
    CHECK(pack.data(*emb) == pack.map[0]);
    CHECK(pack.data(*emb)[5] == (uint8_t) (5 * 31 + 7));
    CHECK(pack.get("blk.99.ffn_down.weight") == nullptr);

    CHECK(pack.eg.rows == 16 && pack.eg.cols() == 2);
    CHECK(pack.eg.module_of(5) == 0 && pack.eg.module_of(3) == -1);
    CHECK(pack.eg.token_map[999] == 3 && pack.eg.primes[1] == 1000003);

    // a pack that is not hy_v4_flash must be refused, with an error
    const std::string bad = make_pack("some_other_arch");
    CHECK(!bad.empty());
    hy4::Pack pack2;
    std::string err2;
    CHECK(!pack2.load(bad, err2));
    CHECK(!err2.empty());

    std::fprintf(stderr, "metal_hy4_pack_test: PASS\n");
    return 0;
}
