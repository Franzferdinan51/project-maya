// e2e_greedy_decode: the model-level capstone of the Qwen parity oracle
// (design: E2E-GREEDY.md in this directory).
//
// The kernel fixtures prove each entry point against a scalar reference. This
// harness proves the COMPOSED engine: a fixed prompt, greedy sampling, and an
// exact token-id sequence as the verdict. Greedy is the point: with
// temperature 0 the whole pipeline (embed, PLE, GDN/QSA layers, MoE, head,
// sampler) collapses to one deterministic sequence, so a single wrong bit in
// any kernel shows up as a token divergence at a specific position, and the
// first diverging position localises the fault to the steps before it.
//
// Modes:
//   --selftest                 round-trip the golden format; no model needed
//   --compare <golden.mqgt> <candidate.txt>
//                              candidate.txt = whitespace-separated token ids
//                              produced by any engine build (CUDA, HIP, Metal,
//                              or serve --dump-tokens); exact-compare + report
//                              the first divergence position
//   --run <pack_dir> <golden.mqgt>
//                              drive the in-process engine. NOT WIRED YET: the
//                              Metal kernels are Stage-1 stubs, so generation
//                              cannot run; this mode prints the integration
//                              contract and exits 77 (the tree's skip code, as
//                              in pf_wmma_parity) so ctest records a skip, not
//                              a pass and not a fail.
//
// Golden capture: on the CUDA build (the reference implementation), run the
// session with the golden prompt + greedy params, dump the produced ids, and
// store them with --write-golden. The Metal run must reproduce them exactly.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

inline constexpr uint32_t kGoldenMagic = 0x5447514Du;  // "MQGT"
inline constexpr uint16_t kGoldenVersion = 1;

struct Golden {
    std::string pack_id;    // e.g. "q2_0" (the pack directory name)
    std::string model_id;   // e.g. "qwen3.8-flash-next-q2_0"
    std::vector<int32_t> prompt;
    std::vector<int32_t> expected;  // greedy continuation, length = n_gen
};

bool write_golden(const std::string& path, const Golden& g) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
    auto wstr = [&](const std::string& s) { w32((uint32_t)s.size()); std::fwrite(s.data(), 1, s.size(), f); };
    w32(kGoldenMagic);
    w16(kGoldenVersion);
    w16(0);
    wstr(g.pack_id);
    wstr(g.model_id);
    w32((uint32_t)g.prompt.size());
    std::fwrite(g.prompt.data(), 4, g.prompt.size(), f);
    w32((uint32_t)g.expected.size());
    std::fwrite(g.expected.data(), 4, g.expected.size(), f);
    const bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

bool read_golden(const std::string& path, Golden& g) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> buf;
    uint8_t chunk[8192];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
    std::fclose(f);
    size_t pos = 0;
    auto get = [&](void* dst, size_t sz) {
        if (pos + sz > buf.size()) return false;
        std::memcpy(dst, buf.data() + pos, sz);
        pos += sz;
        return true;
    };
    uint32_t magic = 0;
    uint16_t ver = 0, rsv = 0;
    if (!get(&magic, 4) || magic != kGoldenMagic || !get(&ver, 2) || ver != kGoldenVersion || !get(&rsv, 2)) return false;
    auto gstr = [&](std::string& s) {
        uint32_t len = 0;
        if (!get(&len, 4) || pos + len > buf.size()) return false;
        s.assign((const char*)buf.data() + pos, len);
        pos += len;
        return true;
    };
    uint32_t np = 0, ne = 0;
    if (!gstr(g.pack_id) || !gstr(g.model_id) || !get(&np, 4)) return false;
    g.prompt.resize(np);
    if (!get(g.prompt.data(), 4ull * np) || !get(&ne, 4)) return false;
    g.expected.resize(ne);
    return get(g.expected.data(), 4ull * ne);
}

std::vector<int32_t> read_candidate(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    std::vector<int32_t> out;
    if (!f) return out;
    long v;
    while (std::fscanf(f, "%ld", &v) == 1) out.push_back((int32_t)v);
    std::fclose(f);
    return out;
}

int compare_tokens(const Golden& g, const std::vector<int32_t>& got) {
    const size_t n = std::min(g.expected.size(), got.size());
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != g.expected[i]) {
            std::printf("DIVERGE at position %zu: got token %d, expected %d (of %zu)\n", i, got[i], g.expected[i],
                        g.expected.size());
            return 1;
        }
    }
    if (got.size() != g.expected.size()) {
        std::printf("LENGTH differs: got %zu tokens, expected %zu (common prefix identical)\n", got.size(),
                    g.expected.size());
        return 1;
    }
    std::printf("MATCH: %zu greedy tokens identical for prompt of %zu tokens (%s / %s)\n", got.size(),
                g.prompt.size(), g.pack_id.c_str(), g.model_id.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--selftest") == 0) {
        Golden g;
        g.pack_id = "q2_0";
        g.model_id = "qwen3.8-flash-next-q2_0";
        g.prompt = {9707, 25, 264, 1094, 312, 220, 17, 15, 28};  // placeholder ids; format test only
        g.expected = {362, 494, 264, 1094, 312, 220, 19, 15, 28, 13};
        const std::string path = "/tmp/e2e_greedy_selftest.mqgt";
        if (!write_golden(path, g)) { std::fprintf(stderr, "selftest: write failed\n"); return 1; }
        Golden back;
        if (!read_golden(path, back)) { std::fprintf(stderr, "selftest: read failed\n"); return 1; }
        if (back.pack_id != g.pack_id || back.prompt != g.prompt || back.expected != g.expected) {
            std::fprintf(stderr, "selftest: round-trip mismatch\n");
            return 1;
        }
        std::printf("e2e_greedy_decode selftest: golden format round-trip OK\n");
        return 0;
    }
    if (argc >= 4 && std::strcmp(argv[1], "--compare") == 0) {
        Golden g;
        if (!read_golden(argv[2], g)) { std::fprintf(stderr, "cannot read golden %s\n", argv[2]); return 2; }
        const auto got = read_candidate(argv[3]);
        if (got.empty()) { std::fprintf(stderr, "cannot read candidate tokens %s\n", argv[3]); return 2; }
        return compare_tokens(g, got);
    }
    if (argc >= 4 && std::strcmp(argv[1], "--run") == 0) {
        // Integration contract for Stage 2's final gate:
        //   1. open the pack in argv[2] through the engine's session path
        //      (the same construction serve uses for the qwen family),
        //   2. feed golden.prompt, then greedy-decode golden.expected.size()
        //      tokens with SamplerParams{greedy=true, temperature=0},
        //   3. compare_tokens() on the produced ids; exit 0/1 as above.
        // Blocked on: real Metal kernels (Stage 2) + a golden captured from
        // the CUDA build. Until both exist this is a skip, per the tree's
        // SKIP_RETURN_CODE convention.
        std::printf("e2e_greedy_decode --run: engine drive not wired yet (Stage 2 kernels + CUDA golden capture "
                    "pending); skipping\n");
        return 77;
    }
    std::fprintf(stderr, "usage: e2e_greedy_decode --selftest | --compare <golden.mqgt> <candidate.txt> | "
                         "--run <pack_dir> <golden.mqgt>\n");
    return 2;
}
