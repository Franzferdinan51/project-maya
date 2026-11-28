// src/hy4-metal/hy4_metal_main.cpp - maya-metal-hy4: Hy4-Flash on Apple Silicon (the CLI
// half; the maya.sh serve wiring drives the same engine through maya.py's server).
//
//   maya-metal-hy4 --pack DIR --serve [...]                the serve protocol (hy4_metal_serve.cpp)
//   maya-metal-hy4 --pack DIR --ids 1,2,3 [--n 64]          greedy continuation of token ids, with timings
//   maya-metal-hy4 --pack DIR --ids 1,2,3 --cpu             the same on the scalar CPU reference (the oracle)
//   maya-metal-hy4 DIR                                      load a pack, print its geometry (stage 0's check)
//
// The forward is hym::Forward (hy4_metal_model.hpp): the ggml-metal graph forward, gated
// token-for-token against the scalar CPU golden by metal_hy4_block_test.  Flags the CUDA
// engine takes that this one does not use (--gpus, --tp, --reserve-gb, --cpu-threads) are
// accepted and noted once; two of them change behaviour and say so loudly: --mtp (MTP
// drafting is NOT ported - generation is plain, the same tokens without the draft's
// speed) and --vision (the tower is stage 5; image requests are refused per request).

#include "hy4_metal_model.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: maya-metal-hy4 --pack DIR [--serve | --ids 1,2,3 [--n N]]\n"
                 "  --max-context N      tokens the caches hold (default 32768, or the pack's own context if smaller)\n"
                 "  --chunk N            prompt tokens per window slice (default 512)\n"
                 "  --engram-cache-gb N  Engram n-gram row cache in RAM, GB (default 1; 0 disables)\n"
                 "  --cpu                run on the scalar CPU reference instead of the GPU\n");
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        v.push_back((int32_t) std::strtol(p, &end, 10));
        p = *end == ',' ? end + 1 : end;
        if (end == p && *p) break;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    // Residency: keep the weight buffers wired across idle gaps (the mm3 main's reasoning:
    // at ~117 GB resident the unwire costs a long stall when generation resumes).
    setenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", "86400", 0);
    std::string pack_dir;
    bool serve = false, cpu = false;
    std::vector<int32_t> ids;
    int n_gen = 32;
    int64_t max_ctx = 0;
    int chunk = 512;
    double eg_gb = 1.0;
    std::vector<std::string> noted;
    bool mtp = false, vision = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "maya-metal-hy4: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") pack_dir = next();
        else if (a == "--serve") serve = true;
        else if (a == "--ids") ids = parse_ids(next());
        else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--cpu") cpu = true;
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--chunk") chunk = std::atoi(next());
        else if (a == "--engram-cache-gb") eg_gb = std::atof(next());
        else if (a == "--mtp") mtp = true;
        else if (a == "--vision") vision = std::strcmp(next(), "off") != 0;
        else if (a == "--gpus" || a == "--tp" || a == "--reserve-gb" || a == "--cpu-threads") {
            noted.push_back(a + " " + next());
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (!a.empty() && a[0] != '-') {
            pack_dir = a;   // stage 0's positional form
        } else {
            noted.push_back(a);
        }
    }
    if (pack_dir.empty()) {
        usage();
        return 2;
    }
    hym::Weights w;
    std::string err;
    if (!w.load(pack_dir, err)) {
        std::fprintf(stderr, "maya-metal-hy4: %s\n", err.c_str());
        return 1;
    }
    const hy4::Geo& g = w.pk.g;
    std::fprintf(stderr,
                 "maya-metal-hy4: pack ok: %d layers + %d mtp, embd %d, vocab %d, %d experts (%d/token), "
                 "engram rows %lld\n",
                 g.n_layer, g.n_mtp, g.n_embd, g.n_vocab, g.n_expert, g.n_used, (long long) w.pk.eg.rows);
    if (!serve && ids.empty()) return 0;   // the stage-0 load check
    if (serve && cpu) {
        std::fprintf(stderr, "maya-metal-hy4: --serve runs the Metal forward; --cpu is the --ids oracle\n");
        return 2;
    }
    if (max_ctx <= 0 || max_ctx > g.context) max_ctx = std::min<int64_t>(g.context, 32768);
    if (mtp)
        noted.push_back("--mtp (MTP drafting is not ported yet: plain generation, the same tokens)");
    if (vision)
        noted.push_back("--vision (the vision tower is stage 5 of the port: text only, image requests refused)");

    ggml_backend_t be = nullptr;
    if (!cpu) {
        ggml_backend_load_all();
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!be) {
            std::fprintf(stderr, "maya-metal-hy4: no GPU backend\n");
            return 1;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<hym::Forward> fwd =
        cpu ? hym::make_ref_forward(w, max_ctx) : hym::make_ggml_forward(w, be, max_ctx, false, eg_gb);
    if (!fwd) {
        std::fprintf(stderr, "maya-metal-hy4: forward init failed\n");
        return 1;
    }
    {
        const hym::TierInfo ti = fwd->tier_info();
        std::fprintf(stderr, "maya-metal-hy4: engram row cache %.2f GB (%lld rows of %zu B), experts resident %.1f GB\n",
                     (double) ti.eg_cache_cap * (double) ti.eg_row_bytes / 1073741824.0,
                     (long long) ti.eg_cache_cap, ti.eg_row_bytes, ti.resident_gb);
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "maya-metal-hy4: forward ready in %.1f s (%s, context %lld)\n", load_s,
                 cpu ? "cpu reference" : "metal", (long long) max_ctx);
    for (const std::string& n : noted)
        if (serve) std::printf("NOTE this engine ignores %s\n", n.c_str());
        else std::fprintf(stderr, "maya-metal-hy4: ignoring %s\n", n.c_str());
    if (serve) {
        const int rc = hym::serve_hy4(*fwd, g, max_ctx, chunk);
        fwd.reset();
        if (be) ggml_backend_free(be);
        return rc;
    }

    // --ids: greedy continuation, timed (the mm3 main's shape).  The reference forward
    // writes every row's logits (the ggml one only the last), so the prompt is fed into a
    // chunk-sized buffer and the last row taken.
    std::vector<float> logits((size_t) g.n_vocab);
    std::vector<float> rows((size_t) chunk * (size_t) g.n_vocab);
    auto run = [&](const int* p, int T) {
        if (!fwd->run(p, T, rows.data(), nullptr, err)) {
            std::fprintf(stderr, "maya-metal-hy4: %s\n", err.c_str());
            std::exit(1);
        }
        // the reference writes all T rows (last at T-1); the ggml forward only the last, at 0
        std::memcpy(logits.data(), rows.data() + (cpu ? (size_t) (T - 1) : 0) * (size_t) g.n_vocab,
                    (size_t) g.n_vocab * sizeof(float));
    };
    auto t1 = std::chrono::steady_clock::now();
    for (size_t at = 0; at < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) chunk, ids.size() - at);
        run((const int*) ids.data() + at, T);
        at += (size_t) T;
    }
    const double pp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("prompt %zu tokens in %.2f s (%.1f tok/s)\n", ids.size(), pp, pp > 0 ? ids.size() / pp : 0.0);
    auto argmax = [&] {
        int32_t best = 0;
        for (int v = 1; v < g.n_vocab; ++v)
            if (logits[(size_t) v] > logits[(size_t) best]) best = v;
        return best;
    };
    std::vector<int32_t> out;
    t1 = std::chrono::steady_clock::now();
    for (int32_t t = argmax(); (int) out.size() < n_gen;) {
        out.push_back(t);
        if ((int) out.size() >= n_gen) break;
        run((const int*) &t, 1);
        t = argmax();
    }
    const double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("tokens:");
    for (int32_t t : out) std::printf(" %d", t);
    std::printf("\ndecode %d tokens in %.2f s (%.2f tok/s)\n", (int) out.size(), tg,
                tg > 0 ? out.size() / tg : 0.0);
    fwd.reset();
    if (be) ggml_backend_free(be);
    return 0;
}
