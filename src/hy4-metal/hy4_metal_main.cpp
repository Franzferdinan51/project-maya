// src/hy4-metal/hy4_metal_main.cpp - maya-metal-hy4: the Metal port of upstream's
// maya-hy4 (src/hy4, CUDA), following the fork's Metal engine layout.
//
// Stage 0 (2026-11-19): the pack loader is upstream's own host code
// (src/hy4/hy4_pack.hpp) and works today - give the engine a pack directory
// (tools/hy4/pack_gguf.py's output) and it loads and reports the geometry.
// The Metal forward pass is NOT ported yet, so the engine then stops loudly:
// the fork's stub rule (name what is not ported, never answer silently wrong).
// Staged plan: HY4-METAL-PORT-PLAN.md in the Maya goal workspace.
//
// Usage: maya-metal-hy4 <pack-dir>

#include "hy4_pack.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    std::fprintf(stderr, "maya-metal-hy4: Hy4-Flash on Metal (port in progress)\n");
    if (argc < 2) {
        std::fprintf(stderr, "usage: maya-metal-hy4 <pack-dir>\n");
        return 2;
    }
    hy4::Pack pack;
    std::string err;
    if (!pack.load(argv[1], err)) {
        std::fprintf(stderr, "maya-metal-hy4: %s\n", err.c_str());
        return 1;
    }
    const hy4::Geo& g = pack.g;
    std::fprintf(stderr,
                 "pack ok: %d layers + %d mtp, embd %d, vocab %d, %d experts (%d/token), "
                 "engram rows %lld - generation NOT PORTED YET (stage 0 scaffold)\n",
                 g.n_layer, g.n_mtp, g.n_embd, g.n_vocab, g.n_expert, g.n_used,
                 (long long) pack.eg.rows);
    return 1;  // loud stop: the pack loaded, but no Metal forward pass exists yet
}
