// src/metal/metal_glm_backoff_test.cpp - the adaptive draft backoff (spec_backoff.hpp) in GLM's
// decode_spec: a scripted drafter whose drafts are all wrong for the first half of a generation and
// perfect for the second.  With the policy on, verify rounds in the bad half must nearly vanish
// (the done-round tally FREEZES while plain tokens flow - no phantom rounds fed to DraftLength),
// drafting must resume and stay resumed in the good half, and the emitted tokens must equal plain
// greedy decoding token for token.  With the policy disabled, every bad round is paid and tallied.
#include "glm_metal.hpp"
#include "spec.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using maya::metal::GlmMetal;
using maya::metal::SpecBackoff;

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "metal_glm_backoff_test: %s\n", what.c_str());
        std::exit(1);
    }
}

int argmax(const float* l, size_t V) {
    return (int) (std::max_element(l, l + V) - l);
}

std::vector<float> fwd(GlmMetal& m, const std::vector<int32_t>& toks) {
    std::vector<float> lg;
    std::string err;
    require(m.forward(toks, lg, err), "forward: " + err);
    return lg;
}

}   // namespace

int main(int argc, char** argv) {
    const std::string gguf = argc > 1 ? argv[1] : "data/glm5-synth.gguf";
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    const int64_t P = (int64_t) prompt.size();
    const int N_GEN = 462, K = 4, SWITCH = 232;

    GlmMetal m;
    {
        maya::metal::GlmOptions o;
        o.max_window = 8;
        const char* cpu = std::getenv("MAYA_METAL_CPU");
        o.use_gpu = !(cpu && *cpu && std::atoi(cpu) != 0);
        m.set_options(o);
        std::string err;
        require(m.load(gguf, 1024, err), "load: " + err);
    }
    const size_t V = (size_t) m.geometry().n_vocab;
    std::string err;

    // plain greedy reference
    std::vector<int32_t> ref;
    std::vector<float> lg = fwd(m, prompt);
    for (int i = 0; i < N_GEN; ++i) {
        ref.push_back(argmax(lg.data(), V));
        if (i + 1 < N_GEN) lg = fwd(m, {ref.back()});
    }

    struct Run {
        std::vector<int32_t> got;
        std::vector<int> tally_at;   // emissions when each done-round fired
        int rounds = 0, trips = 0;
    };
    auto run = [&](bool enabled) {
        Run r;
        m.reset();
        lg = fwd(m, prompt);
        maya::metal::DecodeHooks h;
        h.pick = [&](const float* l) { return argmax(l, V); };
        h.emit = [&](int32_t t) {
            r.got.push_back(t);
            return (int) r.got.size() < N_GEN;
        };
        h.left = [&] { return (int64_t) (N_GEN - (int) r.got.size()); };
        const int32_t y = argmax(lg.data(), V);
        require(h.emit(y), "n_gen < 2");
        const maya::metal::Drafter draft = [&](int32_t, int64_t p0, int room, std::vector<int32_t>& d,
                                               std::string&) {
            const bool low = (int) r.got.size() < SWITCH;
            for (int j = 0; j < room; ++j) {
                const int64_t idx = p0 + 1 + j - P;   // the ref token emitted for that position
                if (idx < 0 || idx >= (int64_t) ref.size()) break;
                const int32_t want = ref[(size_t) idx];
                d.push_back(low ? (int32_t) ((want + 137) % (int) V) : want);
            }
            return true;
        };
        SpecBackoff boff(enabled);
        require(maya::metal::decode_spec(m, y, K, draft,
                                         [&](int, int, double) {
                                             ++r.rounds;
                                             r.tally_at.push_back((int) r.got.size());
                                         },
                                         h, err, &boff),
                "decode_spec: " + err);
        r.trips = boff.trips();
        return r;
    };

    const Run on = run(true);
    require(on.got == ref, "backoff ON: tokens differ from plain greedy");
    require(on.trips >= 2, "backoff ON: expected at least 2 trips in the bad half");
    int low_calls = 0, high_calls = 0, max_gap = 0;
    for (size_t i = 0; i < on.tally_at.size(); ++i) {
        if (on.tally_at[i] < SWITCH) ++low_calls;
        else ++high_calls;
        if (i > 0) max_gap = std::max(max_gap, on.tally_at[i] - on.tally_at[i - 1]);
    }
    require(low_calls <= 20, "backoff ON: verify rounds did not collapse in the bad half");
    require(max_gap >= 32, "backoff ON: no parked stretch (tally froze while plain tokens flowed)");
    require(high_calls >= 30, "backoff ON: drafting did not resume in the good half");
    std::printf("backoff ON : rounds low=%d high=%d, max tally gap=%d, trips=%d, token-IDENTICAL\n",
                low_calls, high_calls, max_gap, on.trips);

    const Run off = run(false);
    require(off.got == ref, "backoff OFF: tokens differ from plain greedy");
    require(off.trips == 0, "backoff OFF: the disabled policy tripped");
    int off_low = 0, off_gap = 0;
    for (size_t i = 0; i < off.tally_at.size(); ++i) {
        if (off.tally_at[i] < SWITCH) ++off_low;
        if (i > 0) off_gap = std::max(off_gap, off.tally_at[i] - off.tally_at[i - 1]);
    }
    require(off_low >= 200, "backoff OFF: expected every bad round paid and tallied");
    require(off_gap <= 8, "backoff OFF: unexpected tally gap (a round went missing)");
    std::printf("backoff OFF: rounds low=%d, max tally gap=%d, trips=0, token-IDENTICAL\n", off_low, off_gap);

    std::printf("GLM BACKOFF TEST: all checks passed\n");
    return 0;
}
