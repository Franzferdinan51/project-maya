// src/metal/metal_mm3_backoff_test.cpp - the adaptive draft backoff (spec_backoff.hpp) on the
// synthetic fixture, in the divergence test's style:
//   part 1: the policy object alone, scripted samples - the trip arithmetic (EWMA, warmup,
//           cooldown drain, escalation, perfect-round reset, floor-cut samples, disabled
//           inertness) checked value by value;
//   part 2: a replica of mm3_serve.cpp's decode loop WITH the backoff hooks, driven by
//           fabricated drafts (K = 4): a low phase where every draft is wrong and a high
//           phase where drafts are the true next tokens. The emitted tokens must be plain
//           greedy's with the policy on and with it disabled, while the round counts show
//           the mechanism working: verify rounds collapse in the low phase, drafting parks
//           and re-probes on the escalating cooldowns, and resumes for good in the high phase.
// The policy changes WHEN drafts are proposed, never WHAT is accepted, so token identity
// against the plain reference is the correctness gate; the fixture is far too small for its
// timing to mean anything (fabricated drafts pay no propose cost), and part 2's wall times
// are printed for shape only.
// usage: metal_mm3_backoff_test <fixture.gguf>
#include "mm3_metal.hpp"
#include "spec_backoff.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

void check_ewma(float got, float want, const char* what) {
    if (std::fabs((double) got - (double) want) > 1e-6) {
        std::printf("FAIL: %s (ewma %.6f, want %.6f)\n", what, (double) got, (double) want);
        ++g_fail;
    }
}

std::string g_err;

int32_t argmax(const std::vector<float>& l) {
    size_t best = 0;
    for (size_t v = 1; v < l.size(); ++v)
        if (l[v] > l[best]) best = v;
    return (int32_t) best;
}

bool prefill_serve_style(maya::metal::Mm3Metal& m, const std::vector<int32_t>& ids,
                         std::vector<float>& logits) {
    std::vector<int32_t> head(ids.begin(), ids.end() - 1);
    return m.forward(head, logits, g_err) && m.forward({ids.back()}, logits, g_err);
}

// ---- part 1: the policy, scripted ----
void part1() {
    using maya::metal::SpecBackoff;

    {   // disabled = inert over zero-yield rounds, floor cuts and plain tokens
        SpecBackoff b(false);
        bool any = false;
        for (int i = 0; i < 20; ++i) any |= b.note_round(0, 7) != 0;
        for (int i = 0; i < 20; ++i) any |= b.note_floor_cut() != 0;
        for (int i = 0; i < 50; ++i) b.note_plain_token();
        check(!any && b.drafting() && b.trips() == 0, "disabled policy must be inert");
    }
    {   // counting-like yields (5/7, 6/7 alternating) never trip
        SpecBackoff b(true);
        bool tripped = false;
        for (int i = 0; i < 200; ++i) {
            tripped |= b.note_round(i % 2 ? 6 : 5, 7) != 0;
            tripped |= !b.drafting();
        }
        check(!tripped && b.trips() == 0, "counting-like yields must never trip");
    }
    {   // zero-yield rounds trip exactly at the 4th sample, cooldown 32, draining token for token
        SpecBackoff b(true);
        check(b.note_round(0, 7) == 0 && b.drafting(), "no trip inside warmup (1)");
        check(b.note_round(0, 7) == 0 && b.drafting(), "no trip inside warmup (2)");
        check(b.note_round(0, 7) == 0 && b.drafting(), "no trip inside warmup (3)");
        check(b.note_round(0, 7) == 32 && !b.drafting() && b.trips() == 1,
              "trip at the 4th zero-yield sample, cooldown 32");
        for (int i = 0; i < 31; ++i) b.note_plain_token();
        check(!b.drafting(), "still parked at 31 plain tokens");
        b.note_plain_token();
        check(b.drafting(), "free at 32 plain tokens");
    }
    {   // nothing is sampled during a cooldown; the next trip escalates to 64
        SpecBackoff b(true);
        for (int i = 0; i < 4; ++i) b.note_round(0, 7);
        check(b.trips() == 1 && !b.drafting(), "tripped before the sampling-during-cooldown probe");
        bool any = false;
        for (int i = 0; i < 10; ++i) any |= b.note_round(0, 7) != 0;
        for (int i = 0; i < 5; ++i) any |= b.note_floor_cut() != 0;
        check(!any && b.trips() == 1, "no sampling during a cooldown");
        for (int i = 0; i < 32; ++i) b.note_plain_token();
        check(b.drafting(), "cooldown drained after the probe");
        check(b.note_round(0, 7) == 0 && b.note_round(0, 7) == 0 && b.note_round(0, 7) == 0,
              "fresh warmup after the cooldown (samples during it were not counted)");
        check(b.note_round(0, 7) == 64, "second trip escalates to 64");
    }
    {   // escalation across re-trips: 32, 64, 128, 256, 256
        SpecBackoff b(true);
        const int want[5] = {32, 64, 128, 256, 256};
        bool ok = true;
        for (int t = 0; t < 5; ++t) {
            int cool = 0;
            for (int i = 0; i < 4; ++i) cool = b.note_round(0, 7);
            ok &= cool == want[t];
            for (int i = 0; i < cool; ++i) b.note_plain_token();
            ok &= b.drafting();
        }
        check(ok && b.trips() == 5, "escalation 32 -> 64 -> 128 -> 256 -> 256");
    }
    {   // a perfect round resets the escalation; the EWMA arithmetic crosses on the 5th sample
        SpecBackoff b(true);
        for (int t = 0; t < 2; ++t) {   // two trips: the next cooldown would be 128
            int cool = 0;
            for (int i = 0; i < 4; ++i) cool = b.note_round(0, 7);
            for (int i = 0; i < cool; ++i) b.note_plain_token();
        }
        check(b.note_round(7, 7) == 0, "perfect round does not trip");
        check_ewma(b.ewma(), 1.0f, "ewma after a perfect round");
        check(b.note_round(0, 7) == 0, "zero round 1 after perfect");
        check_ewma(b.ewma(), 0.75f, "ewma 0.75");
        check(b.note_round(0, 7) == 0, "zero round 2 after perfect");
        check_ewma(b.ewma(), 0.5625f, "ewma 0.5625");
        check(b.note_round(0, 7) == 0, "zero round 3 after perfect: 0.4219 is still above 0.35");
        check_ewma(b.ewma(), 0.421875f, "ewma 0.421875");
        check(b.note_round(0, 7) == 32, "trip on the 5th sample cools 32, not 128 (escalation reset)");
        check_ewma(b.trip_ewma(), 0.31640625f, "trip ewma 0.31640625");
    }
    {   // floor cuts alone trip on the 4th sample
        SpecBackoff b(true);
        check(b.note_floor_cut() == 0 && b.note_floor_cut() == 0 && b.note_floor_cut() == 0 &&
                  b.drafting(),
              "floor cuts inside warmup do not trip");
        check(b.note_floor_cut() == 32 && b.trips() == 1, "floor cuts trip on the 4th sample");
    }
    std::printf("part 1: policy unit checks %s\n", g_fail ? "FAILED" : "passed");
}

// ---- part 2: the serve-loop replica with the backoff hooks ----
struct ReplicaOut {
    std::vector<int32_t> emitted;
    int rounds_low = 0, rounds_high = 0, plain = 0;
    int trips_at_switch = -1, trips_end = 0;
    double seconds = 0.0;
};

ReplicaOut replica(maya::metal::Mm3Metal& model, const std::vector<int32_t>& prompt,
                   const std::vector<int32_t>& ref, bool backoff_enabled) {
    const int K = 4, TARGET = 460, SWITCH = 232;
    const int V = model.geometry().n_vocab;
    maya::metal::SpecBackoff boff(backoff_enabled);
    ReplicaOut out;
    model.reset();
    std::vector<float> logits;
    if (!prefill_serve_style(model, prompt, logits)) return out;
    const auto t0 = std::chrono::steady_clock::now();
    int32_t y = argmax(logits);
    out.emitted.push_back(y);
    while ((int) out.emitted.size() < TARGET) {
        const int64_t pos = model.position();
        if (!boff.drafting()) {
            if (!model.forward({y}, logits, g_err)) return out;
            y = argmax(logits);
            out.emitted.push_back(y);
            boff.note_plain_token();
            ++out.plain;
            continue;
        }
        const int t = (int) out.emitted.size() - 1;   // y == ref[t] while identity holds
        const bool low = (int) out.emitted.size() < SWITCH;
        if (!low && out.trips_at_switch < 0) out.trips_at_switch = boff.trips();
        if (t + 1 + K >= (int) ref.size()) break;
        std::vector<int32_t> drafts;
        for (int j = 0; j < K; ++j)
            drafts.push_back(low ? (ref[(size_t) (t + 1 + j)] + 137) % V : ref[(size_t) (t + 1 + j)]);
        std::vector<int32_t> win(1, y);
        win.insert(win.end(), drafts.begin(), drafts.end());
        std::vector<float> rows;
        if (!model.forward_rows(win, rows, g_err)) return out;
        if (low) ++out.rounds_low; else ++out.rounds_high;
        int kept = 0;
        int32_t next = -1;
        for (int i = 0; i <= K; ++i) {
            const int32_t cand = argmax(std::vector<float>(rows.begin() + (size_t) i * V,
                                                           rows.begin() + (size_t) (i + 1) * V));
            if (i == K || cand != drafts[(size_t) i]) {
                next = cand;
                break;
            }
            out.emitted.push_back(drafts[(size_t) i]);
            kept = i + 1;
        }
        model.rewind_to(pos + 1 + kept);
        boff.note_round(kept, K);
        y = next;
        out.emitted.push_back(y);
    }
    if (out.trips_at_switch < 0) out.trips_at_switch = boff.trips();
    out.trips_end = boff.trips();
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

int first_diff(const std::vector<int32_t>& a, const std::vector<int32_t>& ref) {
    const size_t n = std::min(a.size(), ref.size());
    for (size_t i = 0; i < n; ++i)
        if (a[i] != ref[i]) return (int) i;
    return a.size() == n ? -1 : (int) n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: metal_mm3_backoff_test <fixture.gguf>\n");
        return 2;
    }
    part1();

    const std::string gguf = argv[1];
    const int NREF = 520;
    maya::metal::Mm3Options o;
    if (const char* v = std::getenv("MAYA_MM3_CPU"); v && *v && *v != '0') o.use_gpu = false;
    maya::metal::Mm3Metal model;
    model.set_options(o);
    if (!model.load(gguf, 1200, g_err)) {
        std::fprintf(stderr, "FAIL: load: %s\n", g_err.c_str());
        return 1;
    }
    const int V = model.geometry().n_vocab;
    std::fprintf(stderr, "backoff test on: %s\n", model.describe().c_str());

    std::vector<int32_t> prompt;
    uint32_t s = 12345;
    for (int i = 0; i < 48; ++i) {
        s = s * 1664525u + 1013904223u;
        prompt.push_back((int32_t) ((s >> 16) % (uint32_t) V));
    }
    // the plain greedy reference (single-row forwards, serve-style prefill)
    std::vector<float> logits;
    if (!prefill_serve_style(model, prompt, logits)) {
        std::fprintf(stderr, "FAIL: prefill: %s\n", g_err.c_str());
        return 1;
    }
    std::vector<int32_t> ref;
    {
        int32_t y = argmax(logits);
        ref.push_back(y);
        while ((int) ref.size() < NREF) {
            if (!model.forward({y}, logits, g_err)) {
                std::fprintf(stderr, "FAIL: forward: %s\n", g_err.c_str());
                return 1;
            }
            y = argmax(logits);
            ref.push_back(y);
        }
    }

    const ReplicaOut on = replica(model, prompt, ref, true);
    const int d_on = first_diff(on.emitted, ref);
    std::printf("backoff ON : verify rounds low=%d high=%d, plain=%d, trips at switch=%d end=%d, "
                "%zu tokens in %.2f s, %s\n",
                on.rounds_low, on.rounds_high, on.plain, on.trips_at_switch, on.trips_end,
                on.emitted.size(), on.seconds,
                d_on < 0 ? "token-IDENTICAL to plain greedy" : "DIVERGED");
    if (d_on >= 0) std::printf("  backoff ON first divergence at emitted index %d\n", d_on);
    const ReplicaOut off = replica(model, prompt, ref, false);
    const int d_off = first_diff(off.emitted, ref);
    std::printf("backoff OFF: verify rounds low=%d high=%d, plain=%d, trips at switch=%d end=%d, "
                "%zu tokens in %.2f s, %s\n",
                off.rounds_low, off.rounds_high, off.plain, off.trips_at_switch, off.trips_end,
                off.emitted.size(), off.seconds,
                d_off < 0 ? "token-IDENTICAL to plain greedy" : "DIVERGED");
    if (d_off >= 0) std::printf("  backoff OFF first divergence at emitted index %d\n", d_off);

    check(d_on < 0, "backoff ON tokens must be plain greedy's");
    check(d_off < 0, "backoff OFF (disabled) tokens must be plain greedy's");
    check(on.trips_at_switch >= 2, "the low phase must trip the backoff at least twice");
    check(on.rounds_low > 0 && on.rounds_low <= 20,
          "low-phase verify rounds must collapse (probes only, ~12)");
    check(on.plain >= 150, "the low phase must be mostly plain-decoded under backoff");
    check(off.plain == 0 && off.trips_end == 0, "disabled control: no plain iterations, no trips");
    check(off.rounds_low >= 200, "disabled control drafts every low-phase round");
    check(on.rounds_high >= 30, "drafting must resume and stay on in the high phase");
    check(on.trips_end == on.trips_at_switch, "no trips during the high phase");

    std::printf("part 2: serve-loop replica %s\n", g_fail ? "FAILED" : "passed");
    if (g_fail) {
        std::printf("BACKOFF TEST: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("BACKOFF TEST: all checks passed\n");
    return 0;
}
