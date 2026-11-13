// src/metal/spec_backoff.hpp - adaptive draft backoff: stop proposing when measured yield says
// drafting is losing money, re-probe after a cooldown of plain decoding.
//
// Why this exists (the prose mechanism, AUDIT-MINIMAX): in the mm3 serve loop a round pays the
// draft proposal BEFORE the confidence floor can cut anything, so on low-agreement text (prose)
// every round pays propose + verify-window + rewind and keeps 0-1 tokens - strictly worse than
// the plain 1-row loop at every floor setting. No floor can recover a cost already paid; the
// only fix is to not propose. This policy is that fix, re-applied from the 2026-11-01 design
// (ROUND2-d1-backoff-impl.md; its live failure that night was the 98,304 context cap's memory
// thrash, not the policy - the policy was exonerated and reverted untested on a healthy cap).
//
// The policy, exact. One instance per generation. The serve loop consults it in three places:
//   1. iteration gate: while drafting() is false the loop takes its plain path verbatim and
//      calls note_plain_token() once per plain-decoded token (burns one cooldown token).
//      draft_propose is never called while parked, so the drafter's forward is not paid either.
//   2. verify-round sample: after the keep/rewind walk, note_round(kept, k) feeds the EWMA the
//      round's yield kept/k.
//   3. floor-cut sample: when a paid propose is cut to k == 0 by the floor, note_floor_cut()
//      feeds a zero-yield sample (an empty propose - no room - is never sampled).
// Correctness does not depend on the policy: every emitted token still comes from the target's
// own pick over its own rows on either path; backoff changes WHEN drafts are proposed, never
// WHAT is accepted. With the policy disabled, drafting() is unconditionally true and every
// note_* is a no-op, so the loop executes exactly the pre-backoff code paths.
//
// Break-even derivation (plain-step units: one single-row target forward = 1 step). The
// 2026-11 counting sweep at conf 0.3 made 34.5 tok/s vs 13.1 spec-off (2.63x) at aggregate
// yield 0.774 with ~5.3 drafts offered per round, i.e. ~5.1 tokens per round, so a full spec
// round costs ~5.1 / 2.63 ~ 2.0 plain steps. Decomposed as C_round(k) ~ 1 + C_PROPOSE +
// C_ROW*k that gives C_PROPOSE ~ 0.25 (drafter block forward + tap ingest) and C_ROW ~ 0.14
// (each extra verify row on the batched path). A round emits kept + 1 tokens, so drafting pays
// iff kept > 0.25 + 0.14k, i.e. yield kept/k > 0.14 + 0.25/k - 0.18 at k = 7, 0.22 at k = 3,
// 0.39 at k = 1. The 0.35 trigger sits above the break-even band for k >= 2 with margin
// (borderline-marginal drafting backs off rather than bleeds), far below measured productive
// drafting (counting 0.76-0.98), and the floor-cut zero samples price the propose tax into
// the EWMA.
#pragma once

#include <algorithm>

namespace maya::metal {

class SpecBackoff {
public:
    static constexpr float kYieldThreshold = 0.35f;   // EWMA yield below this trips a cooldown
    static constexpr float kEwmaAlpha = 0.25f;        // per-sample EWMA weight (~7-round window)
    static constexpr int kWarmupRounds = 4;           // samples required before a trip is possible
    static constexpr int kCoolBase = 32;              // first cooldown, in plain-decoded tokens
    static constexpr int kCoolMax = 256;              // escalation cap, tokens

    explicit SpecBackoff(bool enabled = true) : enabled_(enabled) {}

    bool enabled() const { return enabled_; }
    // false while a cooldown is being burned down: the loop must take its plain path
    bool drafting() const { return !enabled_ || cooldown_ <= 0; }
    float ewma() const { return ewma_; }
    float trip_ewma() const { return trip_ewma_; }   // the EWMA that crossed at the last trip
    int trips() const { return trips_; }

    // one plain-decoded token emitted while parked: burns one cooldown token. (Also called on
    // the plain path when spec is off entirely; outside a cooldown it changes nothing.)
    void note_plain_token() {
        if (!enabled_) return;
        if (cooldown_ > 0) --cooldown_;
    }

    // a verify round that kept `kept` of the `k` drafts the floor let through. Returns the
    // cooldown length if this sample tripped the policy, else 0. Nothing is sampled during a
    // cooldown (the loop's shape guarantees it; the guard here keeps it true by construction).
    int note_round(int kept, int k) {
        if (!enabled_ || cooldown_ > 0 || k <= 0) return 0;
        if (kept >= k) cool_next_ = kCoolBase;   // a perfect round resets the escalation
        return sample((float) kept / (float) k);
    }

    // a propose was paid and the confidence floor cut every draft: a zero-yield sample.
    int note_floor_cut() {
        if (!enabled_ || cooldown_ > 0) return 0;
        return sample(0.0f);
    }

private:
    int sample(float yield) {
        ewma_ = samples_ == 0 ? yield : ewma_ + kEwmaAlpha * (yield - ewma_);
        ++samples_;
        if (samples_ >= kWarmupRounds && ewma_ < kYieldThreshold) {
            trip_ewma_ = ewma_;
            cooldown_ = cool_next_;
            cool_next_ = std::min(cool_next_ * 2, kCoolMax);
            ++trips_;
            ewma_ = 0.0f;   // yield is re-measured fresh after the cooldown
            samples_ = 0;
            return cooldown_;
        }
        return 0;
    }

    bool enabled_;
    float ewma_ = 0.0f, trip_ewma_ = 0.0f;
    int samples_ = 0;
    int cooldown_ = 0;              // plain tokens left to burn before drafting resumes
    int cool_next_ = kCoolBase;     // the next trip's cooldown (doubles per trip, capped)
    int trips_ = 0;
};

}  // namespace maya::metal
