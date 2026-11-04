// src/metal/mm3_serve.cpp - maya-metal-mm3 --serve: the serve protocol of src/metal/serve.cpp (itself
// src/program/generate.cpp's loop), so serve/server.py drives the MiniMax Metal engine exactly as it
// drives the GLM one.
//
//   in:  GEN <max_new> [key=value ...] <id,id,...>   STOP   PAUSE [keep]   WRAP   QUIT
//   out: INFO ... / READY <ctx> stop pause, then per request PP ..., T <id> ..., DONE ... (or ERR ...)
//
// Sampling is the shared chain (sampler.cpp): same keys, same defaults, same Philox draws, so a seeded
// request repeats.  The differences from the GLM engine's serve are the model's, not the protocol's:
// DSpark drafting when a draft is loaded (mm3_dspark.cpp; the spec keys steer it, and a verify
// window writes the same tokens plain decoding would), no images (GENI is refused: the vision
// tower is not wired on this backend yet), cross-request restore points (GEN marks=/shared=,
// the GLM engine's P5 protocol: named prefix depths recorded during prefill that a later request
// whose prompt starts with that prefix resumes from - here an entry is just a position, since the
// caches are positional), no expert
// pool (every expert is resident, so the STAT lines report a 1.0 hit rate and no pool), and the stop
// tokens are the GGUF's own eos ids instead of a hardcoded set.  PAUSE and the prefix snapshot are
// cheaper here than anywhere: the caches are positional, so a snapshot is a position and restoring
// it is rewinding to it.  The loop guard is generate.cpp's, with the answer-side force closing the
// turn on the model's first eos id (the GLM original forces its template's user marker to the same
// effect).
#include "mm3_metal.hpp"
#include "sampler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef MAYA_VERSION
#define MAYA_VERSION "dev"
#endif

namespace maya::metal {

namespace {

int env_int(const char* k, int def) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : def;
}

// stdin on its own thread: STOP / PAUSE / WRAP act at once (a running request sees them), the rest queue up
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false}, wrap{false}, pause{false};
    bool eof = false;

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;
            if (line == "STOP" || line.rfind("STOP ", 0) == 0 || line == "PAUSE" || line.rfind("PAUSE ", 0) == 0) {
                if (line.rfind("PAUSE", 0) == 0) pause.store(true);
                stop.store(true);
                continue;
            }
            if (line == "WRAP") {
                wrap.store(true);
                continue;
            }
            if (line == "QUIT") stop.store(true);
            if (line.rfind("GEN", 0) == 0) {
                stop.store(false);
                pause.store(false);
                wrap.store(false);
            }
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(line);
            cv.notify_all();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_all();
    }
    bool next(std::string& line) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        line = std::move(q.front());
        q.pop_front();
        return true;
    }
};

// a request's restore points (GEN keys marks=/shared=, the DS4 engine's spellings, as on the GLM
// engine): positions in the prompt where prefill should record a reusable prefix, shared= the one
// another conversation may resume from
struct Restore {
    std::vector<int64_t> marks;
    int64_t shared = -1;
};

struct Server {
    Mm3Metal& model;
    Input in;
    std::vector<int32_t> cached;      // the previous prompt less its last token: what the snapshot holds
    // cross-request restore points (P5): prefixes an earlier request recorded at its marks=/shared=
    // points.  An entry certifies "the first toks.size() positions hold this prefix" - the caches are
    // positional, so the state at a depth is the buffer itself and restoring is rewind_to(depth).
    // run() keeps that true by dropping every entry deeper than a request's reuse depth before that
    // request's prefill overwrites from there, and by clearing the table wherever it resets.
    struct PrefixEntry {
        std::vector<int32_t> toks;    // the prefix (its size is the position)
        bool shared = false;          // recorded from a shared= point: the eviction victim of last resort
    };
    std::vector<PrefixEntry> entries;
    static constexpr size_t kMaxEntries = 8;
    const int loop_w_think = env_int("MAYA_MM3_LOOP_THINK", 256);
    const int loop_w_answer = env_int("MAYA_MM3_LOOP_ANSWER", 1024);
    int chunk = 512;                  // prompt tokens a forward() takes (serve())
    bool spec_on = true;        // DSpark drafting when a draft is loaded (a GEN's spec= overrides)
    float conf_floor = 0.7f;    // a round verifies the drafts before the first below this confidence
    float draft_cost = 0.0f;    // > 0: ... and before the first whose confidences' product is below it

    void drop_entry(size_t i) { entries.erase(entries.begin() + (long) i); }

    // every entry dies (the engine's state was reset or discarded: the positions no longer certify
    // the prefixes they held)
    void clear_entries() { entries.clear(); }

    // capture the prefix at position p (the prompt's first p tokens, just prefilled) as an entry
    void record_point(const std::vector<int32_t>& prompt, int64_t p, bool is_shared) {
        for (size_t i = 0; i < entries.size();) {   // one entry per depth: this request's prefix replaces it
            if ((int64_t) entries[i].toks.size() == p) drop_entry(i);
            else ++i;
        }
        if (entries.size() >= kMaxEntries) {   // full: the deepest non-shared entry goes (the root stays)
            long victim = -1;
            for (size_t i = 0; i < entries.size(); ++i)
                if (!entries[i].shared && (victim < 0 || entries[i].toks.size() >= entries[(size_t) victim].toks.size()))
                    victim = (long) i;
            if (victim < 0)   // every entry is a shared root: the deepest goes
                for (size_t i = 0; i < entries.size(); ++i)
                    if (victim < 0 || entries[i].toks.size() >= entries[(size_t) victim].toks.size())
                        victim = (long) i;
            if (victim >= 0) drop_entry((size_t) victim);
        }
        PrefixEntry e;
        e.toks.assign(prompt.begin(), prompt.begin() + (long) p);
        e.shared = is_shared;
        entries.push_back(std::move(e));
    }

    void err(const std::string& m) {
        std::printf("ERR %s\n", m.c_str());
        std::fflush(stdout);
    }

    void gen(const std::string& line) {
        std::istringstream ss(line);
        std::string cmd, tok;
        long long max_new = 0;
        ss >> cmd >> max_new;
        if (cmd == "GENI") return err("maya-metal-mm3 serves text only (the vision tower is not wired yet)");
        Sampling sp;
        bool any_key = false, greedy_req = false;
        int64_t think_budget = 0;
        int32_t think_end = -1;
        int spec_req = -1;      // GEN spec: -1 the engine default, 0/1 off, >= 2 on
        float conf_req = -1.0f;   // GEN dspark_conf / spec_min_p: the round's confidence floor
        int pen_n = 0;
        float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
        Restore restore;
        std::string ids_s, bias_err;
        const int n_vocab = model.geometry().n_vocab;
        while (ss >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) {
                ids_s = tok;
                continue;
            }
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "temperature") {
                sp.temperature = (float) std::atof(v.c_str());
                if (!(sp.temperature > 0.0f) || !std::isfinite(1.0f / sp.temperature)) greedy_req = true;
                else any_key = true;
            } else if (k == "top_p") { sp.top_p = (float) std::atof(v.c_str()); any_key = true; }
            else if (k == "top_k") { sp.top_k = std::atoi(v.c_str()); any_key = true; }
            else if (k == "min_p") { sp.min_p = (float) std::atof(v.c_str()); any_key = true; }
            else if (k == "seed") sp.seed = (uint64_t) std::atoll(v.c_str());
            else if (k == "think_budget") think_budget = std::atoll(v.c_str());
            else if (k == "think_end") think_end = std::atoi(v.c_str());
            else if (k == "spec") {   // the CUDA serve's modes: 1 is off, 2 and up are on
                const int sv = std::atoi(v.c_str());
                spec_req = sv == 1 ? 0 : sv >= 2 ? 1 : spec_req;
            }
            else if (k == "dspark_conf" || k == "spec_min_p") conf_req = (float) std::atof(v.c_str());
            else if (k == "marks") {   // restore points to record during prefill: <pos>,<pos>,...
                for (size_t at = 0; at < v.size();) {
                    const size_t end = v.find(',', at);
                    const std::string one = v.substr(at, end == std::string::npos ? end : end - at);
                    char* p = nullptr;
                    const long long pos = std::strtoll(one.c_str(), &p, 10);
                    if (p != one.c_str() && *p == '\0' && pos > 0) restore.marks.push_back(pos);
                    if (end == std::string::npos) break;
                    at = end + 1;
                }
            }
            else if (k == "shared") {   // the restore point another conversation may resume from
                char* p = nullptr;
                const long long pos = std::strtoll(v.c_str(), &p, 10);
                if (p != v.c_str() && *p == '\0' && pos > 0) restore.shared = pos;
            }
            else if (k == "spec_n_min" || k == "spec_cum") {
                // accepted, as on the GLM engine; the DSpark round's cut is the confidence floor
                // (and the survival product, an engine flag) - these two change nothing here
            }
            else if (k == "penalty_last_n") pen_n = std::atoi(v.c_str());
            else if (k == "penalty_repeat") pen_repeat = std::strtof(v.c_str(), nullptr);
            else if (k == "penalty_freq") pen_freq = std::strtof(v.c_str(), nullptr);
            else if (k == "penalty_present") pen_present = std::strtof(v.c_str(), nullptr);
            else if (k == "logit_bias") {   // <id>:<bias>,... - -100 or below bans the token
                for (size_t at = 0; at < v.size() && bias_err.empty();) {
                    size_t end = v.find(',', at);
                    if (end == std::string::npos) end = v.size();
                    const std::string e1 = v.substr(at, end - at);
                    const size_t colon = e1.find(':');
                    char* p1 = nullptr;
                    char* p2 = nullptr;
                    const long id = colon == std::string::npos ? -1 : std::strtol(e1.c_str(), &p1, 10);
                    const float b = colon == std::string::npos ? 0.0f : std::strtof(e1.c_str() + colon + 1, &p2);
                    if (id < 0 || id >= n_vocab || p1 != e1.c_str() + colon || p2 == nullptr || *p2 != '\0' ||
                        !std::isfinite(b))
                        bias_err = "malformed logit_bias entry '" + e1 + "'";
                    else
                        sp.bias.emplace_back((int32_t) id, b <= -100.0f ? -INFINITY : b);
                    at = end + 1;
                }
            }
        }
        if (any_key) sp.greedy = false;
        if (greedy_req) sp.greedy = true;
        if (!std::isfinite(pen_repeat) || pen_repeat <= 0.0f) pen_repeat = 1.0f;
        if (!std::isfinite(pen_freq)) pen_freq = 0.0f;
        if (!std::isfinite(pen_present)) pen_present = 0.0f;
        if (pen_n < 0) pen_n = INT_MAX;
        if (pen_n > 0 && (pen_repeat != 1.0f || pen_freq != 0.0f || pen_present != 0.0f)) {
            sp.pen_n = pen_n;
            sp.pen_repeat = pen_repeat;
            sp.pen_freq = pen_freq;
            sp.pen_present = pen_present;
        }
        if (!bias_err.empty()) return err(bias_err);
        std::vector<int32_t> prompt;
        bool ok = max_new > 0 && !ids_s.empty();
        for (size_t at = 0; ok && at < ids_s.size();) {
            char* end = nullptr;
            const long long t = std::strtoll(ids_s.c_str() + at, &end, 10);
            if (end == ids_s.c_str() + at) {
                ok = false;
                break;
            }
            if (t < 0 || t >= n_vocab)
                return err("token " + std::to_string(t) + " out of range [0, " + std::to_string(n_vocab) + ")");
            prompt.push_back((int32_t) t);
            at = (size_t) (end - ids_s.c_str());
            if (at < ids_s.size()) {
                if (ids_s[at] != ',') {
                    ok = false;
                    break;
                }
                ++at;
            }
        }
        if (!ok || prompt.empty()) return err("malformed GEN line");
        if (max_new > model.max_context() - (int64_t) prompt.size())
            max_new = model.max_context() - (int64_t) prompt.size();
        if (max_new <= 0)
            return err("the prompt (" + std::to_string(prompt.size()) + " tokens) fills the context (" +
                       std::to_string(model.max_context()) + ")");
        run(prompt, max_new, sp, think_budget, think_end, spec_req, conf_req, restore);
    }

    void run(const std::vector<int32_t>& prompt, int64_t max_new, Sampling sp, int64_t think_budget,
             int32_t think_end, int spec_req, float conf_req, const Restore& restore) {
        const int n_vocab = model.geometry().n_vocab;
        const std::vector<int32_t>& eos = model.geometry().eos;
        std::string e;
        const char* finish = "length";
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = prompt.size();

        // the prefix the snapshot holds, when the new prompt starts with all of it
        size_t reuse = 0;
        const int64_t snap = model.snapshot_position();
        if (snap > 0 && (size_t) snap == cached.size() && (size_t) snap <= n - 1 &&
            std::equal(cached.begin(), cached.end(), prompt.begin()) && model.snapshot_restore())
            reuse = (size_t) snap;
        else {
            // a restore point an earlier request recorded: this prompt resumes from the deepest one
            // it starts with (a matching entry is never deeper than a matching snapshot - the
            // invalidation below saw to that - so the snapshot above keeps its first claim).
            // Restoring is a rewind: the caches are positional and the entry's positions still
            // hold its prefix.  The DSpark draft context needs no handling here - snapshot_restore
            // does not touch it either, and the forwards after a restore re-feed from the depth.
            int best = -1;
            for (size_t i = 0; i < entries.size(); ++i) {
                const PrefixEntry& e = entries[i];
                if (e.toks.size() <= n - 1 && (best < 0 || e.toks.size() > entries[(size_t) best].toks.size()) &&
                    std::equal(e.toks.begin(), e.toks.end(), prompt.begin()))
                    best = (int) i;
            }
            if (best >= 0) {
                model.rewind_to((int64_t) entries[(size_t) best].toks.size());
                reuse = entries[(size_t) best].toks.size();
            } else
                model.reset();
        }
        cached.clear();
        // this request writes every position from reuse on: entries standing deeper are overwritten
        // by its prefill, so they die here unless the prefill records them again; entries at or
        // above the reuse depth keep the state they certify
        for (size_t i = 0; i < entries.size();) {
            if (entries[i].toks.size() > reuse) drop_entry(i);
            else ++i;
        }

        std::vector<float> logits;
        auto pp = [&](size_t read) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("PP %zu %zu %.1f %.1f %zu\n", read, n, ms, ms > 0 ? (read - reuse) * 1000.0 / ms : 0.0, reuse);
            std::fflush(stdout);
        };
        // the restore points to record while prefilling: this request's shared= and marks= inside
        // the span prefill actually writes (a point at or below the reuse depth is already behind
        // us; the snapshot itself stands at the last one, n - 1)
        std::vector<std::pair<int64_t, bool>> points;   // (position, is_shared), ascending, unique
        if (restore.shared > (int64_t) reuse && restore.shared <= (int64_t) n - 1)
            points.emplace_back(restore.shared, true);
        for (const int64_t p : restore.marks)
            if (p > (int64_t) reuse && p <= (int64_t) n - 1) points.emplace_back(p, false);
        std::stable_sort(points.begin(), points.end(),
                         [](const std::pair<int64_t, bool>& a, const std::pair<int64_t, bool>& b) {
                             return a.first < b.first;
                         });
        points.erase(std::unique(points.begin(), points.end(),
                                 [](const std::pair<int64_t, bool>& a, const std::pair<int64_t, bool>& b) {
                                     return a.first == b.first;
                                 }),
                     points.end());
        size_t next_pt = 0;
        for (size_t at = reuse; at + 1 < n;) {   // every prompt token but the last
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            size_t k = std::min((size_t) chunk, n - 1 - at);
            if (next_pt < points.size() && points[next_pt].first > (int64_t) at &&
                points[next_pt].first < (int64_t) (at + k))
                k = (size_t) (points[next_pt].first - (int64_t) at);   // land exactly on the point
            if (!model.forward(std::vector<int32_t>(prompt.begin() + (long) at, prompt.begin() + (long) (at + k)),
                               logits, e)) {
                model.reset();
                clear_entries();
                return err(e);
            }
            pp(at + k);
            at += k;
            if (next_pt < points.size() && points[next_pt].first == (int64_t) at) {
                record_point(prompt, points[next_pt].first, points[next_pt].second);
                ++next_pt;
            }
        }
        int64_t produced = 0;
        auto t1 = std::chrono::steady_clock::now();
        if (std::strcmp(finish, "cancel") != 0) {
            model.snapshot_save();
            cached.assign(prompt.begin(), prompt.end() - 1);
            if (!model.forward({prompt.back()}, logits, e)) {
                model.reset();
                cached.clear();
                clear_entries();
                return err(e);
            }
            pp(n);
            t1 = std::chrono::steady_clock::now();
        } else {
            model.reset();   // a cut prompt leaves no state worth keeping
            clear_entries();
        }

        std::vector<int32_t> pen_hist;
        if (sp.pen_n > 0) pen_hist = prompt;
        bool in_think = think_budget > 0 && think_end >= 0;
        std::vector<int32_t> gen_hist;
        int guard_fired = 0;
        int32_t forced = -1;
        auto looping = [&](int w) -> int {
            const int m = (int) gen_hist.size();
            if (w <= 0 || m < w) return 0;
            for (int per = 1; per <= 32 && per < w / 4; ++per) {
                bool rep = true;
                for (int i = m - w + per; i < m && rep; ++i) rep = gen_hist[(size_t) i] == gen_hist[(size_t) (i - per)];
                if (rep) return per;
            }
            return 0;
        };
        auto note = [&](const char* m) {
            std::fprintf(stderr, "mm3 loop guard: %s\n", m);
            std::printf("NOTE %s\n", m);
        };
        std::vector<int32_t> emitted;   // what this request wrote (a PAUSE keeps the state they lead to)
        auto on_token = [&](int32_t t) -> bool {
            std::printf("T %d\n", t);
            std::fflush(stdout);
            emitted.push_back(t);
            ++produced;
            if (sp.pen_n > 0) pen_hist.push_back(t);
            gen_hist.push_back(t);
            if (in_think && t != think_end) {
                if (const int per = looping(loop_w_think)) {
                    forced = think_end;
                    in_think = false;
                    char m[200];
                    std::snprintf(m, sizeof m,
                                  "thinking was repeating a %d-token pattern (%d tokens in): closed, answering", per,
                                  (int) produced);
                    note(m);
                    gen_hist.clear();
                }
            } else if (!in_think) {
                if (const int per = looping(loop_w_answer)) {
                    char m[240];
                    if (guard_fired == 0 && !eos.empty()) {
                        forced = eos[0];
                        std::snprintf(m, sizeof m,
                                      "the answer was repeating a %d-token pattern (%d tokens in): the turn was ended "
                                      "there",
                                      per, (int) produced);
                    } else {
                        std::snprintf(m, sizeof m,
                                      "the answer kept repeating a %d-token pattern (%d tokens in): ended", per,
                                      (int) produced);
                    }
                    note(m);
                    gen_hist.clear();
                    if (guard_fired++ > 0 || eos.empty()) {
                        finish = "length";
                        return false;
                    }
                }
            }
            if (in_think && (t == think_end || produced >= think_budget || in.wrap.load())) {
                if (t != think_end) forced = think_end;
                in_think = false;
                gen_hist.clear();
            }
            if (produced % 8 == 0) {   // the Monitor's live counters
                const auto& st = model.stats();
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=0.000 disk=%.3f promo=0.000 "
                            "vram_used=%zu vram_slots=0 ram_used=0 ram_slots=0 ram_gb=0.0 vram_gb=%.1f\n",
                            produced / std::max(1e-9, el), 1000.0 * el / (double) produced, 1.0,
                            (double) 0 / std::max<double>(1.0, (double) st.tokens), 0,
                            (st.weight_bytes + st.state_bytes) / 1e9);
            }
            for (const int32_t x : eos)
                if (x == t) {
                    finish = "stop";
                    return false;
                }
            if (in.stop.load()) {
                finish = "cancel";
                return false;
            }
            return produced < max_new;
        };
        long long d_acc = 0, d_off = 0;
        if (std::strcmp(finish, "cancel") != 0) {
            bool bad = false;
            auto pick_one = [&](const float* l) -> int32_t {
                for (size_t v = 0; v < (size_t) n_vocab; ++v)
                    if (!std::isfinite(l[v])) {
                        bad = true;
                        return 0;
                    }
                int32_t t = pick(std::vector<float>(l, l + n_vocab), sp, pen_hist);
                if (forced >= 0) {
                    t = forced;
                    forced = -1;
                }
                sp.counter += 1;
                return t;
            };
            // DSpark: a round drafts after the token, verifies the token and its drafts in one
            // forward and keeps the drafts the target agrees with - every pick is the plain
            // loop's own pick over that row (penalties, the loop guard's forcing and the
            // sampler all run per row, in order), so the tokens are plain decoding's, only
            // more of them per pass over the weights (the CUDA serve's loop, mm3_main.cpp)
            const int B = model.draft_block();
            const bool spec = B > 0 && (spec_req < 0 ? spec_on : spec_req != 0);
            const float floor_c = conf_req >= 0.0f ? conf_req : conf_floor;
            if (!in.stop.load()) {
                int32_t y = pick_one(logits.data());
                if (bad) {
                    model.reset();
                    cached.clear();
                    clear_entries();
                    return err("non-finite logits");
                }
                bool go = on_token(y);   // (on_token checked STOP: the last token written is never fed)
                while (go) {
                    const int64_t pos = model.position();   // y sits at pos, not fed yet
                    if (!spec) {
                        if (!model.forward({y}, logits, e)) {
                            model.reset();
                            cached.clear();
                            clear_entries();
                            return err(e);
                        }
                        y = pick_one(logits.data());
                        if (bad) {
                            model.reset();
                            cached.clear();
                            clear_entries();
                            return err("non-finite logits");
                        }
                        go = on_token(y);
                        continue;
                    }
                    long long room = std::min<long long>(max_new - produced, model.max_context() - pos - 1);
                    if (in_think) room = std::min(room, think_budget - produced);
                    if (in.wrap.load()) room = 0;
                    std::vector<int32_t> drafts;
                    std::vector<float> confs;
                    int k = 0;
                    if (room > 0) {
                        if (!model.draft_propose(y, pos, B, drafts, confs, e)) {
                            model.reset();
                            cached.clear();
                            clear_entries();
                            return err(e);
                        }
                        double surv = 1.0;
                        while (k < (int) drafts.size() && k < room && confs[(size_t) k] >= floor_c) {
                            surv *= confs[(size_t) k];
                            if (draft_cost > 0.0f && surv < draft_cost) break;
                            ++k;
                        }
                    }
                    if (k == 0) {
                        if (!model.forward({y}, logits, e)) {
                            model.reset();
                            cached.clear();
                            clear_entries();
                            return err(e);
                        }
                        y = pick_one(logits.data());
                        if (bad) {
                            model.reset();
                            cached.clear();
                            clear_entries();
                            return err("non-finite logits");
                        }
                        go = on_token(y);
                        continue;
                    }
                    std::vector<int32_t> win(1, y);
                    win.insert(win.end(), drafts.begin(), drafts.begin() + k);
                    std::vector<float> rows;
                    if (!model.forward_rows(win, rows, e)) {
                        model.reset();
                        cached.clear();
                        clear_entries();
                        return err(e);
                    }
                    d_off += k;
                    int kept = 0;
                    int32_t next = -1;
                    bool ended = false;
                    for (int i = 0; i <= k; ++i) {
                        const int32_t cand = pick_one(rows.data() + (size_t) i * n_vocab);
                        if (bad) {
                            model.reset();
                            cached.clear();
                            clear_entries();
                            return err("non-finite logits");
                        }
                        if (i == k || cand != drafts[(size_t) i]) {
                            next = cand;
                            break;
                        }
                        ++d_acc;
                        if (!on_token(drafts[(size_t) i])) {   // written, but never fed (the rewind below)
                            ended = true;
                            kept = i;
                            break;
                        }
                        kept = i + 1;
                    }
                    model.rewind_to(pos + 1 + kept);
                    if (ended) break;
                    y = next;
                    go = on_token(y);
                }
            } else {
                finish = "cancel";
            }
        }
        const auto t2 = std::chrono::steady_clock::now();
        // PAUSE: every token written but the last was fed, so the state is the follow-up prompt's less its last
        // token - exactly what the prefix reuse restores
        if (in.pause.load() && std::strcmp(finish, "cancel") == 0 && !emitted.empty() &&
            model.position() == (int64_t) (n + emitted.size() - 1)) {
            model.snapshot_save();
            cached.assign(prompt.begin(), prompt.end());
            cached.insert(cached.end(), emitted.begin(), emitted.end() - 1);
        }
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", (long long) produced, n,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count(), finish, d_acc, d_off, reuse);
        std::fflush(stdout);
        if (d_off > 0)
            std::fprintf(stderr, "maya-metal-mm3: DSpark kept %lld of %lld drafts\n", d_acc, d_off);
        if (produced > 0) {
            const double s = std::chrono::duration<double>(t2 - t1).count();
            std::fprintf(stderr, "maya-metal-mm3: %lld tokens, %.2f tok/s\n", (long long) produced,
                         produced / std::max(1e-9, s));
        }
    }
};

}  // namespace

int serve(Mm3Metal& model, const std::vector<std::string>& args) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Server s{model, {}, {}};
    for (size_t i = 0; i < args.size(); ++i) {   // the CUDA engine's drafting flags, honoured here
        if (args[i] == "--dspark" && i + 1 < args.size()) s.spec_on = args[++i] != "off";
        else if (args[i] == "--dspark-conf" && i + 1 < args.size())
            s.conf_floor = (float) std::atof(args[++i].c_str());
        else if (args[i] == "--dspark-cost" && i + 1 < args.size())
            s.draft_cost = (float) std::atof(args[++i].c_str());
    }
    s.chunk = model.ubatch();
    if (const char* c = std::getenv("MAYA_METAL_PP_CHUNK")) s.chunk = std::max(1, std::atoi(c));
    const auto& st = model.stats();
    std::printf("INFO engine=%s context=%lld logit_bias=1 kv_ctx=%lld kv_gb=%.3f vram_gb=%.1f vram_slots=0 "
                "dspark=%d\n",
                MAYA_VERSION, (long long) model.max_context(), (long long) model.max_context(),
                st.state_bytes / 1e9, (st.weight_bytes + st.state_bytes) / 1e9,
                model.draft_block() > 0 && s.spec_on ? 1 : 0);
    std::printf("NOTE %s\n", model.describe().c_str());
    std::printf("READY %lld stop pause restore\n", (long long) model.max_context());
    std::fflush(stdout);
    std::thread reader([&] { s.in.run(); });
    reader.detach();   // blocked in getline until stdin closes; the process exits around it
    std::string line;
    auto quit = [] {
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
    };
    while (s.in.next(line)) {
        if (line == "QUIT") quit();
        const std::string head = line.substr(0, line.find(' '));
        if (head == "GEN" || head == "GENI") s.gen(line);
        else if (head == "VLEND" || head == "VRECLAIM" || head == "LOGP")
            s.err("unsupported command '" + head + "' (maya-metal-mm3 serves text only)");
        else
            s.err("unsupported command '" + head + "'");
    }
    quit();
    return 0;
}

}  // namespace maya::metal
