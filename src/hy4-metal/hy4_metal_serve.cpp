// src/hy4-metal/hy4_metal_serve.cpp - maya-metal-hy4 --serve: the serve protocol of
// serve/server.py's StrataEngine (the same line protocol src/metal/serve.cpp and
// mm3_serve.cpp speak), so the server drives the Hy4 Metal engine exactly as it drives
// the others.
//
//   in:  GEN <max_new> [key=value ...] <id,id,...>   GENI ...   STOP   PAUSE   WRAP   QUIT
//   out: INFO ... / READY <ctx> stop, then per request PP ..., T <id> ..., STAT ..., DONE ...
//        (or ERR ... followed by a DONE, as maya-ds4 / maya-mm3 do)
//
// Sampling is the shared chain (src/metal/sampler.cpp): same keys, same defaults, same
// Philox draws, so a seeded request repeats.  One conversation (READY names no multi=):
// the caches are positional, so the conversation cache is the fed-id list - a GEN whose
// ids extend it feeds only the new tail (DONE's reused field says how much), and a GEN
// that diverges rewinds the state to the common prefix (Forward::rewind_to).
//
// NOT ported yet, and loud about it: images (GENI is refused - the vision tower is
// stage 5) and MTP drafting (no draft exists on this engine; a config's --mtp is noted
// by the main, and generation is plain - the same tokens, without the draft's speed).

#include "hy4_metal_model.hpp"

#include "../metal/sampler.hpp"

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
#include <vector>

#ifndef MAYA_VERSION
#define MAYA_VERSION "dev"
#endif

namespace hym {

namespace {

using maya::metal::PickScratch;
using maya::metal::Sampling;

// stdin on its own thread: STOP / PAUSE / WRAP act at once (a running request sees them),
// the rest queue up - the mm3 serve loop's shape.
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false}, wrap{false};
    bool quit = false, eof = false;

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;
            if (line == "STOP" || line.rfind("STOP ", 0) == 0 || line == "PAUSE" ||
                line.rfind("PAUSE ", 0) == 0) {
                stop.store(true);   // (READY claims stop only; a PAUSE still ends the request)
                continue;
            }
            if (line == "WRAP") {
                wrap.store(true);
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                if (line == "QUIT") quit = true;
                else q.push_back(line);
            }
            cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
        }
        cv.notify_one();
    }

    // the next queued line, or "" when the input is over (QUIT / EOF)
    std::string next() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || quit || eof; });
        if (q.empty()) return "";
        std::string s = q.front();
        q.pop_front();
        return s;
    }
};

struct Serve {
    Forward& fwd;
    const hy4::Geo& g;
    int64_t max_ctx;
    int chunk;
    Input in;
    std::vector<int32_t> cached;   // the ids the forward has been fed, in order (pos() == its size)
    PickScratch scratch;
    EgStats eg_prev{};   // the Engram tier's counters at the last STAT: the rates are per request

    void out(const std::string& s) {
        std::fputs(s.c_str(), stdout);
        std::fflush(stdout);
    }
    void err(const std::string& m, size_t prompt_n = 0) {
        out("ERR " + m + "\n");
        out("DONE 0 " + std::to_string(prompt_n) + " 0 0 error 0 0 0 0 0\n");
    }

    void gen(const std::string& line) {
        std::istringstream ss(line);
        std::string cmd, tok;
        long long max_new = 0;
        ss >> cmd >> max_new;
        if (cmd == "GENI")
            return err("maya-metal-hy4 serves text only for now (the vision tower is stage 5 of the port)");
        Sampling sp;
        bool any_key = false, greedy_req = false;
        int64_t think_budget = 0;
        int32_t think_end = -1;
        int pen_n = 0;
        float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
        std::string ids_s, bias_err;
        const int n_vocab = g.n_vocab;
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
            // every other key (spec=, dspark_conf=, cvec=, ablate=, the tune keys, ...) is
            // another engine's: accepted and ignored, as the protocol expects
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
        if (max_new > max_ctx - (int64_t) prompt.size()) max_new = max_ctx - (int64_t) prompt.size();
        if (max_new <= 0)
            return err("the prompt (" + std::to_string(prompt.size()) + " tokens) fills the context (" +
                       std::to_string(max_ctx) + ")");
        run(prompt, max_new, sp, think_budget, think_end);
    }

    void run(const std::vector<int32_t>& prompt, int64_t max_new, Sampling sp, int64_t think_budget,
             int32_t think_end) {
        const size_t n = prompt.size();
        const auto t0 = std::chrono::steady_clock::now();
        // the conversation cache: feed only what the state does not already hold
        size_t reuse = 0;
        while (reuse < n && reuse < cached.size() && cached[reuse] == prompt[reuse]) ++reuse;
        if ((int64_t) reuse < fwd.pos()) fwd.rewind_to((int64_t) reuse);
        cached.resize(reuse);
        std::string e;
        std::vector<float> logits((size_t) g.n_vocab);
        size_t fed = reuse;
        if (fed == n) {
            // the whole prompt is already the state: the logits on hand are for a later
            // position than the prompt's end, so rewind one token and re-feed it - its
            // cache rows rewrite identically and the last-row logits are the prompt's own
            fwd.rewind_to((int64_t) n - 1);
            cached.resize(n - 1);
            fed = n - 1;
        }
        const size_t feed_from = fed;
        while (fed < n) {
            const int T = (int) std::min<size_t>((size_t) chunk, n - fed);
            if (!fwd.run((const int*) prompt.data() + fed, T, logits.data(), nullptr, e)) {
                fwd.reset();
                cached.clear();
                return err(e, n);
            }
            cached.insert(cached.end(), prompt.begin() + (std::ptrdiff_t) fed,
                          prompt.begin() + (std::ptrdiff_t) (fed + (size_t) T));
            fed += (size_t) T;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            char buf[160];
            std::snprintf(buf, sizeof buf, "PP %zu %zu %.1f %.1f %zu\n", fed, n, ms,
                          ms > 0 ? (double) (fed - feed_from) * 1000.0 / ms : 0.0, reuse);
            out(buf);
            if (in.stop.load()) break;   // a STOP during the prompt read ends the request below
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (in.stop.load()) {   // stopped while the prompt was still being read
            char buf[256];
            std::snprintf(buf, sizeof buf, "DONE 0 %zu %.1f 0 cancel 0 0 %zu 0 0\n", n, prompt_ms, reuse);
            out(buf);
            return;
        }

        int64_t produced = 0;
        const char* finish = "length";
        bool force_think_end = false;
        std::vector<int32_t> hist = cached;   // the sampler's penalty history: everything fed
        while (produced < max_new) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (in.wrap.exchange(false) && think_end >= 0) force_think_end = true;
            int32_t y;
            if (force_think_end) {
                y = think_end;
                force_think_end = false;
            } else {
                sp.counter = (uint64_t) produced;
                y = maya::metal::pick(logits.data(), (size_t) g.n_vocab, sp, hist, scratch);
                if (y < 0) {
                    fwd.reset();
                    cached.clear();
                    return err("non-finite logits", n);
                }
            }
            ++produced;
            out("T " + std::to_string(y) + "\n");
            if (y == g.eos) {
                finish = "stop";
                break;
            }
            if (think_budget > 0 && think_end >= 0 && produced >= think_budget) force_think_end = true;
            if (produced >= max_new) break;
            // feed the token just written; the last one of the request stays unfed, which is
            // exactly the state the next GEN's prefix reuse expects (as on the mm3 engine)
            if (!fwd.run((const int*) &y, 1, logits.data(), nullptr, e)) {
                fwd.reset();
                cached.clear();
                return err(e, n);
            }
            cached.push_back(y);
            hist.push_back(y);
        }
        const auto t2 = std::chrono::steady_clock::now();
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (produced > 0 && decode_ms > 0) {
            // the truth this engine owes the Monitor: every expert resident in unified memory
            // (vram_gb, vram_hit=1, no slot fields - the unified card), and the Engram tier's
            // real activity: ram_fetch/disk split by page residency, promo the rows promoted
            const TierInfo ti = fwd.tier_info();
            const EgStats d{ti.eg.rows - eg_prev.rows, ti.eg.hits - eg_prev.hits,
                            ti.eg.reads - eg_prev.reads, ti.eg.disk - eg_prev.disk,
                            ti.eg.promo - eg_prev.promo, ti.eg.ms - eg_prev.ms};
            eg_prev = ti.eg;
            const double np = (double) produced;
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "STAT tok_s=%.2f ms_tok=%.2f vram_gb=%.1f vram_hit=1 ram_fetch=%.2f disk=%.2f promo=%.2f "
                          "eg_lookup_rows=%llu eg_lookup_hits=%llu eg_lookup_reads=%llu eg_cache_rows=%lld "
                          "eg_cache_cap=%lld eg_cache_row_bytes=%zu engram_ms=%.2f\n",
                          (double) produced * 1000.0 / decode_ms, decode_ms / np, ti.resident_gb,
                          (double) (d.reads - d.disk) / np, (double) d.disk / np, (double) d.promo / np,
                          (unsigned long long) ti.eg.rows, (unsigned long long) ti.eg.hits,
                          (unsigned long long) ti.eg.reads, (long long) ti.eg_cache_rows,
                          (long long) ti.eg_cache_cap, ti.eg_row_bytes, d.ms / np);
            out(buf);
        }
        char buf[256];
        std::snprintf(buf, sizeof buf, "DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0\n", (long long) produced, n,
                      prompt_ms, decode_ms, finish, reuse);
        out(buf);
        if (produced > 0)
            std::fprintf(stderr, "maya-metal-hy4: %lld tokens, %.2f tok/s\n", (long long) produced,
                         decode_ms > 0 ? (double) produced * 1000.0 / decode_ms : 0.0);
    }
};

}  // namespace

int serve_hy4(Forward& fwd, const hy4::Geo& g, int64_t max_ctx, int chunk) {
    Serve s{fwd, g, max_ctx, chunk > 0 ? chunk : 512, {}, {}, {}};
    std::thread th([&] { s.in.run(); });
    th.detach();
    char buf[320];
    const TierInfo ti0 = fwd.tier_info();
    std::snprintf(buf, sizeof buf,
                  "INFO engine=%s arch=hy_v4_flash engine_kind=hy4-metal context=%lld kv=f16 experts=%d vision=0 "
                  "spec=none engram_lookups=%d engram_cache_gb=%.2f\n",
                  MAYA_VERSION, (long long) max_ctx, g.n_expert, ti0.engram_lookups,
                  (double) ti0.eg_cache_cap * (double) ti0.eg_row_bytes / 1073741824.0);
    s.out(buf);
    std::snprintf(buf, sizeof buf, "READY %lld stop\n", (long long) max_ctx);
    s.out(buf);
    for (;;) {
        const std::string line = s.in.next();
        if (line.empty()) break;   // QUIT or EOF
        s.in.stop.store(false);
        s.in.wrap.store(false);
        if (line.rfind("GEN", 0) == 0) s.gen(line);
        // anything else (a stray command this engine does not know) is ignored, never fatal
    }
    return 0;
}

}  // namespace hym
