// sampler.metal - the sampler's greedy kernel: one threadgroup of 1024 per
// token row over the vocabulary, argmax of the biased, penalized logits with
// ties resolved to the LOWEST index (the CUDA sampler_greedy_kernel's exact
// rule, including its all-(-inf) -> 0 answer). Penalty counts are computed by
// scanning the history tail per candidate - the CUDA kernel's bitmap is a
// speed variant whose counts are identical to the scan's.
//
// The sampled (temperature) chain is NOT here yet: its top_p / draw stage is
// double-precision in the CUDA TU and Apple GPUs expose no FP64 in Metal, so
// a float transcription would silently change draws. The host entries for
// the sampled paths stop loudly until that port is validated against
// sampler_parity (see stubs/sampler.cpp).

struct GreedyArgs {
    int n_vocab;
    int history_len;
    int plen;  // min(penalty_last_n, history_len), the tail window actually counted
    int has_history;
    int has_bias;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
};

kernel void sampler_greedy_k(device const float* logits [[buffer(0)]], device const int* history [[buffer(1)]],
                             device const float* bias [[buffer(2)]], device int* out [[buffer(3)]],
                             constant GreedyArgs& a [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup int si[32];
    device const float* l = logits + tg * a.n_vocab;
    device const int* hrow = history + tg * a.history_len + (a.history_len - a.plen);
    float bv = as_type<float>(0xff800000u);  // -inf
    int best = a.n_vocab;
    for (int v = int(tid); v < a.n_vocab; v += 1024) {
        float s = a.has_bias ? l[v] + bias[v] : l[v];
        if (a.has_history && a.plen > 0) {
            int cnt = 0;
            for (int i = 0; i < a.plen; ++i)
                if (hrow[i] == v) ++cnt;
            if (cnt > 0) {
                if (s <= 0.0f) s *= a.penalty_repeat;
                else s /= a.penalty_repeat;
                s -= float(cnt) * a.penalty_freq + a.penalty_present;
            }
        }
        if (s > bv) {
            bv = s;
            best = v;
        }
    }
    // warp reduction, then warp 0 over the 32 warp winners (the CUDA shape)
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        int oi = simd_shuffle_down(best, off);
        if (ov > bv || (ov == bv && oi < best)) {
            bv = ov;
            best = oi;
        }
    }
    if (lane == 0) {
        sv[sg] = bv;
        si[sg] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float wv = (lane < 32) ? sv[lane] : as_type<float>(0xff800000u);
        int wi = (lane < 32) ? si[lane] : a.n_vocab;
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(wv, off);
            int oi = simd_shuffle_down(wi, off);
            if (ov > wv || (ov == wv && oi < wi)) {
                wv = ov;
                wi = oi;
            }
        }
        if (lane == 0) out[tg] = (wi < a.n_vocab) ? wi : 0;
    }
}
