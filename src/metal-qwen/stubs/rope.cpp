// src/metal-qwen/stubs/rope.cpp - Metal implementation for
// include/strata/kernels/rope.hpp (workstream qsa).
//
// build_rope_table is the CUDA host function ported verbatim (float64
// throughout, in the reference's order). rope_neox_apply launches
// strata_rope_neox_apply from src/metal-qwen/kernels/rope.metal.

#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"

#include "../runtime/metal_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply: n_rot %d must be even and <= head_dim %d\n", n_rot, head_dim);
        std::exit(1);
    }
    const int32_t* tab = mrope_table();
    const int has_mtab = tab != nullptr ? 1 : 0;
    // A null table is passed as a placeholder buffer (never dereferenced:
    // has_mtab == 0 makes the kernel use the position itself).
    const void* tab_arg = has_mtab ? (const void*) tab : (const void*) pos;
    const int64_t rows64 = rows;
    const ::strata::metal::KernelArg args[] = {::strata::metal::buf(x),     ::strata::metal::buf(out),   ::strata::metal::buf(cos_tab),
                                     ::strata::metal::buf(sin_tab), ::strata::metal::buf(pos), ::strata::metal::buf(tab_arg),
                                     ::strata::metal::val(has_mtab), ::strata::metal::val(rows64), ::strata::metal::val(head_dim),
                                     ::strata::metal::val(n_rot)};
    ::strata::metal::launch((cudaStream_t) stream, "strata_rope_neox_apply", (uint32_t) rows, 1, 1, 256, 1, 1, args,
                  sizeof(args) / sizeof(args[0]));
}

}  // namespace strata::kernels
