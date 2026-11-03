// src/metal-qwen/stubs/shared_expert.cpp - Metal implementation for
// include/strata/kernels/shared_expert.hpp. Replaces the stage-1 loud stub.
//
// The composition mirrors src/kernels/cuda/shared_expert.cu exactly: gate/up
// projections through each form's own GEMV entry (the dense workstream's
// entries - s2_gemv_q8 for 2-bit codes, s_gemv_q8k_split for Q8_K activations,
// s_gemv_q8_0_split otherwise), SwiGLU, the down projection over the
// intermediate quantized to the down weight's own contract, then the per-token
// scalar gate and the output scale. Optional native projections go through
// native_quantize_q8_1 / native_mmvq, and the native BF16 scalar gate through
// bf16_gemv_fp32_mmvf, as in the CUDA TU. The small kernels are in
// kernels/shared_expert.metal. No allocation anywhere: the scratch layout is
// shared_expert_scratch_bytes' (a pure host formula, ported below).

#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../runtime/metal_kernel.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"

namespace strata::kernels {
namespace {

bool& native_bf16_flag() {
    static bool f = false;
    return f;
}

void launch_swiglu(const float* gate, const float* up, float* out, int n, void* stream) {
    const metal::KernelArg args[] = {metal::buf(gate), metal::buf(up), metal::buf(out), metal::val(n)};
    metal::launch((cudaStream_t) stream, "shexp_swiglu_k", (uint32_t) n, 1, 1, 256, 1, 1, args, 4);
}
void launch_scalar_gate(const uint16_t* x_bf16, const uint16_t* w_bf16, float* out, int n_embd, void* stream) {
    const metal::KernelArg args[] = {metal::buf(x_bf16), metal::buf(w_bf16), metal::buf(out), metal::val(n_embd)};
    metal::launch((cudaStream_t) stream, "shexp_scalar_gate_k", 256, 1, 1, 256, 1, 1, args, 4);
}
void launch_sigmoid(float* gate, int n, void* stream) {
    const metal::KernelArg args[] = {metal::buf(gate), metal::val(n)};
    metal::launch((cudaStream_t) stream, "shexp_sigmoid_k", (uint32_t) n, 1, 1, 32, 1, 1, args, 2);
}
void launch_scale(float* out, const float* g, int n, void* stream) {
    const metal::KernelArg args[] = {metal::buf(out), metal::buf(g), metal::val(n)};
    metal::launch((cudaStream_t) stream, "shexp_scale_k", (uint32_t) n, 1, 1, 256, 1, 1, args, 3);
}
void launch_scale_rows(float* out, const float* g, int n, int n_tok, void* stream) {
    const metal::KernelArg args[] = {metal::buf(out), metal::buf(g), metal::val(n)};
    metal::launch((cudaStream_t) stream, "shexp_scale_rows_k", (uint32_t) n, (uint32_t) n_tok, 1, 256, 1, 1, args,
                  3);
}

}  // namespace

// Host-state flag (RISKS.md S4): a plain bool in shared_expert.cu, read by the
// shared_expert entries in this TU.
void shared_expert_set_native_bf16(bool enabled) { native_bf16_flag() = enabled; }

// Pure host arithmetic (a *_bytes probe in the RISKS.md S4 sense): the exact
// formula from shared_expert.cu - gate | up (n_ff f32 each) | q8_0 (n_ff/32 x
// 34) | q8k (n_ff/256 x 292) | g (1 f32, in the trailing 32 B), 16-aligned.
uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out,
                         int64_t n_embd, int64_t n_ff, void* stream) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    launch_swiglu(gate, up, gate, (int) (n_ff * n_tok), stream);
    native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    static const bool batch = [] {
        const char* v = std::getenv("STRATA_DEC_BATCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (native_bf16_flag() && batch && n_tok > 1) {
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
        launch_sigmoid(g, n_tok, stream);
    } else {
        for (int t = 0; t < n_tok; ++t) {
            if (native_bf16_flag()) {
                bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
                launch_sigmoid(g + t, 1, stream);
            } else {
                launch_scalar_gate(x_bf16 + (size_t) t * n_embd, gate_inp_bf16, g + t, (int) n_embd, stream);
            }
        }
    }
    launch_scale_rows(out, g, (int) n_embd, n_tok, stream);
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16_flag();
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument(
                "shared_expert native projections require scratch, stream and int32 dimensions");
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        std::fprintf(stderr,
                     "shared_expert: scratch is null; the caller owns it "
                     "(see shared_expert_scratch_bytes)\n");
        std::exit(1);
    }
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    if (native_gate || native_up) native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    launch_swiglu(gate, up, gate, (int) n_ff, stream);

    if (native_down) {
        native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else if (down_form.act_kind == 1) {
        if (n_ff % 256 != 0) {
            std::fprintf(stderr,
                         "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                         "of 256; Q8_K is structurally impossible here\n",
                         (long long) n_ff);
            std::exit(1);
        }
        quantize_q8_K(gate, h_q8k, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else {
        quantize_q8_0(gate, h_q8_0, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    }

    if (use_native) {
        bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
        launch_sigmoid(g, 1, stream);
    } else {
        launch_scalar_gate(x_bf16, gate_inp_bf16, g, (int) n_embd, stream);
    }
    launch_scale(out, g, (int) n_embd, stream);

    if (stream == nullptr) cudaDeviceSynchronize();
}

// The MoE block's final combination (see the header for the two readings it
// pins): ascending sum of weights[e] * parts[e], the shared output added
// PLAIN. The CUDA kernel accumulates in double; the MSL kernel uses
// compensated float in the same order (kernels/moe.metal, moe_combine_k).
void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "moe_combine: k = %lld exceeds 64\n", (long long) k);
        std::exit(1);
    }
    struct Args {
        int32_t n_embd, k, has_shared;
    };
    const Args a{(int32_t) n_embd, (int32_t) k, shared != nullptr ? 1 : 0};
    const metal::KernelArg args[] = {metal::buf(parts), metal::buf(weights),
                                     metal::buf(shared ? shared : parts), metal::buf(y), metal::val(a)};
    metal::launch((cudaStream_t) stream, "moe_combine_k", (uint32_t) n_embd, 1, 1, 256, 1, 1, args, 5);
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
