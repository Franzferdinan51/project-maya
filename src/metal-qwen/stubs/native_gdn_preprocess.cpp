// native_gdn_preprocess.cpp - Metal launchers, pinned native GDN preprocessing.
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "../runtime/metal_kernel.hpp"
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
namespace strata::kernels {
namespace { constexpr int S = 128;
using strata::metal::buf; using strata::metal::val; using strata::metal::KernelArg;
struct Span { const void* p; size_t b; };
void valid(Span s) { auto a = reinterpret_cast<uintptr_t>(s.p);
    if (!s.p || a % 4 || s.b > UINTPTR_MAX - a) throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans"); }
void disjoint(Span a, Span b) { auto ap = reinterpret_cast<uintptr_t>(a.p), bp = reinterpret_cast<uintptr_t>(b.p);
    if (ap < bp + b.b && bp < ap + a.b) throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans"); }
void count_stream(int64_t n, void* st) { if (!st || n <= 0 || n > 65535) throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,65535]"); }
void norm_geom(int64_t rows, int64_t cols, float eps, void* st) { count_stream(rows, st);
    if (cols != S || !std::isfinite(eps) || eps < 0) throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon"); }
void launch(void* s, const char* fn, uint32_t g, uint32_t b, const KernelArg* a, size_t n) {
    strata::metal::launch((cudaStream_t)s, fn, g, 1, 1, b, 1, 1, a, n); } }
void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels, int64_t d_conv, void* stream) {
    count_stream(channels, stream);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * 4;
    const Span w[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span in[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto s : w) valid(s); for (auto s : in) valid(s);
    for (int i = 0; i < 3; ++i) { for (int j = 0; j < i; ++j) disjoint(w[i], w[j]); for (auto s : in) disjoint(w[i], s); }
    int ch = (int)channels;
    KernelArg a[] = {buf(history), buf(input), buf(weights), buf(raw_output), buf(silu_output), val(ch)};
    launch(stream, "native_gdn_conv_silu_kernel", (uint32_t)channels, 256, a, 6);
}
void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geom(rows, cols, epsilon, stream); valid({input, size_t(rows) * S * 4});
    float ed = epsilon / S, sa = 1.0f / sqrtf(float(S));
    KernelArg a[] = {buf(input), val(ed), val(sa)};
    launch(stream, "native_gdn_l2_kernel", (uint32_t)(rows * 256), 256, a, 3);
}
void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_stream(heads, stream); valid({beta, size_t(heads) * 4});
    int n = (int)heads;
    KernelArg a[] = {buf(beta), val(n)};
    launch(stream, "gdn_beta_kernel", (uint32_t)heads, 256, a, 2);
}
void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t heads, void* stream) {
    count_stream(heads, stream);
    const size_t bytes = size_t(heads) * 4; const Span out{gate, bytes}; valid(out);
    for (auto s : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) { valid(s); disjoint(out, s); }
    int n = (int)heads;
    KernelArg a[] = {buf(alpha), buf(dt), buf(ssm_a), buf(gate), val(n)};
    launch(stream, "native_gdn_gate_kernel", (uint32_t)heads, 256, a, 5);
}
void native_gdn_out_norm(const float* output, const float* z, const float* gamma, float* destination,
                         int64_t heads, int64_t cols, float epsilon, void* stream) {
    norm_geom(heads, cols, epsilon, stream);
    const size_t bytes = size_t(heads) * S * 4; const Span w{destination, bytes}; valid(w);
    for (auto s : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * 4}}) { valid(s); disjoint(w, s); }
    KernelArg a[] = {buf(output), buf(z), buf(gamma), buf(destination), val(epsilon)};
    launch(stream, "native_gdn_out_norm_kernel", (uint32_t)(heads * 256), 256, a, 5);
}
}  // namespace strata::kernels
