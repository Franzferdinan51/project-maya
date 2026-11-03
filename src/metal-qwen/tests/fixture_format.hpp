#pragma once
// Shared definition of the .mqfx fixture format (see FIXTURE-FORMAT.md in this
// directory for the prose specification). The generator (gen_fixtures.cpp, host
// only) writes fixtures through this header and the runner
// (metal_kernel_parity.cpp, Mac) reads them through it, so the two sides can
// never drift apart on layout.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace mqfx {

inline constexpr uint32_t kMagic = 0x5846514Du;  // "MQFX" little-endian
inline constexpr uint16_t kVersion = 1;

enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,   // raw IEEE half bits, stored as bytes
    BF16 = 2,  // raw bfloat16 bits, stored as bytes
    I32 = 3,
    I64 = 4,
    U8 = 5,  // raw bytes: quantized blocks, canonical codes, any bit pattern
    I8 = 6,
    U16 = 7,  // raw 16-bit patterns that are neither f16 nor bf16 semantics
};

enum class TolKind : uint8_t {
    Bitwise = 0,  // every output element bit-identical (integers and exact float paths)
    MaxAbs = 1,   // |got - want| <= tol_abs for every element
    MaxRel = 2,   // |got - want| <= tol_abs + tol_rel * |want| for every element
};

inline size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32: case DType::I32: return 4;
        case DType::F16: case DType::BF16: case DType::U16: return 2;
        case DType::I64: return 8;
        case DType::U8: case DType::I8: return 1;
    }
    return 1;
}

struct Tensor {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;   // row-major; empty shape = scalar
    std::vector<uint8_t> bytes;   // exactly numel(shape) * dtype_size(dtype), or
                                  // for U8 the raw byte count in shape's product

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
    const float* f32() const { return reinterpret_cast<const float*>(bytes.data()); }
    const int32_t* i32() const { return reinterpret_cast<const int32_t*>(bytes.data()); }
};

struct Fixture {
    std::string family;     // kernel family, e.g. "dequant", "gemv", "router"
    std::string entry;      // exact host entry point name, e.g. "dequant_f32"
    std::string case_name;  // e.g. "q2_0_real_dims"
    uint64_t seed = 0;      // the deterministic seed every input was drawn from
    TolKind tol_kind = TolKind::Bitwise;
    double tol_abs = 0.0;
    double tol_rel = 0.0;
    std::vector<std::pair<std::string, int64_t>> params_i;  // dims, ggml types, eps*1e6, flags
    std::vector<std::pair<std::string, double>> params_f;   // scalar floats (eps, scales)
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;  // expected outputs

    const Tensor* find(const std::vector<Tensor>& v, const char* name) const {
        for (const auto& t : v)
            if (t.name == name) return &t;
        return nullptr;
    }
    const Tensor* in(const char* name) const { return find(inputs, name); }
    const Tensor* out(const char* name) const { return find(outputs, name); }
    int64_t param_i(const char* name, int64_t dflt = 0) const {
        for (const auto& p : params_i)
            if (p.first == name) return p.second;
        return dflt;
    }
    double param_f(const char* name, double dflt = 0.0) const {
        for (const auto& p : params_f)
            if (p.first == name) return p.second;
        return dflt;
    }
};

namespace detail {
inline void w_u8(FILE* f, uint8_t v) { std::fwrite(&v, 1, 1, f); }
inline void w_u16(FILE* f, uint16_t v) { std::fwrite(&v, 2, 1, f); }
inline void w_u32(FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
inline void w_u64(FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
inline void w_i64(FILE* f, int64_t v) { std::fwrite(&v, 8, 1, f); }
inline void w_f64(FILE* f, double v) { std::fwrite(&v, 8, 1, f); }
inline void w_str(FILE* f, const std::string& s) {
    w_u32(f, (uint32_t)s.size());
    if (!s.empty()) std::fwrite(s.data(), 1, s.size(), f);
}
inline void w_tensor(FILE* f, const Tensor& t) {
    w_str(f, t.name);
    w_u8(f, (uint8_t)t.dtype);
    w_u8(f, (uint8_t)t.shape.size());
    for (int64_t d : t.shape) w_i64(f, d);
    w_u64(f, (uint64_t)t.bytes.size());
    if (!t.bytes.empty()) std::fwrite(t.bytes.data(), 1, t.bytes.size(), f);
}

struct Reader {
    std::vector<uint8_t> buf;
    size_t pos = 0;
    bool ok = true;
    template <typename T> T get() {
        if (pos + sizeof(T) > buf.size()) { ok = false; return T{}; }
        T v;
        std::memcpy(&v, buf.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    std::string str() {
        uint32_t n = get<uint32_t>();
        if (pos + n > buf.size()) { ok = false; return {}; }
        std::string s((const char*)buf.data() + pos, n);
        pos += n;
        return s;
    }
    Tensor tensor() {
        Tensor t;
        t.name = str();
        t.dtype = (DType)get<uint8_t>();
        uint8_t rank = get<uint8_t>();
        for (int i = 0; i < rank; ++i) t.shape.push_back(get<int64_t>());
        uint64_t n = get<uint64_t>();
        if (pos + n > buf.size()) { ok = false; return t; }
        t.bytes.assign(buf.begin() + (long)pos, buf.begin() + (long)(pos + n));
        pos += (size_t)n;
        return t;
    }
};
}  // namespace detail

inline bool write_fixture(const std::string& path, const Fixture& fx) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    using namespace detail;
    w_u32(f, kMagic);
    w_u16(f, kVersion);
    w_u16(f, 0);  // reserved
    w_str(f, fx.family);
    w_str(f, fx.entry);
    w_str(f, fx.case_name);
    w_u64(f, fx.seed);
    w_u8(f, (uint8_t)fx.tol_kind);
    w_f64(f, fx.tol_abs);
    w_f64(f, fx.tol_rel);
    w_u32(f, (uint32_t)fx.params_i.size());
    for (const auto& p : fx.params_i) { w_str(f, p.first); w_i64(f, p.second); }
    w_u32(f, (uint32_t)fx.params_f.size());
    for (const auto& p : fx.params_f) { w_str(f, p.first); w_f64(f, p.second); }
    w_u32(f, (uint32_t)fx.inputs.size());
    for (const auto& t : fx.inputs) w_tensor(f, t);
    w_u32(f, (uint32_t)fx.outputs.size());
    for (const auto& t : fx.outputs) w_tensor(f, t);
    const bool good = std::ferror(f) == 0;
    std::fclose(f);
    return good;
}

inline bool read_fixture(const std::string& path, Fixture& fx, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    detail::Reader r;
    uint8_t chunk[1 << 16];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) r.buf.insert(r.buf.end(), chunk, chunk + n);
    std::fclose(f);
    if (r.get<uint32_t>() != kMagic) { err = path + ": bad magic"; return false; }
    if (r.get<uint16_t>() != kVersion) { err = path + ": unsupported version"; return false; }
    r.get<uint16_t>();  // reserved
    fx.family = r.str();
    fx.entry = r.str();
    fx.case_name = r.str();
    fx.seed = r.get<uint64_t>();
    fx.tol_kind = (TolKind)r.get<uint8_t>();
    fx.tol_abs = r.get<double>();
    fx.tol_rel = r.get<double>();
    const uint32_t npi = r.get<uint32_t>();
    for (uint32_t i = 0; i < npi; ++i) fx.params_i.emplace_back(r.str(), r.get<int64_t>());
    const uint32_t npf = r.get<uint32_t>();
    for (uint32_t i = 0; i < npf; ++i) fx.params_f.emplace_back(r.str(), r.get<double>());
    const uint32_t nin = r.get<uint32_t>();
    for (uint32_t i = 0; i < nin; ++i) fx.inputs.push_back(r.tensor());
    const uint32_t nout = r.get<uint32_t>();
    for (uint32_t i = 0; i < nout; ++i) fx.outputs.push_back(r.tensor());
    if (!r.ok) { err = path + ": truncated"; return false; }
    return true;
}

}  // namespace mqfx
