// src/hy4-metal/hy4_metal_egcache.hpp - the Engram table's RAM tier: the Strata row-cache
// pattern (include/strata/ngram/ple_reader.hpp, src/kernels/ngram.cpp) ported to this
// engine's mmap'd gather.
//
// WHY HERE AND NOT THE EXPERTS. Every routed expert is copied into a Metal buffer at load
// (load_ctx): on this unified-memory machine they are all resident, and the Monitor's
// "SSD only" reading was the STAT line carrying no tier fields, not the truth. The one
// tenant this engine keeps reading from its file is the Engram n-gram table (63 GB, 660M
// rows of head_dim bytes): engram_gather picks rows by a rolling hash of the last tokens,
// a few dozen random rows per token per Engram layer. Under memory pressure those pages
// fault from the SSD. This cache keeps the hot rows in RAM with real promotion, the way
// the Strata expert tier keeps hot experts: bounded, CLOCK-evicted, counted.
//
// The row index is the key: a row's bytes are a function of its index alone (one table),
// so columns and modules sharing an index share the same bytes - no aliasing hazard.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace hym {

struct EgStats {
    uint64_t rows = 0;     // rows asked for
    uint64_t hits = 0;     // rows served from the RAM tier
    uint64_t reads = 0;    // misses: rows read from the mapping (page cache or SSD)
    uint64_t disk = 0;     // misses whose page was NOT resident: a true SSD fault
    uint64_t promo = 0;    // rows copied into the RAM tier
    double ms = 0.0;       // gather wall time, ms
};

// Is the page holding p resident in RAM? mincore answers it; an error claims nothing.
inline bool eg_page_resident(const void* p) {
    const uintptr_t pg = (uintptr_t) ::sysconf(_SC_PAGESIZE);
    const uintptr_t a = (uintptr_t) p / pg * pg;
    char vec = 0;
    if (::mincore((void*) a, (size_t) pg, &vec) != 0) return true;
    return (vec & 1) != 0;   // MINCORE_INCORE
}

class EgCache {
public:
    EgCache() = default;
    EgCache(size_t row_bytes, double gb) : row_(row_bytes) {
        if (row_bytes == 0 || gb <= 0) return;
        stride_ = 16 + row_bytes;
        const uint64_t want = (uint64_t) (gb * 1073741824.0 / (double) stride_);
        uint64_t cap = 1;
        while (cap * 2 <= want && cap < (1ull << 32)) cap *= 2;
        if (cap < 1024) return;   // too small to bother
        cap_ = cap;
        mask_ = cap - 1;
        slots_.assign((size_t) cap_ * stride_, 0);
        for (uint64_t i = 0; i < cap_; ++i) key(i) = kEmpty;
        // best effort: pinned rows survive the pressure that evicts the file's pages
        (void) ::mlock(slots_.data(), slots_.size());
    }
    bool on() const { return cap_ > 0; }
    uint64_t capacity() const { return cap_; }
    uint64_t size() const { return size_; }
    size_t row_bytes() const { return row_; }

    // One row: a hit copies from the tier; a miss reads the mapping (the fault), says
    // whether the page was resident, and promotes the row into the tier.
    void fetch(uint64_t row, const uint8_t* table_row, uint8_t* out, EgStats& st) {
        st.rows++;
        if (cap_) {
            uint64_t i = hash(row) & mask_;
            for (uint64_t n = 0; n < cap_; ++n) {
                if (key(i) == row) {
                    ref(i) = 1;
                    std::memcpy(out, bytes(i), row_);
                    st.hits++;
                    return;
                }
                if (key(i) == kEmpty) break;
                i = (i + 1) & mask_;
            }
        }
        st.reads++;
        if (!eg_page_resident(table_row)) st.disk++;
        std::memcpy(out, table_row, row_);
        if (cap_) insert(row, out, st);
    }

private:
    static constexpr uint64_t kEmpty = UINT64_MAX;
    size_t row_ = 0, stride_ = 0;
    uint64_t cap_ = 0, mask_ = 0, size_ = 0, hand_ = 0;
    std::vector<uint8_t> slots_;

    uint64_t& key(uint64_t i) { return *reinterpret_cast<uint64_t*>(slots_.data() + (size_t) i * stride_); }
    uint8_t& ref(uint64_t i) { return slots_[(size_t) i * stride_ + 8]; }
    uint8_t* bytes(uint64_t i) { return slots_.data() + (size_t) i * stride_ + 16; }

    static uint64_t hash(uint64_t x) {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27; x *= 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    void insert(uint64_t row, const uint8_t* src, EgStats& st) {
        uint64_t i;
        if (size_ < cap_) {
            i = hash(row) & mask_;
            while (key(i) != kEmpty) i = (i + 1) & mask_;
            size_++;
        } else {
            // CLOCK: the hand clears reference bits until it finds an unreferenced slot
            for (;;) {
                i = hand_;
                hand_ = (hand_ + 1) & mask_;
                if (ref(i) == 0) break;
                ref(i) = 0;
            }
        }
        key(i) = row;
        ref(i) = 1;
        std::memcpy(bytes(i), src, row_);
        st.promo++;
    }
};

}  // namespace hym
