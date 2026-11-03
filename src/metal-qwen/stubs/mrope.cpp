// src/metal-qwen/stubs/mrope.cpp - Metal implementation for
// include/strata/kernels/mrope.hpp (workstream qsa).
//
// The M-RoPE table is pure host state: the CUDA implementation keeps one
// pointer per device; Metal has exactly one device, so one atomic pointer.
// Kernels receive the pointer as an argument exactly as on CUDA - the table's
// contents may change between requests; the pointer is set before capture.

#include "strata/kernels/mrope.hpp"

#include <atomic>

namespace strata::kernels {
namespace {
std::atomic<const int32_t*> g_mrope_tab{nullptr};
}  // namespace

void mrope_table_set(const int32_t* device_table) {
    g_mrope_tab.store(device_table, std::memory_order_relaxed);
}
const int32_t* mrope_table() { return g_mrope_tab.load(std::memory_order_relaxed); }

}  // namespace strata::kernels
