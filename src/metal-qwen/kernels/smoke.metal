// src/metal-qwen/kernels/smoke.metal - the dispatch smoke kernel: out[i] = a[i] + b[i].
// It exists so metal_dispatch_smoke can prove the whole dispatch path (embedded MSL,
// pipeline cache, buffer resolution through the allocation map, by-value args) on a
// known answer before any real kernel family lands on it.

kernel void smoke_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                      device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < n) out[id] = a[id] + b[id];
}

// The batching regression kernel (BATCH-NOTES.md): each launch adds exactly 1,
// so N launches with no sync in between must leave exactly N - a result no
// idempotent kernel could fake, and the launch count is the test's variable.
kernel void smoke_inc(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < n) out[id] += 1.0f;
}
