#pragma once
// Included only by STRATA_ENABLE_METAL builds. Host translation units do not use __half (the
// kernel headers carry their own bit-level types in strata/kernels/f16_bits.hpp and
// bf16_bits.hpp); this header exists so an include of <cuda_fp16.h> resolves.
