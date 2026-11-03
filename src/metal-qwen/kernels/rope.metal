// rope.metal - NEOX partial RoPE kernels (see common.metalh for the prelude).
// Semantics: src/kernels/cuda/rope.cu and src/kernels/cuda/native_rope.cu.
// NEOX pairing (i, i + n_rot/2); only the first n_rot of head_dim rotate.

// One thread per row, as in the CUDA kernel: a row's tail copy and rotation
// then have a single writer and no ordering question.
kernel void strata_rope_neox_apply(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                                   device const float* cos_tab [[buffer(2)]],
                                   device const float* sin_tab [[buffer(3)]],
                                   device const int* pos [[buffer(4)]],
                                   device const int* mtab [[buffer(5)]],
                                   constant int& has_mtab [[buffer(6)]], constant long& rows [[buffer(7)]],
                                   constant int& head_dim [[buffer(8)]], constant int& n_rot [[buffer(9)]],
                                   uint tid [[thread_position_in_grid]]) {
    if ((long)tid >= rows) return;
    const int nhalf = n_rot / 2;
    device const float* xr = x + (long)tid * head_dim;
    device float* orow = out + (long)tid * head_dim;
    for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];  // the partial rotation's untouched tail
    const int p = pos[tid];
    for (int i = 0; i < nhalf; ++i) {
        const int rp = strata_mrope_pos(mtab, has_mtab, p, i);
        const size_t toff = (size_t)rp * (size_t)nhalf;
        const float c = cos_tab[toff + i], s = sin_tab[toff + i];
        const float a = xr[i], b = xr[nhalf + i];
        orow[i] = a * c - b * s;
        orow[nhalf + i] = a * s + b * c;
    }
}

// One thread per (row, pair of the row's width/2 pairs), as in the CUDA
// kernel: pairs past n_rot/2 copy their two elements through (when not in
// place); rotated pairs use the NEOX half-split within the first n_rot.
// theta = position * theta_scale^pair, evaluated in float as on CUDA
// (the host computes theta_scale with the same float pow).
kernel void strata_native_rope_apply(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                                     device const int* positions [[buffer(2)]],
                                     device const int* mtab [[buffer(3)]],
                                     constant int& has_mtab [[buffer(4)]], constant int& rows [[buffer(5)]],
                                     constant int& width [[buffer(6)]], constant int& n_rot [[buffer(7)]],
                                     constant float& theta_scale [[buffer(8)]],
                                     uint2 tp [[thread_position_in_grid]]) {
    const int pair = (int)tp.x;
    const int row = (int)tp.y;
    if (row >= rows || pair >= width / 2) return;
    const size_t start = (size_t)row * (size_t)width;
    if (pair >= n_rot / 2) {
        if (x != out) {
            out[start + 2 * pair] = x[start + 2 * pair];
            out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
        }
        return;
    }
    const int rp = strata_mrope_pos(mtab, has_mtab, positions[row], pair);
    const float theta = (float)rp * pow(theta_scale, (float)pair);
    const float c = cos(theta), s = sin(theta);
    const float a = x[start + pair], b = x[start + pair + n_rot / 2];
    out[start + pair] = a * c - b * s;
    out[start + pair + n_rot / 2] = a * s + b * c;
}
