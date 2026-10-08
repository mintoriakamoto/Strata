// Ternary (-1, 0, +1) matrix-vector product by lookup table. NOT BUILT OR RUN ON A GPU YET: written without a GPU or
// nvcc (only syntax-checked against stub CUDA headers). The layout and arithmetic follow tools/ternary_lut.py, whose
// tests pin them down; this kernel still needs a parity run against that reference on a real card (ternary_bench.cu).
//
// Weights: five ternary values per byte (base 3, digit = weight + 1, so byte 121 = five zeros; bytes 243..255 are
// unused), stored group-major: packed[group * rows + row]. rows must be a multiple of 16 and the base pointer 16-byte
// aligned: each thread reads 16 consecutive rows of one group as ONE uint4 load (the Intel port's notes in
// docs/INTEL.md measured aligned 16-byte loads 2.3-4.7x faster than narrow ones on a bandwidth-bound kernel).
// Activations: int8, zero-padded to groups * 5 values. For each group of five, a 256-entry int16 table of the
// product with every possible byte sits in shared memory; a row's dot product is one lookup and one add per byte.
// |table entry| <= 5 * 127, so int16 cannot overflow; accumulation is int32.
//
// Parallelism: a 640-row expert matrix has no parallelism of its own, so the grid is (row tiles, K splits, experts).
// Each K split adds its int32 partial sum into y with atomicAdd. Integer addition is associative, so the result is
// bit-identical for every split count and block order. y must be zeroed first (gemv_i8_launch does) and the scales
// are applied afterwards (scale_i32).
//
// Prior art, so the design is not mistaken for new: five trits per byte with a 256-entry table of dot products is
// QTEA's CUDA GEMV (arXiv 2609.00224, measured on an H200, which also builds only half the table by sign symmetry
// and keeps activations in fp16); the table-lookup idea is T-MAC's (arXiv 2407.00088); BITCOS (arXiv 2609.16338)
// uses a 256-entry shared-memory table on Intel Xe2. T-SAR (arXiv 2511.13676) reports table accesses as over 75% of
// memory requests on CPUs, the same risk as shared-memory bank conflicts here.
#include <cuda_runtime.h>
#include <stdint.h>

namespace strata_ternary {

constexpr int kGroup = 5;
constexpr int kTable = 256;
constexpr int kUsed = 243;
constexpr int kChunk = 32;          // groups per shared-memory tile: 32 * 256 * 2 B = 16 KB
constexpr int kThreads = 128;
constexpr int kRowsPerThread = 16;  // one uint4 of packed bytes
constexpr int kRowsPerBlock = kThreads * kRowsPerThread;

// packed + z * packed_stride, xq + z * x_stride, y + z * y_stride is expert z's matrix, input and output (a stride of
// 0 shares one input between experts, as the gate/up projections of one token do).
__global__ void gemv_i8(const uint8_t* __restrict__ packed, size_t packed_stride, int groups, int rows,
                        const int8_t* __restrict__ xq, size_t x_stride, int32_t* __restrict__ y, size_t y_stride,
                        int groups_per_split) {
    __shared__ int16_t tab[kChunk][kTable];
    const int z = blockIdx.z;
    const uint8_t* p = packed + (size_t)z * packed_stride;
    const int8_t* x = xq + (size_t)z * x_stride;
    int32_t* out = y + (size_t)z * y_stride;

    const int g_begin = blockIdx.y * groups_per_split;
    const int g_end = min(groups, g_begin + groups_per_split);
    const int row_base = blockIdx.x * kRowsPerBlock + threadIdx.x * kRowsPerThread;
    const bool active = row_base < rows;     // rows % 16 == 0, so the whole uint4 is in range

    int acc[kRowsPerThread];
#pragma unroll
    for (int j = 0; j < kRowsPerThread; j++) acc[j] = 0;

    for (int g0 = g_begin; g0 < g_end; g0 += kChunk) {
        const int n = min(kChunk, g_end - g0);
        for (int idx = threadIdx.x; idx < n * kTable; idx += kThreads) {
            const int gl = idx / kTable;
            const int b = idx % kTable;
            const int8_t* xg = x + (size_t)(g0 + gl) * kGroup;
            int s = 0;
            if (b < kUsed) {
                int v = b;
#pragma unroll
                for (int i = 0; i < kGroup; i++) {
                    s += (v % 3 - 1) * (int)xg[i];
                    v /= 3;
                }
            }
            tab[gl][b] = (int16_t)s;
        }
        __syncthreads();
        if (active) {
            for (int gl = 0; gl < n; gl++) {
                const uint4 w = *reinterpret_cast<const uint4*>(p + (size_t)(g0 + gl) * rows + row_base);
                const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int j = 0; j < kRowsPerThread; j++)
                    acc[j] += tab[gl][(words[j >> 2] >> ((j & 3) * 8)) & 0xFF];
            }
        }
        __syncthreads();
    }
    if (active) {
#pragma unroll
        for (int j = 0; j < kRowsPerThread; j++) atomicAdd(out + row_base + j, acc[j]);
    }
}

// y[e * rows + r] (float) = acc[e * rows + r] * x_scale * row_scale[r]; row_scale is shared by experts when
// scale_stride is 0.
__global__ void scale_i32(const int32_t* __restrict__ acc, const float* __restrict__ row_scale, size_t scale_stride,
                          float x_scale, float* __restrict__ y, int rows, int experts) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * experts) return;
    const size_t e = i / rows, r = i % rows;
    y[i] = (float)acc[i] * x_scale * row_scale[e * scale_stride + r];
}

// Zeroes y (experts * rows int32) and runs the kernel with k_splits slices of the K dimension. Returns the CUDA
// error of the launch. packed rows % 16 == 0 and 16-byte alignment are the caller's contract.
cudaError_t gemv_i8_launch(const uint8_t* packed, size_t packed_stride, int experts, int groups, int rows,
                           const int8_t* xq, size_t x_stride, int32_t* y, size_t y_stride, int k_splits,
                           cudaStream_t stream) {
    if (rows % kRowsPerThread != 0 || (reinterpret_cast<uintptr_t>(packed) & 15) || (packed_stride & 15))
        return cudaErrorInvalidValue;
    if (k_splits < 1) k_splits = 1;
    cudaError_t e = cudaMemsetAsync(y, 0, sizeof(int32_t) * y_stride * experts, stream);
    if (e != cudaSuccess) return e;
    const int per_split = (groups + k_splits - 1) / k_splits;
    dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock, (groups + per_split - 1) / per_split, experts);
    gemv_i8<<<grid, kThreads, 0, stream>>>(packed, packed_stride, groups, rows, xq, x_stride, y, y_stride, per_split);
    return cudaGetLastError();
}

}  // namespace strata_ternary
