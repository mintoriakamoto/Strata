// Ternary (-1, 0, +1) matrix-vector product by lookup table. NOT BUILT OR RUN YET: written without a GPU or nvcc.
// The layout and arithmetic follow tools/ternary_lut.py, whose tests pin them down; this kernel still needs a
// parity test against that reference on a real card before anything uses it.
//
// Weights: five ternary values per byte (base 3, digit = weight + 1, so byte 121 = five zeros; bytes 243..255
// are unused), stored group-major: packed[group * rows + row], so threads on consecutive rows read consecutively.
// Activations: int8, zero-padded to groups * 5 values. For each group of five, a 256-entry int16 table of the
// product with every possible byte sits in shared memory; a row's dot product is one lookup and one add per byte.
// |table entry| <= 5 * 127, so int16 cannot overflow; accumulation is int32.
#include <cuda_runtime.h>
#include <stdint.h>

namespace strata_ternary {

constexpr int kGroup = 5;
constexpr int kTable = 256;
constexpr int kUsed = 243;
constexpr int kChunk = 32;          // groups per shared-memory tile: 32 * 256 * 2 B = 16 KB
constexpr int kThreads = 256;
constexpr int kRowsPerThread = 4;   // each block rebuilds the tables, so more rows per block amortize that

__global__ void gemv_i8(const uint8_t* __restrict__ packed, int groups, int rows,
                        const int8_t* __restrict__ xq, const float* __restrict__ row_scale, float x_scale,
                        float* __restrict__ y) {
    __shared__ int16_t tab[kChunk][kTable];
    const int row0 = blockIdx.x * kThreads * kRowsPerThread;
    int acc[kRowsPerThread] = {0, 0, 0, 0};

    for (int g0 = 0; g0 < groups; g0 += kChunk) {
        const int n = min(kChunk, groups - g0);
        for (int idx = threadIdx.x; idx < n * kTable; idx += kThreads) {
            const int gl = idx / kTable;
            const int b = idx % kTable;
            const int8_t* xg = xq + (size_t)(g0 + gl) * kGroup;
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
#pragma unroll
        for (int r = 0; r < kRowsPerThread; r++) {
            const int row = row0 + r * kThreads + threadIdx.x;
            if (row < rows) {
                const uint8_t* p = packed + (size_t)g0 * rows + row;
                for (int gl = 0; gl < n; gl++) acc[r] += tab[gl][p[(size_t)gl * rows]];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kRowsPerThread; r++) {
        const int row = row0 + r * kThreads + threadIdx.x;
        if (row < rows) y[row] = (float)acc[r] * x_scale * row_scale[row];
    }
}

// packed: device, groups * rows bytes (group-major). xq: device, groups * 5 int8, zero-padded.
cudaError_t gemv_i8_launch(const uint8_t* packed, int groups, int rows, const int8_t* xq, const float* row_scale,
                           float x_scale, float* y, cudaStream_t stream) {
    const int per_block = kThreads * kRowsPerThread;
    gemv_i8<<<(rows + per_block - 1) / per_block, kThreads, 0, stream>>>(packed, groups, rows, xq, row_scale,
                                                                        x_scale, y);
    return cudaGetLastError();
}

}  // namespace strata_ternary
