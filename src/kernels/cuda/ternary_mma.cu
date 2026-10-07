// Ternary weights x int8 activations for batches of tokens, on the INT8 tensor cores through the wmma API.
// No dp4a anywhere (the CMP 170HX needs it off). NOT BUILT OR RUN YET: written without a GPU or nvcc; it needs a
// parity test against tools/ternary_lut.py's integer path (and the bench in ternary_bench.cu) on a real card.
//
// Weights use the same packed layout as ternary_lut.cu: five trits per byte, group-major, packed[group * rows + row].
// Each block unpacks a K-chunk of the weights into int8 in shared memory and multiplies it with the activations'
// chunk on 16x16x16 int8 tiles, accumulating int32. Output: y[token * rows + row], int32 (scales applied by the
// caller). Activations: xq[token * kpad + k], int8, kpad = groups * 5, zero-padded.
//
// wmma wants every tile pointer 256-bit aligned, so the shared tiles are stored slab-major: a slab is 16
// consecutive k values of 16 rows (or tokens), 256 bytes, ldm = 16.
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace strata_ternary {

using namespace nvcuda;

constexpr int kTrits = 5;
constexpr int kGroupsPerChunk = 16;                  // 16 groups = 80 weights = 5 slabs of 16
constexpr int kChunkK = kGroupsPerChunk * kTrits;    // 80
constexpr int kSlabs = kChunkK / 16;                 // 5
constexpr int kWarps = 4;                            // each warp owns 16 weight rows
constexpr int kZeroByte = 121;                       // five zero weights

__global__ void mma_i8(const uint8_t* __restrict__ packed, int groups, int rows, const int8_t* __restrict__ xq,
                       int tokens, int32_t* __restrict__ y) {
    __shared__ __align__(32) int8_t sA[kSlabs * 256];
    __shared__ __align__(32) int8_t sB[kWarps][kSlabs * 256];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row0 = (blockIdx.x * kWarps + warp) * 16;
    const int tok0 = blockIdx.y * 16;
    const int kpad = groups * kTrits;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc;
    wmma::fill_fragment(acc, 0);

    for (int g0 = 0; g0 < groups; g0 += kGroupsPerChunk) {
        // activations: 16 tokens x 80 k, slab-major [slab][token][16]
        for (int i = threadIdx.x; i < 16 * kChunkK; i += blockDim.x) {
            const int m = i / kChunkK, k = i % kChunkK;
            const int gk = g0 * kTrits + k;
            int8_t v = 0;
            if (tok0 + m < tokens && gk < kpad) v = xq[(size_t)(tok0 + m) * kpad + gk];
            sA[(k / 16) * 256 + m * 16 + (k % 16)] = v;
        }
        // weights: this warp's 16 rows x 16 groups, unpacked to int8, slab-major [slab][row][16] (col_major B)
        for (int i = lane; i < 16 * kGroupsPerChunk; i += 32) {
            const int n = i / kGroupsPerChunk, g = i % kGroupsPerChunk;
            int v = kZeroByte;
            if (g0 + g < groups && row0 + n < rows) v = packed[(size_t)(g0 + g) * rows + row0 + n];
#pragma unroll
            for (int t = 0; t < kTrits; t++) {
                const int k = g * kTrits + t;
                sB[warp][(k / 16) * 256 + n * 16 + (k % 16)] = (int8_t)(v % 3 - 1);
                v /= 3;
            }
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < kSlabs; s++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
            wmma::load_matrix_sync(a, sA + s * 256, 16);
            wmma::load_matrix_sync(b, sB[warp] + s * 256, 16);
            wmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    // acc[m][n] -> y[(tok0 + m) * rows + row0 + n]; stage through shared memory to guard the edges
    __shared__ __align__(32) int32_t sC[kWarps][16 * 16];
    wmma::store_matrix_sync(sC[warp], acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        const int m = i / 16, n = i % 16;
        if (tok0 + m < tokens && row0 + n < rows) y[(size_t)(tok0 + m) * rows + row0 + n] = sC[warp][i];
    }
}

cudaError_t mma_i8_launch(const uint8_t* packed, int groups, int rows, const int8_t* xq, int tokens, int32_t* y,
                          cudaStream_t stream) {
    dim3 grid((rows + 16 * kWarps - 1) / (16 * kWarps), (tokens + 15) / 16);
    mma_i8<<<grid, kWarps * 32, 0, stream>>>(packed, groups, rows, xq, tokens, y);
    return cudaGetLastError();
}

}  // namespace strata_ternary
