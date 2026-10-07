// Parity and timing for the ternary kernels: the LUT matvec (ternary_lut.cu) and the INT8 tensor-core matmul
// (ternary_mma.cu). NOT BUILT OR RUN YET (written without a GPU). Build, from this directory:
//   nvcc -O3 -arch=sm_80 ternary_bench.cu -o ternary_bench && ./ternary_bench [rows] [cols] [tokens]
// Prints, for each kernel, whether its output matches a CPU integer reference exactly, and the time per call.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include "ternary_lut.cu"
#include "ternary_mma.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

int main(int argc, char** argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 2048;
    const int cols = argc > 2 ? atoi(argv[2]) : 2048;
    const int tokens = argc > 3 ? atoi(argv[3]) : 128;
    const int groups = (cols + 4) / 5, kpad = groups * 5;
    srand(1);

    std::vector<int8_t> w((size_t)rows * kpad, 0);                       // zero beyond cols
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) w[(size_t)r * kpad + c] = rand() % 3 - 1;
    std::vector<uint8_t> packed((size_t)groups * rows);
    for (int g = 0; g < groups; g++) for (int r = 0; r < rows; r++) {
        int b = 0, m = 1;
        for (int i = 0; i < 5; i++) { b += (w[(size_t)r * kpad + g * 5 + i] + 1) * m; m *= 3; }
        packed[(size_t)g * rows + r] = (uint8_t)b;
    }
    std::vector<int8_t> x((size_t)tokens * kpad, 0);
    for (int t = 0; t < tokens; t++) for (int c = 0; c < cols; c++) x[(size_t)t * kpad + c] = (int8_t)(rand() % 255 - 127);

    std::vector<int32_t> ref((size_t)tokens * rows);
    for (int t = 0; t < tokens; t++) for (int r = 0; r < rows; r++) {
        int32_t s = 0;
        for (int c = 0; c < kpad; c++) s += (int)w[(size_t)r * kpad + c] * (int)x[(size_t)t * kpad + c];
        ref[(size_t)t * rows + r] = s;
    }

    uint8_t* dp; int8_t* dx; int32_t* dy; float* dyf; float* dscale;
    CK(cudaMalloc(&dp, packed.size())); CK(cudaMalloc(&dx, x.size()));
    CK(cudaMalloc(&dy, ref.size() * 4)); CK(cudaMalloc(&dyf, rows * 4)); CK(cudaMalloc(&dscale, rows * 4));
    std::vector<float> ones(rows, 1.0f);
    CK(cudaMemcpy(dp, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dscale, ones.data(), rows * 4, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int iters = 50;
    float ms;

    // LUT matvec, token 0
    CK(strata_ternary::gemv_i8_launch(dp, groups, rows, dx, dscale, 1.0f, dyf, 0)); CK(cudaDeviceSynchronize());
    std::vector<float> yf(rows); CK(cudaMemcpy(yf.data(), dyf, rows * 4, cudaMemcpyDeviceToHost));
    int bad = 0; for (int r = 0; r < rows; r++) bad += (int32_t)yf[r] != ref[r];
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) strata_ternary::gemv_i8_launch(dp, groups, rows, dx, dscale, 1.0f, dyf, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
    printf("lut   1 token : %s (%d/%d mismatches)  %.1f us/call  %.1f GB/s of weights\n", bad ? "FAIL" : "ok", bad, rows,
           ms * 1000 / iters, (double)packed.size() * iters / (ms * 1e-3) / 1e9);

    // tensor-core matmul, all tokens
    CK(strata_ternary::mma_i8_launch(dp, groups, rows, dx, tokens, dy, 0)); CK(cudaDeviceSynchronize());
    std::vector<int32_t> y(ref.size()); CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    bad = 0; for (size_t i = 0; i < y.size(); i++) bad += y[i] != ref[i];
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) strata_ternary::mma_i8_launch(dp, groups, rows, dx, tokens, dy, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
    printf("mma %4d tokens: %s (%d/%zu mismatches)  %.1f us/call  %.2f TOPS\n", tokens, bad ? "FAIL" : "ok", bad, y.size(),
           ms * 1000 / iters, 2.0 * rows * kpad * tokens * iters / (ms * 1e-3) / 1e12);
    return bad != 0;
}
