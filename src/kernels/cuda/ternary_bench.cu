// Parity and timing for the ternary kernels: the LUT matvec (ternary_lut.cu) and the INT8 tensor-core matmul
// (ternary_mma.cu). NOT RUN ON A GPU YET (written without one). Build, from this directory:
//   nvcc -O3 -arch=sm_80 ternary_bench.cu -o ternary_bench && ./ternary_bench [rows] [cols] [tokens]
// rows must be a multiple of 16. For each kernel it prints whether the output matches a CPU integer reference
// exactly, the time per call, and the bandwidth or TOPS reached. The LUT kernel is swept over K-split counts (all must
// match bit for bit: integer sums do not depend on the split) and run once as 10 grouped experts, the shape of one
// layer's routed experts in the engine.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include "ternary_lut.cu"
#include "ternary_mma.cu"

#define CK(x) do { cudaError_t ck_err_ = (x); if (ck_err_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(ck_err_)); return 1; } } while (0)

int main(int argc, char** argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 2048;
    const int cols = argc > 2 ? atoi(argv[2]) : 2048;
    const int tokens = argc > 3 ? atoi(argv[3]) : 128;
    if (rows % 16) { printf("rows must be a multiple of 16\n"); return 2; }
    const int groups = (cols + 4) / 5, kpad = groups * 5;
    const int n_exp = 10, iters = 50;
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

    uint8_t *dp, *dpe; int8_t* dx; int32_t *dy, *dye;
    CK(cudaMalloc(&dp, packed.size())); CK(cudaMalloc(&dpe, packed.size() * n_exp));
    CK(cudaMalloc(&dx, x.size())); CK(cudaMalloc(&dy, ref.size() * 4)); CK(cudaMalloc(&dye, (size_t)rows * n_exp * 4));
    CK(cudaMemcpy(dp, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    for (int e = 0; e < n_exp; e++) CK(cudaMemcpy(dpe + e * packed.size(), packed.data(), packed.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;
    int failed = 0;

    // LUT matvec, token 0, one matrix, K-split sweep
    for (int ks : {1, 2, 4, 8, 16, 32}) {
        CK(strata_ternary::gemv_i8_launch(dp, packed.size(), 1, groups, rows, dx, 0, dy, rows, ks, 0));
        CK(cudaDeviceSynchronize());
        std::vector<int32_t> y(rows); CK(cudaMemcpy(y.data(), dy, rows * 4, cudaMemcpyDeviceToHost));
        int bad = 0; for (int r = 0; r < rows; r++) bad += y[r] != ref[r];
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) strata_ternary::gemv_i8_launch(dp, packed.size(), 1, groups, rows, dx, 0, dy, rows, ks, 0);
        cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
        printf("lut  1 matrix, k_splits %2d: %s (%d/%d bad)  %.1f us/call  %.1f GB/s\n", ks, bad ? "FAIL" : "ok", bad, rows,
               ms * 1000 / iters, (double)packed.size() * iters / (ms * 1e-3) / 1e9);
        failed += bad != 0;
    }

    // LUT matvec, 10 experts in one launch, one shared input (x_stride 0)
    for (int ks : {1, 4, 16}) {
        CK(strata_ternary::gemv_i8_launch(dpe, packed.size(), n_exp, groups, rows, dx, 0, dye, rows, ks, 0));
        CK(cudaDeviceSynchronize());
        std::vector<int32_t> y((size_t)rows * n_exp); CK(cudaMemcpy(y.data(), dye, y.size() * 4, cudaMemcpyDeviceToHost));
        int bad = 0; for (int e = 0; e < n_exp; e++) for (int r = 0; r < rows; r++) bad += y[(size_t)e * rows + r] != ref[r];
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) strata_ternary::gemv_i8_launch(dpe, packed.size(), n_exp, groups, rows, dx, 0, dye, rows, ks, 0);
        cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
        printf("lut %2d experts, k_splits %2d: %s (%d bad)  %.1f us/call  %.1f GB/s\n", n_exp, ks, bad ? "FAIL" : "ok", bad,
               ms * 1000 / iters, (double)packed.size() * n_exp * iters / (ms * 1e-3) / 1e9);
        failed += bad != 0;
    }

    // tensor-core matmul, all tokens
    CK(strata_ternary::mma_i8_launch(dp, groups, rows, dx, tokens, dy, 0)); CK(cudaDeviceSynchronize());
    std::vector<int32_t> y(ref.size()); CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0; for (size_t i = 0; i < y.size(); i++) bad += y[i] != ref[i];
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) strata_ternary::mma_i8_launch(dp, groups, rows, dx, tokens, dy, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
    printf("mma %4d tokens: %s (%d/%zu bad)  %.1f us/call  %.2f TOPS\n", tokens, bad ? "FAIL" : "ok", bad, y.size(),
           ms * 1000 / iters, 2.0 * rows * kpad * tokens * iters / (ms * 1e-3) / 1e12);
    failed += bad != 0;
    return failed != 0;
}
