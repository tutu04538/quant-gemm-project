// 第 11 节：使用 CUDA event 计时。
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "01.cu"

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    constexpr int n = 4096, threads = 256, warmup = 10, repeats = 100;
    constexpr int blocks = (n + threads - 1) / threads;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes)); CUDA_CHECK(cudaMalloc(&db, bytes)); CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (c[i] != 3.0f) { std::fprintf(stderr, "正确性验证失败：i=%d\n", i); return EXIT_FAILURE; }
    }
    for (int i = 0; i < warmup; ++i) vector_add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    float ms = -1.0f;

    // TODO 1：记录 start。
    for (int i = 0; i < repeats; ++i) vector_add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    // TODO 2：记录 stop。
    // TODO 3：等待 stop 完成。
    // TODO 4：计算 start 到 stop 的毫秒数，写到 ms。

    if (ms < 0.0f) { std::fprintf(stderr, "请完成 event 计时 TODO。\n"); return EXIT_FAILURE; }
    std::printf("%d 次总计 %.6f ms，平均 %.6f ms/次\n", repeats, ms, ms / repeats);
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    return EXIT_SUCCESS;
}
