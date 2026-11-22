// 第 3 节：只填写 kernel 发射后的两个检查点。
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
    constexpr int n = 37;
    constexpr int threads = 32;
    const int blocks = (n + threads - 1) / threads;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 0.5f;
    }
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    vector_add<<<blocks, threads>>>(da, db, dc, n);
    // TODO 1：检查发射错误，用 CUDA_CHECK 包住相应的 CUDA 调用。
    // TODO 2：等待设备工作完成，并检查执行阶段的错误。

    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i] + b[i]) {
            std::fprintf(stderr, "i=%d: 得到 %g，期望 %g\n", i, c[i], a[i] + b[i]);
            correct = false;
            break;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    if (!correct) return EXIT_FAILURE;
    std::printf("数值验证通过：%d 个元素。错误检查需确认 TODO 已填写。\n", n);
    return EXIT_SUCCESS;
}
