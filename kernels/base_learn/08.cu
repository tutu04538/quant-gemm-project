// 第 8 节：在正确的位置增加 block 内同步。
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void exchange_warps(const float* a, float* b) {
    __shared__ float tile[64];
    const int t = threadIdx.x;
    tile[t] = a[t];
    // TODO：等待 block 中的线程完成写入，再让线程读取 shared memory。
    __syncthreads();

    b[t] = tile[(t + 32) % 64];
}

int main() {
    constexpr int n = 64;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n, -1.0f);
    for (int i = 0; i < n; ++i) a[i] = static_cast<float>(i);
    float *da = nullptr, *db = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    exchange_warps<<<1, n>>>(da, db);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(b.data(), db, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        const float expected = a[(i + 32) % n];
        if (b[i] != expected) {
            std::fprintf(stderr, "B[%d]：得到 %g，期望 %g\n", i, b[i], expected);
            correct = false;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    if (!correct) return EXIT_FAILURE;
    std::printf("通过：%d 个跨 warp 交换值。仍需检查同步位置。\n", n);
    return EXIT_SUCCESS;
}
