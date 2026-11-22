// 第 7 节：通过 shared memory 交换相邻线程的数据。
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

__global__ void shared_neighbor(const float* a, float* b, int n) {
    __shared__ float tile[32];
    const int t = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + t;
    // TODO 1：tile[t] 写入有效输入 a[i]，无效时写 0.0f。

    __syncthreads();  // 所有线程都必须到达这里。

    // TODO 2：只让 i < n 的线程将下一个线程的 tile 值写到 b[i]。
    // 下一个线程编号为 (t + 1) % 32。
}

int main() {
    constexpr int n = 37, threads = 32;
    const int blocks = (n + threads - 1) / threads;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n, -1.0f);
    for (int i = 0; i < n; ++i) a[i] = static_cast<float>(i);
    float *da = nullptr, *db = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    shared_neighbor<<<blocks, threads>>>(da, db, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(b.data(), db, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        const int neighbor = (i / threads) * threads + (i % threads + 1) % threads;
        const float expected = neighbor < n ? a[neighbor] : 0.0f;
        if (b[i] != expected) {
            std::fprintf(stderr, "B[%d]：得到 %g，期望 %g\n", i, b[i], expected);
            correct = false;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    if (!correct) return EXIT_FAILURE;
    std::printf("通过：%d 个 shared memory 邻居值。\n", n);
    return EXIT_SUCCESS;
}
