// 第 9 节：填充、同步、跨 warp 读取、复用前同步。
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

// 本练习约定只启动 1 个 block，blockDim.x == 64，且 n == 96。
__global__ void exchange_tiles(const float* a, float* b, int n) {
    __shared__ float tile[64];
    const int t = threadIdx.x;
    for (int base = 0; base < n; base += 64) {
        const int i = base + t;
        // TODO 1：加载 a[i] 到 tile[t]；输入越界时填 0.0f。
        // TODO 2：等待整个 block 写完 tile。
        // TODO 3：仅当 i 有效时，读取另一个 warp 对应的 tile 并写 b[i]。
        // TODO 4：等待整个 block 读完 tile，再进入下一轮复用。
    }
}

int main() {
    constexpr int n = 96, threads = 64;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n, -1.0f);
    for (int i = 0; i < n; ++i) a[i] = static_cast<float>(i + 1);
    float *da = nullptr, *db = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    exchange_tiles<<<1, threads>>>(da, db, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(b.data(), db, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        const int base = (i / threads) * threads;
        const int peer = base + ((i % threads + 32) % threads);
        const float expected = peer < n ? a[peer] : 0.0f;
        if (b[i] != expected) {
            std::fprintf(stderr, "B[%d]：得到 %g，期望 %g\n", i, b[i], expected);
            correct = false;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    if (!correct) return EXIT_FAILURE;
    std::printf("通过：%d 个跨 warp 交换值。\n", n);
    return EXIT_SUCCESS;
}
