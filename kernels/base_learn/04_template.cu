// 第 4 节：二维线程索引。只填写 kernel 内的 TODO。
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

__global__ void matrix_copy(const float* a, float* b, int rows, int cols) {
    // TODO 1：计算当前线程对应的 row 和 col。
    // TODO 2：检查 row 和 col 均在矩阵范围内。
    // TODO 3：用按行连续存储的索引，将 a 的元素复制到 b。
}

int main() {
    constexpr int rows = 3, cols = 5;
    const size_t bytes = rows * cols * sizeof(float);
    std::vector<float> a(rows * cols), b(rows * cols, -1.0f);
    for (int i = 0; i < rows * cols; ++i) a[i] = static_cast<float>(i);
    float *da = nullptr, *db = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    dim3 block(4, 2);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    matrix_copy<<<grid, block>>>(da, db, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(b.data(), db, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < rows * cols; ++i) {
        if (b[i] != a[i]) {
            std::fprintf(stderr, "元素 %d：得到 %g，期望 %g\n", i, b[i], a[i]);
            correct = false;
            break;
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    if (!correct) return EXIT_FAILURE;
    std::printf("通过：%d × %d 矩阵复制。\n", rows, cols);
    return EXIT_SUCCESS;
}
