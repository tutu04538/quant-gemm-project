// 第 5 节：填写矩阵转置的线性索引。
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

__global__ void transpose(const float* a, float* b, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;

    // TODO：读取 A[row, col]，写入 B[col, row]。
    // A 的形状是 rows × cols；B 的形状是 cols × rows。
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
    transpose<<<grid, block>>>(da, db, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(b.data(), db, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int output_row = 0; output_row < cols; ++output_row) {
        for (int output_col = 0; output_col < rows; ++output_col) {
            float expected = a[output_col * cols + output_row];
            float actual = b[output_row * rows + output_col];
            if (actual != expected) {
                std::fprintf(stderr, "B[%d,%d]：得到 %g，期望 %g\n", output_row, output_col, actual, expected);
                correct = false;
            }
        }
    }
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    if (!correct) return EXIT_FAILURE;
    std::printf("通过：%d × %d 矩阵转置。\n", rows, cols);
    return EXIT_SUCCESS;
}
