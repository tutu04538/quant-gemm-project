// 第 6 节：记录转置时一个 warp 的输入、输出元素索引。
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

__global__ void record_indices(int* input_indices, int* output_indices) {
    const int row = 0;                    // 模拟处理 A 的第 0 行
    const int col = threadIdx.x;          // 本例 lane = threadIdx.x，范围 0～31
    const int rows = 4, cols = 32;
    // TODO 1：将 A[row,col] 的线性元素索引写入 input_indices[col]。
    // TODO 2：将 B[col,row] 的线性元素索引写入 output_indices[col]。
}

int main() {
    constexpr int lanes = 32;
    const size_t bytes = lanes * sizeof(int);
    std::vector<int> input(lanes, -1), output(lanes, -1);
    int *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, output.data(), bytes, cudaMemcpyHostToDevice));
    record_indices<<<1, lanes>>>(d_input, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(input.data(), d_input, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int lane = 0; lane < lanes; ++lane) {
        if (lane < 4) std::printf("lane %d: 输入索引 %d，输出索引 %d\n", lane, input[lane], output[lane]);
        if (input[lane] != lane || output[lane] != lane * 4) correct = false;
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    if (!correct) { std::fprintf(stderr, "索引验证未通过。\n"); return EXIT_FAILURE; }
    std::printf("通过：32 个 lane 的索引。\n");
    return EXIT_SUCCESS;
}
