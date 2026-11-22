// 第 10 节：实现 shared-memory tiled FP32 GEMM。
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

constexpr int T = 4;
__global__ void tiled_gemm(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[T][T];
    __shared__ float bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    // TODO 1：计算 row、col，声明并初始化 float 累加器。
    int row = blockIdx.y * blockDim.y + ty, col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;

    for (int base = 0; base < k; base += T) {
        // TODO 2：加载 As[ty][tx] 和 Bs[ty][tx]；越界时写 0.0f。
        // 注意变量名为小写 as、bs；A 和 B 的 K 索引不同。
        if (ty < T && tx < T && row < m && (base + tx) < k) {
            as[ty][tx] = a[row * k + base + tx];
        }
        else {
            as[ty][tx] = 0.0f;
        }

        if (ty < T && tx < T && (base + ty) < k && col < n) {
            bs[ty][tx] = b[(base + ty) * n + col];
        }
        else {
            bs[ty][tx] = 0.0f;
        }
        // TODO 3：同步，循环累加 T 项，再同步以便复用 shared 内存。
        __syncthreads();
        for (int t = 0; t < T; t++) {
            sum += as[ty][t] * bs[t][tx];
        }
        __syncthreads();
    }
    // TODO 4：仅当 row、col 都有效时写出对应的 C。
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

struct Shape { int m, n, k; };
int main() {
    const Shape cases[] = {{5, 6, 7}, {1, 1, 1}, {4, 4, 4}, {3, 7, 9}};
    bool all_ok = true;
    for (const Shape s : cases) {
        std::vector<float> a(s.m * s.k), b(s.k * s.n), c(s.m * s.n, -999.0f);
        for (int i = 0; i < s.m * s.k; ++i) a[i] = static_cast<float>((i % 5) - 2);
        for (int i = 0; i < s.k * s.n; ++i) b[i] = static_cast<float>((i % 7) - 3);
        float *da = nullptr, *db = nullptr, *dc = nullptr;
        CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dc, c.data(), c.size() * sizeof(float), cudaMemcpyHostToDevice));
        dim3 block(T, T), grid((s.n + T - 1) / T, (s.m + T - 1) / T);
        tiled_gemm<<<grid, block>>>(da, db, dc, s.m, s.n, s.k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int row = 0; row < s.m; ++row) {
            for (int col = 0; col < s.n; ++col) {
                float expected = 0.0f;
                for (int p = 0; p < s.k; ++p) expected += a[row * s.k + p] * b[p * s.n + col];
                if (c[row * s.n + col] != expected) {
                    std::fprintf(stderr, "(%d,%d,%d) C[%d,%d]: 得到 %g，期望 %g\n", s.m, s.n, s.k, row, col, c[row * s.n + col], expected);
                    ok = false;
                }
            }
        }
        CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
        std::printf("M=%d N=%d K=%d：%s\n", s.m, s.n, s.k, ok ? "通过" : "失败");
        all_ok = all_ok && ok;
    }
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
