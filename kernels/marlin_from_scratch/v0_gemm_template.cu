// V0：朴素 FP32 GEMM。只填写 v0_gemm 内的 TODO。
// 每个线程计算一个 C[row,col]；输入输出均为行优先、连续、正尺寸。
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void v0_gemm(const float* a, const float* b, float* c,
                        int m, int n, int k) {
    // TODO 1：由 blockIdx、blockDim、threadIdx 计算输出行 row 和列 col。
    int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO 2：检查 row < m 且 col < n，越界线程不能访问数组。
    if (row >= m || col >= n) return; 
    // TODO 3：初始化 FP32 累加器，遍历 K，计算 A[row,p] * B[p,col]。
    float sum = 0.0f;
    for (int t = 0; t < k; t++) sum += a[row * k + t] * b[t * n + col];
    // TODO 4：将结果写入 C[row,col]；只写当前线程负责的输出。
    c[row * n + col] = sum;
}

struct Shape { int m, n, k; };
constexpr int block_x = 16;
constexpr int block_y = 16;

// 数值选择为小整数：本模板的所有乘积和部分和都可由 FP32 精确表示。
// 因此此处要求逐元素精确相等；不能推广到一般浮点 GEMM。
static bool run_case(Shape shape, bool benchmark) {
    const int m = shape.m, n = shape.n, k = shape.k;
    std::vector<float> a(m * k), b(k * n), c(m * n, -999.0f);
    for (int i = 0; i < m * k; ++i) a[i] = static_cast<float>((i % 7) - 3);
    for (int i = 0; i < k * n; ++i) b[i] = static_cast<float>((i % 5) - 2);
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, c.data(), c.size() * sizeof(float), cudaMemcpyHostToDevice));
    const dim3 block(block_x, block_y);
    const dim3 grid((n + block_x - 1) / block_x, (m + block_y - 1) / block_y);

    v0_gemm<<<grid, block>>>(da, db, dc, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 参考按行列/K 循环计算，不读取 GPU 中间结果。
    int mismatches = 0;
    float max_abs_error = 0.0f;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float expected = 0.0f;
            for (int p = 0; p < k; ++p) expected += a[row * k + p] * b[p * n + col];
            const float actual = c[row * n + col];
            if (!std::isfinite(actual) || actual != expected) {
                if (mismatches < 3)
                    std::fprintf(stderr, "  C[%d,%d]: 得到 %g，期望 %g\n", row, col, actual, expected);
                ++mismatches;
            }
            if (std::isfinite(actual)) max_abs_error = std::max(max_abs_error, std::fabs(actual - expected));
        }
    }
    std::printf("M=%d N=%d K=%d: %s；不匹配=%d，有限值最大绝对误差=%g\n",
                m, n, k, mismatches ? "失败" : "通过", mismatches, max_abs_error);

    // V0c：仅在该形状正确时计时。样本是“每批重复的平均耗时”，并非单次延迟。
    if (benchmark && mismatches == 0) {
        constexpr int warmup = 10, batches = 15, repeats_per_batch = 100;
        for (int i = 0; i < warmup; ++i) v0_gemm<<<grid, block>>>(da, db, dc, m, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<float> samples_ms;
        for (int batch = 0; batch < batches; ++batch) {
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < repeats_per_batch; ++i) v0_gemm<<<grid, block>>>(da, db, dc, m, n, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            samples_ms.push_back(total_ms / repeats_per_batch);
        }
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        std::sort(samples_ms.begin(), samples_ms.end());
        const int p90_index = (9 * batches + 9) / 10 - 1; // 最近秩：ceil(0.9*N)-1
        std::printf("V0c：warmup=%d，批次=%d，每批=%d；批均值 p50=%.6f ms，p90=%.6f ms\n",
                    warmup, batches, repeats_per_batch, samples_ms[batches / 2], samples_ms[p90_index]);
        std::printf("排序后的每批均值（ms）：");
        for (float sample : samples_ms) std::printf(" %.6f", sample);
        std::printf("\n");
    }

    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    return mismatches == 0;
}

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    if (argc > 1 && !benchmark) {
        std::fprintf(stderr, "用法：%s [--bench]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::printf("逻辑 GPU=%d，%s，SM=%d.%d，driver API=%d，runtime=%d；block=(%d,%d)\n",
                device, prop.name, prop.major, prop.minor, driver_version, runtime_version, block_x, block_y);
    const Shape cases[] = {{2, 2, 3}, {1, 1, 1}, {5, 7, 9}, {17, 19, 23}, {16, 128, 256}};
    bool correct = true;
    for (const Shape shape : cases) correct = run_case(shape, false) && correct;
    if (!correct) {
        std::fprintf(stderr, "正确性未通过；不会执行计时。请检查 kernel 的 TODO。\n");
        return EXIT_FAILURE;
    }
    if (benchmark) return run_case({128, 192, 256}, true) ? EXIT_SUCCESS : EXIT_FAILURE;
    return EXIT_SUCCESS;
}
