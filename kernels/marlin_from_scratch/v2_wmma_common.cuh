// V2a/V2b 共用测试框架。输入 BF16，accumulator 和输出 FP32。
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 两份练习文件都实现此函数，每个 block 启动一个完整 warp。
__global__ void v2_wmma(const __nv_bfloat16* a, const __nv_bfloat16* b,
                        float* c, int m, int n, int k);

struct V2Shape { int m, n, k; };

static bool v2_run_case(V2Shape shape, bool fractional) {
    const int m = shape.m, n = shape.n, k = shape.k;
    // 限定范围：三个尺寸为正且均能被 16 整除。当前没有尾部支持。
    if (m <= 0 || n <= 0 || k <= 0 || m % 16 || n % 16 || k % 16) {
        std::fprintf(stderr, "V2 当前仅接受正的 16 倍数形状。\n");
        return false;
    }
    std::vector<__nv_bfloat16> a(m * k), b(k * n);
    std::vector<float> c(m * n, -999.0f);
    for (int i = 0; i < m * k; ++i) {
        float v = fractional ? ((i % 17) - 8) * 0.1f : float((i % 7) - 3);
        a[i] = __float2bfloat16(v);
    }
    for (int i = 0; i < k * n; ++i) {
        float v = fractional ? ((i % 13) - 6) * 0.1f : float((i % 5) - 2);
        b[i] = __float2bfloat16(v);
    }
    __nv_bfloat16 *da = nullptr, *db = nullptr;
    float* dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(a[0])));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(b[0])));
    CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(c[0])));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(a[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(b[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, c.data(), c.size() * sizeof(c[0]), cudaMemcpyHostToDevice));
    const dim3 block(32), grid(n / 16, m / 16);
    v2_wmma<<<grid, block>>>(da, db, dc, m, n, k);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(c[0]), cudaMemcpyDeviceToHost));

    // 独立 CPU 数学参考：先按传入的 BF16 值还原 float，再按 FP32 顺序累加。
    // 本实验预先约定 fractional 时 atol=0.002、rtol=0.0001；整数 case 精确比较。
    const float atol = fractional ? 0.002f : 0.0f;
    const float rtol = fractional ? 0.0001f : 0.0f;
    int errors = 0;
    float max_abs = 0.0f;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float expected = 0.0f;
            for (int p = 0; p < k; ++p)
                expected += __bfloat162float(a[row*k+p]) * __bfloat162float(b[p*n+col]);
            const float actual = c[row*n+col];
            const float err = std::fabs(actual - expected);
            if (!std::isfinite(actual) || err > atol + rtol * std::fabs(expected)) {
                if (errors < 3) std::fprintf(stderr, "C[%d,%d]=%g，期望 %g\n", row, col, actual, expected);
                ++errors;
            }
            if (std::isfinite(actual)) max_abs = std::max(max_abs, err);
        }
    }
    std::printf("M=%d N=%d K=%d %s：%s，超差=%d，最大绝对误差=%g\n",
                m, n, k, fractional ? "BF16 小数" : "整数", errors ? "失败" : "通过", errors, max_abs);
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    return errors == 0;
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) { std::fprintf(stderr, "本练习需要支持 BF16 WMMA 的 SM80 或更新设备。\n"); return EXIT_FAILURE; }
    std::printf("GPU=%s SM=%d.%d；每 block 一个 warp\n", prop.name, prop.major, prop.minor);
    bool correct = true;
#ifdef V2_SINGLE_TILE
    correct = v2_run_case({16,16,16}, false) && correct;
    correct = v2_run_case({16,16,16}, true) && correct;
#else
    for (V2Shape shape : {V2Shape{16,16,16}, V2Shape{16,32,32}, V2Shape{32,16,48}, V2Shape{16,128,256}}) {
        correct = v2_run_case(shape, false) && correct;
        correct = v2_run_case(shape, true) && correct;
    }
#endif
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
