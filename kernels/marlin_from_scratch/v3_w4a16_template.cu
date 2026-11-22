// V3：教学用 W4A16；详细契约见 learning_notes/24_V3_W4A16.md。
// 复制为 v3_w4a16.cu，完成 decode_weight 与 w4a16_gemm 的 TODO。
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err_)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 固定 group size 128；Q 按 K 轴每 8 个打包成 uint32。
constexpr int GROUP = 128;

__device__ __nv_bfloat16 decode_weight(const uint32_t* packed, const uint8_t* zero,
                                        const __nv_bfloat16* scale, int k, int n, int N) {
    // TODO 1：读取包含 Q[k,n] 的 word，并取出对应的 4 bit 无符号 q。
    int q = packed[k / 8 * N + n] >> ((k % 8) * 4) & 0xf;
    // TODO 2：取 group=k/128 对应的 zero 与 BF16 scale，计算 (q-zero)*scale。
    int z = zero[k / 128 * N + n];
    float s = __bfloat162float(scale[k / 128 * N + n]);
    float res = (q - z) * s; 
    // TODO 3：将结果用 __float2bfloat16_rn 转成 BF16 并返回。
    // 注意减法时先将 q、zero 转成 int，避免无符号减法产生大正数。
    return __float2bfloat16_rn(res);
}

// 测试入口：每个线程解码一个逻辑权重，便于单独定位 Q/Z/S 映射。
__global__ void decode_all(const uint32_t* packed, const uint8_t* zero,
                           const __nv_bfloat16* scale, __nv_bfloat16* out, int K, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < K * N) {
        int k = index / N, n = index % N;
        out[index] = decode_weight(packed, zero, scale, k, n, N);
    }
}

__global__ void w4a16_gemm(const __nv_bfloat16* a, const uint32_t* packed,
                            const uint8_t* zero, const __nv_bfloat16* scale,
                            float* c, int M, int N, int K) {
    // TODO 4：用 blockIdx、blockDim 和 threadIdx 求输出 row、col。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO 5：越界线程返回；每个有效线程沿 K 依次调用 decode_weight。
    if (row >= M || col >= N) return;
    // TODO 6：BF16 A 和解码的 BF16 W 分别转 float，FP32 累加，写入 C[row*N+col]。
    // 每个线程只维护一个输出累加值，不需 shared memory，也不需 __syncthreads。
    float sum = 0.0f;
    for (int p = 0; p < K; p++) {
        __nv_bfloat16 q = decode_weight(packed, zero, scale, p, col, N);
        sum += __bfloat162float(a[row * K + p]) * __bfloat162float(q);
    }
    c[row * N + col] = sum;
}

struct Shape { int m, n, k; };

static bool run_case(Shape sh, int pattern) {
    const int M = sh.m, N = sh.n, K = sh.k;
    const int groups = K / GROUP;
    std::vector<uint8_t> logical_q(K * N), z(groups * N);
    std::vector<__nv_bfloat16> a(M * K), s(groups * N), decoded(K * N);
    std::vector<float> c(M * N, std::numeric_limits<float>::quiet_NaN());
    // 原始逻辑 Q 和 Q 打包是测试准备步骤；CPU 参考只使用原始逻辑 Q。
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            logical_q[k * N + n] = static_cast<uint8_t>((k + 3 * n + 5 * (k / GROUP)) % 16);
    for (int g = 0; g < groups; ++g)
        for (int n = 0; n < N; ++n) {
            z[g * N + n] = static_cast<uint8_t>((g * 7 + n * 3) % 16);
            const float scales[] = {0.1f, 0.25f, 0.375f, 0.7f};
            s[g * N + n] = __float2bfloat16_rn(pattern ? scales[(g + n) % 4] : 0.25f);
        }
    for (int i = 0; i < M * K; ++i)
        a[i] = __float2bfloat16_rn(pattern ? ((i % 19) - 9) * 0.1f : float((i % 5) - 2));
    std::vector<uint32_t> packed((K / 8) * N, 0);
    for (int base = 0; base < K; base += 8)
        for (int n = 0; n < N; ++n) {
            uint32_t word = 0;
            for (int j = 0; j < 8; ++j)
                word |= uint32_t(logical_q[(base + j) * N + n]) << (4 * j);
            packed[(base / 8) * N + n] = word;
        }
    // 独立手算锚点：n=0 的前 8 个 q 为 0,1,...,7。
    if (packed[0] != 0x76543210u) {
        std::fprintf(stderr, "测试准备错误：固定 word != 0x76543210\n");
        return false;
    }
    uint32_t* dq = nullptr; uint8_t* dz = nullptr;
    __nv_bfloat16 *da = nullptr, *ds = nullptr, *dw = nullptr;
    float* dc = nullptr;
    CUDA_CHECK(cudaMalloc(&dq, packed.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dz, z.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&ds, s.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dw, decoded.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dq, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, z.data(), z.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds, s.data(), s.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, c.data(), c.size() * sizeof(float), cudaMemcpyHostToDevice));
    decode_all<<<(K * N + 255) / 256, 256>>>(dq, dz, ds, dw, K, N);
    CUDA_CHECK(cudaGetLastError());
    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    w4a16_gemm<<<grid, block>>>(da, dq, dz, ds, dc, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(decoded.data(), dw, decoded.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    int w_errors = 0, c_errors = 0; float max_err = 0.0f, max_math_diff = 0.0f;
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
        int group_index = (k / GROUP) * N + n;
        float w = float(int(logical_q[k * N + n]) - int(z[group_index])) * __bfloat162float(s[group_index]);
        float rounded = __bfloat162float(__float2bfloat16_rn(w));
        float got = __bfloat162float(decoded[k * N + n]);
        if (!std::isfinite(got) || got != rounded) {
            if (w_errors < 3) std::fprintf(stderr, "  W[%d,%d] 得到 %g，期望 %g\n", k, n, got, rounded);
            ++w_errors;
        }
    }
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float expected = 0, math = 0;
        for (int k = 0; k < K; ++k) {
            int group_index = (k / GROUP) * N + n;
            float av = __bfloat162float(a[m * K + k]);
            float w = float(int(logical_q[k * N + n]) - int(z[group_index])) * __bfloat162float(s[group_index]);
            float wr = __bfloat162float(__float2bfloat16_rn(w));
            expected = std::fma(av, wr, expected);
            math = std::fma(av, w, math);
        }
        float got = c[m * N + n];
        float error = std::fabs(got - expected);
        max_math_diff = std::max(max_math_diff, std::fabs(expected - math));
        if (!std::isfinite(got) || error > 0.0001f + 0.00001f * std::fabs(expected)) {
            if (c_errors < 3) std::fprintf(stderr, "  C[%d,%d] 得到 %g，期望 %g\n", m, n, got, expected);
            ++c_errors;
        }
        if (std::isfinite(got)) max_err = std::max(max_err, error);
    }
    std::printf("M=%d N=%d K=%d pattern=%d: W错误=%d C错误=%d 最大误差=%g；BF16权重舍入与数学参考最大差=%g\n",
                M, N, K, pattern, w_errors, c_errors, max_err, max_math_diff);
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dz)); CUDA_CHECK(cudaFree(ds));
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(dc));
    return w_errors == 0 && c_errors == 0;
}

int main() {
    int device; cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU=%s SM=%d.%d；容差 atol=0.0001 rtol=0.00001\n", prop.name, prop.major, prop.minor);
    const Shape cases[] = {{1,1,128}, {3,9,256}, {16,128,256}};
    bool ok = true;
    for (const Shape sh : cases) for (int pattern = 0; pattern < 2; ++pattern)
        ok = run_case(sh, pattern) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
