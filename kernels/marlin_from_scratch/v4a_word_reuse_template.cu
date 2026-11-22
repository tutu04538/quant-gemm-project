// V4a：每线程复用一个包含 8 个 W4 权重的 packed word。
// 复制本文件为 v4a_word_reuse.cu，只填写 v4a_gemm 中的 TODO。
// 测试复用 V3 的逻辑输入和独立 CPU 参考；它不是 V3e 的 checked 入口。
#define main v3_original_test_main
#include "v3_w4a16_template.cu"
#undef main

__global__ void v4a_gemm(const __nv_bfloat16* a, const uint32_t* packed,
                          const uint8_t* zero, const __nv_bfloat16* scale,
                          float* c, int M, int N, int K) {
    // TODO 1：和 V3 一样，每个线程映射一个 row、col；越界时直接 return。
    // TODO 2：每轮处理连续 8 个 k；读取 packed[(base/8)*N+col] 一次。
    //         根据 base/128 和 col 读取本组 zero、BF16 scale 一次。
    // TODO 3：对 nibble 0..7 按顺序取 q，整数减 zero，乘 scale 后转 BF16；
    //         A 和 BF16 权重各转 float，按原 k 顺序累加，最后写 C。
    // K 是 128 的倍数：8 个连续 k 不跨 group，最后一轮也不会越界。
    (void)a; (void)packed; (void)zero; (void)scale; (void)c; (void)M; (void)N; (void)K;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int p = 0; p < K; p += 8) {
        uint32_t packed_q = packed[(p/8)*N+col];
        uint8_t z = zero[(p/128)*N+col];
        __nv_bfloat16 s = scale[(p/128)*N+col];
        for (int t = 0; t < 8; t++) {
            __nv_bfloat16 q = __float2bfloat16(((int)((packed_q >> (t*4)) & 0xf) - (int)z) * __bfloat162float(s));
            sum += __bfloat162float(q) * __bfloat162float(a[row*K+p+t]); 
        }
    }
    c[row*N+col] = sum;
}

static bool run_v4_case(Shape sh, int pattern) {
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
    v4a_gemm<<<grid, block>>>(da, dq, dz, ds, dc, M, N, K);
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

#ifndef V4A_NO_MAIN
int main() {
    int device = 0; cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("V4a GPU=%s SM=%d.%d；容差 atol=0.0001 rtol=0.00001\n", prop.name, prop.major, prop.minor);
    const Shape cases[] = {{1,1,128}, {3,9,256}, {16,128,256}, {17,19,384}};
    bool ok = true;
    for (const Shape sh : cases) for (int pattern = 0; pattern < 2; ++pattern)
        ok = run_v4_case(sh, pattern) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
