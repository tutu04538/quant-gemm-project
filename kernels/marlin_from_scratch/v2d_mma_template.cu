// V2d：单 warp 的 m16n8k16 BF16 MMA；先填写 fragment 映射，再写回 C。
// 练习直接从 global memory 组装寄存器，不使用 ldmatrix，不是性能实现。
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
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

// 两个 BF16 的位模式按 low/high 组合成一个 32-bit 输入寄存器。
__device__ inline uint32_t pack_pair(__nv_bfloat16 low, __nv_bfloat16 high) {
    return uint32_t(__bfloat16_as_ushort(low)) | (uint32_t(__bfloat16_as_ushort(high)) << 16);
}

__global__ void mma_one_warp(const __nv_bfloat16* a, const __nv_bfloat16* b, float* out) {
    // A 是行优先 [16,16]，B 是行优先 [16,8]，输出是行优先 [16,8]。
    // launch 约定：恰好 1 个 block，32 个线程。
    const int lane = threadIdx.x;
    const int group = lane / 4;
    const int in_group = lane % 4;
    uint32_t af[4] = {}, bf[2] = {};
    // TODO 1：按 V2c 文档中 PTX m16n8k16 的 A/B fragment 表，从 a、b 读取 BF16，
    //         用 pack_pair 将每两个值按 low/high 顺序装入 af[0..3]、bf[0..1]。
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    // 已给出 MMA 指令；此指令需要完整 warp 中所有 lane 一致执行。
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
          "r"(bf[0]), "r"(bf[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    // TODO 2：将当前 lane 的 d0..d3 写到它们各自对应的 out 元素。
}

static bool check_case(bool fractional) {
    constexpr int M=16, N=8, K=16;
    std::vector<__nv_bfloat16> a(M*K), b(K*N);
    std::vector<float> out(M*N, -999.0f);
    for (int i=0; i<M*K; ++i) a[i]=__float2bfloat16(fractional ? ((i%17)-8)*0.1f : float((i%7)-3));
    for (int i=0; i<K*N; ++i) b[i]=__float2bfloat16(fractional ? ((i%13)-6)*0.1f : float((i%5)-2));
    __nv_bfloat16 *da=nullptr, *db=nullptr;
    float* dc=nullptr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*sizeof(a[0])));
    CUDA_CHECK(cudaMalloc(&db,b.size()*sizeof(b[0])));
    CUDA_CHECK(cudaMalloc(&dc,out.size()*sizeof(out[0])));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(a[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(b[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc,out.data(),out.size()*sizeof(out[0]),cudaMemcpyHostToDevice));
    mma_one_warp<<<1,32>>>(da,db,dc);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*sizeof(out[0]),cudaMemcpyDeviceToHost));
    const float atol=fractional ? 0.002f : 0.0f, rtol=fractional ? 0.0001f : 0.0f;
    int errors=0;
    for (int row=0;row<M;++row) for (int col=0;col<N;++col) {
        float expected=0.0f;
        for (int p=0;p<K;++p) expected+=__bfloat162float(a[row*K+p])*__bfloat162float(b[p*N+col]);
        float actual=out[row*N+col];
        if (!std::isfinite(actual) || std::fabs(actual-expected)>atol+rtol*std::fabs(expected)) {
            if (errors<3) std::fprintf(stderr,"C[%d,%d]=%g，期望 %g\n",row,col,actual,expected);
            ++errors;
        }
    }
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    std::printf("m16n8k16 %s：%s，超差=%d\n",fractional?"BF16 小数":"整数",errors?"失败":"通过",errors);
    return errors==0;
}

int main() {
    int device=0; cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    if (prop.major<8) { std::fprintf(stderr,"本练习需要支持 BF16 MMA 的 SM80 或更新设备。\n"); return EXIT_FAILURE; }
    bool ok=check_case(false);
    ok=check_case(true) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
