// V2d 衔接：shared → ldmatrix → mma.sync。只填写 shared 行地址与 ready。
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 对非转置的四个 8×8 片段，向每个 lane 返回 4 个 32-bit 寄存器。
__device__ inline void load_a(uint32_t address, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address) : "memory");
}

// 对两个 8×8 片段按 .trans 装载，供本例的 B 操作数使用。
__device__ inline void load_b(uint32_t address, uint32_t (&r)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(address) : "memory");
}

__global__ void ldmatrix_mma(const __nv_bfloat16* a, const __nv_bfloat16* b,
                             float* out, uint32_t* fragments) {
    // 固定 A[16,16]、B[16,8]、C[16,8]，行优先；一个 block、一个 warp。
    __shared__ __align__(16) __nv_bfloat16 as[16][16];
    __shared__ __align__(16) __nv_bfloat16 bs[16][8];
    const int lane = threadIdx.x;
    for (int i = lane; i < 16*16; i += 32) as[i / 16][i % 16] = a[i];
    for (int i = lane; i < 16*8;  i += 32) bs[i / 8][i % 8] = b[i];
    __syncthreads();

    uint32_t addr_a = 0, addr_b = 0;
    // TODO 1：按课程图，为每条 lane 分别选择 A 的 8×8 子块行首地址。
    __nv_bfloat16 *flat_a = &as[0][0], *addr_a_bf16 = NULL;
    if (lane < 16) addr_a_bf16 = flat_a + lane * 16;
    else addr_a_bf16 = flat_a + 8 + (lane % 16) * 16;
    // TODO 2：为每条 lane 分别选择 B 的 8×8 子块行首地址。
    __nv_bfloat16 *flat_b = &bs[0][0], *addr_b_bf16 = NULL;
    addr_b_bf16 = flat_b + (lane % 16) * 8;
    // TODO 3：用 __cvta_generic_to_shared 转换两个地址，分别赋给 addr_a/addr_b。
    addr_a = __cvta_generic_to_shared(addr_a_bf16);
    addr_b = __cvta_generic_to_shared(addr_b_bf16);
    // TODO 4：完成后将 ready 设为全 warp 一致的 true。
    const bool ready = true;
    if (!ready) return;  // 默认全 warp 一起退出，不执行未填写地址的 ldmatrix。

    uint32_t af[4], bf[2];
    load_a(addr_a, af);
    load_b(addr_b, bf);
    // 保存真实寄存器位模式，host 独立按逻辑坐标检查每条 lane 的输入。
    for (int i=0; i<4; ++i) fragments[lane*6+i] = af[i];
    for (int i=0; i<2; ++i) fragments[lane*6+4+i] = bf[i];

    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    // V2d 已练习的 C 坐标映射，这里由框架提供。
    const int group = lane / 4, t = lane % 4;
    out[group*8+2*t]=d0; out[group*8+2*t+1]=d1;
    out[(group+8)*8+2*t]=d2; out[(group+8)*8+2*t+1]=d3;
}

static uint32_t expected_pair(__nv_bfloat16 low, __nv_bfloat16 high) {
    uint16_t l, h;
    std::memcpy(&l, &low, sizeof(l)); std::memcpy(&h, &high, sizeof(h));
    return uint32_t(l) | (uint32_t(h) << 16);
}

// mode 0：小整数；1：BF16 小数；2：逐位置可识别的 BF16 精确整数。
static bool run_case(int mode) {
    constexpr int M=16,N=8,K=16;
    const bool fractional = mode == 1;
    std::vector<__nv_bfloat16> a(M*K), b(K*N);
    std::vector<float> out(M*N, -999.0f);
    std::vector<uint32_t> frag(32*6, 0xFFFFFFFFu);
    for (int i=0;i<M*K;++i) a[i]=__float2bfloat16(mode == 2 ? float(i-128) : fractional ? ((i%17)-8)*0.1f : float((i%7)-3));
    for (int i=0;i<K*N;++i) b[i]=__float2bfloat16(mode == 2 ? float(i-64) : fractional ? ((i%13)-6)*0.1f : float((i%5)-2));
    __nv_bfloat16 *da=nullptr,*db=nullptr; float* dc=nullptr; uint32_t* df=nullptr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*sizeof(a[0]))); CUDA_CHECK(cudaMalloc(&db,b.size()*sizeof(b[0])));
    CUDA_CHECK(cudaMalloc(&dc,out.size()*sizeof(out[0]))); CUDA_CHECK(cudaMalloc(&df,frag.size()*sizeof(frag[0])));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(a[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(b[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc,out.data(),out.size()*sizeof(out[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(df,frag.data(),frag.size()*sizeof(frag[0]),cudaMemcpyHostToDevice));
    ldmatrix_mma<<<1,32>>>(da,db,dc,df);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*sizeof(out[0]),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(frag.data(),df,frag.size()*sizeof(frag[0]),cudaMemcpyDeviceToHost));

    int fragment_errors=0, output_errors=0;
    for (int lane=0;lane<32;++lane) {
        int g=lane/4,t=lane%4;
        const uint32_t expected[6]={
            expected_pair(a[g*K+2*t],a[g*K+2*t+1]),
            expected_pair(a[(g+8)*K+2*t],a[(g+8)*K+2*t+1]),
            expected_pair(a[g*K+2*t+8],a[g*K+2*t+9]),
            expected_pair(a[(g+8)*K+2*t+8],a[(g+8)*K+2*t+9]),
            expected_pair(b[(2*t)*N+g],b[(2*t+1)*N+g]),
            expected_pair(b[(2*t+8)*N+g],b[(2*t+9)*N+g])};
        for (int i=0;i<6;++i) if (frag[lane*6+i]!=expected[i]) {
            if (fragment_errors<3) std::fprintf(stderr,"lane=%d frag=%d: 0x%08x，期望 0x%08x\n",lane,i,frag[lane*6+i],expected[i]);
            ++fragment_errors;
        }
    }
    const float atol=fractional ? 0.002f : 0.0f, rtol=fractional ? 0.0001f : 0.0f;
    for (int r=0;r<M;++r) for (int col=0;col<N;++col) {
        float expected=0.0f;
        for (int k=0;k<K;++k) expected+=__bfloat162float(a[r*K+k])*__bfloat162float(b[k*N+col]);
        float got=out[r*N+col];
        if (!std::isfinite(got) || std::fabs(got-expected)>atol+rtol*std::fabs(expected)) {
            if (output_errors<3) std::fprintf(stderr,"C[%d,%d]=%g，期望 %g\n",r,col,got,expected);
            ++output_errors;
        }
    }
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(df));
    const char* label = mode == 2 ? "坐标标记" : fractional ? "BF16 小数" : "整数";
    std::printf("%s：fragment 错误=%d，输出超差=%d\n",label,fragment_errors,output_errors);
    return fragment_errors==0 && output_errors==0;
}

int main() {
    int device=0; cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    if (prop.major<8) { std::fprintf(stderr,"需要支持 BF16 MMA 的 SM80 或更新设备。\n"); return EXIT_FAILURE; }
    bool ok=run_case(0); ok=run_case(1) && ok; ok=run_case(2) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
