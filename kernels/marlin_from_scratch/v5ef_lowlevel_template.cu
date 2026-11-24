// V5e/f：一个 block/warp 负责 16×8 输出；逐个 K=16 tile 使用 ldmatrix/mma。
// --e 只测固定 16×8×256；--f 再测多 block 与 M/N 尾部。
#include "v5_test_common.cuh"
#include <cstring>
#undef CUDA_CHECK
#define main v2d_old_test_main
#include "v2d_ldmatrix_mma.cu" // 复用 V2d 的 load_a/load_b 包装
#undef main

// 每条 lane 的 4 个 FP32 accumulator，跨 K 循环保留。
__device__ inline void mma_accumulate(const uint32_t (&af)[4],const uint32_t (&bf)[2],float (&acc)[4]) {
    // TODO 1（V5e）：使用 mma.sync.m16n8k16 的 PTX，对 acc 做读写累加。
    // 提示：在 asm 输出里使用 "+f"(acc[i])，四个 C 输入可引用 %0..%3。
    // A/B 分别使用 4 个和 2 个 "r" 输入；语法示例见 V5e 文档。
    asm volatile("mma.aligned.sync.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
    (void)af;(void)bf;(void)acc;
}

__global__ void v5ef_gemm(const __nv_bfloat16* a,const uint32_t* packed,
                          const uint8_t* z,const __nv_bfloat16* s,
                          float* c,int M,int N,int K) {
    __shared__ __align__(16) __nv_bfloat16 as[16][16],bs[16][8];
    const int lane=threadIdx.x,row0=blockIdx.y*16,col0=blockIdx.x*8;
    // 没完成时整个 warp 返回，避免读取未初始化 shared。
    bool ready=true; // TODO 5：完成 V5e 的 TODO 后设为 true。
    if(!ready) return;
    float acc[4]={0.f,0.f,0.f,0.f};
    uint32_t af[4], bf[2];

    for(int base=0;base<K;base+=16) {
        // TODO 2（V5e）：32 lane 协作加载 A[16,16] 到 as；
        // 解码当前 W[16,8] 到 bs，逐元素先转 BF16。
        // TODO 3（V5e）：所有 lane 写完后同步；像 V5d 一样构造各自 A/B
        // shared 行首地址，转 shared 地址，load_a/load_b 并调用 mma_accumulate。
        // TODO 4（V5e）：确保本轮读取完成后同步，再复用 shared。
        // TODO 6（V5f）：加载 A 与 B 时，M/N 越界的位置必须写 BF16 零。
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, t = i % 16;
            if (row0 + r >= M || base + t >= K) as[r][t] = 0;
            else as[r][t] = a[(row0 + r) * K + base + t];
        }
        for (int i = lane; i < 128; i += 32) {
            int r = i / 8, t = i % 8;
            if (base + r >= K || col0 + t >= N) bs[r][t] = 0;
            else bs[r][t] = __float2bfloat16(((int)((packed[(base+r)/8*N+col0+t] >> ((base+r)%8*4)) & 0xf) - (int)(z[(base+r)/128*N+col0+t])) * __bfloat162float(s[(base+r)/128*N+col0+t]));
        }
        __syncwarp();
        auto* addr_a_bf16 = &as[lane%16][lane<16?0:8];
        auto* addr_b_bf16 = &bs[lane%16][0];
        load_a(__cvta_generic_to_shared(addr_a_bf16), af);
        load_b(__cvta_generic_to_shared(addr_b_bf16), bf);
        mma_accumulate(af, bf, acc);
        __syncwarp();
    }
    // TODO 7（V5e）：按 V5d 的 lane 输出映射把 acc[0..3] 写入 C。
    // TODO 8（V5f）：写 C 之前逐元素检查 M/N，不能让越界 lane 提前退出整个 kernel。

    int g=lane/4, t=lane%4;
    if(row0+g<M && col0+t*2<N) c[(row0+g)*N+col0+t*2] = acc[0];
    if(row0+g<M && col0+t*2+1<N) c[(row0+g)*N+col0+t*2+1] = acc[1];
    if(row0+g+8<M && col0+t*2<N) c[(row0+g+8)*N+col0+t*2] = acc[2];
    if(row0+g+8<M && col0+t*2+1<N) c[(row0+g+8)*N+col0+t*2+1] = acc[3];
    (void)a;(void)packed;(void)z;(void)s;(void)c;(void)M;(void)N;(void)K;(void)row0;(void)col0;(void)lane;(void)acc;
}

static bool test_case(int M,int N,int K,bool fractional) {
    V5Inputs h(M,N,K,fractional); V5Device d(h);
    std::vector<float> out(M*N,std::numeric_limits<float>::quiet_NaN());
    float* dc=nullptr; CUDA_CHECK(cudaMalloc(&dc,out.size()*sizeof(*dc)));
    CUDA_CHECK(cudaMemcpy(dc,out.data(),out.size()*sizeof(*dc),cudaMemcpyHostToDevice));
    dim3 block(32),grid((N+7)/8,(M+15)/16);
    v5ef_gemm<<<grid,block>>>(d.a,d.packed,d.z,d.s,dc,M,N,K);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*sizeof(*dc),cudaMemcpyDeviceToHost));
    constexpr float atol=0.002f,rtol=0.0001f;
    int errors=0;float max_abs=0;double sum_abs=0;
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        float expected=0;
        for(int k=0;k<K;++k) expected=std::fma(__bfloat162float(h.a[m*K+k]),h.weight(k,n),expected);
        float got=out[m*N+n],err=std::fabs(got-expected);
        if(!std::isfinite(got)||err>atol+rtol*std::fabs(expected)) {
            if(errors<3) std::fprintf(stderr,"C[%d,%d]: %g != %g\n",m,n,got,expected); ++errors;
        }
        if(std::isfinite(got)) {max_abs=std::max(max_abs,err);sum_abs+=err;}
    }
    std::printf("M=%d N=%d K=%d fractional=%d errors=%d max=%g mean=%g\n",M,N,K,fractional,errors,max_abs,sum_abs/(M*N));
    CUDA_CHECK(cudaFree(dc)); return errors==0;
}
#ifndef V5EF_NO_MAIN
int main(int argc,char** argv) {
    bool e=argc==2 && std::strcmp(argv[1],"--e")==0;
    bool f=argc==2 && std::strcmp(argv[1],"--f")==0;
    if(!e && !f) {std::fprintf(stderr,"用法：%s --e|--f\n",argv[0]);return EXIT_FAILURE;}
    int dev;cudaDeviceProp prop{};CUDA_CHECK(cudaGetDevice(&dev));CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    if(prop.major<8) {std::fprintf(stderr,"需要 SM80+\n");return EXIT_FAILURE;}
    std::printf("GPU=%s SM=%d.%d atol=0.002 rtol=0.0001\n",prop.name,prop.major,prop.minor);
    bool ok=true;for(bool frac:{false,true}) {
        ok=test_case(16,8,256,frac)&&ok;
        if(f) {
            ok=test_case(1,1,128,frac)&&ok;
            ok=test_case(17,19,256,frac)&&ok;
            ok=test_case(31,33,384,frac)&&ok;
            ok=test_case(16,128,256,frac)&&ok;
        }
    }
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
#endif
