// V5d 固定一个 m16n8k16：W4 -> shared BF16 -> ldmatrix -> mma.sync。
#include "v5_test_common.cuh"
#include <cstring>
// 复用 V2d 已完成的汇编包装，重命名其测试 main；本节只补 W4->B shared。
#undef CUDA_CHECK
#define main v2d_old_test_main
#include "v2d_ldmatrix_mma.cu"
#undef main
__global__ void v5d_one_tile(const __nv_bfloat16* a,const uint32_t* packed,
                             const uint8_t* z,const __nv_bfloat16* s,
                             float* c,uint32_t* fragments,int K,int base) {
    constexpr int N=8;
    __shared__ __align__(16) __nv_bfloat16 as[16][16], bs[16][8];
    int lane=threadIdx.x;
    for(int i=lane;i<256;i+=32) as[i/16][i%16]=a[i];
    // TODO 1：32 lane 合作把逻辑 W[base:base+16,0:8] 解码到 bs[16][8]。
    // packed K 轴打包，Z/S 的 group 是 k/128，写入前转换 BF16。
    // TODO 2：填写完后把 ready 改为 true；所有 lane 一致。
    for(int i=lane;i<128;i+=32) {
        int r=i/8,t=i%8;
        bs[r][t] = __float2bfloat16(((int)((packed[(base+r)/8*N+t] >> ((base+r)%8*4)) & 0xf) - (int)(z[(base+r)/128*N+t])) * __bfloat162float(s[(base+r)/128*N+t]));
    }

    bool ready=true;
    if(!ready) return;
    __syncthreads();
    // 以下地址映射、MMA 和输出映射沿用 V2d 的练习结果。
    auto* addr_a_bf16=&as[lane%16][lane<16?0:8];
    auto* addr_b_bf16=&bs[lane%16][0];
    uint32_t af[4],bf[2];
    load_a(static_cast<uint32_t>(__cvta_generic_to_shared(addr_a_bf16)),af);
    load_b(static_cast<uint32_t>(__cvta_generic_to_shared(addr_b_bf16)),bf);
    for(int i=0;i<2;++i) fragments[lane*2+i]=bf[i];
    float d0,d1,d2,d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(af[0]),"r"(af[1]),"r"(af[2]),"r"(af[3]),"r"(bf[0]),"r"(bf[1]),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));
    int g=lane/4,t=lane%4;
    c[g*8+2*t]=d0; c[g*8+2*t+1]=d1;
    c[(g+8)*8+2*t]=d2; c[(g+8)*8+2*t+1]=d3;
    (void)packed;(void)z;(void)s;(void)K;(void)base;
}
static bool test_fragment(int base,bool fractional) {
    constexpr int M=16,N=8,K=256;
    V5Inputs h(M,N,K,fractional); V5Device d(h);
    std::vector<__nv_bfloat16> tile_a(256);
    for(int m=0;m<M;++m) for(int k=0;k<16;++k) tile_a[m*16+k]=h.a[m*K+base+k];
    std::vector<uint32_t> f(64,0xffffffffu);
    std::vector<float> out(128,std::numeric_limits<float>::quiet_NaN());
    __nv_bfloat16* da=nullptr; uint32_t* df=nullptr;float* dc=nullptr;
    CUDA_CHECK(cudaMalloc(&da,tile_a.size()*sizeof(*da))); CUDA_CHECK(cudaMalloc(&df,f.size()*sizeof(*df))); CUDA_CHECK(cudaMalloc(&dc,out.size()*sizeof(*dc)));
    CUDA_CHECK(cudaMemcpy(da,tile_a.data(),tile_a.size()*sizeof(*da),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(df,f.data(),f.size()*sizeof(*df),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc,out.data(),out.size()*sizeof(*dc),cudaMemcpyHostToDevice));
    v5d_one_tile<<<1,32>>>(da,d.packed,d.z,d.s,dc,df,K,base);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(f.data(),df,f.size()*sizeof(*df),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*sizeof(*dc),cudaMemcpyDeviceToHost));
    int fe=0,ce=0;
    for(int lane=0;lane<32;++lane) {
        int g=lane/4,t=lane%4;
        for(int j=0;j<2;++j) {
            int k=base+2*t+8*j;
            auto lo=__float2bfloat16_rn(h.weight(k,g)), hi=__float2bfloat16_rn(h.weight(k+1,g));
            uint32_t expected=expected_pair(lo,hi);
            if(f[lane*2+j]!=expected) { if(fe<3) std::fprintf(stderr,"lane=%d bf[%d]: 0x%x != 0x%x\n",lane,j,f[lane*2+j],expected); ++fe; }
        }
    }
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        float expected=0;
        for(int t=0;t<16;++t) expected=std::fma(__bfloat162float(tile_a[m*16+t]),h.weight(base+t,n),expected);
        float got=out[m*N+n];
        if(!std::isfinite(got)||std::fabs(got-expected)>0.002f+0.0001f*std::fabs(expected)) {
            if(ce<3) std::fprintf(stderr,"C[%d,%d]: %g != %g\n",m,n,got,expected); ++ce;
        }
    }
    std::printf("base=%d fractional=%d B寄存器错误=%d C错误=%d\n",base,fractional,fe,ce);
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(df));CUDA_CHECK(cudaFree(dc));
    return fe==0 && ce==0;
}
int main() {
    int dev;cudaDeviceProp prop{};CUDA_CHECK(cudaGetDevice(&dev));CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    if(prop.major<8) {std::fprintf(stderr,"需要 SM80+\n");return EXIT_FAILURE;}
    bool ok=true;for(bool f:{false,true}) for(int base:{0,112,128,240}) ok=test_fragment(base,f)&&ok;
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
