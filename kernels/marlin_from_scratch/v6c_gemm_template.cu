// V6c：V5ef 的计算、同步、输出不变；B 改为从 V6 重排结果解码。
// 先完成 v6b_repack.cu；复制本文件为 v6c_gemm.cu 并填写 v6_read_q。
#define V5EF_NO_MAIN
#include "v5ef_lowlevel_template.cu"
#define V6B_NO_MAIN
#include "v6b_repack.cu"

__device__ int v6_read_q(const uint32_t* r,int k,int n,int N) {
    // TODO：R 形状为 [K/16,ceil(N/8),16]；n 决定 ntile 和 nibble。
    // 按 V6a 图从 R 中取出该逻辑 Q[k,n]，返回 0..15。
    // r[k/16][n/8][k%16]
    int ntiles = (N + 7) / 8;
    return (r[(k/16)*ntiles*16+n/8*16+k%16] >> (n%8*4)) & 0xf;
    (void)r;(void)k;(void)n;(void)N;
    return 0;
}

__global__ void v6_gemm(const __nv_bfloat16* a,const uint32_t* packed,
                          const uint8_t* z,const __nv_bfloat16* s,
                          float* c,int M,int N,int K) {
    __shared__ __align__(16) __nv_bfloat16 as[16][16],bs[16][8];
    const int lane=threadIdx.x,row0=blockIdx.y*16,col0=blockIdx.x*8;

    float acc[4]={0.f,0.f,0.f,0.f};
    uint32_t af[4], bf[2];

    for(int base=0;base<K;base+=16) {
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, t = i % 16;
            if (row0 + r >= M || base + t >= K) as[r][t] = 0;
            else as[r][t] = a[(row0 + r) * K + base + t];
        }
        for (int i = lane; i < 128; i += 32) {
            int r = i / 8, t = i % 8;
            if (base + r >= K || col0 + t >= N) bs[r][t] = 0;
            else bs[r][t] = __float2bfloat16((v6_read_q(packed, base+r, col0+t, N) - int(z[(base+r)/128*N+col0+t])) * __bfloat162float(s[(base+r)/128*N+col0+t]));
        }
        __syncwarp();
        auto* addr_a_bf16 = &as[lane%16][lane<16?0:8];
        auto* addr_b_bf16 = &bs[lane%16][0];
        load_a(__cvta_generic_to_shared(addr_a_bf16), af);
        load_b(__cvta_generic_to_shared(addr_b_bf16), bf);
        mma_accumulate(af, bf, acc);
        __syncwarp();
    }

    int g=lane/4, t=lane%4;
    if(row0+g<M && col0+t*2<N) c[(row0+g)*N+col0+t*2] = acc[0];
    if(row0+g<M && col0+t*2+1<N) c[(row0+g)*N+col0+t*2+1] = acc[1];
    if(row0+g+8<M && col0+t*2<N) c[(row0+g+8)*N+col0+t*2] = acc[2];
    if(row0+g+8<M && col0+t*2+1<N) c[(row0+g+8)*N+col0+t*2+1] = acc[3];
    (void)a;(void)packed;(void)z;(void)s;(void)c;(void)M;(void)N;(void)K;(void)row0;(void)col0;(void)lane;(void)acc;
}


static bool test_v6(int M,int N,int K,bool frac) {
    V5Inputs h(M,N,K,frac); V5Device d(h);
    int nt=(N+7)/8, count=K*nt;
    uint32_t* dr=nullptr;float *c5=nullptr,*c6=nullptr;
    CUDA_CHECK(cudaMalloc(&dr,size_t(count)*sizeof(*dr)));
    CUDA_CHECK(cudaMalloc(&c5,size_t(M)*N*sizeof(float)));CUDA_CHECK(cudaMalloc(&c6,size_t(M)*N*sizeof(float)));
    CUDA_CHECK(cudaMemset(dr,0xff,size_t(count)*sizeof(*dr)));
    CUDA_CHECK(cudaMemset(c5,0xff,size_t(M)*N*sizeof(float)));CUDA_CHECK(cudaMemset(c6,0xff,size_t(M)*N*sizeof(float)));
    repack_k_to_n<<<(count+255)/256,256>>>(d.packed,dr,K,N);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> r(count);CUDA_CHECK(cudaMemcpy(r.data(),dr,size_t(count)*sizeof(*dr),cudaMemcpyDeviceToHost));
    int pack_errors=0;
    for(int k=0;k<K;++k) for(int tile=0;tile<nt;++tile) {
        uint32_t expected=0;
        for(int j=0;j<8;++j) {int n=tile*8+j;if(n<N) expected|=uint32_t(h.q[k*N+n])<<(4*j);}
        if(r[(k/16)*nt*16+tile*16+(k%16)]!=expected) ++pack_errors;
    }
    if(pack_errors) {
        std::fprintf(stderr,"M=%d N=%d K=%d：R 对照逻辑 Q 失败 %d，不运行 GEMM\n",M,N,K,pack_errors);
        CUDA_CHECK(cudaFree(dr));CUDA_CHECK(cudaFree(c5));CUDA_CHECK(cudaFree(c6));return false;
    }
    dim3 block(32),grid((N+7)/8,(M+15)/16);
    v5ef_gemm<<<grid,block>>>(d.a,d.packed,d.z,d.s,c5,M,N,K);
    v6_gemm<<<grid,block>>>(d.a,dr,d.z,d.s,c6,M,N,K);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> v5(M*N),v6(M*N);
    CUDA_CHECK(cudaMemcpy(v5.data(),c5,v5.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v6.data(),c6,v6.size()*sizeof(float),cudaMemcpyDeviceToHost));
    int e5=0,e6=0,diff=0;float max5=0,max6=0,maxdiff=0;
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        float expected=0;for(int k=0;k<K;++k) expected=std::fma(__bfloat162float(h.a[m*K+k]),h.weight(k,n),expected);
        float x=v5[m*N+n],y=v6[m*N+n],limit=0.002f+0.0001f*std::fabs(expected);
        if(!std::isfinite(x)||std::fabs(x-expected)>limit) ++e5;
        if(!std::isfinite(y)||std::fabs(y-expected)>limit) {if(e6<3) std::fprintf(stderr,"C[%d,%d]: V6 %g expected %g\n",m,n,y,expected);++e6;}
        if(std::isfinite(x)) max5=std::max(max5,std::fabs(x-expected));
        if(std::isfinite(y)) max6=std::max(max6,std::fabs(y-expected));
        if(!std::isfinite(x)||!std::isfinite(y)||std::fabs(x-y)>limit) ++diff;
        if(std::isfinite(x)&&std::isfinite(y)) maxdiff=std::max(maxdiff,std::fabs(x-y));
    }
    std::printf("M=%d N=%d K=%d frac=%d R错误=%d V5错误=%d V6错误=%d 差异超差=%d max5=%g max6=%g maxdiff=%g\n",M,N,K,frac,pack_errors,e5,e6,diff,max5,max6,maxdiff);
    CUDA_CHECK(cudaFree(dr));CUDA_CHECK(cudaFree(c5));CUDA_CHECK(cudaFree(c6));
    return !e5 && !e6 && !diff;
}
#ifndef V6C_NO_MAIN
int main() {
    int dev;cudaDeviceProp prop{};CUDA_CHECK(cudaGetDevice(&dev));CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    if(prop.major<8) {std::fprintf(stderr,"需要 SM80+\n");return EXIT_FAILURE;}
    std::printf("GPU=%s SM=%d.%d atol=0.002 rtol=0.0001\n",prop.name,prop.major,prop.minor);
    bool ok=true;
    for(bool f:{false,true}) {
        ok=test_v6(16,8,256,f)&&ok;
        ok=test_v6(1,1,128,f)&&ok;
        ok=test_v6(17,19,256,f)&&ok;
        ok=test_v6(31,33,384,f)&&ok;
        ok=test_v6(16,128,256,f)&&ok;
    }
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
#endif
