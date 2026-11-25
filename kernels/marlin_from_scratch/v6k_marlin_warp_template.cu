// V6k：固定实例一个 block 的 4 warp，K=128/N=64；输入是参考布局 B 寄存器片段。
// 教学诊断版：不包含参考 shared stage、跨 block 归约或 epilogue。
#include "v5_test_common.cuh"
#include <random>
#include <cstring>
#undef CUDA_CHECK
#define main v2d_old_test_main
#include "v2d_ldmatrix_mma.cu"
#undef main
__global__ void v6k_warps(const __nv_bfloat16* a,const uint32_t* bfrag,float* out,float* debug) {
    __shared__ __align__(16) __nv_bfloat16 as[4][16][16];
    __shared__ float partial[4][16][64];
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    bool ready=true;if(!ready)return; // 必须全 block 一致
    // TODO 1：每 warp K 范围 warp*32..warp*32+31；每 lane acc[4][2][4] 清零。
    // TODO 2：逐 step=0,1，协作加载 A 到各 warp 的 as 并同步，load_a。
    // TODO 3：逐 j=0..3，从 bfrag[warp,step,lane,j,0..3] 加载对应 4 个
    // BF16x2 寄存器；分别执行两条 MMA，更新各 j 的两个 acc。
    // TODO 4：warp 内同步后复用 as；按 lane 输出坐标把 acc 写 partial[warp]。
    float acc[4][2][4] = {};
    uint32_t af[4] = {};
    uint32_t bf[4] = {};
    for (int step = 0; step < 2; step++) {
        const __nv_bfloat16* a_start_addr = a + warp*32 + step*16;
        for (int i = lane; i < 256; i += 32) {
            as[warp][i/16][i%16] = a_start_addr[i/16*128+i%16];
        }
        __syncwarp();
        __nv_bfloat16 *a_local_addr = &as[warp][0][0] + lane%16*16 + (lane<16?0:8);
        load_a(__cvta_generic_to_shared(a_local_addr), af);
        for (int j = 0; j < 4; j++) {
            for (int t = 0; t < 4; t++) {
                bf[t] = bfrag[warp*(2*32*4*4)+step*(32*4*4)+lane*(4*4)+j*4+t];
            }
            __syncwarp();
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[j][0][0]), "+f"(acc[j][0][1]), "+f"(acc[j][0][2]), "+f"(acc[j][0][3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));

            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[j][1][0]), "+f"(acc[j][1][1]), "+f"(acc[j][1][2]), "+f"(acc[j][1][3])
                : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[2]), "r"(bf[3]));
        }
    }
    int g = lane / 4, t = lane % 4;
    for (int j = 0; j < 4; j++) {
        partial[warp][g][j*16+t*2] = acc[j][0][0];
        partial[warp][g][j*16+t*2+1] = acc[j][0][1];
        partial[warp][g+8][j*16+t*2] = acc[j][0][2];
        partial[warp][g+8][j*16+t*2+1] = acc[j][0][3];
        partial[warp][g][j*16+8+t*2] = acc[j][1][0];
        partial[warp][g][j*16+8+t*2+1] = acc[j][1][1];
        partial[warp][g+8][j*16+8+t*2] = acc[j][1][2];
        partial[warp][g+8][j*16+8+t*2+1] = acc[j][1][3];
    }
    __syncthreads();
    for(int i=threadIdx.x;i<4*16*64;i+=128)debug[i]=partial[i/(16*64)][(i/64)%16][i%64];
    // TODO 5：由 warp 0 把四份 partial 相加，写 out[16,64]。
    if (warp == 0) {
        for (int i = lane; i < 16*64; i += 32) {
            float o = 0.0f;
            for (int w = 0; w < 4; w++) {
                o += partial[w][i/64][i%64];
            }
            out[i] = o;
        }
    }
    (void)a;(void)bfrag;(void)out;(void)as;(void)warp;(void)lane;
}
static bool test_case(unsigned seed) {
    constexpr int M=16,N=64,K=128;std::mt19937 rng(seed);
    std::vector<__nv_bfloat16> a(M*K),b(K*N);for(auto& v:a)v=__float2bfloat16_rn(float(int(rng()%13)-6)*0.125f);
    for(auto& v:b)v=__float2bfloat16_rn(float(int(rng()%25)-12)*0.25f);
    std::vector<uint32_t> bf(4*2*32*4*4);
    for(int w=0;w<4;++w)for(int step=0;step<2;++step)for(int lane=0;lane<32;++lane)for(int j=0;j<4;++j){int g=lane/4,t=lane%4;
        for(int x=0;x<4;++x){int k=w*32+step*16+2*t+(x%2)*8,n=j*16+g+(x/2)*8;uint16_t lo,hi;
            std::memcpy(&lo,&b[k*N+n],2);std::memcpy(&hi,&b[(k+1)*N+n],2);
            bf[((((w*2+step)*32+lane)*4+j)*4+x)]=uint32_t(lo)|(uint32_t(hi)<<16);}}
    __nv_bfloat16* da=nullptr;uint32_t* db=nullptr;float *dc=nullptr,*dd=nullptr;
    std::vector<float> out(M*N),parts(4*M*N);
    CUDA_CHECK(cudaMalloc(&da,a.size()*2));CUDA_CHECK(cudaMalloc(&db,bf.size()*4));CUDA_CHECK(cudaMalloc(&dc,out.size()*4));CUDA_CHECK(cudaMalloc(&dd,parts.size()*4));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,bf.data(),bf.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dc,0xff,out.size()*4));CUDA_CHECK(cudaMemset(dd,0xff,parts.size()*4));
    v6k_warps<<<1,128>>>(da,db,dc,dd);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(parts.data(),dd,parts.size()*4,cudaMemcpyDeviceToHost));
    int pe=0,ce=0;float maxerr=0;
    for(int w=0;w<4;++w)for(int m=0;m<M;++m)for(int n=0;n<N;++n){float expected=0;
        for(int k=w*32;k<w*32+32;++k)expected=std::fma(__bfloat162float(a[m*K+k]),__bfloat162float(b[k*N+n]),expected);
        float v=parts[w*M*N+m*N+n];if(!std::isfinite(v)||std::fabs(v-expected)>0.002f+0.0001f*std::fabs(expected))++pe;}
    for(int m=0;m<M;++m)for(int n=0;n<N;++n){float expected=0;
        for(int k=0;k<K;++k)expected=std::fma(__bfloat162float(a[m*K+k]),__bfloat162float(b[k*N+n]),expected);
        float v=out[m*N+n],err=std::fabs(v-expected);if(!std::isfinite(v)||err>0.002f+0.0001f*std::fabs(expected))++ce;if(std::isfinite(v))maxerr=std::max(maxerr,err);}
    std::printf("seed=%u partial错误=%d C错误=%d max_abs=%g\n",seed,pe,ce,maxerr);
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));CUDA_CHECK(cudaFree(dd));return !pe&&!ce;
}
int main(){bool ok=true;for(unsigned s:{1u,456u,2026u})ok=test_case(s)&&ok;return ok?EXIT_SUCCESS:EXIT_FAILURE;}
