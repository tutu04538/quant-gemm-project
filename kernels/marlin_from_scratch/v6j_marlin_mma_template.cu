// V6j：固定实例 warp 0 的 N=0..15、K=0..31，两次 K16，每步两条 MMA。
// B 片段以 V6i 的 4 个寄存器形式传入；A 经 shared + ldmatrix。
// PTX 指令与 fragment 映射对照第三方 Marlin/文档；完整 reference pipeline 尚未实现。
#include "v5_test_common.cuh"
#include <random>
#include <cstring>
#undef CUDA_CHECK
#define main v2d_old_test_main
#include "v2d_ldmatrix_mma.cu"
#undef main
__global__ void v6j_mma(const __nv_bfloat16* a,const uint32_t* bfrag,float* out) {
    __shared__ __align__(16) __nv_bfloat16 as[16][16];
    int lane=threadIdx.x;float c0[4]={0,0,0,0},c1[4]={0,0,0,0};
    bool ready=true;if(!ready)return; // 完成 TODO 后全 warp 一致改为 true
    for(int step=0;step<2;++step) {
        // TODO 1：32 lane 协作加载 A[16,32] 的当前 16 列到 shared；同步。
        // TODO 2：按 V2d 的地址映射 load_a，加载本 lane 对应的 bf0[2],bf1[2]。
        // bfrag 的形状是 [2 steps,32 lanes,4 uint32]；先 2 个用于第一个 N8。
        // TODO 3：两条 mma.sync，分别对 c0/c1 跨 step 累加；复用 shared 前同步。
        for (int i = lane; i < 256; i += 32) {
            as[i/16][i%16] = a[i/16*32+step*16+i%16];
        }
        __syncwarp();
        __nv_bfloat16* flat = &as[0][0];
        __nv_bfloat16* a_addr = flat + lane%16*16 + (lane<16?0:8);
        uint32_t af[4] = {0, 0, 0, 0};
        uint32_t bf[4] = {0, 0, 0, 0};
        load_a(__cvta_generic_to_shared(a_addr), af);
        for (int i = 0; i < 4; i++) {
            bf[i] = bfrag[step*32*4+lane*4+i];
        }
        __syncwarp();
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0[0]), "+f"(c0[1]), "+f"(c0[2]), "+f"(c0[3])
            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));

        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c1[0]), "+f"(c1[1]), "+f"(c1[2]), "+f"(c1[3])
            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[2]), "r"(bf[3]));
    }
    // TODO 4：将 c0 写到 C 前 8 列、c1 写到 C 后 8 列，按 V2d 输出映射。
    int g=lane/4, t=lane%4;
    out[g*16+2*t] = c0[0]; out[g*16+2*t+1] = c0[1];
    out[(g+8)*16+2*t] = c0[2]; out[(g+8)*16+2*t+1] = c0[3];
    out[g*16+2*t+8] = c1[0]; out[g*16+2*t+9] = c1[1];
    out[(g+8)*16+2*t+8] = c1[2]; out[(g+8)*16+2*t+9] = c1[3];

    (void)a;(void)bfrag;(void)out;(void)lane;(void)c0;(void)c1;
}
static bool run_case(unsigned seed) {
    constexpr int M=16,N=16,K=32;std::mt19937 rng(seed);
    std::vector<__nv_bfloat16> a(M*K),w(K*N);for(auto& v:a)v=__float2bfloat16_rn(float(int(rng()%13)-6)*0.125f);
    for(auto& v:w)v=__float2bfloat16_rn(float(int(rng()%25)-12)*0.25f);
    // 按独立的逻辑 B 坐标构造 V6i 期望寄存器，非从被测 GPU 结果生成。
    std::vector<uint32_t> bf(2*32*4,0);
    for(int step=0;step<2;++step)for(int lane=0;lane<32;++lane){int g=lane/4,t=lane%4;
        for(int j=0;j<4;++j){int k=step*16+2*t+(j%2)*8,n=g+(j/2)*8;uint16_t lo,hi;
            std::memcpy(&lo,&w[k*N+n],2);std::memcpy(&hi,&w[(k+1)*N+n],2);
            bf[(step*32+lane)*4+j]=uint32_t(lo)|(uint32_t(hi)<<16);}}
    __nv_bfloat16* da=nullptr;uint32_t* db=nullptr;float* dc=nullptr;std::vector<float> out(M*N);
    CUDA_CHECK(cudaMalloc(&da,a.size()*2));CUDA_CHECK(cudaMalloc(&db,bf.size()*4));CUDA_CHECK(cudaMalloc(&dc,out.size()*4));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,bf.data(),bf.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(dc,0xff,out.size()*4));
    v6j_mma<<<1,32>>>(da,db,dc);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*4,cudaMemcpyDeviceToHost));
    int errors=0;float maxerr=0;for(int m=0;m<M;++m)for(int n=0;n<N;++n){float ref=0;
        for(int k=0;k<K;++k)ref=std::fma(__bfloat162float(a[m*K+k]),__bfloat162float(w[k*N+n]),ref);
        float v=out[m*N+n],e=std::fabs(v-ref);if(!std::isfinite(v)||e>0.002f+0.0001f*std::fabs(ref)){if(errors<3)std::fprintf(stderr,"C[%d,%d]=%g expected=%g\n",m,n,v,ref);++errors;}if(std::isfinite(v))maxerr=std::max(maxerr,e);}
    std::printf("seed=%u 16x16 两步 MMA：错误=%d max_abs=%g\n",seed,errors,maxerr);
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));return !errors;
}
int main(){bool ok=true;for(unsigned s:{1u,456u,2026u})ok=run_case(s)&&ok;return ok?EXIT_SUCCESS:EXIT_FAILURE;}
