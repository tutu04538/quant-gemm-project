// V6m：参考 Q/Z/S 布局 + 四 warp 解码/MMA，固定 N tile 0 的两个 K group。
// 诊断版本：直接 global 读取，输出每 warp 部分和；不含参考 cp.async、归约/epilogue。
#include "v5_test_common.cuh"
#include <cstring>
#include <random>
#undef CUDA_CHECK
#define main v2d_old_test_main
#include "v2d_ldmatrix_mma.cu"
#undef main
__device__ uint32_t v6m_lop3(uint32_t a,uint32_t b,uint32_t c){uint32_t o;asm volatile("lop3.b32 %0,%1,%2,%3,0xea;":"=r"(o):"r"(a),"r"(b),"r"(c));return o;}
__global__ void v6m_compute(const __nv_bfloat16* a,const uint32_t* q,const uint32_t* z,
                             const __nv_bfloat16* s,float* partial) {
    // 两个 block 分别处理 kr=0/1，各 4 warp；nr 固定为 0。
    __shared__ __align__(16) __nv_bfloat16 as[4][16][16];
    int kr=blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x%32;
    bool ready=true;if(!ready)return; // 两个 block 各自全部线程一致
    // TODO 1：每 lane 准备 acc[4 j][2 MMA][4 float]；warp 的两个 K16 step。
    // TODO 2：加载 A 并用 ldmatrix 得到 af，按 V6i 的参考物理 Q/Z/S 取四对 B。
    // TODO 3：对每个 j 的 bf0、bf1 各做 MMA，累加；每次复用 as 前保证 warp 同步。
    // TODO 4：按每 lane 的坐标把各 warp 的结果写入 partial[kr][warp][16][64]。
    // 不做跨 warp/block 归约；全部 partial 由 host 独立验证。
    float acc[4][2][4] = {};
    uint32_t af[4] = {};
    uint32_t bf[4] = {};
    uint32_t bfrag[16];
    uint32_t MASK = 0x000f000f, EX = 0x43004300;
    for (int step = 0; step < 2; step++) {
        const __nv_bfloat16* a_start_addr = a + kr*128 + warp*32 + step*16;
        for (int i = lane; i < 256; i += 32) {
            as[warp][i/16][i%16] = a_start_addr[i/16*256+i%16];
        }
        __syncwarp();
        __nv_bfloat16 *a_local_addr = &as[warp][0][0] + lane%16*16 + (lane<16?0:8);
        load_a(__cvta_generic_to_shared(a_local_addr), af);
        uint32_t q_z = z[kr*16+lane/4];
        __nv_bfloat162 qweight[4], qzero[2], qscale[2];
        for (int j = 0; j < 4; j++) {
            uint32_t q_weight = q[kr*(256*8)+warp*256*2+step*256+lane*4+j];
            __nv_bfloat16 s1 = s[kr*128+lane/4*8+j*2];
            __nv_bfloat16 s2 = s[kr*128+lane/4*8+j*2+1];
            qscale[0] = __halves2bfloat162(s1, s1);
            qscale[1] = __halves2bfloat162(s2, s2);
            uint32_t q_zj = v6m_lop3(q_z >> (4 * j), MASK, EX);
            qzero[0] = __halves2bfloat162(__ushort_as_bfloat16(q_zj&0xffff), __ushort_as_bfloat16(q_zj&0xffff));
            qzero[1] = __halves2bfloat162(__ushort_as_bfloat16((q_zj>>16)&0xffff), __ushort_as_bfloat16((q_zj>>16)&0xffff));
            for (int p = 0; p < 4; p++) {
                uint32_t t = v6m_lop3(q_weight >> (4 * p), MASK, EX);
                qweight[p] = __halves2bfloat162(__ushort_as_bfloat16(t&0xffff), __ushort_as_bfloat16((t>>16)&0xffff));
                __nv_bfloat162_raw raw = __hmul2(__hsub2(qweight[p], qzero[p/2]), qscale[p/2]); 
                bfrag[j*4+p] = static_cast<uint32_t>(raw.x) | (static_cast<uint32_t>(raw.y) << 16);
            }
        }

        for (int j = 0; j < 4; j++) {
            for (int t = 0; t < 4; t++) {
                bf[t] = bfrag[j*4+t];
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
        partial[kr*4*16*64+warp*16*64+g*64+j*16+t*2] = acc[j][0][0];
        partial[kr*4*16*64+warp*16*64+g*64+j*16+t*2+1] = acc[j][0][1];
        partial[kr*4*16*64+warp*16*64+(g+8)*64+j*16+t*2] = acc[j][0][2];
        partial[kr*4*16*64+warp*16*64+(g+8)*64+j*16+t*2+1] = acc[j][0][3];
        partial[kr*4*16*64+warp*16*64+g*64+j*16+8+t*2] = acc[j][1][0];
        partial[kr*4*16*64+warp*16*64+g*64+j*16+8+t*2+1] = acc[j][1][1];
        partial[kr*4*16*64+warp*16*64+(g+8)*64+j*16+8+t*2] = acc[j][1][2];
        partial[kr*4*16*64+warp*16*64+(g+8)*64+j*16+8+t*2+1] = acc[j][1][3];
    }

    (void)a;(void)q;(void)z;(void)s;(void)partial;(void)as;(void)kr;(void)warp;(void)lane;
}
static bool run_case(unsigned seed){
    constexpr int M=16,N=128,K=256;std::mt19937 rng(seed);
    std::vector<__nv_bfloat16> a(M*K),s(2*N),rs(2*N);std::vector<uint8_t> q(K*N),z(2*N);
    for(auto& x:a)x=__float2bfloat16_rn(float(int(rng()%9)-4)*0.125f);
    for(auto& x:q)x=uint8_t(rng()%16);for(auto& x:z)x=uint8_t(rng()%16);
    const float choices[]={0.125f,0.25f,0.5f};for(auto& x:s)x=__float2bfloat16_rn(choices[rng()%3]);
    std::vector<uint32_t> rq(K*N/8,0),rz(2*N/8,0);
    for(int k=0;k<K;++k)for(int n=0;n<N;++n){int kl=k%16,nl=n%64,l=(nl%8)*4+(kl%8)/2,w=nl/16,pos=(kl%2)*4+((nl%16)/8)*2+kl/8;
        rq[((k/16)*2+n/64)*128+l*4+w]|=uint32_t(q[k*N+n])<<(4*pos);}
    for(int g=0;g<2;++g)for(int n=0;n<N;++n){int chunk=n/64,x=n%8,y=(n%64)/8;
        rs[g*N+chunk*64+x*8+y]=s[g*N+n];int nib=y%2?4+y/2:y/2;rz[g*(N/8)+chunk*8+x]|=uint32_t(z[g*N+n])<<(4*nib);}
    __nv_bfloat16 *da=nullptr,*ds=nullptr;uint32_t *dq=nullptr,*dz=nullptr;float* dp=nullptr;
    std::vector<float> got(2*4*16*64);
    CUDA_CHECK(cudaMalloc(&da,a.size()*2));CUDA_CHECK(cudaMalloc(&ds,rs.size()*2));CUDA_CHECK(cudaMalloc(&dq,rq.size()*4));CUDA_CHECK(cudaMalloc(&dz,rz.size()*4));CUDA_CHECK(cudaMalloc(&dp,got.size()*4));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(ds,rs.data(),rs.size()*2,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq,rq.data(),rq.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dz,rz.data(),rz.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(dp,0xff,got.size()*4));
    v6m_compute<<<2,128>>>(da,dq,dz,ds,dp);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(got.data(),dp,got.size()*4,cudaMemcpyDeviceToHost));
    int errors=0;float maxerr=0;
    for(int kr=0;kr<2;++kr)for(int w=0;w<4;++w)for(int m=0;m<16;++m)for(int n=0;n<64;++n){float ref=0;
        for(int k=kr*128+w*32;k<kr*128+w*32+32;++k){float weight=__bfloat162float(__float2bfloat16_rn(float(int(q[k*N+n])-int(z[kr*N+n]))*__bfloat162float(s[kr*N+n])));
            ref=std::fma(__bfloat162float(a[m*K+k]),weight,ref);}
        float v=got[((kr*4+w)*16+m)*64+n],err=std::fabs(v-ref);
        if(!std::isfinite(v)||err>0.002f+0.0001f*std::fabs(ref)){if(errors<3)std::fprintf(stderr,"kr=%d warp=%d C[%d,%d]=%g expected=%g\n",kr,w,m,n,v,ref);++errors;}
        if(std::isfinite(v))maxerr=std::max(maxerr,err);
    }
    std::printf("seed=%u 8 warp部分和错误=%d max=%g\n",seed,errors,maxerr);
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(ds));CUDA_CHECK(cudaFree(dq));CUDA_CHECK(cudaFree(dz));CUDA_CHECK(cudaFree(dp));return errors==0;
}
int main(){bool ok=true;for(unsigned seed:{1u,456u,2026u})ok=run_case(seed)&&ok;return ok?EXIT_SUCCESS:EXIT_FAILURE;}
