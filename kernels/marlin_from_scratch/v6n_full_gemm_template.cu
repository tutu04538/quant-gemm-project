// V6n：由用户完成的 V6m 改编，固定 M=16,N=128,K=256，W4A16 group128。
// Q/Z/S 参考 Marlin 重排布局；非参考流水线、调度或归约。来源及许可见 third_party/vllm_marlin/README.md。
#include "v5_test_common.cuh"
#include <random>
#include <cstring>
#undef CUDA_CHECK
#define main v2d_old_test_main
#include "v2d_ldmatrix_mma.cu"
#undef main
__device__ uint32_t v6n_lop3(uint32_t a,uint32_t b,uint32_t c){uint32_t o;asm volatile("lop3.b32 %0,%1,%2,%3,0xea;":"=r"(o):"r"(a),"r"(b),"r"(c));return o;}
__global__ void v6n_compute(const __nv_bfloat16* a,const uint32_t* q,const uint32_t* z,
                            const __nv_bfloat16* s,const __nv_bfloat16* bias,__nv_bfloat16* out,float* diagnostic) {
    __shared__ __align__(16) __nv_bfloat16 as[4][16][16];
    __shared__ float part[4][16][64];
    int nr=blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x%32;
    float acc[4][2][4] = {};
    uint32_t af[4] = {};
    uint32_t bf[4] = {};
    uint32_t bfrag[16];
    uint32_t MASK = 0x000f000f, EX = 0x43004300;
    for (int kr = 0; kr < 2; ++kr) for (int step = 0; step < 2; step++) {
        const __nv_bfloat16* a_start_addr = a + kr*128 + warp*32 + step*16;
        for (int i = lane; i < 256; i += 32) {
            as[warp][i/16][i%16] = a_start_addr[i/16*256+i%16];
        }
        __syncwarp();
        __nv_bfloat16 *a_local_addr = &as[warp][0][0] + lane%16*16 + (lane<16?0:8);
        load_a(__cvta_generic_to_shared(a_local_addr), af);
        uint32_t q_z = z[kr*16+nr*8+lane/4];
        __nv_bfloat162 qweight[4], qzero[2], qscale[2];
        for (int j = 0; j < 4; j++) {
            uint32_t q_weight = q[kr*(256*8)+warp*256*2+step*256+nr*128+lane*4+j];
            __nv_bfloat16 s1 = s[kr*128+nr*64+lane/4*8+j*2];
            __nv_bfloat16 s2 = s[kr*128+nr*64+lane/4*8+j*2+1];
            qscale[0] = __halves2bfloat162(s1, s1);
            qscale[1] = __halves2bfloat162(s2, s2);
            uint32_t q_zj = v6n_lop3(q_z >> (4 * j), MASK, EX);
            qzero[0] = __halves2bfloat162(__ushort_as_bfloat16(q_zj&0xffff), __ushort_as_bfloat16(q_zj&0xffff));
            qzero[1] = __halves2bfloat162(__ushort_as_bfloat16((q_zj>>16)&0xffff), __ushort_as_bfloat16((q_zj>>16)&0xffff));
            for (int p = 0; p < 4; p++) {
                uint32_t t = v6n_lop3(q_weight >> (4 * p), MASK, EX);
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
        part[warp][g][j*16+t*2] = acc[j][0][0];
        part[warp][g][j*16+t*2+1] = acc[j][0][1];
        part[warp][g+8][j*16+t*2] = acc[j][0][2];
        part[warp][g+8][j*16+t*2+1] = acc[j][0][3];
        part[warp][g][j*16+8+t*2] = acc[j][1][0];
        part[warp][g][j*16+8+t*2+1] = acc[j][1][1];
        part[warp][g+8][j*16+8+t*2] = acc[j][1][2];
        part[warp][g+8][j*16+8+t*2+1] = acc[j][1][3];
    }

    __syncthreads();
    if (warp == 0) for (int i=lane;i<16*64;i+=32) {
        int row=i/64, col=i%64, n=nr*64+col;
        float sum=0.f;
        for (int w=0;w<4;++w) sum+=part[w][row][col];
        diagnostic[row*128+n]=sum;
        out[row*128+n]=__hadd(__float2bfloat16_rn(sum),bias[n]);
    }
}

static bool run_case(unsigned seed, int mode) {
    constexpr int M=16,N=128,K=256;
    std::mt19937 rng(seed);
    std::vector<__nv_bfloat16> a(M*K),s(2*N),rs(2*N),bias(N),out(M*N);
    std::vector<uint8_t> q(K*N),z(2*N);
    const float scales[]={0.1f,0.125f,0.25f,0.7f,1.f};
    for(auto& x:a)x=__float2bfloat16_rn(float(int(rng()%17)-8)*0.125f);
    for(auto& x:q)x=uint8_t(rng()%16);
    for(auto& x:z)x=uint8_t(rng()%16);
    for(auto& x:s)x=__float2bfloat16_rn(scales[rng()%5]);
    for(auto& x:bias)x=__float2bfloat16_rn(float(int(rng()%17)-8)*0.125f);
    // 极值与 group、N tile 边界：两个 group/tile 使用不同值。
    if(mode==1) for(int g=0;g<2;++g)for(int n=0;n<N;++n) {
        z[g*N+n]=uint8_t((g+n/64)%2?15:0);
        for(int k=g*128;k<(g+1)*128;++k) q[k*N+n]=uint8_t((k+n)%2?15:0);
    }
    if(mode==2) for(auto& x:bias)x=__float2bfloat16_rn(0.f);
    std::vector<uint32_t> rq(K*N/8,0),rz(2*N/8,0);
    for(int k=0;k<K;++k)for(int n=0;n<N;++n){
        int kl=k%16,nl=n%64,l=(nl%8)*4+(kl%8)/2,w=nl/16;
        int pos=(kl%2)*4+((nl%16)/8)*2+kl/8;
        rq[((k/16)*2+n/64)*128+l*4+w]|=uint32_t(q[k*N+n])<<(4*pos);
    }
    for(int g=0;g<2;++g)for(int n=0;n<N;++n){
        int chunk=n/64,x=n%8,y=(n%64)/8;
        rs[g*N+chunk*64+x*8+y]=s[g*N+n];
        int nib=y%2?4+y/2:y/2;
        rz[g*(N/8)+chunk*8+x]|=uint32_t(z[g*N+n])<<(4*nib);
    }
    __nv_bfloat16 *da=nullptr,*ds=nullptr,*db=nullptr,*dc=nullptr;
    uint32_t *dq=nullptr,*dz=nullptr;float* df=nullptr;
    std::vector<float> diagnostic(M*N);
    CUDA_CHECK(cudaMalloc(&da,a.size()*sizeof(a[0]))); CUDA_CHECK(cudaMalloc(&ds,rs.size()*sizeof(rs[0])));
    CUDA_CHECK(cudaMalloc(&db,bias.size()*sizeof(bias[0]))); CUDA_CHECK(cudaMalloc(&dc,out.size()*sizeof(out[0])));
    CUDA_CHECK(cudaMalloc(&dq,rq.size()*sizeof(rq[0]))); CUDA_CHECK(cudaMalloc(&dz,rz.size()*sizeof(rz[0])));
    CUDA_CHECK(cudaMalloc(&df,diagnostic.size()*sizeof(diagnostic[0])));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(a[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds,rs.data(),rs.size()*sizeof(rs[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,bias.data(),bias.size()*sizeof(bias[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq,rq.data(),rq.size()*sizeof(rq[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz,rz.data(),rz.size()*sizeof(rz[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dc,0xff,out.size()*sizeof(out[0]))); CUDA_CHECK(cudaMemset(df,0xff,diagnostic.size()*sizeof(diagnostic[0])));
    v6n_compute<<<2,128>>>(da,dq,dz,ds,db,dc,df);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*sizeof(out[0]),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(diagnostic.data(),df,diagnostic.size()*sizeof(diagnostic[0]),cudaMemcpyDeviceToHost));
    // 独立从逻辑坐标计算，每 warp 的 K 顺序与 kernel 一致，再按 warp 编号归约。
    int ferr=0,berr=0,bitdiff=0; float maxerr=0;
    for(int m=0;m<M;++m)for(int n=0;n<N;++n){
        float sums[4]={};
        for(int w=0;w<4;++w)for(int group=0;group<2;++group)
            for(int k=group*128+w*32;k<group*128+w*32+32;++k){
                float weight=__bfloat162float(__float2bfloat16_rn(float(int(q[k*N+n])-int(z[group*N+n]))*__bfloat162float(s[group*N+n])));
                sums[w]=std::fma(__bfloat162float(a[m*K+k]),weight,sums[w]);
            }
        float ref=0;for(float x:sums)ref+=x;
        float actual=diagnostic[m*N+n],err=std::fabs(actual-ref);
        if(!std::isfinite(actual)||err>0.002f+0.0001f*std::fabs(ref)){if(ferr<3)std::fprintf(stderr,"FP32 [%d,%d] got=%g expected=%g\n",m,n,actual,ref);++ferr;}
        if(std::isfinite(actual))maxerr=std::max(maxerr,err);
        auto rounded=__float2bfloat16_rn(ref);
        auto expected=__float2bfloat16_rn(__bfloat162float(rounded)+__bfloat162float(bias[n]));
        float expected_f=__bfloat162float(expected),actual_f=__bfloat162float(out[m*N+n]);
        if(!std::isfinite(actual_f)||std::fabs(actual_f-expected_f)>0.002f+0.01f*std::fabs(expected_f)){if(berr<3)std::fprintf(stderr,"BF16 [%d,%d] got=%g expected=%g\n",m,n,actual_f,expected_f);++berr;}
        uint16_t eb,ob;std::memcpy(&eb,&expected,2);std::memcpy(&ob,&out[m*N+n],2);bitdiff+=eb!=ob;
    }
    std::printf("seed=%u mode=%d FP32错误=%d max_abs=%g BF16错误=%d BF16逐位不同=%d\n",seed,mode,ferr,maxerr,berr,bitdiff);
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dz)); CUDA_CHECK(cudaFree(df));
    return ferr==0&&berr==0;
}
int main(){bool ok=true;for(unsigned seed:{1u,456u,2026u})for(int mode=0;mode<3;++mode)ok=run_case(seed,mode)&&ok;return ok?EXIT_SUCCESS:EXIT_FAILURE;}
