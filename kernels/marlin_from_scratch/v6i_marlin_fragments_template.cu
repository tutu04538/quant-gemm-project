// V6i 固定实例：参考 repacked Q/Z/S → 每 lane 的 4 个 BF16x2 B 寄存器。
// 位构造和片段语义参考 third_party/vllm_marlin，许可/来源见该目录 README。
#include "v5_test_common.cuh"
#include <random>
#include <cstring>
__device__ uint32_t v6i_lop3(uint32_t a,uint32_t b,uint32_t c) {
    uint32_t o;asm volatile("lop3.b32 %0,%1,%2,%3,0xea;" : "=r"(o):"r"(a),"r"(b),"r"(c));return o;
}
__global__ void trace_fragment(const uint32_t* q,const uint32_t* zp,const __nv_bfloat16* scale,uint32_t* trace) {
    int block=blockIdx.x,tid=threadIdx.x,warp=tid/32,lane=tid%32;
    int nr=block/2,kr=block%2,group=lane/4;
    for(int step=0;step<2;++step)for(int j=0;j<4;++j) {
        // 注意：此实验为观察数据而直接从 global 读；实际 kernel 先经 shared。
        int qrow=kr*8+warp*2+step,qcol=nr*128+lane*4+j;
        uint32_t qword=q[qrow*256+qcol];
        uint32_t zword=zp[kr*16+nr*8+group];
        int spos=kr*128+nr*64+group*8+2*j;
        int dst=((((block*128+tid)*2+step)*4+j)*4);
        // TODO：参考固定分支：两次解码 qword/>>8 得到 4 个 BF16x2；
        // zero word 的第 j 与 j+4 个 nibble 对应两列；scale[spos],scale[spos+1] 对应两列。
        // 对每个 BF16x2 先减相应 128+zero，再乘 BF16 scale，存 trace[dst..dst+3] 原始位。
        // 可使用 v6i_lop3、__ushort_as_bfloat16、__halves2bfloat162、__hsub2、__hmul2。
        uint32_t MASK = 0x000f000f, EX=0x43004300;
        __nv_bfloat162 qweight[4], qzero[2], qscale[2];
        for (int i = 0; i < 4; i++) {
            uint32_t t = v6i_lop3(qword >> (i*4), MASK, EX);
            qweight[i] = __halves2bfloat162(__ushort_as_bfloat16(t&0xffff), __ushort_as_bfloat16((t>>16)&0xffff));
        }
        uint32_t t = v6i_lop3(zword >> (j*4), MASK, EX);
        qzero[0] = __halves2bfloat162(__ushort_as_bfloat16(t&0xffff), __ushort_as_bfloat16(t&0xffff));
        qzero[1] = __halves2bfloat162(__ushort_as_bfloat16((t>>16)&0xffff), __ushort_as_bfloat16((t>>16)&0xffff));
        qscale[0] = __halves2bfloat162(scale[spos], scale[spos]);
        qscale[1] = __halves2bfloat162(scale[spos+1], scale[spos+1]);
        for (int i = 0; i < 4; i++) {
            __nv_bfloat162_raw raw = __hmul2(__hsub2(qweight[i], qzero[i/2]), qscale[i/2]);
            trace[dst+i] = static_cast<uint32_t>(raw.x) | (static_cast<uint32_t>(raw.y) << 16);
        }
        (void)qword;(void)zword;(void)spos;(void)dst;(void)scale;(void)trace;
    }
}
static bool run_case(unsigned seed) {
    constexpr int K=256,N=128,groups=2,count=4*128*2*4*4;
    std::mt19937 rng(seed);std::vector<uint8_t> q(K*N),z(groups*N);std::vector<__nv_bfloat16> s(groups*N);
    for(auto& v:q)v=uint8_t(rng()%16);for(auto& v:z)v=uint8_t(rng()%16);
    const float values[]={0.1f,0.25f,0.375f,0.7f,1.f};for(auto& v:s)v=__float2bfloat16_rn(values[rng()%5]);
    std::vector<uint32_t> rp(K*N/8,0),rz(groups*N/8,0);std::vector<__nv_bfloat16> rs(groups*N);
    // CPU 分别从逻辑 Q、Z、S 推导物理位置，不调用 GPU 解码路径。
    for(int k=0;k<K;++k)for(int n=0;n<N;++n){int kl=k%16,nl=n%64,lane=(nl%8)*4+(kl%8)/2,warp=nl/16;
        int pos=(kl%2)*4+((nl%16)/8)*2+kl/8;
        rp[((k/16)*2+n/64)*128+lane*4+warp]|=uint32_t(q[k*N+n])<<(4*pos);}
    for(int g=0;g<groups;++g)for(int n=0;n<N;++n){int chunk=n/64,a=n%8,b=(n%64)/8;
        rs[g*N+chunk*64+a*8+b]=s[g*N+n];int nib=(b%2==0)?b/2:4+b/2;
        rz[g*(N/8)+chunk*8+a]|=uint32_t(z[g*N+n])<<(4*nib);}
    uint32_t *dq=nullptr,*dz=nullptr,*dt=nullptr;__nv_bfloat16* ds=nullptr;std::vector<uint32_t> got(count,0xffffffffu);
    CUDA_CHECK(cudaMalloc(&dq,rp.size()*4));CUDA_CHECK(cudaMalloc(&dz,rz.size()*4));CUDA_CHECK(cudaMalloc(&ds,rs.size()*2));CUDA_CHECK(cudaMalloc(&dt,got.size()*4));
    CUDA_CHECK(cudaMemcpy(dq,rp.data(),rp.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dz,rz.data(),rz.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds,rs.data(),rs.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dt,got.data(),got.size()*4,cudaMemcpyHostToDevice));
    trace_fragment<<<4,128>>>(dq,dz,ds,dt);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(got.data(),dt,got.size()*4,cudaMemcpyDeviceToHost));
    int errors=0;
    for(int block=0;block<4;++block)for(int warp=0;warp<4;++warp)for(int lane=0;lane<32;++lane)
      for(int step=0;step<2;++step)for(int j=0;j<4;++j){
        int kr=block%2,nr=block/2,k0=kr*128+warp*32+step*16+2*(lane%4),n0=nr*64+j*16+lane/4;
        int ks[4]={k0,k0+8,k0,k0+8},ns[4]={n0,n0,n0+8,n0+8};
        int dst=((((block*128+warp*32+lane)*2+step)*4+j)*4);
        for(int x=0;x<4;++x){uint16_t bits[2];for(int h=0;h<2;++h){int k=ks[x]+h,n=ns[x];
            float f=float(int(q[k*N+n])-int(z[kr*N+n]))*__bfloat162float(s[kr*N+n]);
            auto bf=__float2bfloat16_rn(f);std::memcpy(&bits[h],&bf,2);}
          uint32_t expected=uint32_t(bits[0])|(uint32_t(bits[1])<<16);
          if(got[dst+x]!=expected){if(errors<4)std::fprintf(stderr,"block=%d warp=%d lane=%d step=%d word=%d frag=%d: %08x != %08x\n",block,warp,lane,step,j,x,got[dst+x],expected);++errors;}}
      }
    std::printf("seed=%u：参考 Q/Z/S 到 4 blocks 全部片段错误=%d\n",seed,errors);
    CUDA_CHECK(cudaFree(dq));CUDA_CHECK(cudaFree(dz));CUDA_CHECK(cudaFree(ds));CUDA_CHECK(cudaFree(dt));return errors==0;
}
int main(){bool ok=true;for(unsigned seed:{1u,456u,2026u})ok=run_case(seed)&&ok;return ok?EXIT_SUCCESS:EXIT_FAILURE;}
