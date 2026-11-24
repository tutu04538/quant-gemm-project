// V6h 参考实例：repacked Q -> lane 的两个 BF16x2 编码片段（带 128 偏置）。
// 参考 NVIDIA/vLLM Marlin 的 dequant.h（Apache-2.0，来源见 third_party/vllm_marlin/README.md）。
#include "v5_test_common.cuh"
#include <random>

// 返回 (a & b) | c；请按课程填写 lop3 内联 PTX。
__device__ uint32_t lop3_extract(uint32_t a,uint32_t b,uint32_t c) {
    // TODO 1：使用 lop3.b32、真值表立即数 0xea，返回结果。
    uint32_t out;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;"
        : "=r"(out)
        : "r"(a), "r"(b), "r"(c));
    return out;
    (void)a;(void)b;(void)c;return 0;
}
__global__ void trace_encoded_q(const uint32_t* repacked,uint32_t* encoded) {
    // 仅复现固定例 4 个有效 block 的语义，直接读取已重排 global Q；不含参考 shared/pipeline。
    int block=blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x%32;
    int nr=block/2,kr=block%2;
    for(int step=0;step<2;++step)for(int j=0;j<4;++j) {
        int physical_row=kr*8+warp*2+step,physical_col=nr*128+lane*4+j;
        uint32_t word=repacked[physical_row*256+physical_col];
        // 该索引只是测试输出，不代表参考 kernel 的寄存器物理地址。
        int dst=((((block*128+threadIdx.x)*2+step)*4+j)*4);
        // TODO 2：按照 dequant.h 用 MASK=0x000f000f、EX=0x43004300。
        // 每次 lop3_extract 生成一个 BF16x2；总共生成 4 个，按顺序写入：
        // 每个 nibble 为 4 bit：想一想要将 word 分别右移多少位。
        uint32_t MASK = 0x000f000f, EX=0x43004300;
        encoded[dst] = lop3_extract(word, MASK, EX);
        encoded[dst+1] = lop3_extract(word>>4, MASK, EX);
        encoded[dst+2] = lop3_extract(word>>8, MASK, EX);
        encoded[dst+3] = lop3_extract(word>>12, MASK, EX);
        (void)word;(void)dst;
    }
}
static bool run_case(unsigned seed) {
    constexpr int K=256,N=128,words=K*N/8,count=4*128*2*4*4;
    std::mt19937 rng(seed);std::vector<uint8_t> q(K*N);for(auto& x:q)x=uint8_t(rng()%16);
    // 独立按逻辑 Q 反向构建参考 repack：每个逻辑元素写一个目标 nibble。
    std::vector<uint32_t> packed(words,0);
    for(int k=0;k<K;++k)for(int n=0;n<N;++n){int kl=k%16,nl=n%64;
        int lane=(nl%8)*4+(kl%8)/2,warp=nl/16;
        int pos=(kl%2)*4+((nl%16)/8)*2+kl/8;
        packed[((k/16)*2+n/64)*128+lane*4+warp] |= uint32_t(q[k*N+n])<<(4*pos);
    }
    std::vector<uint32_t> got(count,0xffffffffu);uint32_t *dq=nullptr,*dout=nullptr;
    CUDA_CHECK(cudaMalloc(&dq,packed.size()*4));CUDA_CHECK(cudaMalloc(&dout,got.size()*4));
    CUDA_CHECK(cudaMemcpy(dq,packed.data(),packed.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dout,got.data(),got.size()*4,cudaMemcpyHostToDevice));
    trace_encoded_q<<<4,128>>>(dq,dout);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
    int errors=0;
    for(int block=0;block<4;++block)for(int warp=0;warp<4;++warp)for(int lane=0;lane<32;++lane)
      for(int step=0;step<2;++step)for(int j=0;j<4;++j) {
        int kr=block%2,nr=block/2,k0=kr*128+warp*32+step*16+2*(lane%4),n0=nr*64+j*16+lane/4;
        // 独立参考直接读取逻辑 Q，不解包 repacked Q。
        int ks[4]={k0,k0+8,k0,k0+8},ns[4]={n0,n0,n0+8,n0+8};
        int dst=((((block*128+warp*32+lane)*2+step)*4+j)*4);
        for(int x=0;x<4;++x){uint32_t lo=0x4300u|q[ks[x]*N+ns[x]],hi=0x4300u|q[(ks[x]+1)*N+ns[x]];
            uint32_t expected=lo|(hi<<16);if(got[dst+x]!=expected){if(errors<4)std::fprintf(stderr,"block=%d warp=%d lane=%d step=%d word=%d frag=%d: %08x != %08x\n",block,warp,lane,step,j,x,got[dst+x],expected);++errors;}}
    }
    std::printf("seed=%u：4 blocks × 4 warps × 32 lanes × 2 steps × 4 words × 4 pairs，错误=%d\n",seed,errors);
    CUDA_CHECK(cudaFree(dq));CUDA_CHECK(cudaFree(dout));return errors==0;
}
int main(){bool ok=true;for(unsigned seed:{1u,456u,2026u})ok=run_case(seed)&&ok;return ok?EXIT_SUCCESS:EXIT_FAILURE;}
