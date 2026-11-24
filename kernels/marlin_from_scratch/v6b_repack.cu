// V6b：原布局 P[K/8,N] -> 自定义布局 R[K/16,ceil(N/8),16]。
// 复制为 v6b_repack.cu，只填写 kernel TODO。两者都按 uint32 数值低位优先。
#include "v5_test_common.cuh"
__global__ void repack_k_to_n(const uint32_t* p,uint32_t* out,int K,int N) {
    const int ntiles=(N+7)/8;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=K*ntiles) return;
    // TODO 1：一个线程负责输出 R[ktile,ntile,kin]，从 i 推导三个坐标；k=ktile*16+kin。
    // TODO 2：从 P 的对应 word 逐一提取 8 个逻辑 Q[k,ntile*8+j]；
    //         将它们放入输出 word 的第 j 个 nibble（低位优先）。
    // TODO 3：当列 >= N 时，该 nibble 置 0，不能读取越界 P。
    // TODO 4：将输出 word 写 out[i]。
    int ktile = i / (ntiles * 16), ntile = (i % (ntiles * 16)) / 16, kin = i % (ntiles * 16) % 16;
    uint32_t out_word = 0;
    for (int i = 0; i < 8; i++) {
        if (ntile*8+i >= N) continue;
        out_word |= ((p[(ktile*16+kin)/8*N+ntile*8+i] >> (((ktile*16+kin)%8)*4)) & 0xf) << (i * 4);
    }
    out[ktile*(ntiles*16)+ntile*16+kin] = out_word;
    (void)p;(void)out;
}
static bool test_repack(int K,int N) {
    V5Inputs h(1,N,K,true);
    V5Device d(h);
    const int nt=(N+7)/8, count=K*nt;
    std::vector<uint32_t> got(count,0xdeadbeefu);
    uint32_t* dr=nullptr; CUDA_CHECK(cudaMalloc(&dr,count*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dr,got.data(),count*sizeof(uint32_t),cudaMemcpyHostToDevice));
    repack_k_to_n<<<(count+255)/256,256>>>(d.packed,dr,K,N);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(),dr,count*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    int word_errors=0,nibble_errors=0,padding_errors=0;
    // 固定数据在 k=0 的前 8 列为 0,3,6,9,12,15,2,5。
    if(N>=8 && got[0]!=0x52fc9630u) {
        std::fprintf(stderr,"固定手算锚点 R[0,0,0] 错误：得到 0x%08x，期望 0x52fc9630\n",got[0]);
        ++word_errors;
    }
    for(int k=0;k<K;++k) for(int tile=0;tile<nt;++tile) {
        uint32_t expected=0;
        // CPU expected 直接从原始逻辑 Q 按列顺序打包，不调用 GPU repack。
        for(int j=0;j<8;++j) {
            int n=tile*8+j;
            if(n<N) expected |= uint32_t(h.q[k*N+n])<<(4*j);
        }
        uint32_t word=got[(k/16)*nt*16+tile*16+(k%16)];
        if(word!=expected) {
            if(word_errors<3) std::fprintf(stderr,"R[%d,%d]=0x%08x expected=0x%08x\n",k,tile,word,expected);
            ++word_errors;
        }
        for(int j=0;j<8;++j) {
            int n=tile*8+j, actual=int((word>>(4*j))&15u);
            if(n<N) { if(actual!=h.q[k*N+n]) ++nibble_errors; }
            else if(actual!=0) ++padding_errors;
        }
    }
    std::printf("K=%d N=%d word错误=%d 逻辑Q错误=%d padding错误=%d\n",K,N,word_errors,nibble_errors,padding_errors);
    CUDA_CHECK(cudaFree(dr));return word_errors==0 && nibble_errors==0 && padding_errors==0;
}
#ifndef V6B_NO_MAIN
int main() {
    bool ok=true;
    for(int N:{1,7,8,9,17,128}) ok=test_repack(256,N)&&ok;
    ok=test_repack(384,19)&&ok;
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
#endif
