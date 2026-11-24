// V6f：固定 W4A16、无 act-order 的 Marlin 兼容权重重排练习。
// 依据 third_party/vllm_marlin 的 repack 布局自行实现；不是移植其 CUDA 算法。
#include "v5_test_common.cuh"
#include <cstring>
__global__ void v6f_repack(const uint32_t* p,uint32_t* out,int K,int N) {
    // 一个 block 处理一个 16×64 tile；128 线程即 4 warp。
    // 这里真实 warp/lane 与参考 repack_tile 的 warp_id/th_id 对应。
    const int ktile=blockIdx.y,ntile=blockIdx.x;
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    // TODO 1：由 lane/warp 推导这个线程对应的 tile 内 K/N 坐标。
    // TODO 2：按文档给出的 8 个位置顺序，从 P[K/8,N] 取 nibble，组装一个 word。
    // TODO 3：按 tile 的物理起点和 lane*4+warp，将 word 写到 out。
    // 输入形状已在测试中限定；不能先构造完整 Q，也不能读取越界。
    uint32_t r = 0;
    // pl[lane%4*2][warp*16+lane/4]
    // pl[lane%4*2+8][warp*16+lane/4]
    // pl[lane%4*2][warp*16+lane/4+8]
    // pl[lane%4*2+8][warp*16+lane/4+8]
    // pl[lane%4*2+1][warp*16+lane/4]
    // pl[lane%4*2+9][warp*16+lane/4]
    // pl[lane%4*2+1][warp*16+lane/4+8]
    // pl[lane%4*2+9][warp*16+lane/4+8]
    int x[8] = {lane%4*2, lane%4*2+8, lane%4*2, lane%4*2+8, lane%4*2+1, lane%4*2+9, lane%4*2+1, lane%4*2+9};
    int y[8] = {warp*16+lane/4, warp*16+lane/4, warp*16+lane/4+8, warp*16+lane/4+8, warp*16+lane/4, warp*16+lane/4, warp*16+lane/4+8, warp*16+lane/4+8};
    const uint32_t* pl = p + ktile * 2 * N + ntile * 64;
    for (int i = 0; i < 8; i++) {
        r |= ((pl[x[i]/8*N+y[i]] >> (x[i]%8*4)) & 0xf) << (i*4);
    }
    int ntiles=gridDim.x;
    out[(ktile*ntiles+ntile)*128+lane*4+warp] = r;

    (void)p;(void)out;(void)K;(void)N;(void)ktile;(void)ntile;(void)warp;(void)lane;
}
// CPU 参考走相反方向：对每个逻辑 Q 直接定位目标 word/nibble。
// 这与 GPU 中按 word 收集 8 个 Q 的实现互相独立。
static std::vector<uint32_t> cpu_oracle(const V5Inputs& h) {
    int K=h.K,N=h.N,nt=N/64;
    std::vector<uint32_t> out(K*N/8,0);
    for(int k=0;k<K;++k) for(int n=0;n<N;++n) {
        int kl=k%16,nl=n%64,warp=nl/16;
        int lane=(nl%8)*4+(kl%8)/2;
        int nibble=(kl%2)*4+((nl%16)/8)*2+kl/8;
        int word=((k/16)*nt+n/64)*128+lane*4+warp;
        out[word]|=uint32_t(h.q[k*N+n])<<(4*nibble);
    }
    return out;
}
static bool test_case(int K,int N,const char* dump) {
    V5Inputs h(1,N,K,true); V5Device d(h);
    auto expected=cpu_oracle(h);std::vector<uint32_t> got(expected.size(),0xdeadbeef);
    uint32_t* dout=nullptr;CUDA_CHECK(cudaMalloc(&dout,got.size()*sizeof(*dout)));
    CUDA_CHECK(cudaMemcpy(dout,got.data(),got.size()*sizeof(*dout),cudaMemcpyHostToDevice));
    dim3 grid(N/64,K/16);v6f_repack<<<grid,128>>>(d.packed,dout,K,N);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(),dout,got.size()*sizeof(*dout),cudaMemcpyDeviceToHost));
    int errors=0;for(size_t i=0;i<got.size();++i) if(got[i]!=expected[i]) {
        if(errors<3)std::fprintf(stderr,"word[%zu]=0x%08x expected=0x%08x\n",i,got[i],expected[i]);++errors;
    }
    if(dump && !errors) {FILE* fp=std::fopen(dump,"wb");if(!fp){std::perror("dump");std::exit(EXIT_FAILURE);}
        if(std::fwrite(got.data(),sizeof(uint32_t),got.size(),fp)!=got.size()){std::fclose(fp);std::exit(EXIT_FAILURE);}std::fclose(fp);}
    std::printf("K=%d N=%d word错误=%d\n",K,N,errors);CUDA_CHECK(cudaFree(dout));return errors==0;
}
int main(int argc,char** argv) {
    if(argc!=1 && !(argc==3 && std::strcmp(argv[1],"--dump")==0)){std::fprintf(stderr,"用法：%s [--dump FILE]\n",argv[0]);return EXIT_FAILURE;}
    bool ok=test_case(256,128,argc==3?argv[2]:nullptr);
    ok=test_case(128,64,nullptr)&&ok;ok=test_case(384,192,nullptr)&&ok;
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
