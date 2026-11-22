// V4b：实验 global 的 4×uint32 向量搬运与标量尾部；不是 GEMM kernel。
// 复制为 v4b_vector_load.cu，只填 kernel 中的 TODO。
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e_=(call); if (e_!=cudaSuccess) { \
    std::fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e_)); \
    std::exit(EXIT_FAILURE); } } while(0)

__global__ void copy_words(const uint32_t* src, uint32_t* dst, int rows, int N) {
    // 每个线程负责同一行的最多四个相邻 word，block=(32,1)。
    int row = blockIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (row >= rows || col >= N) return;
    // TODO 1：求 src/dst 里本组首地址，注意每行 stride 是 N 个 uint32。
    // TODO 2：如果本组有完整的 4 个 word，且 src 和 dst 首地址均 16 字节对齐，
    //         通过 const uint4* 读取一个向量，再通过 uint4* 写出。
    // TODO 3：否则逐 word 标量复制本组有效元素，禁止访问超出当前行的部分。
    const uint32_t* sp = &src[row*N+col];
    uint32_t* dp = &dst[row*N+col];

    if (reinterpret_cast<uintptr_t>(sp) % 0x10 || reinterpret_cast<uintptr_t>(dp) % 0x10 || N - col < 4) {
        for (int p = 0; p < min(4, N - col); p++) dp[p] = sp[p];
    } else {
        uint4 t = *reinterpret_cast<const uint4*>(sp);
        *reinterpret_cast<uint4*>(dp) = t;
    }
}

static bool run_case(int rows, int N, int src_offset, int dst_offset) {
    const int size = rows*N, guard=4;
    std::vector<uint32_t> src(src_offset+size+guard, 0xdeadbeefu);
    std::vector<uint32_t> dst(dst_offset+size+guard, 0xcafebabeu);
    for (int i=0;i<size;++i) src[src_offset+i] = 0x12340000u + uint32_t(i * 17 + N);
    uint32_t *dsrc=nullptr,*ddst=nullptr;
    CUDA_CHECK(cudaMalloc(&dsrc,src.size()*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ddst,dst.size()*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dsrc,src.data(),src.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ddst,dst.data(),dst.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    dim3 block(32),grid((N+127)/128,rows);
    copy_words<<<grid,block>>>(dsrc+src_offset,ddst+dst_offset,rows,N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dst.data(),ddst,dst.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    int errors=0;
    for (size_t i=0;i<dst.size();++i) {
        uint32_t expected = (i>=size_t(dst_offset) && i<size_t(dst_offset+size)) ? src[src_offset + i-dst_offset] : 0xcafebabeu;
        if (dst[i]!=expected) { if (errors<3) std::fprintf(stderr,"  dst[%zu]=0x%x，期望 0x%x\n",i,dst[i],expected); ++errors; }
    }
    std::printf("rows=%d N=%d src偏移=%d dst偏移=%d 错误=%d\n",rows,N,src_offset,dst_offset,errors);
    CUDA_CHECK(cudaFree(dsrc)); CUDA_CHECK(cudaFree(ddst)); return errors==0;
}
int main() {
    const int columns[] = {1,3,4,5,9,16,17,129};
    bool ok=true;
    for (int n:columns) for (int offset=0;offset<2;++offset) {
        ok=run_case(3,n,offset,0)&&ok;
        ok=run_case(3,n,0,offset)&&ok;
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
