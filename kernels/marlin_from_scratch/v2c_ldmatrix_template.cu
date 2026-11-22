// V2c：单 warp ldmatrix.x4，从 shared 取四个 8×8 的原始 16-bit 矩阵。
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void load_four(const uint16_t* input, uint32_t* output) {
    // 每个矩阵 8 行，每行 8 个 16-bit 值；每行 16 字节且按 16 字节对齐。
    __shared__ __align__(16) uint16_t tile[4][8][8];
    const int lane = threadIdx.x;
    uint16_t* flat = &tile[0][0][0];
    for (int i = lane; i < 4*8*8; i += 32) flat[i] = input[i];
    __syncthreads();

    // TODO 1：让 lane 0..7、8..15、16..23、24..31 分别提供矩阵
    //         0、1、2、3 的各行地址。每个地址要指向该行的第一个 uint16_t。
    // TODO 2：使用 __cvta_generic_to_shared 把该 shared 指针转为 PTX 地址。
    // TODO 3：完成后将 ready 设为 true；它必须在全 warp 中取同一个值。
    const bool ready = false;
    uint32_t address = 0;
    if (!ready) return;  // 模板未填写时所有 lane 一起退出，不访问空 shared 地址。

    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(address));
    output[lane*4+0]=r0; output[lane*4+1]=r1;
    output[lane*4+2]=r2; output[lane*4+3]=r3;
}

int main() {
    std::vector<uint16_t> input(4*8*8);
    std::vector<uint32_t> output(32*4, 0xFFFFFFFFu);
    for (int mat=0;mat<4;++mat) for (int row=0;row<8;++row) for (int col=0;col<8;++col)
        input[mat*64+row*8+col]=uint16_t(1000*mat+100*row+col);
    uint16_t* di=nullptr; uint32_t* dout=nullptr;
    CUDA_CHECK(cudaMalloc(&di,input.size()*sizeof(input[0])));
    CUDA_CHECK(cudaMalloc(&dout,output.size()*sizeof(output[0])));
    CUDA_CHECK(cudaMemcpy(di,input.data(),input.size()*sizeof(input[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dout,output.data(),output.size()*sizeof(output[0]),cudaMemcpyHostToDevice));
    load_four<<<1,32>>>(di,dout);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(),dout,output.size()*sizeof(output[0]),cudaMemcpyDeviceToHost));
    int errors=0;
    for (int lane=0;lane<32;++lane) {
        int row=lane/4, col=(lane%4)*2;
        for (int mat=0;mat<4;++mat) {
            uint32_t expected=uint32_t(input[mat*64+row*8+col]) |
                              (uint32_t(input[mat*64+row*8+col+1])<<16);
            if (output[lane*4+mat]!=expected) {
                if (errors<3) std::fprintf(stderr,"lane=%d 矩阵=%d: 得到 0x%08x，期望 0x%08x\n",lane,mat,output[lane*4+mat],expected);
                ++errors;
            }
        }
    }
    CUDA_CHECK(cudaFree(di)); CUDA_CHECK(cudaFree(dout));
    std::printf("ldmatrix.x4：%s，错误=%d\n",errors?"失败":"通过",errors);
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
