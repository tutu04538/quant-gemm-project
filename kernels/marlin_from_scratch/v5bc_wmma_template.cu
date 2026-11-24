// V5b/c：一个 warp 完成一个 16×16 输出 tile，循环解码当前 W tile 并用 WMMA。
#include "v5_test_common.cuh"
#include <mma.h>
namespace wmma=nvcuda::wmma;
__global__ void v5_w4a16(const __nv_bfloat16* a,const uint32_t* packed,
                          const uint8_t* z,const __nv_bfloat16* s,float* c,int M,int N,int K) {
    __shared__ __align__(32) __nv_bfloat16 as[16][16];
    __shared__ __align__(32) __nv_bfloat16 bs[16][16];
    __shared__ __align__(32) float cs[16][16];
    const int lane=threadIdx.x, row0=blockIdx.y*16, col0=blockIdx.x*16;
    // 尚未填好时全 warp 一起退出，避免读取未初始化的 shared。
    bool ready=true; // TODO 5：其余 TODO 全部完成后，将它改为 true。
    if(!ready) return;
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bf;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.0f);
    for(int base=0;base<K;base+=16) {
        // TODO 1：lane 以 32 为步长合作填写 256 个 as 元素：
        // as[r][t] <- A[row0+r,base+t]，越界补 BF16 0。
        // TODO 2：同样合作填写 bs[t][n]：从 packed 解码 W[base+t,col0+n]，
        // 在写 bs 前转 BF16；K/N 越界补 0。每次查对应 group 的 Z/S。
        // TODO 3：先 __syncwarp() 确保 tile 写完；再用 wmma::load_matrix_sync
        // 加载 as 和 bs（两者行步长都是 16 个 BF16），调用 wmma::mma_sync。
        // TODO 4：在下一轮覆盖 shared 之前执行 __syncwarp()。
        for (int i = lane; i < 256; i += 32) {

            int r = i / 16, t = i % 16;

            if (row0 + r >= M || base + t >= K) as[r][t] = 0;
            else as[r][t] = a[(row0 + r) * K + base + t];

            // packed[(base+r)/8][col0+t]
            if (base + r >= K || col0 + t >= N) bs[r][t] = 0;
            else bs[r][t] = __float2bfloat16(((int)((packed[(base+r)/8*N+col0+t] >> ((base+r)%8*4)) & 0xf) - (int)(z[(base+r)/128*N+col0+t])) * __bfloat162float(s[(base+r)/128*N+col0+t]));
        }
        __syncwarp();
        wmma::load_matrix_sync(af, &as[0][0], 16);
        wmma::load_matrix_sync(bf, &bs[0][0], 16);
        wmma::mma_sync(acc, af, bf, acc);
        __syncwarp();
    }
    // store_matrix_sync 要求完整 16×16 输出空间，所以先写 shared cs，
    // 再由各 lane 按实际 M/N 把有效元素写回 global。
    wmma::store_matrix_sync(&cs[0][0],acc,16,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<256;i+=32) {
        int r=i/16,n=i%16;
        if(row0+r<M && col0+n<N) c[(row0+r)*N+col0+n]=cs[r][n];
    }
}
static bool test_case(int M,int N,int K,bool fractional) {
    V5Inputs h(M,N,K,fractional); V5Device d(h);
    std::vector<float> out(M*N,std::numeric_limits<float>::quiet_NaN());
    float* dc=nullptr; CUDA_CHECK(cudaMalloc(&dc,out.size()*sizeof(*dc)));
    CUDA_CHECK(cudaMemcpy(dc,out.data(),out.size()*sizeof(*dc),cudaMemcpyHostToDevice));
    dim3 block(32),grid((N+15)/16,(M+15)/16);
    v5_w4a16<<<grid,block>>>(d.a,d.packed,d.z,d.s,dc,M,N,K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*sizeof(*dc),cudaMemcpyDeviceToHost));
    const float atol=0.002f,rtol=0.0001f;
    int errors=0; float max_abs=0,max_math_diff=0; double mean_abs=0;
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        float expected=0,math=0;
        for(int k=0;k<K;++k) {
            float av=__bfloat162float(h.a[m*K+k]);
            expected=std::fma(av,h.weight(k,n),expected);
            math=std::fma(av,h.weight_math(k,n),math);
        }
        max_math_diff=std::max(max_math_diff,std::fabs(expected-math));
        float got=out[m*N+n],err=std::fabs(got-expected);
        if(!std::isfinite(got)||err>atol+rtol*std::fabs(expected)) {
            if(errors<3) std::fprintf(stderr,"C[%d,%d]: %g != %g\n",m,n,got,expected); ++errors;
        }
        if(std::isfinite(got)) { max_abs=std::max(max_abs,err); mean_abs+=err; }
    }
    std::printf("M=%d N=%d K=%d fractional=%d：超差=%d 最大误差=%g 平均误差=%g BF16权重舍入与数学参考最大差=%g\n",M,N,K,fractional,errors,max_abs,mean_abs/(M*N),max_math_diff);
    CUDA_CHECK(cudaFree(dc)); return errors==0;
}
int main() {
    int dev;cudaDeviceProp prop{};CUDA_CHECK(cudaGetDevice(&dev));CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    if(prop.major<8) { std::fprintf(stderr,"需要支持 BF16 WMMA 的 SM80+\n"); return EXIT_FAILURE; }
    std::printf("GPU=%s SM=%d.%d；atol=0.002 rtol=0.0001\n",prop.name,prop.major,prop.minor);
    bool ok=true;
    for(bool f:{false,true}) {
        ok=test_case(16,16,128,f)&&ok;
        ok=test_case(16,128,256,f)&&ok;
        ok=test_case(1,1,128,f)&&ok;
        ok=test_case(17,19,256,f)&&ok;
    }
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
