// V6g：固定 group=128、K>128、N%64=0；scale BF16、逻辑 Z uint8。
#include "v5_test_common.cuh"
#include <cstring>
__global__ void permute_scale(const __nv_bfloat16* in,__nv_bfloat16* out,int groups,int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=groups*N)return;
    // TODO：每个线程负责一个物理输出 scale；按 64 元素块转置 8×8，见文档。
    __nv_bfloat16 scale = in[i];
    int local = i % 64;
    int local_x = local % 8, local_y = local / 8;
    out[(i-local)+local_x*8+local_y] = scale;
    (void)in;(void)out;
}
__global__ void permute_pack_zero(const uint8_t* in,uint32_t* out,int groups,int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=groups*(N/8))return;
    // TODO：每线程负责一个 word；先按 scale 的 8×8 置换，再按 nibble 顺序
    // [0,2,4,6,1,3,5,7] 打包 8 个 zero；见 V6g 文档。
    uint32_t z[8];
    int local = (i * 8) % 64;
    for (int j = 0; j < 8; j++) {
        int local_x = j, local_y = local / 8;
        z[j] = in[(i*8-local)+local_x*8+local_y];
    }
    out[i] = z[0] | (z[2] << 4) | (z[4] << 8) | (z[6] << 12) | (z[1] << 16) | (z[3] << 20) | (z[5] << 24) | (z[7] << 28);
    (void)in;(void)out;
}
static bool test_params(int K,int N) {
    V5Inputs h(1,N,K,true);int groups=K/128;
    // BF16 精确标签，覆盖每个 64 元素块的不同位置，便于发现重排错误。
    for(int g=0;g<groups;++g) for(int n=0;n<N;++n) {
        h.s[g*N+n]=__float2bfloat16_rn(1.0f+float(n%64)/128.0f+float(g)/2.0f);
        h.z[g*N+n]=uint8_t((n*7+n/8+g*3)%16);
    }
    V5Device d(h);
    std::vector<__nv_bfloat16> expected_s(groups*N),actual_s(groups*N);
    std::vector<uint32_t> expected_z(groups*N/8,0),actual_z(groups*N/8,0xffffffffu);
    // 从每个逻辑 n 反向定位，独立于 kernel 的物理位置->输入查找。
    for(int g=0;g<groups;++g) for(int n=0;n<N;++n) {
        int chunk=n/64,a=n%8,b=(n%64)/8;
        int physical=chunk*64+a*8+b;
        expected_s[g*N+physical]=h.s[g*N+n];
        int nibble=(b%2==0)?b/2:4+b/2;
        expected_z[g*(N/8)+chunk*8+a] |= uint32_t(h.z[g*N+n])<<(4*nibble);
    }
    __nv_bfloat16* ds=nullptr;uint32_t* dz=nullptr;
    CUDA_CHECK(cudaMalloc(&ds,actual_s.size()*sizeof(*ds)));CUDA_CHECK(cudaMalloc(&dz,actual_z.size()*sizeof(*dz)));
    CUDA_CHECK(cudaMemset(ds,0xff,actual_s.size()*sizeof(*ds)));CUDA_CHECK(cudaMemset(dz,0xff,actual_z.size()*sizeof(*dz)));
    permute_scale<<<(groups*N+255)/256,256>>>(d.s,ds,groups,N);
    permute_pack_zero<<<(groups*(N/8)+255)/256,256>>>(d.z,dz,groups,N);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual_s.data(),ds,actual_s.size()*sizeof(*ds),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_z.data(),dz,actual_z.size()*sizeof(*dz),cudaMemcpyDeviceToHost));
    int es=0,ez=0;
    for(size_t i=0;i<actual_s.size();++i) {uint16_t a,b;std::memcpy(&a,&actual_s[i],2);std::memcpy(&b,&expected_s[i],2);
        if(a!=b){if(es<3)std::fprintf(stderr,"scale[%zu] bits=0x%04x expected=0x%04x\n",i,a,b);++es;}}
    for(size_t i=0;i<actual_z.size();++i)if(actual_z[i]!=expected_z[i]){if(ez<3)std::fprintf(stderr,"zero[%zu]=0x%08x expected=0x%08x\n",i,actual_z[i],expected_z[i]);++ez;}
    std::printf("K=%d N=%d scale位错误=%d zero word错误=%d\n",K,N,es,ez);
    CUDA_CHECK(cudaFree(ds));CUDA_CHECK(cudaFree(dz));return !es&&!ez;
}
int main(){bool ok=true;for(int K:{256,384})for(int N:{64,128,192})ok=test_params(K,N)&&ok;return ok?EXIT_SUCCESS:EXIT_FAILURE;}
