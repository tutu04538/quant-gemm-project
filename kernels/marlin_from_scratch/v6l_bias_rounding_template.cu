// V6l：固定参考分支的一个输出元素：FP32 acc 先转 BF16，再以 BF16 加 bias。
#include "v5_test_common.cuh"
#include <cstring>
__global__ void v6l_epilogue(const float* acc,const __nv_bfloat16* bias,__nv_bfloat16* out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    // TODO：将 acc[i] 先用 RN 转 BF16，再用 BF16 __hadd 与 bias[i] 相加并写出。
    // 这只模拟参考 write_result 的一对数值操作，不模拟其完整 shared 输出布局。
    (void)acc;(void)bias;(void)out;
}
int main() {
    std::vector<float> a={1.00390625f, -1.00390625f, 0.1f, 2.75f, -0.875f};
    std::vector<__nv_bfloat16> b(a.size()),out(a.size());
    const float bias[]={0.00390625f,0.00390625f,0.7f,-0.5f,0.25f};
    for(size_t i=0;i<a.size();++i)b[i]=__float2bfloat16_rn(bias[i]);
    float *da=nullptr;__nv_bfloat16 *db=nullptr,*dc=nullptr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*4));CUDA_CHECK(cudaMalloc(&db,b.size()*2));CUDA_CHECK(cudaMalloc(&dc,out.size()*2));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(dc,0xff,out.size()*2));
    v6l_epilogue<<<1,32>>>(da,db,dc,int(a.size()));CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*2,cudaMemcpyDeviceToHost));
    int errors=0,different=0;for(size_t i=0;i<a.size();++i){
        auto rounded=__float2bfloat16_rn(a[i]);
        auto expected=__float2bfloat16_rn(__bfloat162float(rounded)+__bfloat162float(b[i]));
        auto once=__float2bfloat16_rn(a[i]+__bfloat162float(b[i]));
        uint16_t gotbits,expbits,oncebits;std::memcpy(&gotbits,&out[i],2);std::memcpy(&expbits,&expected,2);std::memcpy(&oncebits,&once,2);
        if(gotbits!=expbits)++errors;if(expbits!=oncebits)++different;
        std::printf("i=%zu 先舍入再加 bias=0x%04x 一次舍入=0x%04x 输出=0x%04x\n",i,expbits,oncebits,gotbits);
    }
    std::printf("参考次序位错误=%d，两种次序不同=%d\n",errors,different);
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));return errors==0&&different>0?EXIT_SUCCESS:EXIT_FAILURE;
}
