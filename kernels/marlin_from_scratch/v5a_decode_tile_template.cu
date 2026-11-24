// V5a: 一个 warp 把一个 16×16 W4 tile 解码到 shared；再复制到 global 供测试。
#include "v5_test_common.cuh"
__global__ void decode_tile(const uint32_t* packed,const uint8_t* z,const __nv_bfloat16* s,
                            __nv_bfloat16* out,int K,int N,int base,int col0) {
    __shared__ __align__(32) __nv_bfloat16 tile[16][16];
    int lane=threadIdx.x;
    // TODO 1：每 lane 处理 i=lane, lane+32, ... < 256。
    // TODO 2：i 映射 tile 内 (k_local,n_local)，再映射逻辑 (base+k_local,col0+n_local)。
    // TODO 3：合法坐标下从 packed/Z/S 解码并转 BF16，写 tile[k_local][n_local]；
    //         逻辑 K 或 N 越界时写 BF16 零。注意 q-z 应先转为有符号整数。
    // TODO 4：所有 lane 写完 tile 后同步 warp，合作把 tile 复制到 out；测试用。
    // 未填写时 out 仍是测试预填的 NaN，避免假通过。
    (void)packed;(void)z;(void)s;(void)out;(void)K;(void)N;(void)base;(void)col0;(void)lane;
    for (int i = lane; i < 256; i += 32) {
        int k_local = i / 16, n_local = i % 16;
        int k_logic = base + k_local, n_logic = col0 + n_local;
        if (k_logic >= K || n_logic >= N) {
            tile[k_local][n_local] = 0;
        } else {
            uint32_t q_weight = packed[k_logic / 8 * N + n_logic];
            uint8_t q_z = z[k_logic / 128 * N + n_logic];
            __nv_bfloat16 q_s = s[k_logic / 128 * N + n_logic];
            __nv_bfloat16 w = __float2bfloat16(((int)((q_weight >> ((k_logic % 8) * 4)) & 0xf) - (int)q_z) * __bfloat162float(q_s));
            tile[k_local][n_local] = w;
        }
    }
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        int k_local = i / 16, n_local = i % 16;
        out[i] = tile[k_local][n_local];
    }
}
static bool test_tile(int K,int N,int base,int col0,bool fractional) {
    V5Inputs h(1,N,K,fractional); V5Device d(h);
    std::vector<__nv_bfloat16> out(256,__float2bfloat16_rn(std::numeric_limits<float>::quiet_NaN()));
    __nv_bfloat16* dout=nullptr; CUDA_CHECK(cudaMalloc(&dout,out.size()*sizeof(*dout)));
    CUDA_CHECK(cudaMemcpy(dout,out.data(),out.size()*sizeof(*dout),cudaMemcpyHostToDevice));
    decode_tile<<<1,32>>>(d.packed,d.z,d.s,dout,K,N,base,col0);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(),dout,out.size()*sizeof(*dout),cudaMemcpyDeviceToHost));
    int errors=0;
    for(int x=0;x<16;++x) for(int y=0;y<16;++y) {
        int k=base+x,n=col0+y;
        float expected=(k<K && n<N) ? h.weight(k,n) : 0.0f;
        float got=__bfloat162float(out[x*16+y]);
        if(!std::isfinite(got)||got!=expected) {
            if(errors<3) std::fprintf(stderr,"tile[%d,%d]: %g != %g\n",x,y,got,expected); ++errors;
        }
    }
    std::printf("K=%d N=%d base=%d col0=%d fractional=%d：tile 错误=%d\n",K,N,base,col0,fractional,errors);
    CUDA_CHECK(cudaFree(dout)); return errors==0;
}
int main() {
    int dev; cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::printf("GPU=%s SM=%d.%d；tile 每元素精确比较\n",prop.name,prop.major,prop.minor);
    bool ok=true;
    for(bool f:{false,true}) for(int base:{0,112,128,240}) for(int col:{0,16})
        ok=test_tile(256,19,base,col,f)&&ok;
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
