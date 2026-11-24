// V6c：相同逻辑输入的 V5/V6 steady-state 和单次离线重排 GPU 时间。
#define V6C_NO_MAIN
#include "../kernels/marlin_from_scratch/v6c_gemm_template.cu"

static bool measure_case(int M,int N,int K) {
    constexpr int warmup=20,batches=15,repeats=100;
    if(!test_v6(M,N,K,true)) { std::fprintf(stderr,"正确性失败；跳过计时\n");return false; }
    V5Inputs h(M,N,K,true);V5Device d(h);
    int nt=(N+7)/8,count=K*nt;
    uint32_t* dr=nullptr;float *c5=nullptr,*c6=nullptr;
    CUDA_CHECK(cudaMalloc(&dr,size_t(count)*sizeof(*dr)));
    CUDA_CHECK(cudaMalloc(&c5,size_t(M)*N*sizeof(float)));CUDA_CHECK(cudaMalloc(&c6,size_t(M)*N*sizeof(float)));
    dim3 block(32),grid((N+7)/8,(M+15)/16);
    auto launch=[&](int which) {
        if(which==0) v5ef_gemm<<<grid,block>>>(d.a,d.packed,d.z,d.s,c5,M,N,K);
        else if(which==1) v6_gemm<<<grid,block>>>(d.a,dr,d.z,d.s,c6,M,N,K);
        else repack_k_to_n<<<(count+255)/256,256>>>(d.packed,dr,K,N);
    };
    launch(2);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    for(int i=0;i<warmup;++i) {launch(0);launch(1);launch(2);}
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start,stop;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));
    auto measure=[&](int which) {
        CUDA_CHECK(cudaEventRecord(start));
        for(int j=0;j<repeats;++j) launch(which);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(stop));CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));return ms*1000/repeats;
    };
    std::vector<float> a,b,p;
    for(int j=0;j<batches;++j) {
        float x,y;
        if(j%2==0) {x=measure(0);y=measure(1);} else {y=measure(1);x=measure(0);}
        float r=measure(2);
        a.push_back(x);b.push_back(y);p.push_back(r);
        std::printf("batch=%02d order=%s v5_us=%.5f v6_us=%.5f repack_us=%.5f\n",j,j%2?"6,5":"5,6",x,y,r);
    }
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());std::sort(p.begin(),p.end());
    int p90=(9*batches+9)/10-1;
    size_t pbytes=h.packed.size()*sizeof(uint32_t),rbytes=size_t(count)*sizeof(uint32_t);
    float gain=a[batches/2]-b[batches/2];
    std::printf("summary M=%d N=%d K=%d V5_p50=%.5f V5_p90=%.5f V6_p50=%.5f V6_p90=%.5f repack_p50=%.5f repack_p90=%.5f ratio=%.4f P_bytes=%zu R_bytes=%zu P_plus_R=%zu",
        M,N,K,a[batches/2],a[p90],b[batches/2],b[p90],p[batches/2],p[p90],a[batches/2]/b[batches/2],pbytes,rbytes,pbytes+rbytes);
    if(gain>0) std::printf(" estimated_break_even_calls=%.2f",p[batches/2]/gain);
    else std::printf(" estimated_break_even_calls=none");
    std::printf("\n");
    CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dr));CUDA_CHECK(cudaFree(c5));CUDA_CHECK(cudaFree(c6));return true;
}
int main() {
    int dev,driver,runtime;cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    CUDA_CHECK(cudaDriverGetVersion(&driver));CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("GPU=%s CC=%d.%d driverAPI=%d runtime=%d warmup=20 batches=15 repeats=100 batch_means_us fractional_inputs\n",prop.name,prop.major,prop.minor,driver,runtime);
    struct Shape {int m,n,k;};
    bool ok=true;for(const auto sh:{Shape{1,128,256},Shape{16,128,256},Shape{128,128,256},Shape{128,512,1024}})
        ok=measure_case(sh.m,sh.n,sh.k)&&ok;
    return ok?EXIT_SUCCESS:EXIT_FAILURE;
}
