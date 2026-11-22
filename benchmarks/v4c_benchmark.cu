// V4c: same inputs, independent CPU reference, alternating CUDA-event batches.
// Includes the user's V3/V4a kernels directly, with only the V4a test main disabled.
#define V4A_NO_MAIN
#include "../kernels/marlin_from_scratch/v4a_word_reuse_template.cu"
#include <cstring>

static bool bench_case(int M, int N, int K, bool check_only) {
    constexpr int warmup=20, batches=15, repeats=100;
    const int groups=K/GROUP;
    std::vector<__nv_bfloat16> a(M*K), s(groups*N);
    std::vector<uint8_t> q(K*N), z(groups*N);
    std::vector<uint32_t> p((K/8)*N,0);
    for (int i=0;i<M*K;++i) a[i]=__float2bfloat16_rn(float((i*7)%31-15)*0.0625f);
    for (int k=0;k<K;++k) for(int n=0;n<N;++n) q[k*N+n]=(k*3+n*7+k/GROUP)%16;
    for (int g=0;g<groups;++g) for(int n=0;n<N;++n) {
        z[g*N+n]=(g*7+n*3)%16;
        const float scales[]={0.1f,0.25f,0.375f,0.7f};
        s[g*N+n]=__float2bfloat16_rn(scales[(g+n)%4]);
    }
    for (int k=0;k<K;k+=8) for(int n=0;n<N;++n) for(int t=0;t<8;++t)
        p[(k/8)*N+n] |= uint32_t(q[(k+t)*N+n])<<(4*t);
    __nv_bfloat16 *da=nullptr,*ds=nullptr; uint32_t* dp=nullptr; uint8_t* dz=nullptr;
    float *dc3=nullptr,*dc4=nullptr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*sizeof(*da))); CUDA_CHECK(cudaMalloc(&ds,s.size()*sizeof(*ds)));
    CUDA_CHECK(cudaMalloc(&dp,p.size()*sizeof(*dp))); CUDA_CHECK(cudaMalloc(&dz,z.size()*sizeof(*dz)));
    CUDA_CHECK(cudaMalloc(&dc3,size_t(M)*N*sizeof(float))); CUDA_CHECK(cudaMalloc(&dc4,size_t(M)*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(*da),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds,s.data(),s.size()*sizeof(*ds),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp,p.data(),p.size()*sizeof(*dp),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz,z.data(),z.size()*sizeof(*dz),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dc3,0xff,size_t(M)*N*sizeof(float)));
    CUDA_CHECK(cudaMemset(dc4,0xff,size_t(M)*N*sizeof(float)));
    dim3 block(16,16),grid((N+15)/16,(M+15)/16);
    auto launch=[&](bool is_v4) {
        if (is_v4) v4a_gemm<<<grid,block>>>(da,dp,dz,ds,dc4,M,N,K);
        else w4a16_gemm<<<grid,block>>>(da,dp,dz,ds,dc3,M,N,K);
    };
    launch(false); launch(true); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> c3(M*N),c4(M*N);
    CUDA_CHECK(cudaMemcpy(c3.data(),dc3,c3.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c4.data(),dc4,c4.size()*sizeof(float),cudaMemcpyDeviceToHost));
    int errors3=0,errors4=0; float max3=0,max4=0; double sum3=0,sum4=0;
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        float expected=0;
        for(int k=0;k<K;++k) {
            int gi=(k/GROUP)*N+n;
            float w=float(int(q[k*N+n])-int(z[gi]))*__bfloat162float(s[gi]);
            w=__bfloat162float(__float2bfloat16_rn(w));
            expected=std::fma(__bfloat162float(a[m*K+k]),w,expected);
        }
        float x=c3[m*N+n],y=c4[m*N+n];
        float e3=std::fabs(x-expected),e4=std::fabs(y-expected),limit=0.0001f+0.00001f*std::fabs(expected);
        if (!std::isfinite(x) || e3>limit) ++errors3;
        if (!std::isfinite(y) || e4>limit) ++errors4;
        if (std::isfinite(x)) { max3=std::max(max3,e3); sum3+=e3; }
        if (std::isfinite(y)) { max4=std::max(max4,e4); sum4+=e4; }
    }
    std::printf("case M=%d N=%d K=%d correctness V3 errors=%d max=%g mean=%g V4a errors=%d max=%g mean=%g\n",
        M,N,K,errors3,max3,sum3/(M*N),errors4,max4,sum4/(M*N));
    if (errors3 || errors4) {
        std::fprintf(stderr,"correctness failed; no timing for case\n");
    } else if (!check_only) {
        for(int i=0;i<warmup;++i) { launch(false); launch(true); }
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start,stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<float> times3,times4;
        auto measure=[&](bool v4) {
            CUDA_CHECK(cudaEventRecord(start));
            for(int i=0;i<repeats;++i) launch(v4);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop)); return ms*1000.0f/repeats;
        };
        for(int b=0;b<batches;++b) {
            float t3,t4;
            if(b%2==0) { t3=measure(false); t4=measure(true); }
            else { t4=measure(true); t3=measure(false); }
            times3.push_back(t3); times4.push_back(t4);
            std::printf("batch=%02d order=%s V3_us=%.5f V4a_us=%.5f\n",b,b%2?"4,3":"3,4",t3,t4);
        }
        auto sorted3=times3,sorted4=times4; std::sort(sorted3.begin(),sorted3.end()); std::sort(sorted4.begin(),sorted4.end());
        float p50v3=sorted3[batches/2],p50v4=sorted4[batches/2];
        int p90=(9*batches+9)/10-1;
        std::printf("summary M=%d N=%d K=%d V3_p50_us=%.5f V3_p90_us=%.5f V4a_p50_us=%.5f V4a_p90_us=%.5f ratio=%.4f\n",
            M,N,K,p50v3,sorted3[p90],p50v4,sorted4[p90],p50v3/p50v4);
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    }
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(dp)); CUDA_CHECK(cudaFree(dz)); CUDA_CHECK(cudaFree(dc3)); CUDA_CHECK(cudaFree(dc4));
    return errors3==0 && errors4==0;
}
int main(int argc, char** argv) {
    const bool check_only = argc == 2 && std::strcmp(argv[1], "--check") == 0;
    if (argc > 1 && !check_only) { std::fprintf(stderr, "usage: %s [--check]\n", argv[0]); return EXIT_FAILURE; }
    int dev,driver,runtime; cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("GPU=%s CC=%d.%d driverAPI=%d runtime=%d warmup=20 batches=15 repeats_per_batch=100 event_scope=100_kernel_launches batch_means_us\n",prop.name,prop.major,prop.minor,driver,runtime);
    const Shape cases[]={{1,128,256},{16,128,256},{128,128,256},{128,512,1024}};
    bool ok=true; for(const auto& sh:cases) ok=bench_case(sh.m,sh.n,sh.k,check_only)&&ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
