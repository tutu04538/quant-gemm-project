// V5a/b/c 共用的独立测试数据；只用于测试，不用于 GPU 核心解码。
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#define CUDA_CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess) { \
    std::fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e_)); std::exit(EXIT_FAILURE); } } while(0)
constexpr int V5_GROUP=128;
struct V5Inputs {
    int M,N,K;
    std::vector<__nv_bfloat16> a,s;
    std::vector<uint8_t> q,z;
    std::vector<uint32_t> packed;
    V5Inputs(int m,int n,int k,bool fractional): M(m),N(n),K(k),a(m*k),s((k/128)*n),q(k*n),z((k/128)*n),packed((k/8)*n,0) {
        for(int i=0;i<M*K;++i) a[i]=__float2bfloat16_rn(fractional ? float((i*7)%19-9)*0.1f : float(i%5-2));
        for(int k0=0;k0<K;++k0) for(int n0=0;n0<N;++n0) q[k0*N+n0]=(k0+3*n0+5*(k0/128))%16;
        for(int g=0;g<K/128;++g) for(int n0=0;n0<N;++n0) {
            z[g*N+n0]=(g*7+3*n0)%16;
            const float scales[]={0.1f,0.25f,0.375f,0.7f};
            s[g*N+n0]=__float2bfloat16_rn(fractional ? scales[(g+n0)%4] : 0.25f);
        }
        for(int k0=0;k0<K;k0+=8) for(int n0=0;n0<N;++n0) for(int t=0;t<8;++t)
            packed[(k0/8)*N+n0] |= uint32_t(q[(k0+t)*N+n0])<<(4*t);
        if(packed[0]!=0x76543210u) { std::fprintf(stderr,"手算 word 锚点错误\n"); std::exit(EXIT_FAILURE); }
    }
    // 基于原始逻辑 Q 的 CPU 参考，不读取 packed，也不使用 GPU 解码函数。
    float weight_math(int k,int n) const {
        int j=(k/128)*N+n;
        return float(int(q[k*N+n])-int(z[j]))*__bfloat162float(s[j]);
    }
    float weight(int k,int n) const { return __bfloat162float(__float2bfloat16_rn(weight_math(k,n))); }
};
struct V5Device {
    __nv_bfloat16 *a=nullptr,*s=nullptr;
    uint32_t* packed=nullptr; uint8_t* z=nullptr;
    explicit V5Device(const V5Inputs& h) {
        CUDA_CHECK(cudaMalloc(&a,h.a.size()*sizeof(*a))); CUDA_CHECK(cudaMalloc(&s,h.s.size()*sizeof(*s)));
        CUDA_CHECK(cudaMalloc(&packed,h.packed.size()*sizeof(*packed))); CUDA_CHECK(cudaMalloc(&z,h.z.size()));
        CUDA_CHECK(cudaMemcpy(a,h.a.data(),h.a.size()*sizeof(*a),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s,h.s.data(),h.s.size()*sizeof(*s),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(packed,h.packed.data(),h.packed.size()*sizeof(*packed),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(z,h.z.data(),h.z.size(),cudaMemcpyHostToDevice));
    }
    ~V5Device() { CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(s)); CUDA_CHECK(cudaFree(packed)); CUDA_CHECK(cudaFree(z)); }
    V5Device(const V5Device&)=delete; V5Device& operator=(const V5Device&)=delete;
};
