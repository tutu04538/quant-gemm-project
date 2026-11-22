// V3e：受限、同步的教学调用入口。复用用户在 V3c/V3d 填写的 kernel。
// 编译：nvcc -arch=sm_89 -O2 -lineinfo v3e_checked.cu -lcuda -o /tmp/v3e_checked
#define main v3_original_test_main
#include "v3_w4a16_template.cu"
#undef main
#include <cuda.h>
#include <climits>
#include <cstring>

// 元数据由调用者提供，dtype 是声明，不是对内存内容的自动识别。
enum class DType { BF16, U32, U8, F32 };
struct Matrix {
    void* data;
    int rows, cols;
    int row_stride;  // 元素数，不是字节数
    DType dtype;
};
enum class V3Status {
    OK, SHAPE, TYPE, STRIDE, POINTER, DEVICE, CAPACITY, ALIAS,
    ZERO, SCALE, CUDA_ERROR
};
static const char* status_name(V3Status s) {
    switch (s) {
    case V3Status::OK: return "OK";
    case V3Status::SHAPE: return "SHAPE";
    case V3Status::TYPE: return "TYPE";
    case V3Status::STRIDE: return "STRIDE";
    case V3Status::POINTER: return "POINTER";
    case V3Status::DEVICE: return "DEVICE";
    case V3Status::CAPACITY: return "CAPACITY";
    case V3Status::ALIAS: return "ALIAS";
    case V3Status::ZERO: return "ZERO";
    case V3Status::SCALE: return "SCALE";
    case V3Status::CUDA_ERROR: return "CUDA_ERROR";
    }
    return "UNKNOWN";
}
struct Interval { uintptr_t begin, end; };

// 仅支持当前设备上普通 cudaMalloc 的 device allocation（不支持 managed 等）。
// 利用实际 allocation 范围检查偏移指针与缓冲容量。
static V3Status check_buffer(const Matrix& t, DType type, int rows, int cols,
                             size_t element_bytes, int device, Interval& range) {
    if (t.rows != rows || t.cols != cols) return V3Status::SHAPE;
    if (t.dtype != type) return V3Status::TYPE;
    if (t.row_stride != cols) return V3Status::STRIDE;
    if (!t.data) return V3Status::POINTER;
    uintptr_t p = reinterpret_cast<uintptr_t>(t.data);
    if (p % element_bytes != 0) return V3Status::POINTER;
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, t.data);
    if (err != cudaSuccess) { (void)cudaGetLastError(); return V3Status::POINTER; }
    if (attrs.type != cudaMemoryTypeDevice) return V3Status::POINTER;
    if (attrs.device != device) return V3Status::DEVICE;
    CUdeviceptr allocation = 0; size_t allocation_bytes = 0;
    CUresult driver_err = cuMemGetAddressRange(&allocation, &allocation_bytes, static_cast<CUdeviceptr>(p));
    if (driver_err != CUDA_SUCCESS) return V3Status::POINTER;
    uintptr_t base = static_cast<uintptr_t>(allocation);
    if (p < base || p - base > allocation_bytes) return V3Status::CAPACITY;
    size_t needed = size_t(rows) * size_t(cols) * element_bytes;
    if (needed > allocation_bytes - (p - base) || p > UINTPTR_MAX - needed)
        return V3Status::CAPACITY;
    range = {p, p + needed};
    return V3Status::OK;
}

// 这是同步教学接口：调用者需保证输入就绪、无并发访问/释放，入口只用默认 stream。
// 检查 Z/S 时同步复制到 host；成功后同步等待 kernel，包含异步执行错误检查。
// 参数拒绝时不会发射 GEMM；CUDA_ERROR 后输出的状态不保证。
V3Status v3e_checked(const Matrix& A, const Matrix& P, const Matrix& Z,
                     const Matrix& S, const Matrix& C, int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0 || K % GROUP != 0) return V3Status::SHAPE;
    // kernel 使用 int 线性下标；同时避免 grid 计算溢出。
    if (int64_t(M) * K > INT_MAX || int64_t(K) * N > INT_MAX ||
        int64_t(M) * N > INT_MAX || M > INT_MAX - 15 || N > INT_MAX - 15)
        return V3Status::SHAPE;
    int device = 0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return V3Status::CUDA_ERROR;
    if (prop.major < 8) return V3Status::DEVICE;
    // block=(16,16)；检查设备的 grid y 上限。
    if ((int64_t(M) + 15) / 16 > prop.maxGridSize[1] ||
        (int64_t(N) + 15) / 16 > prop.maxGridSize[0]) return V3Status::SHAPE;
    Interval ranges[5]; V3Status st;
    st = check_buffer(A, DType::BF16, M, K, 2, device, ranges[0]); if (st != V3Status::OK) return st;
    st = check_buffer(P, DType::U32, K/8, N, 4, device, ranges[1]); if (st != V3Status::OK) return st;
    st = check_buffer(Z, DType::U8, K/GROUP, N, 1, device, ranges[2]); if (st != V3Status::OK) return st;
    st = check_buffer(S, DType::BF16, K/GROUP, N, 2, device, ranges[3]); if (st != V3Status::OK) return st;
    st = check_buffer(C, DType::F32, M, N, 4, device, ranges[4]); if (st != V3Status::OK) return st;
    for (int i = 0; i < 5; ++i) for (int j = i+1; j < 5; ++j)
        if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end)
            return V3Status::ALIAS;
    // 对同一调用，本接口不允许其他线程或 stream 并发改变这些缓冲区。
    if (cudaDeviceSynchronize() != cudaSuccess) return V3Status::CUDA_ERROR;
    const size_t count = size_t(K/GROUP) * N;
    std::vector<uint8_t> hz(count);
    std::vector<__nv_bfloat16> hs(count);
    if (cudaMemcpy(hz.data(), Z.data, count, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(hs.data(), S.data, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) != cudaSuccess)
        return V3Status::CUDA_ERROR;
    for (size_t i = 0; i < count; ++i) {
        if (hz[i] > 15) return V3Status::ZERO;
        float s = __bfloat162float(hs[i]);
        if (!std::isfinite(s) || s < 0) return V3Status::SCALE;
    }
    dim3 block(16, 16), grid((N + 15)/16, (M + 15)/16);
    w4a16_gemm<<<grid, block>>>(static_cast<const __nv_bfloat16*>(A.data),
        static_cast<const uint32_t*>(P.data), static_cast<const uint8_t*>(Z.data),
        static_cast<const __nv_bfloat16*>(S.data), static_cast<float*>(C.data), M, N, K);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return V3Status::CUDA_ERROR;
    return V3Status::OK;
}

static bool test_v3e() {
    constexpr int M=3, N=9, K=256;
    std::vector<__nv_bfloat16> a(M*K, __float2bfloat16_rn(1)), s((K/GROUP)*N, __float2bfloat16_rn(0.25f));
    std::vector<uint32_t> p((K/8)*N, 0x11111111u);
    std::vector<uint8_t> z((K/GROUP)*N, 0);
    std::vector<float> out(M*N, -999.0f);
    void *da=nullptr,*dp=nullptr,*dz=nullptr,*ds=nullptr,*dc=nullptr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*2)); CUDA_CHECK(cudaMalloc(&dp,p.size()*4));
    CUDA_CHECK(cudaMalloc(&dz,z.size())); CUDA_CHECK(cudaMalloc(&ds,s.size()*2)); CUDA_CHECK(cudaMalloc(&dc,out.size()*4));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp,p.data(),p.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz,z.data(),z.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds,s.data(),s.size()*2,cudaMemcpyHostToDevice));
    Matrix A{da,M,K,K,DType::BF16}, P{dp,K/8,N,N,DType::U32}, Z{dz,K/GROUP,N,N,DType::U8};
    Matrix S{ds,K/GROUP,N,N,DType::BF16}, C{dc,M,N,N,DType::F32};
    int failures=0, checks=0;
    auto expect = [&](const char* label, V3Status got, V3Status wanted) {
        ++checks; if (got != wanted) { ++failures; std::fprintf(stderr,"%s: %s != %s\n",label,status_name(got),status_name(wanted)); }
    };
    auto call = [&]() { return v3e_checked(A,P,Z,S,C,M,N,K); };
    expect("合法调用", call(), V3Status::OK);
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*4,cudaMemcpyDeviceToHost));
    for (float x:out) if (x != 64.0f) { ++failures; std::fprintf(stderr,"正确性期望 64 得到 %g\n",x); break; }
    Matrix small_a=A, small_c=C; small_a.rows=1; small_c.rows=1;
    expect("不同 M 交替：M=1",v3e_checked(small_a,P,Z,S,small_c,1,N,K),V3Status::OK);
    CUDA_CHECK(cudaMemcpy(out.data(),dc,N*4,cudaMemcpyDeviceToHost));
    for (int i=0;i<N;++i) if (out[i] != 64.0f) { ++failures; std::fprintf(stderr,"M=1 结果错误\n"); break; }
    expect("不同 M 交替：M=3",call(),V3Status::OK);
    std::fill(z.begin(),z.end(),1); CUDA_CHECK(cudaMemcpy(dz,z.data(),z.size(),cudaMemcpyHostToDevice));
    expect("重复调用更换 Z",call(),V3Status::OK);
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*4,cudaMemcpyDeviceToHost));
    for (float x:out) if (x != 0.0f) { ++failures; std::fprintf(stderr,"重复调用期望 0 得到 %g\n",x); break; }
    // 拒绝测试时 C 仍应保持 -999：任何参数错误都不能启动 GEMM。
    std::fill(out.begin(),out.end(),-999.0f); CUDA_CHECK(cudaMemcpy(dc,out.data(),out.size()*4,cudaMemcpyHostToDevice));
    expect("空 M",v3e_checked(A,P,Z,S,C,0,N,K),V3Status::SHAPE);
    expect("未对齐 K",v3e_checked(A,P,Z,S,C,M,N,129),V3Status::SHAPE);
    expect("整数溢出",v3e_checked(A,P,Z,S,C,INT_MAX,N,K),V3Status::SHAPE);
    Matrix old=A; A.dtype=DType::F32; expect("dtype",call(),V3Status::TYPE); A=old;
    old=A; A.row_stride=K+1; expect("stride",call(),V3Status::STRIDE); A=old;
    old=A; A.cols=K-1; expect("形状",call(),V3Status::SHAPE); A=old;
    old=A; A.data=nullptr; expect("空指针",call(),V3Status::POINTER); A=old;
    old=A; A.data=a.data(); expect("host 指针",call(),V3Status::POINTER); A=old;
    old=A; A.data=static_cast<char*>(da)+1; expect("地址未对齐",call(),V3Status::POINTER); A=old;
    old=A; A.data=static_cast<char*>(da)+2; expect("allocation 容量",call(),V3Status::CAPACITY); A=old;
    old=C; C.data=da; expect("输出与输入重叠",call(),V3Status::ALIAS); C=old;
    z[0]=16; CUDA_CHECK(cudaMemcpy(dz,z.data(),z.size(),cudaMemcpyHostToDevice));
    expect("非法 zero",call(),V3Status::ZERO);
    z[0]=1; CUDA_CHECK(cudaMemcpy(dz,z.data(),z.size(),cudaMemcpyHostToDevice));
    s[0]=__float2bfloat16_rn(-1.0f); CUDA_CHECK(cudaMemcpy(ds,s.data(),s.size()*2,cudaMemcpyHostToDevice));
    expect("负 scale",call(),V3Status::SCALE);
    s[0]=__float2bfloat16_rn(std::numeric_limits<float>::quiet_NaN()); CUDA_CHECK(cudaMemcpy(ds,s.data(),s.size()*2,cudaMemcpyHostToDevice));
    expect("NaN scale",call(),V3Status::SCALE);
    CUDA_CHECK(cudaMemcpy(out.data(),dc,out.size()*4,cudaMemcpyDeviceToHost));
    for (float x:out) if (x != -999.0f) { ++failures; std::fprintf(stderr,"拒绝后 C 发生变化\n"); break; }
    // 比当前设备不同的 GPU 上的内存必须拒绝；仅在至少有两个 GPU 时执行。
    int original=0, count=0; CUDA_CHECK(cudaGetDevice(&original)); CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count > 1) {
        void* other=nullptr; int other_device=(original+1)%count;
        CUDA_CHECK(cudaSetDevice(other_device)); CUDA_CHECK(cudaMalloc(&other,a.size()*2));
        CUDA_CHECK(cudaSetDevice(original)); old=A; A.data=other; expect("其他设备",call(),V3Status::DEVICE); A=old;
        CUDA_CHECK(cudaSetDevice(other_device)); CUDA_CHECK(cudaFree(other)); CUDA_CHECK(cudaSetDevice(original));
    } else std::printf("其他设备测试跳过（可见 GPU 少于 2）\n");
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(dp)); CUDA_CHECK(cudaFree(dz)); CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(dc));
    std::printf("V3e：%d 项状态检查，失败 %d；另检查正确性、重复调用和拒绝时 C 不变\n", checks, failures);
    return failures == 0;
}
int main() {
    bool ok = test_v3e();
    // 同一用户 kernel 的原始独立量化参考回归；不通过 checked 入口。
    return (v3_original_test_main() == EXIT_SUCCESS && ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
