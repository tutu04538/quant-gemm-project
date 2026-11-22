// 第 2 节练习：填写 TODO。用 nvcc 编译，不要用普通 C++ 编译器。
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 复用你在第 1 节写的 kernel。
#include "01.cu"

// 检查 CUDA API 返回值；失败时打印错误并结束程序。
#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    constexpr int n = 37;
    constexpr int threads = 32;
    const int blocks = (n + threads - 1) / threads;
    const size_t bytes = n * sizeof(float);

    // vector 持有 CPU 内存，data() 提供指向第一个元素的指针。
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 0.5f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // TODO 1：为 d_a、d_b、d_c 各分配 bytes 字节的 device 内存。
    // 将每次 CUDA API 调用放在 CUDA_CHECK(...) 内检查返回值。

    // 未完成分配时提前结束，避免空指针传给后续代码。
    if (!d_a || !d_b || !d_c) {
        std::fprintf(stderr, "请先完成 TODO 1：设备内存分配。\n");
        return EXIT_FAILURE;
    }

    // TODO 2：将 h_a.data() 和 h_b.data() 分别复制到 d_a 和 d_b。

    // TODO 3：用 blocks 和 threads 发射 vector_add。

    // 检查发射配置；同步后检查执行阶段的错误。
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO 4：将 d_c 复制到 h_c.data()。

    // 验证每个元素。当前输入选用能精确表示的数，故可以直接比较。
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            std::fprintf(stderr, "i=%d: 得到 %g，期望 %g\n", i, h_c[i], expected);
            correct = false;
            break;
        }
    }

    // TODO 5：释放 d_a、d_b、d_c 各自的 device 内存。

    if (!correct) return EXIT_FAILURE;
    std::printf("通过：%d 个元素。\n", n);
    return EXIT_SUCCESS;
}
