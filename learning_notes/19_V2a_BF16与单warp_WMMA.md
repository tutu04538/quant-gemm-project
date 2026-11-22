# V2a：BF16 与单 warp 的 WMMA

## 1. 目标与输入

用一个 warp 的 32 个线程合作计算一个 `16×16×16` 矩阵乘：

```text
BF16 A [16,16] × BF16 B [16,16] → FP32 C [16,16]
```

这里的 `16×16×16` 依次表示 M、N、K；不是一次 Marlin W4 解码，也不包含 quantization、bias 或 padding。本节输出 FP32 是为了先观察累加，不代表最终 W4A16 项目的 BF16 输出契约。

## 2. 为什么先看 BF16

FP32、FP16、BF16 都是浮点格式，分别用不同数量的 bit 存符号、指数和小数精度。BF16 的指数位数与 FP32 相同，但小数精度更低；把一般 FP32 小数转换为 BF16 时可能舍入。例如十进制 0.1 通常不能精确表示，转换到 BF16 后再转回 float 得到的是 BF16 可表示的近似值。

模板在 CPU 上先将测试输入转成 `__nv_bfloat16`；CPU 参考再把**已经舍入的 BF16 输入**转换回 FP32，按 K 累加。这样测试的是 GPU 对实际 BF16 输入的计算。累计为 FP32 也不会恢复 BF16 转换前丢失的输入精度。

整数 case 用的是小整数，结果在本例可精确比较；小数 case 在测试前约定 `atol=0.002, rtol=0.0001`，规则为 `abs(actual-reference) <= atol + rtol*abs(reference)`，并检查 finite。这是本练习的测试门槛，不是所有 BF16 GEMM 的通用容差。

## 3. 一个 block、一个 warp 与数据流

```text
GPU grid = 1 个 block；block = 32 个线程 = 1 个 warp

            global A [16,16]          global B [16,16]
                   │                       │
                   └── 32 条 lane 协作加载 ─┘
                                ↓
                    A fragment、B fragment
                    （分布在各 lane 的寄存器中）
                                ↓
             32 条 lane 协作执行 WMMA：A×B + accumulator
                                ↓
                    FP32 accumulator fragment
                                ↓ warp 协作写回
                         global C [16,16]
```

`fragment` 表示 warp 协作的矩阵片段；不是一条线程独占一整块矩阵。WMMA 的 fragment 内部 lane→元素布局不能靠猜测固定下来，也不能随意把它当普通的二维数组访问。所有 32 条 lane 必须一致地执行 warp 级 WMMA 调用。

手算：假如 A 是 `16×16` 单位矩阵，B 任意，则 `C=A×B=B`；比如 B[2,3]=5，则 C[2,3]=5。模板会用独立 CPU 参考检查整个矩阵，实际测试输入不是这个示意矩阵。

## 4. 认识 WMMA 语法

[练习模板](../kernels/marlin_from_scratch/v2a_wmma_template.cu) 已声明三个 fragment：A、B 为 BF16、row-major；acc 为 FP32。`namespace wmma = nvcuda::wmma;` 是给较长命名空间起别名。

### `nvcuda::wmma` 是什么？

`nvcuda::wmma` 是 CUDA C++ 提供的 **warp 矩阵乘接口所在的命名空间**。这里的“命名空间”是 C++ 给一组名字加的前缀；`::` 用来访问其中的成员，例如：

```cpp
nvcuda::wmma::fragment<...>  // 接口里的片段类型
nvcuda::wmma::mma_sync(...)  // 接口里的 warp 协作矩阵乘函数
```

这不是一个单独的矩阵变量，也不是你需要自己创建的 warp。我们的公共头文件包含 `#include <mma.h>`，由 CUDA 工具链提供接口；代码中的 `namespace wmma = nvcuda::wmma;` 只是定义别名，因此 `wmma::mma_sync` 和 `nvcuda::wmma::mma_sync` 指向同一接口。两者调用的是同一份功能，不是两个版本。

`nvcuda` 本身是 C++ 命名空间的名字，不是单独安装的库。这里的 `nvcuda::wmma` 来自 NVIDIA CUDA Toolkit，并不属于 CUTLASS。本机 CUDA 12.1 的 `/usr/local/cuda-12.1/include/mma.h` 包含了 `crt/mma.h`，后者声明了这组命名空间和接口。CUTLASS 是 NVIDIA 的另一套 CUDA C++ 模板组件，提供更完整的矩阵计算构件；单独使用本练习的 WMMA 接口不需要安装 CUTLASS。

```text
你的 CUDA C++：wmma::load_matrix_sync / mma_sync / store_matrix_sync
       ↓ 编译器为目标 GPU 编译
GPU 上：一个 warp 的 32 条 lane 协作执行对应操作
       ↓
对于本机已验证的 BF16 示例，编译结果包含 HMMA 指令
```

这里的 `_sync` 是该 warp 级接口的协作要求，调用时必须满足完整 warp 一致参与等约束；它不是 CPU 的 `cudaDeviceSynchronize()`，也不是让整个 GPU 或所有 block 同步。不同数据类型、形状和 GPU 的支持情况及最终指令应以工具链和实际编译结果为准，不能只看到接口名就断言性能。

依次需要：

1. `wmma::fill_fragment(片段, 初值)`：初始化 accumulator。
2. `wmma::load_matrix_sync(片段, 指针, 行步长)`：从内存加载 A 或 B；行优先时步长指**相邻行相隔多少个元素**，不是字节。
3. `wmma::mma_sync(输出片段, A片段, B片段, 原累加片段)`：执行乘加。
4. `wmma::store_matrix_sync(指针, 片段, 行步长, wmma::mem_row_major)`：写回 FP32 结果。

本模板分配的设备内存和尺寸满足此例的地址及步长要求；实际 WMMA 对齐和 stride 约束不能对任意 view 假设成立。接口及 warp 同步要求可对照 [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#warp-matrix-functions)。

## 5. 动手

复制 `v2a_wmma_template.cu` 为同目录 `v2a_wmma.cu`，不要移动它，因为它需要同目录的 `v2_wmma_common.cuh`。填写 4 处 TODO：清零、加载两个输入、MMA、写回。模板 host 已提供输入、测试和错误检查。

```bash
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
    kernels/marlin_from_scratch/v2a_wmma.cu -o /tmp/quant_v2a
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v2a
```

先确认 GPU 1 当时空闲；否则选其他空闲卡。模板只支持一个 warp 和固定 `M=N=K=16`，不包含非整除尾部处理。写完后先 review 并运行验证，再进入 V2b；不要根据写了 WMMA 就先声称性能或特定机器指令已验证。
