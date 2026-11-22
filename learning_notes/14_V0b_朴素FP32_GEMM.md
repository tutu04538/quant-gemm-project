# V0b：每个线程计算一个输出元素

## 1. 目标和输入

实现 V0a 的 FP32 `C=A×B`。每个 CUDA 线程负责一个 `C[row,col]`，自己遍历全部 K；本版没有 shared memory、线程间归约和 Tensor Core。

同一 [V0 模板](../kernels/marlin_from_scratch/v0_gemm_template.cu)已经提供 host 输入输出、CPU 参考和 CUDA 检查；你只需写 `v0_gemm` 中的四处 TODO。

## 2. block、warp、lane 和输出

模板使用 `block=(16,16)`，也就是 x 方向 16 个线程、y 方向 16 个线程，共 256 个线程，即 8 个 warp。线程在 block 内按 x 先变化排列：

```text
一个 block 负责一个最多 16×16 的 C 子块

                 threadIdx.x / 输出列方向 →
                 0   1   ... 15
threadIdx.y 0   t0  t1   ... t15   ┐ warp 0
            1   t16 t17  ... t31   ┘ lane 0～31
            2   t32 t33  ... t47   ┐ warp 1
            3   t48 t49  ... t63   ┘
            ...
           15   t240 ...     t255  warp 7 的后 16 条 lane
```

在这个二维 block 里，一维线程编号为 `t = threadIdx.y*16 + threadIdx.x`，`warp=t/32`、`lane=t%32`。这是线程编号与输出坐标的图，不是显存地址图。

第一个固定调试形状 `M=16,N=128,K=256` 使用 `grid=(8,1)`：8 个 block 分别负责 C 的 16 列；每个 block 在行方向覆盖 16 行。形状不是 16 的倍数时，边缘 block 会有不对应有效输出的线程。

```text
global A[row, 0:K] ─┐
                     ├→ 一个有效线程：逐项乘加到自己的 FP32 累加器
global B[0:K, col] ─┘                            │
                                                  ▼
                                      global C[row,col] 写一次
```

行和列由二维 block、二维 thread 坐标计算；先检查行列边界，再访问 A、B、C。越界线程可以提前返回，因为 V0 kernel 没有需要整个 block 参与的屏障。未来 shared-memory 版本不能不加分析地照搬这个提前返回。

## 3. 手算和语法

对于 V0a 的 `2×3` 乘 `3×2` 例子，block 0 中 `(threadIdx.x,threadIdx.y)=(1,1)` 对应 `row=1,col=1`。它初始化 `float` 累加器为 0，然后按 `p=0、1、2` 依次计算 `4×0、5×1、6×1`，得到 11 并写 C[1,1]。

你会用到的 C++ 结构：

```cpp
int value = 0;                         // 声明并初始化整数
if (condition) { /* 条件成立时执行 */ }
for (int p = 0; p < k; ++p) { /* 依次处理每个 p */ }
```

`blockIdx.x/y` 是当前 block 的 x/y 编号，`blockDim.x/y` 是每个 block 在该方向的线程数，`threadIdx.x/y` 是线程在 block 内的坐标。可参照已经完成的一维 vector add 和二维矩阵复制，自己推导全局行列。

## 4. 动手和测试

建议把模板复制成 `kernels/marlin_from_scratch/v0_gemm.cu` 再填写，以便保留原始模板。只填写 kernel 中的 TODO：

1. 计算当前线程对应的行、列。
2. 对输出行和列做边界检查。
3. 从 0 初始化 FP32 累加器，遍历全部 K；注意 A 和 B 的行优先步长不同。
4. 将累加结果写入对应 C 元素。

当前机器 PATH 中默认的 CUDA 11.5 与默认 GCC 编译这个模板时在 C++ 标准库头文件报错；已验证本机 CUDA 12.1 可以编译。完成后可执行：

```bash
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 \
    kernels/marlin_from_scratch/v0_gemm.cu -o /tmp/quant_gemm_v0
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_gemm_v0
```

上述 GPU 1 是示例；先检查当时的占用，再选择空闲 GPU。`CUDA_VISIBLE_DEVICES=1` 表示程序只看到宿主机的第 1 号 GPU；程序内部通常将它编号为逻辑 GPU 0。本次不修改系统驱动或其他项目环境。

模板测试的 `(M,N,K)` 为 `(2,2,3)`、`(1,1,1)`、`(5,7,9)`、`(17,19,23)` 和 `(16,128,256)`。输入是较小的整数值，以 FP32 表示；在这些具体形状下乘积和累加均可精确表示，因此测试要求与 CPU FP32 参考逐元素精确相等。**这不是一般 FP32 矩阵乘都应逐位相等的结论。**

模板未填写时会保留 `-999` 哨兵并故意报失败；这是提醒你尚未完成，不代表已测得有效 GEMM 结果。写完后告诉我，我会先 review，再检查边界和必要的 CUDA 工具。
