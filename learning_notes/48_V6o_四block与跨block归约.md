# V6o：四个 block 与跨 block 部分和

## 1. 为什么需要这一节

V6n 用两个 block，每个 block 串行算两个 K group；虽然得到了完整输出，但 block 分工与固定参考实例不同。本节在你完成的 V6m 解码和 MMA 基础上，改成**四个 block，每个 block 负责一个 N64 tile 和一个 K128 group**，再实现跨 block 合并和 BF16 写回。

已实现的代码与测试：[v6o_four_block_template.cu](../kernels/marlin_from_scratch/v6o_four_block_template.cu)。保持固定形状 `A[16,256]`、逻辑 `Q[256,128]`、group=128、输出 `C[16,128]` 和 BF16 bias；物理 Q/Z/S 与 V6m/n 相同。没有 padding 与 act-order。

## 2. 布局：4 个 block 与 2 个输出 tile

```text
                  group 0：K 0..127        group 1：K 128..255
                ┌─────────────────────┬─────────────────────┐
N tile 0        │ block 0             │ block 1             │
N 0..63         │ nr=0 kr=0           │ nr=0 kr=1           │
                ├─────────────────────┼─────────────────────┤
N tile 1        │ block 2             │ block 3             │
N 64..127       │ nr=1 kr=0           │ nr=1 kr=1           │
                └─────────────────────┴─────────────────────┘

block 0 与 1 都贡献 C[:,0..63]；block 2 与 3 都贡献 C[:,64..127]。
每 block 的 128 个线程 = 4 warp × 32 lane。每 warp 计算其 group 的连续 32 个 K：
warp 0: kr*128+0..31       warp 1: kr*128+32..63
warp 2: kr*128+64..95      warp 3: kr*128+96..127
```

`nr=blockIdx.x/2`、`kr=blockIdx.x%2`。Q/Z/S 的物理地址沿用 V6m/V6n；每 lane 解码 4 个 BF16x2，分别提供给两条 N8 MMA。每个 warp 得到同一 16×64 输出区域的一份局部和。

## 3. 为什么需要跨 block 共享数据

以 N tile 0 的 `C[m,n]` 为例：

```text
block 0：four warp 的 FP32 和 P0[m,n] = sum K 0..127
block 1：four warp 的 FP32 和 P1[m,n] = sum K 128..255

最终值：BF16(BF16(P0 + P1) + bias[n])
```

各 block 的寄存器和 shared memory 均只能由本 block 访问，所以必须经 global memory 交换；`__syncthreads()` 只能同步一个 block，无法等待另一个 block。

本节为每个 N tile 准备一个全局 `locks[nr]` 和一个 FP32 临时区域 `tmp[m, n]`。每次调用前锁必须初始化为 0；并发调用不能复用同一个 `tmp` 和 `locks`。例子在同一 stream 连续运行时，每次最终 block 将锁复位，kernel 结束后才能复用。

## 4. 两级归约与时序

```text
                    N tile 0，lock[0] 初始为 0

block 0 (kr=0)                                      block 1 (kr=1)
4 warp 各自 MMA                                     4 warp 各自 MMA
    │                                                    │
写 shared part[warp][16][64]                         写各自 shared part
    │  __syncthreads()                                   │  __syncthreads()
warp 0 将四份相加得到 P0                            等 lock[0] == 1 (acquire)
    │                                                    │
等待 lock[0] == 0 (acquire)                         读取 tmp 中 P0
    │                                                    │
写 global tmp[:,0..63]                              P1 + P0 → diagnostic
    │                                                    │
全 block 同步，release：lock[0] 从 0 → 1            转 BF16，加 BF16 bias，写 C
                                                         │
                                                   全 block 同步，lock[0] 复位 0
```

图中简写了代码的实际次序：block 0 在取得锁后才由 warp 0 汇总并写入 tmp；block 1 也在取得锁后汇总自己的四 warp 部分和。`lock[1]` 独立控制 block 2 与 3。两个 N tile 的 tmp 和输出地址不重叠。

锁等待由 lane 0（`threadIdx.x==0`）执行，之后 `__syncthreads()` 让该 block 的其他线程等待 lane 0；输出合并由 warp 0 的 32 lane 分配完成。释放前的 `__syncthreads()` 确保本 block 的相关线程完成写入。不要把 `__syncwarp()` 当作等待其他 block 的机制。

## 5. acquire / release 是什么？

仅看到 `lock==1` 还需要确保也能正确读取之前发布的 `tmp`。这个教学实现借鉴参考的 acquire/release 次序：

```text
生产者 block 0：写 tmp → block 同步 → fence.acq_rel.gpu → 原子增加锁
消费者 block 1：ld.global.acquire.gpu 看到锁为 1 → block 同步 → 读 tmp
```

- `ld.global.acquire.gpu.b32`：读取全局锁，并提供 GPU 范围的 acquire 语义；`do ... while` 反复读直到等于自己的 `kr`。
- `fence.acq_rel.gpu`：发布前的 GPU 范围内存顺序屏障。
- `red.relaxed.gpu.global.add.s32`：对锁原子加 1；与前面的 fence 配合。
- 内联 PTX 的 `"memory"` 告诉编译器这些汇编涉及内存，避免编译器跨汇编随意重排相关内存操作；它本身不是 GPU 同步指令。

源码参考：[barrier_acquire / barrier_release](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L177)、[global_reduce_fp32](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L1582)、[调用位置](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L2003)。本例把锁协议的关键指令用于固定四 block；tmp 是普通行列顺序，先将 warp 和 group 分别归约，且在输出写完后复位锁。**它不是上游 fragment 布局、完整锁生命周期或完整 slice 调度的原样实现**。忙等可能依赖 block 能够取得执行资源，不能不加分析地推广到任意 grid 或作为通用全局 barrier。

## 6. 手算索引

`blockIdx.x=3`：`nr=1, kr=1`，它处理 N=64..127、K=128..255。`warp=2` 负责 K=192..223；它的第一个 step 是 K=192..207。对于 `lane=5, j=0`，`g=1`：

```text
Qphys 行 = 8*1 + 2*2 + 0 = 12
Qphys 列 = 128*1 + 4*5 + 0 = 148
Zphys    = [1, 8+1] = [1,9]
Sphys    = [1, 64+8*1+0]、[1,73]
```

对应逻辑两列 N=65、73；物理列 148 不等于逻辑列。group 1 对这两列使用同组 zero 和 scale。

## 7. 测试与范围

```bash
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo kernels/marlin_from_scratch/v6o_four_block_template.cu -o /tmp/quant_v6o_four
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v6o_four
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-12.1/compute-sanitizer/compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/quant_v6o_four
```

CPU 从逻辑 Q/Z/S 建立独立权重参考，按照每 group 内 4 warp 求和、再合并 group 的顺序计算 FP32；最后转换 BF16、加 BF16 bias。3 seed × 3 模式含随机 Q/Z、0/15 极值、零 bias、非精确 scale；覆盖 N=63/64 和 K=127/128。FP32 容差 `0.002 + 0.0001*abs(ref)`，BF16 容差 `0.002 + 0.01*abs(ref)`，并统计 BF16 逐位差异。每组连续启动 5 次检查复用，结束后检查锁均为 0。

2026-11-25 在 RTX 4090、CUDA 12.1、SM89：9 组均 FP32 错误 0、最大误差 0、BF16 容差错误 0、逐位不同 0；memcheck、racecheck、synccheck 无报告。racecheck 主要检查 shared memory 危害，这些测试**不能单独证明全局锁协议在所有调度和硬件上的正确性**。本次没有做上游 kernel 的运行时差分或 benchmark。
