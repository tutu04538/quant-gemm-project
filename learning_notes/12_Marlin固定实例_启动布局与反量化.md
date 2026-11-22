# Marlin 固定实例：启动配置、线程布局与 warp 内反量化

这份文档只研究**一次实际运行中选中的一个 kernel 实例**。我们先从整个矩阵和 block 讲起，再进入一个 block 的 warp、一个 warp 的 lane，最后追踪 lane 取到的 packed 权重如何解码。

## 0. 固定例子与证据

本次运行使用同级 `vllm-from-scratch` 的参考实现，输入为：

```text
A：BF16 激活                [M,K] = [16,256]
Q：逻辑 4-bit 无符号权重    [K,N] = [256,128]
S：BF16 scale               [K/128,N] = [2,128]
Z：4-bit zero point         [K/128,N] = [2,128]
bias：BF16                  [N] = [128]
C：BF16 输出                [M,N] = [16,128]
```

数学语义是：

```text
W[k,n] = (Q[k,n] - Z[k//128,n]) * S[k//128,n]
C[m,n] = Σ(k=0..255) A[m,k] * W[k,n] + bias[n]
```

`//` 是向下取整。group 0 覆盖 K=0～127，group 1 覆盖 K=128～255；每个输出列 `n` 各有自己的 scale 和 zero。

**区分两种索引：** `Q[k,n]` 是逻辑权重；kernel 实际访问的是 repack 后的物理 qweight，不能把逻辑 `k,n` 直接当作物理数组下标。本文说明 bit 位序时会明确写成 `bit[高:低]`；地址偏移会注明单位。

本例 K、N 已满足当前路径的对齐要求，**没有 padding**。权重先在加载准备阶段完成 checkpoint → standard → Marlin repack；每次 GEMM 前向并不重做整层 repack。

2026-11-21 的实测使用 RTX 4090（SM89、128 SM），捕获到本节列出的 kernel 和 launch；结果与独立 FP32 反量化参考相比通过 `rtol=0.02, atol=0.05`，最大绝对误差约 0.01533。该检查只覆盖一次输入，不是性能测试。原始记录见 [实测记录](Marlin固定例子_实测记录.md)，[profiler trace](Marlin固定例子_profiler.json)。

下文的 **launch 是 profiler 实测**；block/warp/lane 的内部对应关系是对该实例的源码推导。另用固定 seed 重建逻辑 Q，逐一检查了四个有效 block 的 32,768 个 4-bit 值与实际 GPU repack 结果一致；这验证了物理位序映射，不等于在 GEMM kernel 内逐 lane 捕获寄存器。

## 1. kernel 启动配置

实际捕获到的实例可简写为：

```cpp
Marlin<BF16, U4, BF16, BF16,
       128, 1, 4, 8, false, 4, 8, false>
    <<<128, 128, 101376, stream>>>(...);
```

这里为了可读性把前四个类型 ID 写成了名称；实际 profiler 的模板名称是编码后的整数 ID。`101376` 字节是本次 trace 报告的 shared memory，第三个 launch 参数来自 host 侧动态 shared memory 配置。

| 参数 | 本例 | 作用 |
|---|---:|---|
| grid | `(128,1,1)` | 总共启动 128 个 CUDA block |
| block | `(128,1,1)` | 每 block 128 个线程 |
| warp 数 | 4 / block | 每 warp 32 个 lane |
| `thread_m_blocks` | 1 | M 方向单位 16 行，tile 高 16 |
| `thread_n_blocks` | 4 | N tile 宽 `4×16=64` 列 |
| `thread_k_blocks` | 8 | 一个 K tile 宽 `8×16=128` |
| `m_block_size_8` | false | 不使用 8 行的特殊 M tile |
| `stages` | 4 | shared pipeline 的 stage 配置 |
| `group_blocks` | 8 | 一个量化 group 包含 `8×16=128` 个 K |
| `is_zp_float` | false | zero point 使用 packed 整数表示 |

本次实际调用还使用 `has_bias=true`、`use_atomic_add=false`、`use_fp32_reduce=true`。这些是**运行时参数**，不是上面最后几个模板参数。

启动语法对应 [marlin.cu 中的发射点](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin.cu#L528)：

```cpp
kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(...);
```

四个参数依次表示：grid 的 block 数、每个 block 的线程数、每个 block 的动态 shared memory 字节数、CUDA stream。stream 可以理解为一条 GPU 工作队列，按该 stream 的顺序安排工作；这里不展开多 stream 并发。

## 2. 先画整个矩阵：block 负责什么

输出 C 高 16、宽 128；一个输出 tile 为 `16×64`，因此有两个输出 tile。每个 tile 都需要完整 K=0～255 的贡献，但本实例把 K 拆成两个 128 宽的部分：

```text
                        输出矩阵 C [16,128]
                     N=0                         N=128
               ┌───────────────────┬───────────────────┐
       M=0     │ 输出 tile 0       │ 输出 tile 1       │
               │ C[:, 0:64]        │ C[:, 64:128]      │
       M=16    └───────────────────┴───────────────────┘
                   ↑           ↑        ↑           ↑
                block 0     block 1   block 2     block 3
                K 0～127    K 128～255 K 0～127   K 128～255
```

| CUDA block | 输出的逻辑范围 | 计算的逻辑 K | group | 在跨 block 归约中的顺序 |
|---:|---|---|---:|---|
| 0 | C[0:16, 0:64] | 0～127 | 0 | 第二个：读先前部分和，完成写回 |
| 1 | C[0:16, 0:64] | 128～255 | 1 | 第一个：写 FP32 临时部分和 |
| 2 | C[0:16, 64:128] | 0～127 | 0 | 第二个：读先前部分和，完成写回 |
| 3 | C[0:16, 64:128] | 128～255 | 1 | 第一个：写 FP32 临时部分和 |
| 4～127 | 无有效 slice | — | — | 不处理有效输出 tile |

这里的“第一个、第二个”是锁约束的**归约次序**，不是保证 CUDA 调度器先启动哪个 block。block 0 和 block 1 对同一个 C 子块计算不同的 K 部分，不能简单各自把结果当完整 C。

例如 `C[2,70]` 在输出 tile 1：

```text
block 2：Σ(k=0..127)   A[2,k] × W[k,70]   ← group 0
block 3：Σ(k=128..255) A[2,k] × W[k,70]   ← group 1
                         ↓ 跨 block 合并，加 bias[70]
                      C[2,70]
```

### 为什么是这四个 block？

[调度源码](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L370)计算：

```text
M tile 高 = 16×1 = 16；M 方向组数 parallel = 1
N tile 宽 = 16×4 = 64；n_tiles = 128/64 = 2
K tile 宽 = 16×8 =128；k_tiles = 256/128 = 2
输出 M/N tile 总数 global_mn_tiles = 1×2 = 2
每 block 分配的 K-tile 工作量 iters = ceil(2×2/128) = 1
```

实际代码还将 `iters` 对齐 group；本例 group 与 K tile 同宽，对齐后仍是 1。每个 block 的初始位置：

```cpp
slice_col_par = (iters * blockIdx.x) / k_tiles;
slice_row     = (iters * blockIdx.x) % k_tiles;
```

这两行在 [init_slice](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L509)。代入 `iters=1,k_tiles=2`：block 2 得到 `(slice_col_par=1,slice_row=0)`，即第二个输出 tile、第一段 K；block 3 得到 `(1,1)`。block 4 得到输出 tile 编号 2，超出本例 0、1 两个有效 tile，所以它没有有效 slice。`slice_count` 和 `slice_idx` 的具体计算在 [init_part2_slice](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L428)。

## 3. 放大一个 block：4 个 warp 如何分 K

以 **block 0** 为例，它处理输出 `C[0:16,0:64]` 在 K=0～127 的部分和。每个 block 有 128 线程，线程号从 0 开始：

```text
                    block 0：K = 0～127
   ┌─────────────────────────────────────────────────────┐
   │ warp 0：threadIdx.x =  0～31，lane = 0～31，K  0～31 │
   │ warp 1：threadIdx.x = 32～63，lane = 0～31，K 32～63 │
   │ warp 2：threadIdx.x = 64～95，lane = 0～31，K 64～95 │
   │ warp 3：threadIdx.x = 96～127，lane = 0～31，K 96～127│
   └─────────────────────────────────────────────────────┘
                  4 个 warp 的部分和 → block 内归约
```

这个 K 范围是从当前实例的 shared 读取公式推导的；每个 warp 内还有两个 16 宽的步骤。以 `w` 表示 warp 编号、`s` 表示步骤（0 或 1），block 0 的逻辑 K 子范围起点为 `32*w + 16*s`：

| warp | step 0 | step 1 |
|---:|---|---|
| 0 | K 0～15 | K 16～31 |
| 1 | K 32～47 | K 48～63 |
| 2 | K 64～79 | K 80～95 |
| 3 | K 96～111 | K 112～127 |

block 1 的对应范围整体加 128；block 2 与 block 0 的 K 相同但输出列加 64；block 3 与 block 1 的 K 相同但输出列加 64。

在一维 block 中，`threadIdx.x = 32*w + lane`。一个 warp 的 32 条 lane **协作**处理一个 MMA 片段，不是每个 lane 都独立算完整 16×8 小矩阵。warp 对应 K 子范围的依据是 [A shared 读取索引](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L600)与 [预计算的读取地址](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L731)。

## 4. 线程、数据与内存流动

看单个有效 block 的一段 K：

```text
GPU global memory
 ├─ A 的 BF16 tile：16×128
 ├─ repack 后 Q：逻辑对应 128×64 个 4-bit 值
 ├─ 本 group 的 64 个 BF16 scale
 └─ 本 group 的 64 个 4-bit zero
                  │ 128 个线程合作搬运
                  ▼
block 私有 shared memory
 ├─ shared A：XOR 重排后的 A tile
 ├─ shared B：repack 物理顺序的 packed Q
 ├─ shared S：重排后的 scale
 └─ shared Z：重排、打包后的 zero
                  │ warp/lane 各取各自片段
                  ▼
线程寄存器：A fragment、packed Q、zero、scale
                  │ 解码 Q、减 zero、乘 scale
                  ▼
BF16 B fragment ──与 BF16 A fragment 一起──→ warp MMA
                  ▼
每个线程持有自己的 FP32 accumulator fragment
                  ▼
block 内 4 warp 归约 → 两个 block 的 FP32 部分和归约 → BF16 输出
```

### global 到 shared：A 的具体线程例子

源码把 A 指针视为 `int4*`；**一个 int4 是 16 字节，即 8 个 BF16**。不是 4-bit 权重。对 `t=threadIdx.x`、搬运轮次 `i=0 或 1`、当前 K slice `kr=0 或 1`：

```text
A 的逻辑行 = t // 16 + 8*i
读取的 K 起点 = 128*kr + 8*(t % 16)
每次读取 8 个连续 BF16
```

例如 block 0 的 thread 0：第一次读 `A[0,0:8]`，第二次读 `A[8,0:8]`。block 1 的 thread 0：分别读 `A[0,128:136]` 和 `A[8,128:136]`。

128 线程各读两次，每次 8 个 BF16，共 `128×2×8=2048` 个元素，正好是 `16×128` 的 A tile。数据写 shared 时还做 XOR 地址重排；这个重排改变 shared **物理位置**，不改变 A 的逻辑坐标。参见 [A 索引](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L592)、[XOR 变换](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L720)。

### Q、scale 和 zero 的搬运

一个 Q tile 的大小是 `128×64×4 bit = 4096 byte`。本实例每个线程搬两个 16-byte 的 qweight 物理向量，共 `128×2×16=4096 byte`。前 8 个线程搬 scale：8×16 byte = 64 个 BF16；前 2 个线程搬 zero：2×16 byte = 64 个 4-bit 值。

scale、zero 已经按 kernel 所需顺序重排；这里的“前 8 个线程”是物理搬运职责，不意味着它们搬的每个值在逻辑 N 上按原序排列。具体搬运在 [fetch_to_shared](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L842)。

## 5. 先弄清逻辑权重与 packed 权重，再看 lane 分工

本节我们从一个非常具体的小区域开始：**block 0、warp 0 的第 0 个 K 步骤**。它涉及逻辑 Q 的 `K=0～15、N=0～63`。先把这个区域画清楚，再推广到其他 warp 和 block。

### 5.1 逻辑 Q 和物理 packed Q 是两张不同的图

逻辑矩阵 Q 表示“第 k 个输入通道与第 n 个输出通道之间的权重是多少”。例如 `Q[8,0]` 是一项 4-bit 的逻辑值：

```text
逻辑 Q，形状 [K,N] = [256,128]；这里只画左上角的一部分
                         逻辑输出通道 n
                   0     1    ...    7     8    ...   15
              ┌───────────────────────────────────────────┐
逻辑输入 k=0  │ Q[0,0]  Q[0,1]      ...   Q[0,8]         │
          k=1 │ Q[1,0]  Q[1,1]      ...   Q[1,8]         │
          ... │                                           │
          k=8 │ Q[8,0]  Q[8,1]      ...   Q[8,8]         │
          k=9 │ Q[9,0]  Q[9,1]      ...   Q[9,8]         │
          ... │                                           │
              └───────────────────────────────────────────┘
```

物理存储需要把多个 4-bit 值装进 32-bit 容器。这里要区分**准备过程中的两种布局**：

```text
standard qweight：形状 [K/8,N] = [32,128] 个 int32
  沿逻辑 K 打包：standard_qweight[0,0] 包含 Q[0,0]～Q[7,0]
         │
         │ Marlin repack：改变这些 4-bit 值的排列
         ▼
Marlin qweight：形状 [K/16,2N] = [16,256] 个 int32
  一个 int32 仍装 8 个 4-bit 值，但这 8 个值不再是逻辑上连续的 8 个 K
```

上面两个形状都是**物理 int32 数组的形状**，不是逻辑 Q 的形状；内存里没有 256×128 个 int32。checkpoint 本身还有自己的打包轴和位序，本节从已转换好的 standard 到 Marlin 布局，不把 checkpoint 位序与下面的 Marlin 位序混用。

### 5.2 看 Marlin 中第一个 32-bit 容器

先看物理 `marlin_qweight[0,0]`。它是一个 32-bit word，其中每 4 bit 对应一个逻辑 Q：

```text
物理：marlin_qweight[0,0]  ── 一个 32-bit word

bit 位置：  31:28   27:24   23:20   19:16   15:12   11:8    7:4     3:0
           ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
逻辑位置： │Q[9,8] │Q[1,8] │Q[9,0] │Q[1,0] │Q[8,8] │Q[0,8] │Q[8,0] │Q[0,0] │
           └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
                                                                          ↑
                                                                      最低 4 bit
```

因此，如果只看逻辑矩阵，**同一个物理 word** 把四个 K 行、两个 N 列的值收在一起：

```text
                逻辑 N=0        逻辑 N=8
             ┌─────────────┬─────────────┐
逻辑 K=0     │ Q[0,0]      │ Q[0,8]      │
逻辑 K=1     │ Q[1,0]      │ Q[1,8]      │
逻辑 K=8     │ Q[8,0]      │ Q[8,8]      │
逻辑 K=9     │ Q[9,0]      │ Q[9,8]      │
             └─────────────┴─────────────┘
                   这 8 个逻辑值 → 物理 qweight[0,0]
```

这一组并不是二维矩阵里的一个连续长方形内存片段；图只标识 8 个逻辑坐标，物理上它们被打进一个 word 的不同 bit。上游源码在 [repack 的 vals 和 pack_idx](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/gptq_marlin_repack.cu#L179) 中完成这种组合。

本次固定 seed 的实际 repack 输出中，这个 word 的 32-bit 位模式是 `0x2DBAEABA`。十六进制通常把**高位写在左边**：

```text
bit 31:28 → 3:0：    2   D   B   A   E   A   B   A
                                      ↑                   ↑
                                 Q[8,8] = 14          Q[0,0] = 10
```

这个例子只是确认实际 bit pattern 和坐标对应；不能把 `0x2DBAEABA` 推广到其他随机输入。

### 5.3 再看物理 qweight 的第一行

对这个 `K=0～15、N=0～63` 的 tile，Marlin qweight 第一行的前 128 个 int32 这样安排：

```text
Marlin qweight，物理 int32 数组 [16,256]

物理行 0：逻辑 K=0～15 的 repack 结果
           ┌──────────────────────── N=0～63 的 tile ────────────────────────┐
物理列：    0       1       2       3      4       5       6       7    ...
         ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
内容：   │lane 0 │lane 0 │lane 0 │lane 0 │lane 1 │lane 1 │lane 1 │lane 1 │ ...
         │word 0 │word 1 │word 2 │word 3 │word 0 │word 1 │word 2 │word 3 │
         └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
           ←────────── 16 byte int4 ──────────→

物理列 0～127 对应本次 N=0～63 的 tile；128～255 对应 N=64～127。
物理行 1 对应逻辑 K=16～31，物理行 2 对应逻辑 K=32～47，依此类推。
```

图中的“lane”是本实例 GEMM 消费该 word 的 lane；repack 时也有自己的线程，它们是另一次 kernel 的线程，不要把两次执行中的线程当成同一个。一个 `int4` 是 16 字节，包含四个连续 int32；在当前 block 0、warp 0、step 0 中，lane 0 读物理列 0～3，lane 1 读 4～7，直到 lane 31 读 124～127。这些是**物理数组下标**，并不表示 lane 0 只负责逻辑输出列 0～3。

### 5.4 把每条 lane 对应的逻辑区域画出来

现在固定 **block 0、warp 0、step 0、word 0**。下图中每个格子代表一个逻辑 `Q[k,n]`，格子里的数字是**负责读取并解码这个 Q 的 lane 编号**。例如数字 `00` 表示 lane 0，不是 Q 的数值。

```text
                  逻辑 N 列（不是物理 qweight 的列）
         0  1  2  3  4  5  6  7 │  8  9 10 11 12 13 14 15
       ──────────────────────────┼─────────────────────────
K= 0  00 04 08 12 16 20 24 28  │  00 04 08 12 16 20 24 28
K= 1  00 04 08 12 16 20 24 28  │  00 04 08 12 16 20 24 28
K= 2  01 05 09 13 17 21 25 29  │  01 05 09 13 17 21 25 29
K= 3  01 05 09 13 17 21 25 29  │  01 05 09 13 17 21 25 29
K= 4  02 06 10 14 18 22 26 30  │  02 06 10 14 18 22 26 30
K= 5  02 06 10 14 18 22 26 30  │  02 06 10 14 18 22 26 30
K= 6  03 07 11 15 19 23 27 31  │  03 07 11 15 19 23 27 31
K= 7  03 07 11 15 19 23 27 31  │  03 07 11 15 19 23 27 31
K= 8  00 04 08 12 16 20 24 28  │  00 04 08 12 16 20 24 28
K= 9  00 04 08 12 16 20 24 28  │  00 04 08 12 16 20 24 28
K=10  01 05 09 13 17 21 25 29  │  01 05 09 13 17 21 25 29
K=11  01 05 09 13 17 21 25 29  │  01 05 09 13 17 21 25 29
K=12  02 06 10 14 18 22 26 30  │  02 06 10 14 18 22 26 30
K=13  02 06 10 14 18 22 26 30  │  02 06 10 14 18 22 26 30
K=14  03 07 11 15 19 23 27 31  │  03 07 11 15 19 23 27 31
K=15  03 07 11 15 19 23 27 31  │  03 07 11 15 19 23 27 31
```

读图示例：

- **lane 0**：图中 8 个 `00`，即 `Q[0,0]、Q[1,0]、Q[8,0]、Q[9,0]` 和同样 4 行在 N=8 的值；这正是上一小节的物理 `marlin_qweight[0,0]`。
- **lane 1**：8 个 `01`，位于 K=2、3、10、11，N=0、8；对应物理 `marlin_qweight[0,4]`，因为 lane 1 的四个 word 从物理列 4 开始。
- **lane 4**：8 个 `04`，位于 K=0、1、8、9，N=1、9；对应物理 `marlin_qweight[0,16]`。
- **lane 31**：8 个 `31`，位于 K=6、7、14、15，N=7、15；对应物理 `marlin_qweight[0,124]`。

图里每个 lane 负责 8 个逻辑 Q；32 条 lane 共负责 `32×8=256` 个权重，刚好覆盖图中的 `16×16`。每个 lane 负责的是图中分散的 8 个格子，不是从左到右连续的一小条。

### 5.5 一条 lane 还有另外三个 word

刚才只画了每条 lane 的 **word 0**。一个 int4 有 4 个 word，因此 block 0、warp 0、step 0 实际覆盖 `K=0～15、N=0～63`：

```text
                         逻辑 N 方向
                 0～15       16～31       32～47       48～63
              ┌───────────┬───────────┬───────────┬───────────┐
 K=0～15      │ word 0 的 │ word 1 的 │ word 2 的 │ word 3 的 │
              │ 32 lane 图│ 32 lane 图│ 32 lane 图│ 32 lane 图│
              └───────────┴───────────┴───────────┴───────────┘
                上面已画      同样的 lane 分布，只是 N 坐标整体右移
```

以 lane 0 为例：word 0 涉及列 `0、8`，word 1 涉及列 `16、24`，word 2 涉及列 `32、40`，word 3 涉及列 `48、56`；四个 word 都涉及 K=0、1、8、9。四个 word 在物理存储中连续，但它们对应的逻辑 N 坐标并不连续。

同一 warp 的 **step 1** 使用物理 qweight 的下一行，覆盖 K=16～31；lane 图的形状完全相同，只需把图中的每个 K 坐标加 16。四个 warp 的两个 step 联合覆盖 block 0 的 K=0～127 和 N=0～63：

```text
                 每个小格表示一个 16×16 逻辑 Q 区域
                      N 0～15  16～31  32～47  48～63
                    ┌────────┬────────┬────────┬────────┐
warp 0  step 0 K 0  │ 32 lane│ 32 lane│ 32 lane│ 32 lane│
        step 1 K16  │ 同一映射，K 整体加 16              │
warp 1  step 0 K32  │ 同一映射，K 整体加 32              │
        step 1 K48  │ 同一映射，K 整体加 48              │
warp 2  step 0 K64  │ 同一映射，K 整体加 64              │
        step 1 K80  │ 同一映射，K 整体加 80              │
warp 3  step 0 K96  │ 同一映射，K 整体加 96              │
        step 1 K112 │ 同一映射，K 整体加 112             │
                    └────────┴────────┴────────┴────────┘
```

图中的 K 号码表示每个 16 行小块的起点，**不是 CUDA 线程号**。block 1 的上述 K 全部加 128；block 2 的 N 全部加 64；block 3 的 K 加 128、N 加 64。

### 5.6 最后再把图写成可核对的公式

图理解后，如果需要算任意 lane，用下面的符号：

```text
nr = N tile 编号（block 0、1 为 0；block 2、3 为 1）
kr = K slice 编号（block 0、2 为 0；block 1、3 为 1）
w = warp 编号 0～3；l = lane 编号 0～31
s = 该 warp 的 step 0 或 1；j = int4 中 word 0～3

k0 = 128*kr + 32*w + 16*s + 2*(l % 4)
n0 =  64*nr + 16*j    + (l // 4)
```

这个 word 对应四行 `k0、k0+1、k0+8、k0+9` 与两列 `n0、n0+8`；其 nibble 从低到高是：

```text
Q[k0,n0],   Q[k0+8,n0],   Q[k0,n0+8],   Q[k0+8,n0+8],
Q[k0+1,n0], Q[k0+9,n0],   Q[k0+1,n0+8], Q[k0+9,n0+8]
```

对应的**物理** int32 位置（本例无 padding）是：

```text
marlin_qweight 的物理行 = 8*kr + 2*w + s
marlin_qweight 的物理列 = 128*nr + 4*l + j
```

例如 block 0 的 warp 0、lane 1、step 0、word 0：物理位置是 `[0,4]`；逻辑上 `k0=2,n0=0`，对应 K=2、3、10、11 与 N=0、8，恰好是图里的 8 个 `01`。这两组索引服务于不同目的，不能互换。

这些映射由 [shared→寄存器读取](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L933)和 [repack 位序](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/gptq_marlin_repack.cu#L122)推导，并已与本次参考实现实际 repack 输出逐值核对。四个有效 block 的全部 `(w,l,s,j)` 总计 `4×4×32×2×4×8=32768` 个权重；kernel 运行中的寄存器内容尚未逐 lane 插桩。

## 6. 每个 warp 如何反量化？

四个 warp 使用**同一套指令和位操作**，区别是读到的 K 范围不同；每个 warp 的 32 个 lane 各自对寄存器里的 packed word 执行解码，再作为一个 warp 协作执行 MMA。

```text
同一 warp 的 lane 0                 lane 1              ... lane 31
    │ 各自读取自己的 int32 word        │                       │
    ▼                                ▼                       ▼
提取 nibble 并构造 BF16 对       提取自己的 nibble        提取自己的 nibble
    │                                │                       │
    ├─ 读取该列对应的 zero           ├─ 读取自己的 zero      ├─ ...
    ├─ BF16 减 zero                  ├─ BF16 减 zero         ├─ ...
    └─ BF16 乘 scale                 └─ BF16 乘 scale        └─ ...
    │                                │                       │
    └────────────────────── 32 lane 合作 MMA ────────────────┘
                                    │
                        每 lane 保留自己的 FP32 累加片段
```

### 6.1 一个 int32 怎样变成 8 个 BF16 权重？

先不要看位运算，先回答：**`frag_b_quant` 是什么？**

它是 Marlin kernel 给**每个线程**准备的一个局部变量，用来暂存这个线程从 shared memory 读取到的、仍然压缩着的 B/Q 权重片段。`frag` 是 fragment（片段）的缩写，`b` 指矩阵乘的右侧权重，`quant` 表示数据还处于量化打包状态。

它的数据来源和去向是：

```text
global memory                 block 的 shared memory           当前线程的局部变量
repack 后的 qweight ──搬运──→ sh_b 中的 packed Q ──读取──→ frag_b_quant
                                                               │
                                       取一个 int32 → 解码成 BF16 片段
                                                               │
                                       减 zero、乘 scale → 参与 MMA
```

这里的“片段”只是从更大矩阵里取出的一小部分。`frag_b_quant` 的声明见 [marlin_template.h](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L769)：

```cpp
I4 frag_b_quant[2][b_thread_vecs];
```

- `I4` 是含 4 个 `int` 的向量类型；本例每个 `int` 是 32 bit，所以一个 `I4` 共 16 字节。
- 对本次 4-bit 权重，`b_thread_vecs=1`。因此每个线程的逻辑局部变量可以画成两个槽位：`frag_b_quant[0][0]` 和 `frag_b_quant[1][0]`；每个槽位都有 4 个 int32 word。
- 第一个下标的 0、1 是交替使用的**两个槽位**，不是 warp 编号，也不是 block 编号。源码先把下一片读入一个槽位，再使用另一个槽位的片段计算；局部变量通常由编译器分配到寄存器，最终分配以编译结果为准。
- 每个线程都有自己的 `frag_b_quant`；它不是一个在 block 内由所有线程共同使用的 shared 数组。

例如 block 0、warp 0、lane 0 第一次取到的槽位可以理解为：

```text
这个线程的 frag_b_quant[0][0]：一个 I4 = 4 个 int32

             j=0             j=1             j=2             j=3
       ┌──────────────┬──────────────┬──────────────┬──────────────┐
       │ 0x2DBAEABA   │ 另一个 word  │ 另一个 word  │ 另一个 word  │
       └──────────────┴──────────────┴──────────────┴──────────────┘
             ↑
       本节只展开这个 word
```

这个具体 word 来自上一节的 `marlin_qweight[0,0]`。实际 repack 输出中的位模式已经核对；它进入这个线程的哪个局部变量，是对 [shared→寄存器加载代码](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L933)的静态推导，并非运行时逐寄存器抓取。

接下来，`matmul` 内的 `j=0..3` 循环依次处理 I4 中的 4 个 word；**本节先只看 `j=0`**。

[matmul 中的实际路径](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L1222)每次从 `frag_b_quant` 取一个 int32。对于 U4：

```cpp
b_quant_0 = frag_b_quant[k2][0][j];
b_quant_1 = b_quant_0 >> 8;
dequant_data(b_quant_0, ...frag_b0...);
dequant_data(b_quant_1, ...frag_b1...);
```

`k2 = k % 2` 选择两个寄存器缓冲之一；这里的 `k` 是内层步骤编号，不是整个 Q 的逻辑 K 坐标。`b_quant_1` 右移 8 bit，是为了让第二次解码取到原 word 中另外四个 nibble。

用 `k2=0,j=0` 逐项读这几行：

```cpp
b_quant_0 = frag_b_quant[0][0][0];  // 槽位 0、第 0 个 I4、第 0 个 int32
b_quant_1 = b_quant_0 >> 8;         // 把同一个 word 右移 8 位
```

`b_quant_0` 和 `b_quant_1` 都是临时的 32-bit 整数；它们不是两个独立从 global memory 读出的权重。随后两次 `dequant_data` 分别将这两个整数中需要的 nibble 解释成 BF16。`reinterpret_cast<scalar_32bit_t*>(&frag_b0)` 把输出片段的地址按目标类型解释，以便函数往该片段里写；它本身不执行数值转换。

对我们的这个 word（最高 nibble 写左边）：

```text
b_quant_0 = 0x2DBAEABA
                   └──── 8 个 nibble；每个 nibble 是一个 4-bit 的 Q

b_quant_1 = 0x002DBAEA  （b_quant_0 右移 8 bit）
```

本例最高位为 0，按 32-bit 右移后高位补 0；对于其他位模式，源码中 `int` 的右移语义及编译器行为应结合具体平台考虑，不能仅凭这个正数例子概括所有情况。

当前 BF16/U4 使用 [dequant.h 的特化](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/dequant.h#L174)：

```cpp
MASK = 0x000f000f;
EX   = 0x43004300;
lo = (q & MASK) | EX;  // 源码用 lop3 实现
q >>= 4;
hi = (q & MASK) | EX;
```

一个 32-bit `lo` 或 `hi` 被解释为两个 BF16：每个 16-bit 半字的 `0x4300` 表示 BF16 数值 128；放入低四位后，变成 `128 + nibble`。这是一种位级转换技巧，**不是**在数学权重中额外加了 128：后续 zero 也采用相同方式构造，两者相减时偏移相消。

以 `b_quant_0 = 0x2DBAEABA` 手算，`MASK=0x000F000F` 只保留 bit `[19:16]` 和 `[3:0]`：

```text
第一次 dequant_data：输入 b_quant_0 = 0x2DBAEABA

lo = (0x2DBAEABA & 0x000F000F) | 0x43004300
   =  0x000A000A                | 0x43004300
   =  0x430A430A  → 两个 BF16： (138,138) = (128+10,128+10)

输入先右移 4 bit 后再执行同样操作：
hi = 0x430B430B   → 两个 BF16： (139,139) = (128+11,128+11)

第二次 dequant_data：输入 b_quant_1 = 0x002DBAEA
lo = 0x430D430A   → 低 16 bit 对应 138，高 16 bit 对应 141
hi = 0x4302430E   → 低 16 bit 对应 142，高 16 bit 对应 130
```

因此，一个 packed int32 经**两次**解码调用，得到 `frag_b0` 和 `frag_b1` 两个片段；每个片段有两个 BF16x2，合计 8 个 BF16 数值。此时这些值仍带有 128 偏移，还不是最终反量化权重。下面才是与 zero 和 scale 的组合：

```text
一个 int32：8 个 4-bit Q
       │ 两次 dequant_data
       ▼
frag_b0：4 个 BF16（带 128）  +  frag_b1：4 个 BF16（带 128）
       │ 每个 Q 片段减去匹配的、也带 128 的 zero
       ▼
       Q - Z
       │ 乘以匹配的 BF16 scale
       ▼
       8 个可供 MMA 的 BF16 权重值
```

按第 5 节的 `k0,n0`，第一次调用处理列 n0，第二次调用处理列 n0+8：

```text
原 word 的第一个解码调用：
  lo → (128+Q[k0,n0],   128+Q[k0+1,n0])
  hi → (128+Q[k0+8,n0], 128+Q[k0+9,n0])

原 word 右移 8 bit 后的第二次解码调用：
  lo → (128+Q[k0,n0+8],   128+Q[k0+1,n0+8])
  hi → (128+Q[k0+8,n0+8], 128+Q[k0+9,n0+8])
```

这里 `(x,y)` 是一个 BF16x2 形式的寄存器值，表示两个 BF16；不是两个 CUDA 线程。

### 6.2 zero 和 scale 怎样匹配到列？

对于 warp 内 lane `l`，令 `r = l // 4`。同一个 warp 的连续四条 lane 有相同 r；它们处理不同的 K 局部位置，但本次选择同一组输出列：

```text
j=0 → 列 64*nr + r，      以及该列 +8
j=1 → 列 64*nr + 16 + r， 以及该列 +8
j=2 → 列 64*nr + 32 + r， 以及该列 +8
j=3 → 列 64*nr + 48 + r， 以及该列 +8
```

在本例 `tb_n_warps=1`，源码中 scale 的 shared **int4** 读取索引和 zero 的 shared **int32** 读取索引均可化简成 `l//4`；虽然数值相同，**单位不同**：

```text
scale: shared S 的 int4[l//4] → 8 个 BF16 scale
zero : shared Z 的 int32[l//4] → 8 个 4-bit zero
```

scale 经重排后，这 8 个 BF16 对应逻辑列：

```text
64*nr + r + [0, 8, 16, 24, 32, 40, 48, 56]
```

zero word 的 nibble 从低到高对应逻辑列：

```text
64*nr + r + [0, 16, 32, 48, 8, 24, 40, 56]
```

两种物理位序不同，随后源码分别解码、选择，使每个 B fragment 使用相应列的 Z 和 S。它们都使用当前 block 的 group `kr`：block 0/2 用 group 0，block 1/3 用 group 1。已将两个 group、全部 128 列的上述物理重排结果与实际准备后的张量逐项核对；读取和寄存器内对应关系仍是源码推导。可对照 [scale/zero 重排](../third_party/vllm_marlin/minivllm/layers/quantization/marlin.py#L42)、[shared 读取索引](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L661) 和 [zero 加载](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L1089)。

### 6.3 减 zero、乘 scale、参与 MMA

该特化的 `dequant_skip_flop=true`；U4、整数 zero 的这条 BF16 路径在 [matmul](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L1244)中依次执行：

```text
构造含 128 偏移的 Q BF16 对
       ↓
减去含同样 128 偏移的 Z BF16 对：__hsub2
       ↓
乘以相应 BF16 scale：__hmul2
       ↓
得到供 MMA 使用的 BF16 权重 fragment
```

假设某个对应位置 `Q=11,Z=7,S=0.25`：`(128+11)-(128+7)=4`，再乘 `0.25` 得 `1`。这只是手算值，不声称随机输入某位置恰好如此。乘法发生在 BF16 类型并产生舍入；后续 MMA 是 BF16 输入、FP32 累加。因此数学公式有助理解，但不能假设所有中间值都按 FP32 计算。

接下来 warp 的 32 条 lane 协作执行 [BF16 MMA](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_mma.h#L68)：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
逻辑上：A[16,16] × B[16,8] + FP32 accumulator[16,8]
```

`m16n8k16` 是一条指令的数学片段，不是完整的 16×64 输出 tile；完成一个 warp 的工作需要多条 MMA。A/B fragment 的**完整逐 lane 寄存器表**在这次分析中尚未通过设备端插桩或独立指令布局验证，因此不把 packed word 的 lane 映射直接冒充为 MMA 操作数的完整映射。

## 7. 反量化以后：结果如何完成

```text
同一 block：warp 0 / 1 / 2 / 3 的 K 局部和
               ↓ shared memory + block 内同步
         一个 block 的 FP32 部分和
               ↓
block 1 → C_tmp ─┐        block 3 → C_tmp ─┐
block 0 ─────────┴→ C[:,0:64]    block 2 ─┴→ C[:,64:128]
               ↓ 锁控制次序、FP32 临时归约
               ↓ 最后一个 slice 加 bias
               ↓ BF16 写回
```

同一个 block 的四个 warp 先在 `thread_block_reduce` 中通过 shared memory 合并；之后本例的两个 block 因 `slice_count=2`、`use_atomic_add=false`、`use_fp32_reduce=true`，通过锁和 FP32 临时缓冲合并。最终 slice 写输出。具体见 [block 内归约](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L1406) 和 [跨 block 归约调用](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L2001)。

这里存在两类“部分和”，不要混淆：**warp 部分和**是一个 block 内四段 K 的结果；**block 部分和**是整个 128 宽 K slice 的结果。最终输出路径会进行 BF16 转换并按代码顺序加 bias，因此与 CPU 端全 FP32 计算后再转换不必逐位相同。

## 8. 本例的局限

- `stages=4` 表示模板的 pipeline 配置。本例每个有效 block 只有一个 K tile，不能据此声称观察到了四个有效 tile 的搬运与计算重叠。
- profiler 确认 kernel 特化、grid 和 block；本篇的 block/warp/lane 细节依赖源码推导，其中 packed Q 坐标已与实际 repack 结果逐值验证，但没有逐 lane 抓取 GEMM 寄存器。
- 当 M/N/K、group、dtype 或设备变化时，dispatch、调度和布局都可能改变，本文数值不可直接套用。
