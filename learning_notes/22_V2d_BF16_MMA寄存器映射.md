# V2d：手动组装 BF16 MMA 的 lane 寄存器

## 1. 目标与边界

使用一个 warp 执行一条 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`：

```text
A：BF16 [16,16] × B：BF16 [16,8] → C：FP32 [16,8]
```

为聚焦 lane→逻辑元素，[模板](../kernels/marlin_from_scratch/v2d_mma_template.cu)让各 lane **直接从 global memory 读取**自己的 BF16 值并装到寄存器；它没有使用 ldmatrix，也不作为高性能 GEMM。V2c 独立学了 ldmatrix；今后还需要研究怎样把它的 shared 布局正确衔接到 MMA fragment，不能宣称本节已完成这个融合。

完成本节后，继续做 [V2d 衔接实验：shared → ldmatrix → MMA](23_V2d_ldmatrix衔接MMA.md)。

```text
1 个 block = 1 个 warp = 32 条 lane

global A [16,16] ──各 lane 取各自元素──→ 每 lane 4 个 32-bit A 寄存器
global B [16,8]  ──各 lane 取各自元素──→ 每 lane 2 个 32-bit B 寄存器
                                                  │
                                       全 warp 一起执行 MMA
                                                  ▼
                              每 lane 得到 4 个 FP32 输出寄存器
                                                  │
                                    按映射写 C [16,8]
```

每个 32-bit A/B 寄存器装两个 BF16 位模式；模板的 `pack_pair(low,high)` 将前者放低 16 bit、后者放高 16 bit。输入 B 在 global memory 中是**行优先**；PTX 指令中的 `.row.col` 描述指令操作数的 fragment 布局，不代表可以直接把 B 的地址当列优先内存。

## 2. 逐 lane 的输入和输出坐标

对 lane `l=0..31`，设 `g=l//4`、`t=l%4`，则每 lane 寄存器包含下面的逻辑元素，按 low、high 顺序：

| 寄存器 | low 16 bit | high 16 bit |
|---|---|---|
| `af[0]` | A[g, 2t] | A[g, 2t+1] |
| `af[1]` | A[g+8, 2t] | A[g+8, 2t+1] |
| `af[2]` | A[g, 2t+8] | A[g, 2t+9] |
| `af[3]` | A[g+8, 2t+8] | A[g+8, 2t+9] |
| `bf[0]` | B[2t, g] | B[2t+1, g] |
| `bf[1]` | B[2t+8, g] | B[2t+9, g] |

四个 FP32 输出的逻辑坐标：

```text
d0 → C[g,   2t]       d1 → C[g,   2t+1]
d2 → C[g+8, 2t]       d3 → C[g+8, 2t+1]
```

这是特定 `m16n8k16` BF16 指令的寄存器布局，不是 WMMA fragment 的通用布局。依据 [NVIDIA PTX ISA 对该指令的 fragment 定义](https://docs.nvidia.com/cuda/archive/11.5.2/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type)；仓库参考的 [BF16 MMA 封装](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_mma.h#L68)也使用相同指令。

例如 lane 1：`g=0,t=1`。它的 `af[0]` 含 A[0,2] 和 A[0,3]，`bf[0]` 含 B[2,0] 和 B[3,0]；它的四个输出是 C[0,2]、C[0,3]、C[8,2]、C[8,3]。即便此 lane 的 B fragment 属于列 0，最终输出列是 2/3，也不矛盾：MMA 是全 warp 合作操作，不是每条 lane 独立拿自己的 A/B 相乘。

如果 A 所有元素是 1、B 每列 n 都填 n，则每个输出 C[m,n]=16n；lane 1 对应的四个结果应为 32、48、32、48。这是手算例子，模板测试输入使用另外的确定性数据。

## 3. 你来完成

复制模板为同目录 `v2d_mma.cu`：

1. 按上表把各 lane 的 A/B 元素读取并用 `pack_pair` 装入 `af` 和 `bf`。
2. 按输出表写回 `d0..d3`。

模板已给出 `asm volatile` 指令和输入输出约束。`r` 表示 32-bit 整数寄存器，`f` 表示 FP32 寄存器；所有 32 条 lane 必须一致执行此 warp 级指令。先理解数据分布，不需要自己从零编写汇编约束。

CPU 测试参考使用实际 BF16 输入值转为 FP32 后计算；小整数要求精确，小数预先约定 `atol=0.002, rtol=0.0001`。编译使用 CUDA 12.1、`-arch=sm_89 -O2 -lineinfo`；运行前确认 GPU 空闲。完成后除数值/内存检查，还可使用 `cuobjdump` 检查编译后的 HMMA 指令；静态指令存在与每次运行的逐指令执行轨迹是不同证据。

## 4. MMA 指令各段与四个零

PTX 矩阵乘加的抽象格式是 `mma ... d, a, b, c`，数学上 `D = A × B + C`。这里的 C 是**输入累加片段**，D 是输出片段；不要把它与练习代码里最终的 `out` 指针混为一谈。对一整个 warp：A 为 `[16,16]`、B 为 `[16,8]`、C/D 为 `[16,8]`，数据分布在 32 条 lane 的寄存器中。

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 │     │    │       │       │   │  │    │    │    │
 │     │    │       │       │   │  │    │    │    └─ 输入 C：FP32
 │     │    │       │       │   │  │    │    └────── B：BF16
 │     │    │       │       │   │  │    └─────────── A：BF16
 │     │    │       │       │   │  └──────────────── 输出 D：FP32
 │     │    │       │       └───┴─────────────────── A/B 指令片段布局：row/col
 │     │    │       └─────────────────────────────── M=16,N=8,K=16
 │     │    └─────────────────────────────────────── warp lane 一致参与的要求
 │     └──────────────────────────────────────────── warp 级同步限定
 └────────────────────────────────────────────────── 矩阵乘加
```

`.sync` 与 `.aligned` 描述这条 warp 级指令的执行要求，不是全 GPU 屏障。`.row.col` 是 PTX 的操作数布局约定；寄存器片段如何对应 A/B 坐标以上面的表为准，本模板从行优先的 B 内存按表取值，不因此要求 global B 改成列优先。

本例的内联汇编把 14 个占位符绑定为：

```text
{%0,%1,%2,%3}        → d0,d1,d2,d3，输出 D 的 4 个 FP32 值
{%4,%5,%6,%7}        → af[0..3]，A 的 4 个 32-bit packed BF16 寄存器
{%8,%9}              → bf[0..1]，B 的 2 个 32-bit packed BF16 寄存器
{%10,%11,%12,%13}    → 4 个 FP32 输入累加值 C
```

最后四个 `"f"(0.0f)` 把**每条 lane 的四个输入 C 分量都设为 0**，因此这个小实验算的是 `D=A×B+0`。它们不是 A、B 的补零，也不是四个线程或四个 K tile。32 条 lane 各自的 4 个 C 分量共同组成整个 `[16,8]` 的累加矩阵。

`"=f"(d0)` 是 FP32 输出寄存器约束；`"f"(0.0f)` 是 FP32 输入约束；`"r"(af[0])` 则把两个 BF16 位模式组成的 32-bit 值作为整数寄存器输入。C++ 中事先声明 `float d0=0.0f` 不会自动让 MMA 使用它的旧值：当前代码的输入明确是四个 `0.0f`，输出变量用 `=f` 表示由指令写入。如果要在多次 K 片段之间累加，需要将**先前的四个 FP32 结果作为下一次指令的 C 输入**，并正确设置输入输出约束，不能只重复执行当前“C 恒为零”的指令。

指令格式与类型顺序参见 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/11.7.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)；内联汇编寄存器约束参见 [NVIDIA Inline PTX Assembly](https://docs.nvidia.com/cuda/archive/12.8.2/inline-ptx-assembly/index.html)。
