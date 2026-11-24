# V5e：让一个 warp 完成完整 K 循环

## 1. 从 V5d 到 V5e，要解决什么问题

V5d 做一次 `m16n8k16`：A 是 `[16,16]`，B 是 `[16,8]`，得到 `[16,8]` 的 C。真实 GEMM 的 K 往往大于 16，例如本节 `K=256`；于是要把 K 分成 16 轮，并把每轮的贡献保留下来：

```text
C[16,8] = A[:,  0:16 ] × W[  0:16, :]  ──→ 累加
        + A[:, 16:32 ] × W[ 16:32, :]  ──→ 累加
        + ...
        + A[:,240:256] × W[240:256,:]  ──→ 最终 C
```

V5d 的 MMA 把输入 C 固定为 0，只适用于一次计算。V5e 的核心是**每条 lane 持有 4 个 FP32 累加寄存器，把它们作为下一轮 MMA 的输入和输出**。这里仍先将当前 B tile 解码到 shared，不是 Marlin 的寄存器级直接解码。

## 2. 输入和输出契约

固定 `M=16,N=8,K=256`，一个 block、32 个线程（一个 warp）：

```text
A：BF16 [16,256]        逻辑 Q：uint4 [256,8]
P：uint32 [32,8]        Z：uint8 [2,8]       S：BF16 [2,8]
C：float [16,8]         group=128            无 bias、无 padding
```

P 沿 K 每 8 个 q 打包，低 nibble 放低 k；所有数组行连续。每个权重 `float(q-z)*float(S)` 转 BF16 后再参与 MMA，累加及输出 FP32。K 是 128 的正整数倍是本练习的限定；V5e 当前只测试 256。调用者保证输入合法，本模板不是 V3e checked 接口。CPU 参考从原始逻辑 Q/Z/S 计算，不从 packed 反解。

## 3. block、warp、lane 和内存

```text
C [16,8] ← 1 block × 32 lane；block 拥有整个 C tile

每个 base=0,16,...,240：
  32 lane 合作  A[row,base:base+16] ─────→ shared as[16][16] BF16
  32 lane 合作  P/Z/S → 解码本轮 W ──────→ shared bs[16][8] BF16
                   ↓ 同步
        ldmatrix.x4  → 每 lane af[4]     ldmatrix.x2.trans → 每 lane bf[2]
                   └────────────┬─────────────┘
                                ↓ mma.sync
                         每 lane acc[0..3]
                                ↓ 同步后复用 shared
最后每 lane 把 4 个 acc 写回自己的 C 元素
```

每 lane 向 A shared 写 `i=lane,lane+32,...<256`，其局部坐标是 `(i/16,i%16)`；向 B shared 写 `i=lane,lane+32,...<128`，坐标是 `(i/8,i%8)`。这两种坐标是 shared 中的写入位置，不是 MMA 寄存器在 lane 中的分配。

由 V2d 的布局，令 `g=lane/4,t=lane%4`，一条 lane 的输出是：

```text
acc[0] → C[g,   2t]      acc[1] → C[g,   2t+1]
acc[2] → C[g+8, 2t]      acc[3] → C[g+8, 2t+1]
```

例如 lane 5 的 g=1、t=1，它在 16 轮中维护 C[1,2]、C[1,3]、C[9,2]、C[9,3] 的部分和。**不要每轮重新把 acc 初始化为 0。**

## 4. 手算：为什么不能每轮用 0

假设只看 C[1,2]，第 0 轮 16 个乘积的和为 3，第 1 轮的和为 5：

```text
初始 acc = 0
第 0 轮：acc = 3 + 0 = 3
第 1 轮：acc = 5 + 3 = 8
```

如果第 1 轮仍把 MMA 的 C 输入写成 0，得到的是 5，前一轮的 3 就丢了。本节固定例实际上有 16 轮；base=112 处理 group 0 的最后 16 项，base=128 处理 group 1 的前 16 项。

## 5. `+f` 与 MMA 输入输出

PTX 的 `mma.sync` 可理解为 `D = A × B + C`。在本例中，D 和 C 使用同一组 C++ `acc` 变量。内联 PTX 的 `"+f"(acc[0])` 告诉编译器：对应的 float 寄存器既是输入，也是输出；`"=f"` 则只声明输出。模板里 `mma_accumulate` 需要你实现。

```text
输出操作数按顺序绑定：%0..%3 ↔ acc[0..3]（"+f"）
输入操作数按顺序绑定：%4..%7 ↔ af[0..3]  （"r"）
                    %8..%9 ↔ bf[0..1]  （"r"）

PTX 四组寄存器的位置：
  D = {%0,%1,%2,%3}
  A = {%4,%5,%6,%7}
  B = {%8,%9}
  C = {%0,%1,%2,%3}   ← 同一组旧 acc 作为累加输入
```

`af`、`bf` 分别来自 V2d 的 `load_a` 和 `load_b`，他们接受通过 `__cvta_generic_to_shared` 转换后的 shared 地址。所有 lane 必须一致执行 `ldmatrix.sync.aligned` 和 `mma.sync.aligned`。先 `__syncwarp()` 保证 shared 写入完成，再加载并 MMA；下一轮覆盖 shared 前还要让本 warp 完成上一轮读取。

### `aligned`、`row`、`col`

在 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 中，`.aligned` 要求同一个 warp 的线程一致执行同一条 MMA；有条件分支时，所有 lane 必须一致走到该指令。它描述 warp 执行要求，不是声明指针的字节对齐；shared 中 `ldmatrix` 地址的对齐需另外保证。`.sync` 是该 warp 协作指令自身的同步语义，不能替代 shared 写入前后的必要同步。

`.row` 对应 A 的 row-major 布局，`.col` 对应 B 的 column-major 布局；在本条低层 `mma` 中，A/B 已经是寄存器 fragment，使用者必须按该指令规定的 lane/寄存器映射提供元素。修饰符不自动加载、转置 global/shared 内存，也不表示 C 的布局。本例 shared 的 `bs[16][8]` 仍按 C++ 行优先存放；`load_b` 中 `ldmatrix ... .trans` 将其装配到符合该 MMA 的 B 寄存器布局。`as` 使用不带 `.trans` 的加载。具体映射可对照 V5d 的 lane 公式。[NVIDIA PTX ISA 的 MMA 说明](https://docs.nvidia.com/cuda/archive/11.4.4/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)给出了布局与 warp 参与要求。

## 6. 你来完成

V5e 和 V5f 共用 [v5ef_lowlevel_template.cu](../kernels/marlin_from_scratch/v5ef_lowlevel_template.cu)。复制为 `v5ef_lowlevel.cu`，先完成标为 **V5e** 的 TODO：

1. 实现 `mma_accumulate`，按第 5 节给出的寄存器关系声明读写约束。
2. 将固定例的 A、P/Z/S 逐 tile 放入 as、bs；可以参考你已完成的 V5d 解码。
3. 每轮同步后按 V5d 构造 A/B 行首地址，调用已有 `load_a`、`load_b` 和 `mma_accumulate`；同步后进入下一轮。
4. 按第 3 节输出映射写 C；完成后将 `ready` 设为 true。暂时只要求固定例，不要求 M/N 尾部。

本模板通过 include 复用 [V2d 文件](../kernels/marlin_from_scratch/v2d_ldmatrix_mma.cu)中的两个汇编包装；其测试入口被重命名，方便单文件练习。测试输出预填 NaN；未填写的模板不会误判为通过。

```bash
cp kernels/marlin_from_scratch/v5ef_lowlevel_template.cu kernels/marlin_from_scratch/v5ef_lowlevel.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
  kernels/marlin_from_scratch/v5ef_lowlevel.cu -o /tmp/quant_v5ef
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v5ef --e
```

确认 GPU 空闲。`--e` 测小整数与 BF16 小数两组，固定 `atol=0.002,rtol=0.0001`；CPU 参考先将 W 舍入 BF16，逐 K 做 FP32 累加。注意这是浮点容差比较，不要求 Tensor Core 与 CPU 一般情况下 bitwise 一致。

2026-11-23，RTX 4090、CUDA 12.1、SM89 上，临时参考填充的 `--e` 两组通过（最大误差 `1.90735e-6`）；`--f` 全部测试通过且 memcheck/racecheck/synccheck 无错误或危害。编译产物的 `v5ef_gemm` 包含 `LDSM` 和 BF16 `HMMA`。这只验证框架和临时实现，不代表你还未填写的版本通过，也没有性能结论。

用户实现验证（2026-11-23）：同一 GPU 和工具链下，`--e` 两组和 `--f` 十组全部通过；`--f` 中最大误差为 `9.53674e-6`。memcheck、racecheck、synccheck 均未报告错误或危害；`v5ef_gemm` 的 SASS 包含 `LDSM` 和 BF16 `HMMA`。CUDA 12.1 的 ptxas 接受用户当前的 `mma.aligned.sync` 写法；文档示例使用 PTX 规范展示的 `mma.sync.aligned`。
