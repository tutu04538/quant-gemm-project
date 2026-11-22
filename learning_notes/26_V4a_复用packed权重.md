# V4a：每个线程复用一个 packed word

## 1. 解决什么问题

V3 中，一个线程负责一个输出 C[row,col]，沿 K 逐项计算。对同一列，每 8 个 Q 共存在一个 uint32 里；V3 的源码每计算一个 k 都调用解码函数。V4a 让线程把这 8 项组织为一组：**先读取一个 word，再从它依次取出 8 个 nibble**，对应的 Z/S 也在组内复用。

这是源代码层面的复用。编译器可能已经优化 V3 的重复读取，也可能改变 V4a 的指令；缓存、寄存器压力和其他开销都影响结果。**是否减少机器访存、是否加速，等 V4c 用实际编译结果和计时验证。**

## 2. 输入、布局和计算契约

沿用 [V3](24_V3_W4A16.md)：A BF16 `[M,K]`；逻辑 Q uint4 `[K,N]`；P uint32 `[K/8,N]`，沿 K 打包、低位放低 k；Z uint8 和 S BF16 均为 `[K/128,N]`；输出 C FP32 `[M,N]`。连续行优先，M/N 为正，K 为 128 的正整数倍，不补 padding；不包含 bias。每次解码先计算 `(int(q)-int(z))*float(s)`，舍入到 BF16，再与 BF16 A 转 float 相乘并 FP32 累加；保持 k 从小到大的顺序。

```text
逻辑 Q 中同一列 col：                    物理 packed：
Q[base+0,col] ─┐                        P[base/8,col]
Q[base+1,col] ─┤                        ┌───────────────────┐
 ...           ├── 8 个 4 bit ───────→ │ q7 ... q2 q1 q0   │ 32 bit
Q[base+7,col] ─┘                        └───────────────────┘

base = 0, 8, 16, ...；一个线程读取该 word → 依次提取 q0 ... q7
Z/S 的 group = base/128，Z/S 对这一组 8 个 k 相同
```

例如 base=120 时，8 个 k 是 120～127，都属于 group 0；下一轮 base=128，8 个 k 为 128～135，都属于 group 1。因为 128 是 8 的倍数，且 K 是 128 的倍数，任何一组 8 个元素都不跨 group，也没有 K 尾部。**这是本节契约下的结论，不能推广到任意 group size 或任意 K。**

手算：word=`0x76543210`，Z=3，S=0.25，A 的这 8 项都是 1：q 依次为 0、1、...、7；对应 W 为 -0.75、-0.5、-0.25、0、0.25、0.5、0.75、1。这 8 项总贡献为 1。要保留每项先转 BF16 的步骤，即使这个例子中的数恰好可精确表示。

## 3. block、warp、lane 和数据流

沿用 V3 的 block `(16,16)`：

```text
输出 C tile：[16,16]，一个线程拥有一个 C[row,col]

warp 0   lane  0..15  → ty=0, tx=0..15 → C[block_y*16+0, block_x*16+0..15]
         lane 16..31  → ty=1, tx=0..15 → C[block_y*16+1, block_x*16+0..15]
warp 1   lane  0..31  → ty=2、3 的两行输出
 ...
```

对某条有效线程：

```text
寄存器 sum=0
  base=0,8,16,...
    P[base/8,col] ─→ 1 个 uint32 word ─→ 提取第 0..7 个 nibble
    Z/S[base/128,col] ─────────────────→ 逐个解码为 BF16 W
    A[row,base..base+7] ───────────────→ 与 W 逐项乘加到 sum
  sum ─→ C[row,col]
```

这里只复用**同一线程**读到的数据，没有使用 shared memory，也没有线程间同步。不同线程可能读取相同的 P、A；源码层面相同不代表一定访问了同一层物理内存。

## 4. 你来写

复制 [v4a_word_reuse_template.cu](../kernels/marlin_from_scratch/v4a_word_reuse_template.cu) 为 `v4a_word_reuse.cu`，填写 `v4a_gemm` 的三个 TODO：

1. 二维索引和越界判断沿用 V3。
2. 外层按 8 步进，获取该 word 的 P、对应 group 的 Z 和 S；Z 先转 int，S 用 `__bfloat162float`。
3. 内层 `t=0..7`：无符号右移 `4*t`、掩码 `0xF`，计算 BF16 W；按 k 顺序累加，最后写 C。

一个常见错误是把 8 个 q 的和乘一次 A 或只把总和转 BF16；每个权重对应不同的 A，且每个 W 都需要单独 BF16 舍入。

```bash
cp kernels/marlin_from_scratch/v4a_word_reuse_template.cu kernels/marlin_from_scratch/v4a_word_reuse.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
  kernels/marlin_from_scratch/v4a_word_reuse.cu -o /tmp/quant_v4a
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v4a
```

测试使用与 V3 一样的独立逻辑 Q CPU 参考，以及固定容差 `atol=0.0001, rtol=0.00001`；覆盖 `(1,1,128)`、`(3,9,256)`、`(16,128,256)`、`(17,19,384)` 两类数值。输出中的 `W错误` 是调用原 V3 的诊断解码检查；**V4a 新写的内层解码由 C 结果验证，不是由 W 逐元素直接验证。** C 未写时是 NaN，空模板会失败。

临时参考填充在 CUDA 12.1、RTX 4090、SM89 上八组均通过，memcheck 0 错误；这只验证测试框架和一份临时实现，尚不代表你的实现通过。V4a 是原始 kernel 实验，未自动接入 V3e 的 checked host 接口。

你的实现验证：2026-11-22 用 CUDA 12.1、`-arch=sm_89 -O2 -lineinfo` 在 RTX 4090 上直接编译填写后的 `v4a_word_reuse_template.cu`；4 种形状、2 类输入的 C 均通过既定参考和容差，memcheck 报告 0 个错误。本次未测试性能；输出中的 W 检查仍来自 V3 解码函数。

## 5. 想一想

1. 为什么 k=127 与 k=128 不在同一轮 8 项中？
2. 假如 group size 改为 12，8 项是否仍然肯定不跨 group？
3. 即使源码将 word 加载写在外层，为什么还不能直接断言 DRAM 流量减为原来的 1/8？
