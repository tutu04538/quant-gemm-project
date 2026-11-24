# V5a：把当前 W4 权重 tile 解码到 shared memory

## 1. 为什么需要这一步

V4a 中每个输出线程自己解码权重并做标量乘加。现在要逐步走向 Tensor Core：先让一个 warp 把**当前要算的权重小块**从 W4 解码成 BF16，放进 shared memory。下一节再让 WMMA 读取它。我们只解码一次要用的 tile；测试为了查看结果，会把这个 tile 拷到 global，但这只是诊断。

V5 顺序：本节 V5a → V5b/c 将 tile 接到 WMMA 并处理 M/N 尾部 → V5d 用 `ldmatrix`/`mma.sync` 检查低层片段。每次只完成一个模板，写完后先 review。

## 2. 输入和布局

沿用 V3 的约定：A 是 BF16 `[M,K]`；逻辑 Q 是无符号 4 bit `[K,N]`；P 是 uint32 `[K/8,N]`，**沿 K 打包**，每个 word 低 4 位放较小 k；Z uint8、S BF16 均为 `[K/128,N]`；group=128，无 padding。K 是 128 的正整数倍，N 为正；本例只有测试代码分配并传入合法连续张量，没有通用参数检查。

```text
逻辑 Q [256,19]                packed P [32,19]（每格是 1 个 uint32）
 k ↓   n →                    word 行 ↓      n →
      ┌──────────────┐                 ┌──────────────┐
  0   │              │          0      │ Q[0..7,n]    │
 ...  │              │          1      │ Q[8..15,n]   │
 112  ├──────────────┤          ...    │              │
 ...  │ 当前 tile    │         14      │ Q[112..119,n]│
 127  ├──────────────┤         15      │ Q[120..127,n]│
 128  │ 下一 group   │         16      │ Q[128..135,n]│
      └──────────────┘                 └──────────────┘
```

如果当前 `base=112, col0=16`，逻辑 tile 是 `k=112..127, n=16..31`，但实际 N=19，只有列 16、17、18 有效；其他列在 shared 里补 0。下一次 base=128 切换到 group 1。因为 128 能被 16 整除，每个本实验对齐的 K tile 自身不跨 group，但请仍以逻辑 k 获取 group。

## 3. block、warp、lane 如何合作

```text
launch：1 block × 32 threads = 1 warp
shared tile[16][16]：行 = tile 内 k，列 = tile 内 n（BF16，行连续）

lane 0  写线性编号 0,32,64,...,224   → (0,0),(2,0),...,(14,0)
lane 1  写线性编号 1,33,65,...,225   → (0,1),(2,1),...,(14,1)
...
lane 15 写 (0,15),(2,15),...,(14,15)
lane 16 写 (1,0),(3,0),...,(15,0)
...
lane 31 写 (1,15),(3,15),...,(15,15)
```

每条 lane 用 `i=lane; i<256; i+=32` 处理 8 个元素。tile 行列 `x=i/16, y=i%16`；逻辑坐标是 `k=base+x, n=col0+y`。这个图表示 shared 元素的所有权，不是 packed word 的线性位置。

数据流：

```text
lane: 逻辑 (k,n) → P[k/8,n] 中第 k%8 个 nibble → q
                   Z[k/128,n]、S[k/128,n]
             → float(q-z)*float(S) → BF16 舍入
             → shared tile[x][y]
32 lane 全部写完 → __syncwarp() → 32 lane 复制 tile 到 global 测试缓冲
```

例如 `P[0,0]=0x76543210`，`k=3,n=0` 得到 q=3；如果 Z=7、S=0.25，则写入的 BF16 是 -1。对于越界列，即使该地址附近存在其他有效数据，也不能读取 P/Z/S；直接写 BF16 0。

## 4. 自己完成

复制 [v5a_decode_tile_template.cu](../kernels/marlin_from_scratch/v5a_decode_tile_template.cu) 为 `v5a_decode_tile.cu`。只写 kernel 中 TODO：合作遍历 256 个元素、推导逻辑坐标、有效时解码并写 shared、无效时补零、warp 同步后复制到 `out`。`v5_test_common.cuh` 已提供数据、内存管理和独立 CPU 参考。

`__shared__ __align__(32) __nv_bfloat16 tile[16][16]` 声明 block 内共享的 BF16 二维数组并指定起始地址对齐；`__syncwarp()` 在本例的完整 32-lane warp 中等待参与线程到达，并为同步前写入与同步后的读取提供 warp 内所需的顺序。这个 block 只有一个 warp，所有线程都必须参加同步；不要让越界 lane 提前 return。

```bash
cp kernels/marlin_from_scratch/v5a_decode_tile_template.cu kernels/marlin_from_scratch/v5a_decode_tile.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
  kernels/marlin_from_scratch/v5a_decode_tile.cu -o /tmp/quant_v5a
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v5a
```

先确认 GPU 空闲。测试 N=19、K=256，base=0/112/128/240、col0=0/16、小整数和 BF16 小数；每个 BF16 tile 元素与从逻辑 Q 出发的 CPU 参考精确比较。未填模板输出为 NaN，测试会失败。

我用临时参考填充在 RTX 4090、CUDA 12.1、SM89 上验证：16 个 tile 用例全部通过，memcheck 0 错误。此记录只验证框架，不代表你的实现已通过。

你的实现验证（2026-11-23）：在同一 GPU 和工具链下，16 个模板用例逐元素比较全部通过；memcheck、racecheck 和 synccheck 均未报告错误或危害。额外用临时测试覆盖 base=120 的跨 group tile、base=248 的 K 尾部，以及 col0=18 的 N 尾部；8 组逐元素比较通过，memcheck 0 错误。这些额外位置用于验证解码和边界，不改变 V5b/c 使用 16 对齐 K tile 的约定。

练习思考：`base=128,col0=16,i=1` 的逻辑坐标、packed word、nibble 与 group 分别是什么？
