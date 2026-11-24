# V5b/c：从 W4 tile 到 WMMA，处理 M/N 尾部

请先完成 V5a。本节在一次 kernel 中把每轮解码后的权重 tile 接到 WMMA，按 K 循环累加。我们选择**一个 block、一个 warp、一个 16×16 C tile**，保持结构清晰；这不是 Marlin 的 tile 布局或优化策略。

## 1. 契约和要解决的问题

A BF16 `[M,K]`，P uint32 `[K/8,N]` 沿 K 打包，Z uint8 / S BF16 `[K/128,N]`；C FP32 `[M,N]`，行连续、无 bias，K 是 128 的正整数倍，M/N 正数且允许不整除 16。数据计算语义沿用 V3：每个 W 在乘 A 前先转 BF16；WMMA 的 accumulator 是 FP32。只提供合法输入的测试程序，不提供 V3e checked 接口，也不支持任意 K 尾部、stride、padding 或 stream 契约。

WMMA 要求一个完整的 16×16 tile，即使矩阵右边或下边不足 16。解决方法是让 warp 把有效元素写到 shared，无效元素补 0，MMA 后只把有效 C 元素写回。

## 2. 画出 block、warp、内存和 tile

固定例 M=17、N=19、K=256：

```text
C [17,19]：grid=(ceil(19/16),ceil(17/16)) = (2,2)
                N
           0..15    16..18
         ┌────────┬────────┐
 M 0..15 │ block  │ block  │    每 block 32 threads = 一个 warp
         │ (0,0)  │ (1,0)  │
   16    ├────────┼────────┤
         │ (0,1)  │ (1,1)  │    block(1,1) 仅有 1×3 有效输出
         └────────┴────────┘
```

每个 block 沿 K 每次推进 16：

```text
当前 base ──► global A[row0:row0+16, base:base+16]
                     │ 32 lane 协作 + 越界补零
                     ▼
              shared as[16][16] ───────┐
                                         │ WMMA load → af
             global packed P/Z/S         │
                     │ 32 lane 解码      │
                     ▼                   ▼
              shared bs[16][16] → bf → mma_sync → FP32 acc
                                         │
              下一 base 复用 as/bs       │ 16 次循环（K=256）
                                         ▼
              shared cs[16][16] ← store_matrix_sync(acc)
                     │ 32 lane 按 M/N 越界判断
                     ▼
              global C 的有效位置
```

K 的 base 依次为 0、16、...、112、128、...、240。base=112 时使用 group 0，base=128 时使用 group 1。每次解码使用相应的 `P[k/8,n]` 和 bit 位位置，具体公式与 V5a 相同。

## 3. lane 分工及同步时间线

32 条 lane 都执行 `for (i=lane; i<256; i+=32)`。对于 A，`r=i/16,t=i%16`，写 `as[r][t]`；对于 B，同一 i 可作为 `t=i/16,n=i%16`，写 `bs[t][n]`。这只是**shared 写入的分工**，并不是 WMMA 内部 fragment 与 lane 的逐元素映射；WMMA fragment 在本练习里不透明。

```text
所有 lane 写 as、bs（无效 A/W 写 0）
    ↓ __syncwarp()：等待这一 warp 的写入完成并保证后续读取顺序
全 warp load_matrix_sync(as)、load_matrix_sync(bs)
    ↓ 全 warp mma_sync(acc += af × bf)
    ↓ __syncwarp()：确保本轮 warp 的读取完成后再复用 shared
下一轮覆盖 as、bs
    ↓ K 循环结束
全 warp store_matrix_sync 到 cs → __syncwarp() → 各 lane 写合法的 C
```

不要让越界输出的 lane 提前 return：即使 block 只有一个有效 C，这 32 条 lane 都要参与 tile 加载、同步和 WMMA。模板仅允许在最前面用整个 warp 一致的 `ready=false` 提前退出，这是为了让空模板安全失败。

为什么不直接把完整 WMMA 结果写到 global C？像 block(1,1) 只有 1×3 的有效区域；直接 store 16×16 会写出界。因此先将完整 16×16 写到对齐的 shared `cs`，再由 lane 检查行列后逐元素写 C。

## 4. 手算一个边缘 block

M=17、N=19，`blockIdx=(1,1)`：`row0=16,col0=16`。对 A tile：只有 shared 行 0 可读取 A[16,k]，行 1～15 补 0；对 B tile：shared 列 0～2 对应逻辑列 16～18，列 3～15 补 0。计算结束后只有 C[16,16]、C[16,17]、C[16,18] 写回。某个 warp 的其他 lane 即使最终不写 C，仍参与前面的工作。

## 5. 自己完成

复制 [v5bc_wmma_template.cu](../kernels/marlin_from_scratch/v5bc_wmma_template.cu) 为 `v5bc_wmma.cu`。填写循环中的四项 TODO：

1. 协作加载 `as`，按 M/K 越界补零。
2. 参考 V5a 解码 `bs`，按 K/N 越界补零；每个 W 独立转 BF16。
3. warp 同步，按 V2b 的用法用 `load_matrix_sync` 加载两个行优先 16×16 tile、行步长均为 16 个 BF16；再 `mma_sync` 累加。
4. 同步后复用 shared；全部写好后把 `ready` 设为 true。

`wmma::fragment`、`fill_fragment`、最终 shared store 与边界写回已经给出。`acc` 初始化在 K 循环外，不能每次循环清零。`as`、`bs`、`cs` 用 `__align__(32)` 指定 32 字节对齐，满足本模板 WMMA tile 地址对齐需求。

```bash
cp kernels/marlin_from_scratch/v5bc_wmma_template.cu kernels/marlin_from_scratch/v5bc_wmma.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
  kernels/marlin_from_scratch/v5bc_wmma.cu -o /tmp/quant_v5bc
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v5bc
```

测试覆盖 `(16,16,128)`、`(16,128,256)`、`(1,1,128)`、`(17,19,256)` 与两类数据；对 BF16 权重舍入后的 CPU FP32 参考预先固定 `atol=0.002,rtol=0.0001`。同时报告**不将 W 舍入到 BF16 的数学参考**与 BF16 权重参考之间的最大差值；该差值不是 GPU 的误差。不能把两个参考混淆。

临时参考填充在 RTX 4090、CUDA 12.1、SM89 上 8 组通过；最大测试误差约 `2.87e-6`。先前同一 kernel 的 memcheck、racecheck、synccheck 都报告 0 错误/危害；编译产物中看到 BF16 `HMMA`。这些是框架验证，不是你尚未填写的实现结果，也不代表性能收益。

你的实现验证（2026-11-23）：RTX 4090、CUDA 12.1、`sm_89 -O2 -lineinfo` 下，模板的 8 组测试全部通过，最大绝对误差 `2.86102e-6`；memcheck、racecheck、synccheck 均未报告错误或危害。额外的 `(M,N,K)=(31,33,384)` 小整数与 BF16 小数测试也通过，覆盖三个 group 和更多尾部 block，小数 case 最大误差 `9.53674e-6`，memcheck 0 错误。编译产物包含 BF16 `HMMA`；这只说明指令存在，不是性能结果。
