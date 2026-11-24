# V5d：W4 解码后的 BF16 权重如何进入 MMA 寄存器

请先完成 V5a/b/c。V5b 使用 WMMA，fragment 的逐 lane 布局不透明；现在回到 V2d 的固定指令 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，验证 W4 解码后的 BF16 B tile 经过 `ldmatrix` 后，每条 lane 具体收到了什么。

本节是**一个固定的 16×8 输出 tile、一次 K=16 MMA**，不是完整 W4A16 GEMM，也不是 Marlin 的 packed 布局。K 总长度 256、N=8；base 从 0、112、128、240 中选择，P 仍是 V3 的沿 K 打包布局；Z/S 分别按 group 0/1 查找。A 是测试框架从逻辑 A 取出的 `[16,16]` BF16 tile；输出是 `[16,8]` FP32。

```text
32 lane 合作读取 global packed Q/Z/S
      ↓ 每个逻辑 W 都先舍入 BF16
shared bs[16][8]                 shared as[16][16] ← 测试提供 A tile
      ↓ __syncthreads()                   ↓
ldmatrix.x2.trans → 每 lane bf[2]    ldmatrix.x4 → 每 lane af[4]
      └──────────────────┬───────────────────┘
                         ↓ mma.sync.m16n8k16
             每 lane 4 个 FP32 输出 → C[16,8]
```

`bs` 的每一行有 8 个 BF16，恰好 16 字节；它是由 W4 **先解码到 shared**，不是直接在寄存器里按 Marlin 的方式解码。每条 lane 写 4 个 B 元素：`i=lane, lane+32, lane+64, lane+96`，对应 `(k_local=i/8,n=i%8)`；例如 lane 0 写 `(0,0),(4,0),(8,0),(12,0)`。所有 32 条 lane 同步后，现有 V2d 包装将这个 tile 装进寄存器。

注意提供 `ldmatrix` 行地址的 lane 和收到某个 B 元素的 lane 是两种映射。设 `g=lane/4,t=lane%4`，V2d 实验证明本例中：

```text
bf[0] 的低/高 16 bit：W[base+2t,   g], W[base+2t+1, g]
bf[1] 的低/高 16 bit：W[base+2t+8, g], W[base+2t+9, g]
```

例如 lane 5：g=1、t=1，bf[0] 保存 `(W[base+2,1], W[base+3,1])`，bf[1] 保存 `(W[base+10,1], W[base+11,1])`；每对前一个位于低 16 位。CPU 测试不从 packed 反解，而从原始逻辑 Q/Z/S 计算每个 BF16 权重，再独立构造这两个寄存器期望值，逐 lane 逐寄存器精确比较；另用 CPU 参考检查最终 C。

复制 [v5d_fragment_template.cu](../kernels/marlin_from_scratch/v5d_fragment_template.cu) 为 `v5d_fragment.cu`，只需填写 `bs` 的 W4 解码，然后把 `ready` 改为 true。`load_a/load_b`、`ldmatrix` 行地址、MMA 指令和输出映射都由 V2d 的代码提供。`__syncthreads()` 在所有 lane 完成 shared 写入后执行；所有 lane 都必须参与。

```bash
cp kernels/marlin_from_scratch/v5d_fragment_template.cu kernels/marlin_from_scratch/v5d_fragment.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
  kernels/marlin_from_scratch/v5d_fragment.cu -o /tmp/quant_v5d
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v5d
```

RTX 4090、CUDA 12.1、SM89 上，临时参考填充的 8 组 B fragment/C 测试通过、memcheck 0 错误；编译产物包含 `LDSM` 和 BF16 `HMMA`。这是框架验证，不能代表你的实现或更广范围正确。后续如果要做无需 shared 暂存的寄存器级直接解码，还要独立设计并验证 packed 权重与每 lane 输入寄存器之间的对应关系。

你的实现验证（2026-11-23）：RTX 4090、CUDA 12.1、`sm_89 -O2 -lineinfo` 下，4 个 base、两类数值共 8 组测试全部通过，每组检查 32 条 lane 的 2 个 B 寄存器以及 16×8 输出；memcheck、racecheck、synccheck 均未报告错误或危害。针对 `v5d_one_tile` 的 SASS 中可见 `LDSM.16.MT88.2`、`LDSM.16.M88.4` 和 `HMMA.16816.F32.BF16`；这说明编译产物包含相应指令，不构成性能或完整 GEMM 的结论。
