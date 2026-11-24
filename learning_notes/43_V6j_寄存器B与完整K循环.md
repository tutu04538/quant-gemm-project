# V6j：把参考 B fragment 送入 MMA

## 1. 目标与本节的裁剪

V6i 得到了固定实例中每 lane 对一个 packed word 的 4 个 BF16x2 寄存器；现在验证它们如何进入两条 `mma.sync.m16n8k16`。本节取参考 block 0、warp 0、j=0，对应 N=0..15、K=0..31，两次 K16 step。

```text
每 step：
  32 lane： A[16,16] → shared A → ldmatrix.x4 → af[4]
  32 lane： 参考 B fragment 的 4 个 uint32
                ├─ bf[0:2] → MMA → acc0[4] → C 的 N 0..7
                └─ bf[2:4] → MMA → acc1[4] → C 的 N 8..15
                          ↓
                  两个 K16 step 分别累加，不清零
```

测试框架从逻辑 BF16 B 独立生成与 V6i **同坐标和位序**的寄存器输入；为独立检查 MMA 接线，本节 GPU kernel 不调用 V6i，也没有运行参考 kernel 的 pipeline。这是固定实例局部的片段/计算实验，不是完整 Marlin。之后必须把真实 repacked Q/Z/S→V6i→本节 MMA 连成同一计算路径并做端到端差分；当前测试不能证明这一集成已经完成。

## 2. 按 lane 画两条 MMA 的输入输出

令 `g=lane/4,t=lane%4`，`base=16*step`：

```text
bf[0]：B[base+2t,   g]、  B[base+2t+1, g]
bf[1]：B[base+2t+8, g]、  B[base+2t+9, g]
bf[2]：B[base+2t,   g+8]、B[base+2t+1, g+8]
bf[3]：B[base+2t+8, g+8]、B[base+2t+9, g+8]
```

每行是一个 uint32，前者为低 16 位、后者为高 16 位；每个值是已舍入 BF16 权重。第一次 MMA 用 bf[0]/bf[1]，输出列 g；第二次用 bf[2]/bf[3]，输出列 g+8。

以 lane 5 为例，g=1,t=1：第一次读 B[2,1]/B[3,1]/B[10,1]/B[11,1]；第二次读 B[2,9]/B[3,9]/B[10,9]/B[11,9]。step=1 时 K 全部加 16。

输出映射：第一条的 4 个 acc 对应行 g/g+8，列 2t/2t+1；第二条同样的行，列再加 8。warp 全体一致执行每条 MMA。

## 3. 动手

复制 [v6j_marlin_mma_template.cu](../kernels/marlin_from_scratch/v6j_marlin_mma_template.cu)，按 V2d/V5ef 写 A 协作加载、`load_a`、两条 MMA 和输出。`+f` 的 accumulator 在 K 循环外初始化；每次读取 shared 后同步再覆盖。测试预先在 host 构造 B fragment，CPU 用逻辑 A/B 对照。

```bash
cp kernels/marlin_from_scratch/v6j_marlin_mma_template.cu kernels/marlin_from_scratch/v6j_marlin_mma.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo kernels/marlin_from_scratch/v6j_marlin_mma.cu -o /tmp/quant_v6j_marlin
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v6j_marlin
```

3 个随机 seed、FP32 输出；事先固定 `atol=0.002,rtol=0.0001`。临时参考填充通过全部测试和三种 sanitizer；这些输入是 BF16 的二进制精确小数，不是上游模型数据。`j=1..3` 与其他 warp 的拼装在 V6k 学习。
