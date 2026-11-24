# V6m：把参考格式的 Q/Z/S 直接接到四 warp MMA

## 1. 为什么还要这一节

V6i 能从参考布局得到每 lane 的 B 寄存器；V6k 验证了四个 warp 的部分和，但 B 由 host 预先构造。V6m 把两者接在**同一个 kernel**中：warp/lane 实际读取参考 Q/Z/S 并解码，再做 K 循环的 MMA。这样可以检查完整的 `packed → BF16 fragment → MMA partial` 数值链。

仍裁剪了固定实例的一部分：固定 N tile 0（N=0..63），两个 block 分别计算 group 0/1；每个 block 4 warp，每 warp 两轮 K16。输入 `A[16,256]` BF16、repacked Q `[16,256]` uint32、repacked Z `[2,16]` uint32、repacked S `[2,128]` BF16。只支持本例，既没有 N tile 1、bias 和最终输出，也没有参考的 `cp.async`/shared pipeline、block 内/跨 block 的参考归约协议。

## 2. 线程、数据和内存

```text
block 0 (kr=0)                         block 1 (kr=1)
  warp0: K 0..31                         warp0: K 128..159
  warp1: K 32..63                        warp1: K 160..191
  warp2: K 64..95                        warp2: K 192..223
  warp3: K 96..127                       warp3: K 224..255
      ↓ 每 warp 的 32 lane 各算 16×64 的部分和
      ↓ global diagnostic partial[kr][warp][16][64]
CPU 独立按各 warp 的 32 个 K 验证每一个部分和
```

每 warp：step 0/1，各 step 先协作加载 A `[16,16]` 到该 warp 的 shared A，`ldmatrix` 得 af。对 j=0..3：

```text
Qphys[8*kr+2*warp+step, 4*lane+j] → 一个 word
Zphys[kr, lane/4]                  → 一个 word（nr=0）
Sphys[kr, 8*(lane/4)+2*j : +2]    → 两个 BF16
        ↓ 依 V6i 的 q/z/s 配对与 BF16 操作
        4 个 BF16x2 → 两个 MMA B fragment
        ↓ 两次 mma.sync，累计到 acc[j][0/1][4]
```

上式只适用 N tile 0；若 nr=1，物理 Q、Z、S 的列偏移分别需要增加 128、8、64。注意 Q 的四个 word 在本例中由同一 lane 根据 j 逐个读取，不是 4 个不同 lane。

本实验直接 global 标量读取 Q/Z/S 以暴露逻辑；参考 kernel 是向量化 global→shared、片段寄存器读取和流水线，不应将本模板的地址指令序列、寄存器用量或性能等同于参考实现。诊断 partial 的写回布局也只为测试。

## 3. 练习

复制 [v6m_marlin_integrated_template.cu](../kernels/marlin_from_scratch/v6m_marlin_integrated_template.cu)，完成：

1. 每 lane 在循环外初始化 `acc[4][2][4]`；每 warp 根据 `kr*128+warp*32` 进行两个 K16 step。
2. A 写入 `as[warp]`，全 warp 同步后 `load_a`；从给出的物理 Q/Z/S 按 V6i 生成 4 个 BF16x2。
3. 每 j 的前/后两个寄存器各做一次 MMA，将结果累加到对应的 acc；确保 warp 完成本轮 shared 读取后再复用。
4. 按 V6k 的 lane 行列映射写入各自 `partial[kr][warp]`。不在本节做 block 间归约。

```bash
cp kernels/marlin_from_scratch/v6m_marlin_integrated_template.cu kernels/marlin_from_scratch/v6m_marlin_integrated.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo kernels/marlin_from_scratch/v6m_marlin_integrated.cu -o /tmp/quant_v6m_marlin
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v6m_marlin
```

测试从原始逻辑 Q/Z/S 独立构造物理输入，逐元素比较 2 group×4 warp×16×64 的部分和；3 个随机 seed，预先固定 `atol=0.002,rtol=0.0001`。输入刻意选择 BF16 精确小数；后续必须补非精确 scale、两个 N tile、bias、参考内核差分和所有归约。

2026-11-24 在 RTX 4090、CUDA 12.1、SM89，临时参考填充的三组测试全部通过，memcheck/racecheck/synccheck 未报告错误或危害；该 kernel 的编译产物包含 `LDSM` 和 BF16 `HMMA`。它是诊断 kernel 的测试结果，不代表你的实现已通过，也不是整条 Marlin kernel 的性能或 bitwise 一致性结论。
