# V6k：固定实例一个 block 的四 warp 分工

## 1. 目标与输入

追固定实例 block 0 的一个 K=128 部分计算：一个 block 128 线程，4 warp 共同计算 16×64 输出的 FP32 部分和。K 的分工与参考实例一致：

```text
warp 0：K  0..31    warp 1：K 32..63
warp 2：K 64..95    warp 3：K 96..127
每 warp：2 个 K16 step × 4 个 j × 每 j 两条 N8 MMA
```

每 warp 的 lane 对第 j 个 word 的逻辑权重分别来自 N=`16j+g` 和 `16j+g+8`；按 V6j 的 4 个 BF16x2 寄存器进入两条 MMA。测试预先从逻辑 BF16 B 构造 `[warp,step,lane,j,4]` 输入，保持 V6i 的寄存器坐标和位序；实际 Marlin 中这些寄存器是 Q/Z/S 解码所得。本节**没有把 V6i 解码 kernel 直接融合进 MMA**，也没有参考 4-stage shared pipeline 或跨 block 归约。

## 2. 每个 warp、每条 lane 做什么

```text
每个 warp w：
  32 lane 合作加载各自的 A[16,16]（两个 K step）
  每 lane 保有 acc[4 j][2 个 N8][4 个 FP32 输出]
  每 step 对 j=0..3 执行 2 次 MMA，共 8 次
  两个 step 的 acc 相加 → 该 warp 的整个 16×64 部分和

所有 warp 的部分和 → block 内合并 → block 0 的 K=0..127 部分输出
```

例如 warp2 的 K=64..95；lane5 的 g=1,t=1；j=2 时第一条 MMA 的列范围为 32..39，第二条为 40..47。4 个 acc 的输出列是 34/35 和 42/43，输出行是 1/9。各 warp 计算相同输出位置但不同 K 区域。

本实验使用 `partial[warp][16][64]` 和完整 `debug` 数组，目的是逐 warp 验证；这个布局与归约顺序**不是**上游实现。参考源码 [thread_block_reduce](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h#L1406) 使用 shared 中的分层归约与 fragment 布局。后续需要单独逐阶段复现和验证该协议，不能把这里的诊断求和称为参考归约。

## 3. 同步

`as[warp]` 为各 warp 私有使用的 shared 区域；`__syncwarp()` 保证该 warp 的加载和复用顺序。把各 warp 的 acc 写入 partial 后，**所有 128 个线程必须参加 `__syncthreads()`**，再由 warp 0 读取所有 warp 的部分和；仅用 `__syncwarp()` 不能等待其他 warp。

归约顺序与 CPU 顺序 K 累加不同，FP32 结果一般不保证逐位相同。

## 4. 动手

复制 [v6k_marlin_warp_template.cu](../kernels/marlin_from_scratch/v6k_marlin_warp_template.cu)，完成各 warp 的两轮 A 加载、8 组 MMA accumulator、输出 partial 和由 warp0 做教学诊断归约。host 已独立构造实际坐标对应的 B 寄存器；参考 B 数据的 GPU 解码与整个 block 的真实 pipeline 尚未整合。

```bash
cp kernels/marlin_from_scratch/v6k_marlin_warp_template.cu kernels/marlin_from_scratch/v6k_marlin_warp.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo kernels/marlin_from_scratch/v6k_marlin_warp.cu -o /tmp/quant_v6k_marlin
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v6k_marlin
```

CPU 逐 warp 对各自 32 个 K 的 partial 单独核对，再核对四份合并结果；预先固定 `atol=0.002,rtol=0.0001`、3 个随机 seed。临时参考填充全部通过，memcheck/racecheck/synccheck 未报告错误或危害；不是用户实现已通过，也不是整段参考内核正确性结论。
