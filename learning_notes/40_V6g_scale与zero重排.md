# V6g：为什么 scale 和 zero 需要分别重排

> 主线说明：本节 S/Z 输出排列与固定参考分支对照；模板的输入是已解码的逻辑 Z，使用独立 CUDA kernel 完成转换，线程分工不等同参考 Python/PyTorch 准备步骤。AWQ checkpoint → standard zeros 的转换不在本模板内。

## 1. 目标与适用范围

固定参考路径的逻辑输入为 `S BF16 [groups,N]`、`Z uint8 [groups,N]`，`groups=K/128`，K>128 且为 128 的倍数、N 为 64 的倍数；无 padding、无 act-order。我们从**已经解码的逻辑 Z**出发，重排后每 8 个 4-bit zero 存入一个 uint32；这与 AWQ checkpoint packed zeros 的输入格式不同。

参考 [marlin.py 的 scale/zero 准备](../third_party/vllm_marlin/minivllm/layers/quantization/marlin.py)；其中 `group_size == K` 的 single-group scale 使用另一种 permutation，本节不覆盖。Q 的 V6f 重排也不能直接用于 S 或 Z。

## 2. scale：每 64 个元素做 8×8 转置

把某一 group 的连续 64 列分成下面的逻辑小表：

```text
           原逻辑列 n（以一个 64 列块为起点）
        0  1  2  3  4  5  6  7
        8  9 10 11 12 13 14 15
       16 17 18 19 20 21 22 23
       ...
       56 57 58 59 60 61 62 63

物理输出 scale 顺序：
  0, 8,16,24,32,40,48,56, 1,9,17,25,33,41,49,57, ...
```

每个物理 scale 元素由一个线程写；CUDA kernel 是一维索引，不涉及 MMA warp/lane 分工。原逻辑 n=10 在矩阵中行 1、列 2，转置后位于块内物理下标 `2*8+1=17`。group 和 N 的每个 64 列块独立转置，不跨 group。

## 3. zero：先同样转置，再调整每个 word 的 nibble

对物理 scale 顺序中的每组 8 个 Z，再按索引顺序 `[0,2,4,6,1,3,5,7]` 放入一个 32 bit word，低 nibble 在前：

```text
第一组转置后的逻辑列：[0,8,16,24,32,40,48,56]
打包的 nibble 0..7：  [0,16,32,48,8,24,40,56]（这里数字表示逻辑列）
```

手算假设 Z[n]=n%16：上述 zero 数值是 `[0,0,0,0,8,8,8,8]`，word 数值为 `0x88880000`。注意 Z 列顺序与 scale 的顺序不同，不能用 scale 的下标直接当 zero nibble 下标。

## 4. 练习和验证

复制 [v6g_scale_zero_template.cu](../kernels/marlin_from_scratch/v6g_scale_zero_template.cu) 为 `v6g_scale_zero.cu`，填写两个独立 kernel：`permute_scale` 每线程写一个 BF16；`permute_pack_zero` 每线程写一个 uint32。测试从逻辑 n **反向**确定物理位置，比较每个 BF16 原始 16 bit 和每个 zero word。输入 scale 用可识别 BF16 标签；Z 使用 0..15。

```bash
cp kernels/marlin_from_scratch/v6g_scale_zero_template.cu kernels/marlin_from_scratch/v6g_scale_zero.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo kernels/marlin_from_scratch/v6g_scale_zero.cu -o /tmp/quant_v6g
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v6g
```

测试 K=256/384、N=64/128/192。2026-11-24 临时参考填充的 6 组测试通过；另将 CPU 反向参考与本机上游 Python 输出在 K=256/N=128 对比，scale BF16 位和 zero word 均一致。参考脚本 [v6g_compare_upstream.py](../tests/v6g_compare_upstream.py) 依赖完整本地参考项目，**比较的是 CPU 映射与上游，不是你的 GPU 输出直接比较**；你的 GPU 输出经上述独立 CPU 参考验证。不同 K/N、padding、single-group 路径仍需各自验证。

用户实现验证（2026-11-24）：RTX 4090、CUDA 12.1、`sm_89 -O2 -lineinfo` 下 6 组逐位参考检查通过，memcheck 报告 0 错误。对 K=256/N=128，另用临时诊断程序保存用户 GPU 输出，直接与本机参考 Python 的 GPU 输出比较：256 个 BF16 scale 位模式和 32 个 zero word 全部一致。该上游直接比较仅覆盖这一组输入；未验证 single-group 或 padding 路径。
