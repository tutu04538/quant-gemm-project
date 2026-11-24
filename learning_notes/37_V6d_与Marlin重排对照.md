# V6d：我们的重排与 Marlin 有什么不同

这一节是**对照阅读**，不需要再写模板。请先完成 V6b/c；这里对照的是工作区学习快照中指定分支，不意味着 V6 的结果可传给 Marlin，也不声称已完整验证 Marlin 每条路径。

## 1. 同一个问题，不同布局

两个方案都在算子调用前调整 4-bit 权重的物理排列，让后续 tile 的读取和解码适配计算；逻辑 Q 与物理 word 地址必须区分。

```text
我们 V6：
  Q → 沿 K 打包 P[K/8,N] → 按 [16K,8N] 切 tile，R[K/16,ceil(N/8),16]
  每个 R word = 同一 k 的 8 个连续 n，按 n 递增放到低→高 nibble
  GEMM：从 R 解码到 shared BF16，再 ldmatrix → MMA

Marlin 的本地参考路径（下方限定分支）：
  沿 K 打包的输入 → 16K×64N 的重排 tile → 线程收集 8 个特定逻辑位置
  → 按特定的 nibble 次序组 word → kernel 按对应布局加载和解码
```

参考源码位置：[repack kernel](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/gptq_marlin_repack.cu)、[tile 常量](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin.cuh)、[GEMM 模板](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/marlin_template.h)和[反量化实现](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/dequant.h)。参考快照来源和许可证见 [README](../third_party/vllm_marlin/README.md)；它不是我们的独立实现。

## 2. 具体追一个输出 word

只看 `num_bits=4, is_a_8bit=false, has_perm=false`、第一个逻辑 16×64 tile。tile 常量来自 `marlin.cuh`，四个 warp 执行 `repack_tile` 的重排部分；`warp_id=threadIdx.x/32`、`th_id=threadIdx.x%32`。不要把 repack 的 warp 分工误认为 GEMM kernel 的 warp 分工。

从源码 `repack_tile` 中取 `warp_id=0,th_id=0`：`tc_col=0,tc_row=0,cur_n=0`；按 `tc_offsets={0,1,8,9}` 收集的逻辑坐标是：

```text
vals[0..3]： Q[0,0], Q[1,0], Q[8,0], Q[9,0]
vals[4..7]： Q[0,8], Q[1,8], Q[8,8], Q[9,8]
```

这 8 个值还要按源码 `pack_idx={0,2,4,6,1,3,5,7}` 排列，最终 word 的 nibble **低→高**依次是：

```text
Q[0,0] Q[8,0] Q[0,8] Q[8,8] Q[1,0] Q[9,0] Q[1,8] Q[9,8]
```

图中坐标是 tile 内逻辑 Q；实际全局位置还要加 tile 的 K/N 起点。这个分支的 word 输出位置是 `out_offset + th_id*4 + warp_id`；上述 lane/warp 均为 0，因此是该 tile 的第 0 个 word。逻辑列 0 和 8 被组合进同一个 word，并非 V6 的“同一行、连续 8 列”。

```text
同一 tile 中的逻辑元素 Q[k,n]：
           n=0         n=8
k=0       nibble 0    nibble 2
k=1       nibble 4    nibble 6
k=8       nibble 1    nibble 3
k=9       nibble 5    nibble 7
```

这个例子是按本地源码分支静态推导；未在本节另跑上游 repack 的逐 word 差分实验。其他 warp、其他分支、`has_perm=true`、8-bit 激活或不同 bit 数需要分别追踪，不能套用本图。

## 3. scale、zero 和实验边界

V6 中 Z/S 保持 `[K/128,N]`、按逻辑 group/列读取。参考路径中 scale/zero 也可能经历准备与布局变换；本节没有重新推导其完整变换，不能把 V6 的 Z/S 物理指针直接交给 Marlin。也不能因两者都叫“repack”就认为能互换输出。

完成 V6c 后可以用**相同原始 Q/Z/S**分别准备 V5/V6 布局，检查正确性，再分别记录重排、显存和稳态 GEMM；如需比较 Marlin，必须另行确认上游构建、完整权重格式与输出舍入。当前 V6 文档没有性能实测，也不把这段参考源码描述为自己实现。
