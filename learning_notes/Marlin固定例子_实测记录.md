# Marlin 固定例子的实际运行记录

## 目的与边界

为后续课程选定一个实际运行的 kernel 实例；本记录不作性能基准。执行的是同级 `../vllm-from-scratch/` 中的参考实现与 `tests/step122` 的造数函数，**不是本学习项目独立实现的 kernel**。测试使用 BF16 输入、W4、G128、zero point、无 act-order，M=16、K=256、N=128，并带 bias。

## 环境与身份

- 运行时间：2026-11-21（本机时间）；宿主机 GPU 1，NVIDIA RTX 4090，SM 8.9，128 个 SM，UUID `GPU-deb35ff6-c14d-e1d9-5b24-8a7e4eb1bbd9`。使用 `CUDA_VISIBLE_DEVICES=1`，进程内设备号为 0。
- 驱动 535.309.01；PyTorch 2.13.0+cu129；CUDA 工具链 12.9；编译配置含 `-O2` 和 `-gencode=arch=compute_89,code=sm_89`。
- 参考项目 HEAD `3ae5eefb0ab3eecdbc75c16b7d7e733c5116aec0`；本次完整扩展构建身份 SHA-256：`be69fade8e7dcd95df49f6705b6e67590b8b0fbc3dd5a5aa6dbd5605591b462e`。
- 本地快照与参考项目的 `marlin.cu`、`marlin_template.h`、`marlin_mma.h` 的 SHA-256 分别对比一致；快照 selector 与本次构建缓存 selector 的 SHA-256 均为 `6915cb70c8e0638d7f77c7a21246409d77d385770d04fce4c805c5f84407f897`。这验证了列出的文件，而非声称两份项目所有文件相同。

## 正确性与实际 launch

运行 `step122` 造数函数准备带 bias 的层，GPU 结果与独立构造的 FP32 反量化矩阵乘参考比较，通过 `rtol=0.02, atol=0.05`；最大绝对误差 `0.015332937240600586`。这只是单个输入的正确性检查，不能代表所有形状正确。

PyTorch profiler 捕获到：

```text
marlin::Marlin<..., 128, 1, 4, 8, false, 4, 8, false>
grid  = (128, 1, 1)
block = (128, 1, 1)
```

根据 [kernel.h](../third_party/vllm_marlin/minivllm/csrc/quantization/marlin/kernel.h:24) 的模板参数顺序，数值参数依次为 `threads=128, thread_m_blocks=1, thread_n_blocks=4, thread_k_blocks=8, m_block_size_8=false, stages=4, group_blocks=8, is_zp_float=false`。所以该实例对应 `thread_n=64, thread_k=128`；每 block 4 个 warp。profiler 中前四个模板参数显示为整数类型 ID，本记录结合 BF16/W4 输入及源码说明类型，不假称仅靠整数名称能直观看出类型。

这次 trace 还报告每线程 154 个寄存器、shared memory 101376 字节；这是该次 profiler 元数据，不等于性能结论。trace 中 occupancy 相关字段包含 0，尚未核实其含义，不能用它断言实际占用率。

## 证据与复现

- [运行脚本](Marlin固定例子_运行脚本.py)：使用同级参考项目的 `.venv`、测试辅助函数和 CUDA 环境；它会将临时 trace 写到 `/tmp/quant_marlin_trace.json`。
- [本次 profiler 原始 trace](Marlin固定例子_profiler.json)：包含 kernel 名称及 launch 元数据；其中的 device 0 是本进程可见设备编号。
- 权重、激活、偏置和参考的生成方式见脚本及同级项目 `tests/step122/test_step122_cuda_integration.py` 的 `make_layer`。本次未保存输入张量；重新运行可按固定 seed 重建，但跨版本 bitwise 一致尚未验证。

当前已确认 launch 与模板实例；block 的计算分配可据此追踪源码。**每个 warp/lane 的数据布局、解码、MMA 和归约尚未逐项验证**，后续按实际分支继续推导，不能把已拿到 launch 当成已理解全部 kernel。
