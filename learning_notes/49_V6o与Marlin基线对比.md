# V6o 与参考 Marlin：固定形状基线

## 本次要回答的问题

V6o 对其固定契约是完整的 W4A16 GEMM 路径：输入 BF16 A、预先重排的 uint4 Q 和整数 zero、BF16 scale/bias，输出 BF16 C。当前不仅 K=256、N=128 固定，**M 也固定为 16**；没有通用形状、tail、padding、act-order、跨 stream 共享 workspace 等支持。本次对比不把它称为通用 Marlin 实现。

## 实验设置

- 日期 2026-11-25；NVIDIA RTX 4090，SM89；GPU 1 测试开始时空闲，GPU 0 当时另有任务。
- A `[16,256]` BF16，逻辑 Q `[256,128]` uint4，整数 Z `[2,128]`、BF16 S `[2,128]`，group=128，BF16 bias `[128]`，C `[16,128]` BF16；无 act-order，FP32 reduce。
- 两侧使用**同一** A、逻辑 Q/Z/S/bias；通过同一参考 repack/重排生成两侧相同的 Q/Z/S，参考 bias 按 Marlin 入口要求重排，V6o 使用逻辑 bias。重排不计入时间。
- 参考是本机 `../vllm-from-scratch` 的 `minivllm_quant::marlin_gemm`，源自 vLLM Marlin；本次其 `marlin.cu`、`marlin_template.h` 的 SHA256 与工作区快照相同。参考扩展 build identity 为 `be69fade8e7dcd95df49f6705b6e67590b8b0fbc3dd5a5aa6dbd5605591b462e`；两者来源与许可证见 `third_party/vllm_marlin/README.md`。
- 参考扩展由 CUDA 12.9、PyTorch 2.13.0+cu129 构建；V6o 桥接库由 CUDA 12.1、`-O2 -arch=sm_89` 构建。两侧都是 SM89 GPU 代码，但工具链不同是比较的限制。
- 输出预分配；计时前各 warmup 50 次。每条 CUDA Graph 含 128 次调用，预热 graph 5 次；交替两侧顺序测 21 个 batch，CUDA event 记录 graph 总 GPU 时间后除以 128。结果是稳态 GPU 时间，不包含输入准备、repack、graph 构建、Python 调用和常规端到端开销。V6o 计时中仍包含 FP32 diagnostic 额外写入以及锁/临时缓冲的操作；参考 kernel 有自己的 C_tmp 和归约工作，两者具体执行细节不同。

## 正确性

两组输入分别是：seed 20261125、binary 精确 scale；seed 20261126、从 0.1/0.7 等转为 BF16 的 scale。两组各有 2048 个输出：V6o 与参考 Marlin 的 BF16 逐位差异均为 0；独立逻辑 Q/Z/S 的 BF16 反量化加 FP32 矩阵乘法 oracle 对两者最大绝对误差也均为 0。锁最终是 `[0,0]`。

通过阈值为 `atol=0.125, rtol=0.02`；逐位相同是这两组输入的观察，不保证其他输入、归约顺序或所有边界均逐位相同。oracle 是 Torch FP32 matmul，因此与 Marlin 不强制相同的累加顺序。

## 测得的稳态时间

| 输入 | V6o 中位数 | 参考 Marlin 中位数 | V6o / 参考 |
|---|---:|---:|---:|
| 精确 scale | 7.840 μs | 3.239 μs | 2.42× |
| 非精确 scale | 7.832 μs | 3.240 μs | 2.42× |

V6o 在本次口径下约为参考的 41% 吞吐，或者说耗时约 2.42 倍。实验规模很小、固定四 block，不能外推其他 M/N/K；没有通过 profiler 测出瓶颈，不能把时间差归因到某个单一因素。性能结果只是优化起点，不是最终公开结论。

可能值得分开测量的差异包括：逐标量读取 Q/Z/S 与参考的向量化/共享内存路径；同步、跨 block 锁和额外诊断写入；寄存器使用和占用率；参考流水线等。这些只是待验证的假设。建议先建立 V6o 的无诊断写入对照、记录寄存器/本地内存和编译代码，再逐项做正确性与性能对比；避免一次改变多个因素。

## 复现

```bash
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo -shared -Xcompiler -fPIC benchmarks/v6o_benchmark_wrapper.cu -o /tmp/quant_v6o_benchmark.so
nvidia-smi
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/data/syh/infra/cuda-12.9 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 ../vllm-from-scratch/.venv/bin/python benchmarks/benchmark_v6o_vs_marlin.py
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/data/syh/infra/cuda-12.9 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 BENCH_SCALES=inexact BENCH_SEED=20261126 BENCH_OUTPUT=learning_notes/v6o_results/benchmark_inexact.json ../vllm-from-scratch/.venv/bin/python benchmarks/benchmark_v6o_vs_marlin.py
```

原始 batch 值在 [精确 scale 数据](v6o_results/benchmark.json)与[非精确 scale 数据](v6o_results/benchmark_inexact.json)，源代码在 [benchmark](../benchmarks/benchmark_v6o_vs_marlin.py) 与 [CUDA 桥接](../benchmarks/v6o_benchmark_wrapper.cu)。
