# V6c：让 V5 低层 GEMM 读取重排后的 R

## 1. 本节只改变一个主要变量

V6b 验证 P→R 后，保留 V5f 的 block、warp、shared、`ldmatrix`、`mma.sync`、BF16 舍入和 C 写回；只把每个逻辑 Q 的读取换成从 R 定位。这样可以用同一逻辑输入把差异归因于布局与对应读取代码，但是否变快仍需实测。

```text
同一逻辑 Q → P[K/8,N] ─────→ V5f GEMM ─┐
                   └→ GPU 重排 → R ─→ V6 GEMM ─┼→ 对照同一个 CPU 参考
同一 A/Z/S ────────────────────────────────────┘
```

R 是 `[K/16,ceil(N/8),16]`，每个 word 存同一个 k、相邻 8 个 n。lane 的分工仍然是 [V6a](34_V6a_设计离线重排布局.md)第 3 节所画的映射。R 的 N padding 为 0，但 Z/S 没有 padding，V5 的越界判断仍要在读 R/Z/S 前执行。

## 2. 手算读取

例如 `N=19`，nt=3，`k=17,n=10`：ktile=1、kin=1、ntile=1、nibble=2。由 R 的三维行连续下标确定 word，取出第 2 个 nibble；该值必须等于逻辑 `Q[17,10]`。位运算可参考 V5，但打包轴从 K 改为 N。

这一节仍是 `M,N>0`、`K>0 且 K%128==0`、连续数据、A/S BF16、Z uint8、输出 FP32。每次先将解码 W 转 BF16；每 block 一个 warp 计算 C 的 16×8 tile，M/N 尾部补零/掩码，sync 和 accumulator 不变。

## 3. 你来完成

先确保已通过 V6b，并把完成的重排实现保存为 `kernels/marlin_from_scratch/v6b_repack.cu`。复制 [v6c_gemm_template.cu](../kernels/marlin_from_scratch/v6c_gemm_template.cu) 为 `v6c_gemm.cu`，只填写 `v6_read_q`：根据 R 的三维布局计算 word 的线性下标，右移并掩码取逻辑 Q。

模板复用了你完成的 `v5ef_lowlevel_template.cu` 的 MMA 包装以及 `v6b_repack.cu`，同时给出 V6 GEMM 的加载、同步和写回。为便于复用，V5ef 测试入口加了宏保护；不改变你已写的 kernel。若之后你修改了 V5ef 的 kernel，需要同步检查 V6c 中的相同部分。

```bash
cp kernels/marlin_from_scratch/v6c_gemm_template.cu kernels/marlin_from_scratch/v6c_gemm.cu
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
  kernels/marlin_from_scratch/v6c_gemm.cu -o /tmp/quant_v6c
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v6c
```

测试先实际运行你的 GPU repack，并用原始逻辑 Q 检查每一个 R word 和 padding；如果 R 有误会拒绝继续 GEMM。随后对同一组 A/P/R/Z/S 运行 V5f 和 V6，并分别与从 Q 得到的 CPU 参考比较，同时比较两者输出。测试固定 `atol=0.002,rtol=0.0001`；覆盖 `(16,8,256)`、`(1,1,128)`、`(17,19,256)`、`(31,33,384)`、`(16,128,256)`，每个形状两类输入。这里无计时，测试内的重排和逐 word 检查不属于稳态 GEMM。

RTX 4090、CUDA 12.1、SM89 上，临时参考填充的 V6b/V6c 通过上述测试；V6c 的 memcheck/racecheck/synccheck 均无错误或危害。这是框架验证；你的版本需另行 review，不能预设性能收益。

正确性通过后我会准备同输入的 V5/V6 稳态比较，并单列一次性重排耗时、显存及复用次数，不把重排隐藏在性能比较之外。

## 4. 你的实现验证与同口径初测

2026-11-24，在 RTX 4090、CUDA 12.1、SM89，编译选项 `-O2 -lineinfo` 下，你的实现通过 5 个形状 × 2 类数据的检查：逐 word 重排、V5/V6 分别与 CPU 参考、V5/V6 相互对比均无超差；本次两版输出的最大差为 0，不能推广到其他实现或所有输入。`memcheck`、`racecheck`、`synccheck` 均未报告错误或危害。

随后用 [v6c_benchmark.cu](../benchmarks/v6c_benchmark.cu) 测量相同 BF16 小数输入，物理 GPU 1：`GPU-deb35ff6-c14d-e1d9-5b24-8a7e4eb1bbd9`，驱动 535.309.01、CUDA runtime 12.1。运行前 GPU 无其他进程、利用率 0%；未锁频。每个形状先通过独立 CPU 参考检查，每版本 warmup 20 次；15 批交替测两种 GEMM，每批 100 次，用 CUDA event 计时。重排单独测每批 100 次并求批均值。表中是批均值的 p50，单位微秒，**不是单次延迟分布**；短 kernel 可能受到 launch 供给和时钟等影响。

| M | N | K | V5 p50 | V6 p50 | V5/V6 | 重排 p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 256 | 16.131 | 16.425 | 0.982 | 2.055 |
| 16 | 128 | 256 | 20.603 | 20.879 | 0.987 | 1.946 |
| 128 | 128 | 256 | 18.962 | 19.190 | 0.988 | 1.963 |
| 128 | 512 | 1024 | 73.165 | 73.758 | 0.992 | 1.669 |

这次四组的 V6 p50 都略高于 V5，差距约 0.8%～1.8%；只是一轮同机实验，不能排除不同运行条件或测量波动，不据此推广到其他形状。按本次 p50 的简单摊销模型，V6 没有节省稳态时间，所以**没有正的回本次数**：重复使用也不能摊回额外重排时间。这不代表其他实现或场景一定如此。

本例 N 均为 8 的倍数，P 和 R 大小相同：K=256、N=128 各为 16,384 字节，K=1024、N=512 各为 262,144 字节；如果准备期间同时保留 P 和 R，Q 相关缓冲合计分别为 32,768 和 524,288 字节，不包含 A/Z/S/C、分配器与其他工作区。不同 N 的 padding 开销见 V6a。重排时间是已经准备好 P/R 缓冲的 GPU kernel 时间，不含分配、复制或完整端到端准备。

`ptxas` 对本次两版 GEMM 均报告 40 个寄存器/线程、768 字节静态 shared、0 spill；重排 kernel 为 20 个寄存器、0 spill。这些静态资源不能说明 DRAM 访问量或具体瓶颈；本轮没有取得 profiler 硬件计数器。原始各批数据、编译输出与源码哈希见 [v6c_results](v6c_results/)。

```bash
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo -Xptxas=-v \
  benchmarks/v6c_benchmark.cu -o /tmp/quant_v6c_benchmark
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v6c_benchmark
```

基准通过 `V6C_NO_MAIN` 复用你填写的 V6c 模板；该宏只屏蔽模板的测试入口，未改变 kernel 计算。运行前仍需检查 GPU 空闲。
