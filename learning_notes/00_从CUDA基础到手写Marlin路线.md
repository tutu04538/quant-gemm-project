# 从 CUDA 基础到手写量化 GEMM：学习与项目路线

> 目标：在理解当前项目的 Marlin W4A16 代码之后，自己从一个可运行的基础矩阵乘出发，逐步实现、验证和优化一个量化 GEMM；用 profiler 解释每一步为什么快或为什么没变快，最终在明确的设备和输入范围内与 Marlin 比较，形成可以在面试中讲清楚的项目。
>
> 学习方式：我带你一关一关走。每次固定一个小例子，先画数据在内存、block、warp、lane 之间的流动，再读少量代码、手算索引、自己写实现、做测试和测量。**不以看完源码或复制代码为毕业标准。**
>
> 当前机器：RTX 4090，SM 8.9；仓库记录的 CUDA 工具链为 12.9。实际每次测试仍需记录当时的 GPU、驱动、CUDA、编译选项和占用情况。本文是计划，不代表其中的算子、性能结果或验收已经完成。

> 路径约定：本文已迁入 `quant-gemm-project`。文中对 `minivllm/` 的 Marlin 相关源码参考可在工作区 `third_party/vllm_marlin/minivllm/` 找到；具体快照及生成文件见 `third_party/vllm_marlin/README.md`。`tests/step121/`、`tests/step122/` 和 `scripts/activate-4090.sh` 仍指本机同级的 `../vllm-from-scratch/`，没有复制。自己的实现放在 `kernels/`、`tests/` 等目录；`learning_notes/` 随仓库同步。

## 1. 我们最终要做出什么

最小目标不是实现一个通吃所有量化格式的推理框架，而是先独立做出一个**支持指定输入范围的 W4A16 CUDA GEMM**：

```text
A：激活，BF16，形状 [M, K]
Q：量化权重，逻辑形状 [K, N]，每个权重 4 bit，实际按 int32 打包
S：量化 scale
Z：量化 zero point

对每个逻辑权重：W[k,n] = (Q[k,n] - Z[k/128,n]) * S[k/128,n]
输出：C[m,n] = Σ_k A[m,k] * W[k,n]，输出 BF16
```

这里的 `k/128` 是向下取整：每 128 个输入通道为一组，对每个输出通道各有自己的 scale 和 zero point。先支持 **W4、BF16 激活、group size 128、非对称 zero point、无 act-order、单 GPU**，对应本项目当前 AWQ 主线。计算参考使用 FP32 累加，并单独记录实现和上游的实际舍入顺序；上面的公式表达数学语义，不预先规定所有中间值怎样舍入。

最终交付至少包括：

1. 一个别人能独立构建和调用的 CUDA/C++ 算子，写清楚输入布局、shape、dtype、对齐要求和不支持的情况。
2. 可复现的正确性测试，包括独立 FP32 参考、小矩阵可手算例子、随机测试和边界测试。
3. 从基础实现到优化版本的代码或可复现实验记录：每次只改一个主要变量，记录结果，而不是只保留最终版本。
4. 公平的性能比较：基准输入、算子边界、权重布局、预处理、硬件、计时方式、误差和原始结果均可追溯。
5. profiler 分析：能指出瓶颈、提出假设、运行实验，并解释数据是否支持假设。
6. 项目文档：画出数据流、warp/lane 布局，解释取舍、局限、对照和个人贡献，作为简历材料。

**性能目标**：先在若干指定 case 上不断缩小和本机冻结 Marlin 的差距；能否达到或超过，必须实测。不同 M/K/N、预处理口径和 GPU 上结果可能不同。最终即使没有追平，也可以诚实展示优化过程、原因及数据，不能用未经验证的倍数写简历。

## 2. 固定例子：后面所有概念先用它讲

先约定 `M=16, K=256, N=128, group=128`：

```text
输入 A                  逻辑权重 Q                  输出 C
      K=256                   N=128                       N=128
   ┌────────────┐          ┌────────────┐              ┌────────────┐
M  │            │    ×   K │ 4-bit 权重 │      =     M │            │
16 │            │      256 │            │           16 │            │
   └────────────┘          └────────────┘              └────────────┘
     [16,256]                [256,128]                   [16,128]

沿 K 的分组：
  k =   0 ... 127  → group 0，使用 S[0,n]、Z[0,n]
  k = 128 ... 255  → group 1，使用 S[1,n]、Z[1,n]

S、Z 的逻辑形状都是 [2,128]。
Q 的 [256,128] 是逻辑形状，不代表显存里有 32768 个 int32。
```

一个计算例子：假设 `Q[3,5]=11`、`Z[0,5]=7`、`S[0,5]=0.25`，则该权重的数学值是 `(11-7)×0.25=1`；计算 `C[2,5]` 时还要把 `A[2,3]×1` 与其他 255 个 K 项累加。为了便于手算，本例使用简单数值，不代表真实 checkpoint 的 scale。

本项目中常见的三种权重表示：

```text
一、AWQ checkpoint：qweight [K, N/8]    int32，沿 N 每 8 个 4-bit 值打包
   配套 qzeros [K/128, N/8]、scales [K/128, N]
          ↓  重排 AWQ nibble 顺序并转换打包轴
二、standard：       qweight [K/8, N]    int32，沿 K 打包
   标准 qzeros 为 [N/8, K/128]；scale 仍以具体变换阶段为准
          ↓  padding + Marlin repack；scale / zero 也各自重排
三、Marlin kernel：  qweight [K/16, N×16/8] = [K/16, 2N] int32
```

这是当前项目支持的 AWQ W4/G128 具体路径，**不是所有 AWQ/GPTQ 工具都采用的通用格式**。上述形状对于 Marlin 输出应使用实际 padding 后的 K、N。已有详细图解见 [Marlin repack 数学映射](Marlin_repack数学映射.md)；后续讲 kernel 内布局时，我们也按这种画图方式展开。

## 3. 全程路线图

```text
阶段 A：补 CUDA 基础                    阶段 B：看懂当前 Marlin 的主线
┌──────────────────────────┐        ┌──────────────────────────────────┐
│ 线程/内存/同步 → 基础 GEMM │ ─────→ │ Python 入口 → 权重布局 → launch │
│ 正确性 → CUDA 计时/工具  │        │ → warp 数据流 → 解量化 → MMA   │
└──────────────────────────┘        └──────────────────────────────────┘
                     └─────────────────┬────────────────┘
                                       ↓
阶段 C：自己手写，逐版测量                    阶段 D：复盘与项目化
┌─────────────────────────────────────┐  ┌───────────────────────────────┐
│ BF16 GEMM → 最简单 W4A16 → 线程协作 │  │ 固定矩阵与形状测量、profiler │
│ → Tensor Core → 重排 → 异步流水线   │→ │ 消融、局限、可复现 README     │
│ → 小 M 并行与归约 → 调参和边界      │  │ 简历：用实测事实讲个人贡献   │
└─────────────────────────────────────┘  └───────────────────────────────┘
```

阅读与动手交错进行：不要求先把复杂 Marlin 内核全部读完，才允许写代码。每到一个优化点，先学够用的 CUDA 机制，再看 Marlin 对应部分，然后自己做一个可比较的小实验。

## 4. 阶段 A：把 CUDA 语法和基础补回来

### 第 0 关：认识工具和一次 kernel 调用

- 学：host 与 device；`.cpp` / `.cu`；`nvcc`；`__global__`、`__device__`、`__host__`；`kernel<<<grid, block, shared_bytes, stream>>>(...)`；CUDA 错误检查；同步为什么会影响计时。
- 动手：写 vector add；从 Python 传 GPU tensor 调用，画出 Python → C++ binding → CUDA kernel 的调用链；检查结果和错误传播。先独立运行，再决定用 PyTorch extension 接入项目。
- 过关：能解释“CPU 发起 kernel”和“GPU 执行 kernel”的区别；知道 launch 是异步的，不能用不包含同步的普通 CPU 计时当 GPU 耗时。
- 不在这一关学完整 PyTorch dispatcher、模板元编程或 inline PTX。

### 第 1 关：线程、warp 和地址

- 学：grid、block、thread、warp、lane；`blockIdx`、`blockDim`、`threadIdx`；一维/二维索引；越界 mask；warp 中哪些线程访问相邻地址。
- 动手：对长度为 37 的向量，在 block 大小为 32 时画出 2 个 block 的 thread → 元素映射；最后 27 个线程怎样避免越界。再写矩阵转置，观察读取与写入哪个方向连续。
- 过关：给定 block/tile 配置，能独立画出一个元素由谁读写；理解合并访问是 warp 内相邻线程访问模式的问题，而不是“用了 shared memory 就自动合并”。

### 第 2 关：shared memory 与同步

- 学：global/shared/register 的作用域和生命周期；`__shared__`；`__syncthreads()`；数据竞争；shared memory bank conflict 的直觉；寄存器压力和溢出到 local memory 的可能性。
- 动手：让 block 合作读一块数据；写一个正确和一个故意缺同步的实验。用 `compute-sanitizer` 的 `memcheck`、`racecheck`、`synccheck` 排查（工具支持和当前环境权限先实测）。
- 过关：能画出“写 shared → 全 block 同步 → 读 shared → 再复用缓冲前同步”的依赖；知道 `__syncthreads()` 是 block 内协作，不是跨 block 同步。

### 第 3 关：从普通 GEMM 重新建立直觉

- 学：`C=A×B` 的 M/N/K；block tile、thread tile、K 循环；全局读、shared 复用、寄存器累加；边界 tile 与同步。
- 动手：先写最简单的 FP32 GEMM，再写 shared-memory tiled GEMM；画一张 `A tile × B tile → C tile` 图，明确每个 block 算哪个 C 子块。
- 过关：支持非整除 shape，与 PyTorch/独立参考核对；能指出每次循环搬了什么、哪些值被复用。**本关目标是恢复手感，不把基础 CUDA core 实现当 Tensor Core 基线。**

### 第 4 关：测量和 profiler 最小入门

- 学：warmup、CUDA event、重复次数、median/p90；异步执行、首次 JIT/构建、频率与功耗、显存占用和后台任务对结果的影响；Nsight Systems 与 Nsight Compute 的不同用途。
- 动手：对第 3 关 kernel 做计时；同时用 Nsight Systems 看 CPU launch 和 GPU 时间线，用 Nsight Compute 对单个指定 kernel 看带宽、计算和访存。每项先检查工具及权限是否可用；有工具不代表当前账号一定能访问 GPU 计数器。
- 过关：能给出一份带设备、shape、dtype、warmup、重复次数、计时边界的实验记录，并解释 profiler 结果本身可能扰动运行时间。

## 5. 阶段 B：从入口到内核，分层读懂当前 Marlin

### 先看整体调用链

```text
模型 Linear.forward
  └─ AutoAWQMarlinLinearMethod.apply                    awq.py
       └─ apply_marlin_w4a16                            marlin.py
            └─ marlin_gemm_w4a16                        quant_ops/ops.py
                 └─ torch.ops.minivllm_quant.marlin_gemm
                      ├─ schema / 注册                   bindings.cpp、marlin.cu
                      ├─ 检查参数、选配置、发射          marlin.cu
                      └─ __global__ Marlin<...>          marlin_template.h
                           ├─ 搬运/同步工具               marlin.cuh
                           ├─ 4-bit 解码                 dequant.h
                           └─ Tensor Core 指令           marlin_mma.h
```

加载模型时还有一条**只在准备权重时执行**的路径：AWQ 参数加载 → `_convert_awq_to_standard_format` → `prepare_marlin_weights` → `gptq_marlin_repack`。它和每次前向调用 GEMM 是两条路径；不要把 repack 算作每次 GEMM 都会执行，也不能比较性能时悄悄只给其中一方排除 repack。

### 第 5 关：Python 入口和参数契约

- 按顺序读：`minivllm/layers/quantization/awq.py` 的 `process_weights_after_loading`、`apply` → `minivllm/layers/quantization/marlin.py` 的 `prepare_marlin_weights`、`apply_marlin_w4a16` → `minivllm/quant_ops/ops.py` 的 `marlin_gemm_w4a16`。
- 动手：对固定例子的每个张量画出准备前后形状、dtype、设备、生命周期；标记 padding 的逻辑维度和实际 kernel 维度；追踪 bias、workspace、g_idx 和 perm 的含义。
- 过关：能从 Python 调用准确指到 CUDA 入口，并解释 qweight、scale、zero point 谁在何时重排；知道量化 GEMM 不需要先把**整层**权重还原成 BF16 再调用 `F.linear`。

### 第 6 关：repack 与布局，不急着读 GEMM 内循环

- 对照已有 [repack 图解](Marlin_repack数学映射.md)，读 `minivllm/csrc/quantization/marlin/gptq_marlin_repack.cu` 的 `gptq_marlin_repack_kernel`：先输入/输出，再 tile、warp、lane、每个 int32 的 8 个 nibble。
- 动手：用小整数给逻辑 `Q[k,n]` 填可识别的编号；独立写 CPU reference，证明 AWQ packed → standard → Marlin 的映射。每个阶段列出 tensor 的物理轴、逻辑轴、位序与 padding。
- 过关：不看答案，能画出一个选定 tile 中某个 `q[k,n]` 被哪个 lane 装入哪个 int32；能解释 GEMM 如何按相同约定读回来。测试整数应逐元素精确；**不能只测自己 pack 后自己 unpack 的往返**。
- 单独讨论 `perm`、`g_idx`、GPTQ act-order；它们在当前无 act-order 主线不是必需输入，不要和表示权重位置的 permutation 混为一谈。

### 第 7 关：从 C++ 发射点倒推实际特化

- 读 `bindings.cpp`、`marlin.cu` 的 `marlin_gemm`、配置检查/选择与 `kernel<<<...>>>`；再读 `kernel.h` 的模板参数和 `generate_kernels.py` 的实例化。
- 动手：对一个真实输入记录 M/N/K、实际 padding、线程数、thread tile、stages、group 等选择；逐个解释 `thread_m_blocks`、`thread_n_blocks`、`thread_k_blocks` 以 16 为单位表达什么。生成模板参数是编译期特化，运行时配置选择决定实际命中哪个 kernel，二者不要混为一谈。
- 过关：不用读完整个主循环，也能画出输入 → 校验 → 配置选择 → 具体 kernel → 输出的路线；能区分 Python 包装、C++ host 代码、GPU kernel。

### 第 8 关：Tensor Core 的最小可运行例子

- 学：warp 协作执行矩阵乘；MMA 的输入 fragment 和累加 fragment；FP16/BF16 输入与 FP32 accumulation；先用 CUDA 提供的 WMMA 接口建立模型，再按本机 SM89 能力学习 `ldmatrix` 和 `mma.sync`。
- 动手：写一个小型 BF16/FP16 Tensor Core GEMM；画出一个 warp 的 fragment 如何对应矩阵 tile，比较非 Tensor Core 基础版本。
- 过关：用工具检查实际使用的指令或 kernel，并解释 PTX 是虚拟指令、最终执行以编译后的机器指令为准；不能仅因代码写了 WMMA 或 PTX 就断言达到了理想吞吐。局部索引按具体指令及布局逐项推导，不凭经验猜 lane 对应关系。

### 第 9 关：按一次 K 循环读 Marlin 主计算

- 聚焦 `marlin_template.h`：输入/输出指针 → shared 布局 → 从 global 取 A/Q/scale/zero → 等待 → fragment → 解码与 MMA → 输出。只追当前 W4A16、BF16、group128、zero point、无 act-order 的分支。
- 配合阅读 `dequant.h` 的 `kU4`/BF16 路径、`marlin_mma.h` 的 BF16 MMA、`marlin.cuh` 的 `cp_async*` 封装。文件里其他 FP8/INT8、不同架构与量化配置先跳过。
- 动手：画时间线与寄存器/内存图；选一个线程实际读取的 packed int32，跟踪它的 8 个 4-bit 值如何进入 fragment。把 `lop3`、位操作、zero 和 scale 分成几步，并用手算小样本核对。
- 过关：能解释“packed 权重到达 fragment、如何解码、何时使用 scale/zero、如何参与 MMA”，并能指出源码中具体分支。四舍五入/混合精度顺序必须根据实际分支验证，不凭公式推断 bitwise 相等。

### 第 10 关：异步流水线与跨 block 归约

- 学：`cp.async` 的发起、commit、wait；shared buffer 何时可复用；`stages` 与共享内存/寄存器/占用率的取舍。先画单缓冲、双缓冲，再看当前实现的多 stage，区分依赖关系和真实硬件重叠是否发生。
- 学：小 M 下为什么可能需要把 K 方向拆给多个 block；局部结果如何归约；workspace/locks、atomic 与非 atomic 方案的正确性和代价。以实际 dispatch 和 kernel 路径为准，不能假设所有 shape 都走同一种归约。
- 过关：画出至少两轮 K tile 时间线；能说清某块 shared buffer 被覆盖前需要等待什么、哪些线程需同步；用最小实验验证竞争和不正确归约，而不是只记 API 名称。

## 6. 阶段 C：从头实现，版本化推进

### 6.1 目标、目录和学习方式

目标是独立实现一个**受 Marlin 启发的 W4A16 GEMM**，而不是把参考源码改名后称为从零实现。最终主线先限定为单 GPU、BF16 激活、4-bit 无符号权重、group size 128、非对称 zero point、无 act-order；输出和 bias 的支持按各版契约明确。先有正确实现，再按实际瓶颈逐项优化；不承诺追平 Marlin。

- 新算子和小实验放在 `kernels/marlin_from_scratch/`；此前基础练习保留在 `kernels/base_learn/`。
- 每节学习文档放在 `learning_notes/`，随仓库同步。值得公开展示的设计和真实实验结果，再整理到 `docs/`。
- 需要用户手写算子的节，在 `kernels/marlin_from_scratch/` 提供本节 template：输入输出、编译、基础错误检查与测试框架由助手准备，**只留下本节学习目标相关代码**。源码阅读、推导或实验分析节只写文档，不强行为简单公式或位操作创建练习。
- 每节先定义目标、张量和物理布局，再画 block/warp/lane 与内存流、手算小例子、解释必要语法，然后由用户完成；助手先给提示和 review，再按约定运行测试。不能把整份 Marlin 内核一次交给用户。
- 实现、测试、模板及实验记录按版本命名，避免后续覆盖前一版。辅助测试可以共享，但关键 bit 布局、数学语义与实现应有独立来源；目录、构建工具等具体约定从 V0 开始确定，不要求现在预建全部文件。
- `kernels/base_learn/10.cu` 是已完成的 shared-memory GEMM 学习练习及参考，不自动等于新目录下已经完成 V1；后续从 V0 开始按项目契约重写或复用经过确认的思路。

### 6.2 先约定数据语义，不预先承诺全部形状

贯穿的手算调试例为 `M=16,K=256,N=128,group=128`：

```text
A [M,K]：BF16 激活，按行连续
Q [K,N]：逻辑无符号整数，Q[k,n] ∈ [0,15]
Z [K/128,N]：每组每列的 zero point，逻辑整数 ∈ [0,15]
S [K/128,N]：每组每列的 BF16 scale
W[k,n] = (Q[k,n] - Z[k//128,n]) * S[k//128,n]
C[m,n] = Σ(k=0..K-1) A[m,k] * W[k,n]
```

`Q`、`Z` 的逻辑下标不是物理 packed 地址；最初版本可以由我们自己约定简单的 int32 打包轴与 bit 位序，但必须画图和写成契约，不能默认与 checkpoint、standard 或 Marlin 重排后的格式兼容。各阶段分别声明 scale/zero 的物理布局、是否包含 bias、输出 dtype、矩阵 stride、对齐及 padding；不支持的情况明确拒绝。最初可只支持 group/pack 对齐的有限正尺寸，随后逐步扩大范围；不能把尚未实现的 padding 写成已有功能。

对输入 BF16、输出 BF16 的主线，数学参考由原始逻辑 Q/Z/S 独立构造 FP32 W 并以 FP32 累加；同时记录 kernel 中解码、减 zero、乘 scale、MMA、输出与 bias 的**实际中间类型和舍入顺序**。由于量化后权重可能先舍入为 BF16，kernel 与 FP32 数学参考不必逐位相同；在测试前登记容差，并报告最大/平均误差和越界数。FP32 GEMM 初级版本按自己的 FP32 契约检查，不能直接套用量化 BF16 的容差。

### 6.3 版本与依赖：主里程碑下设检查点

V0～V9 是项目里程碑，字母后缀是逐步学习和验收的检查点；**不是要求每个检查点都做一个生产 kernel，也不是保证每一步都变快**。需要不同机制才能定位问题时再拆分；对已经掌握的部分可通过画图和测试合并，不为增加编号而增加机械练习。

```text
V0 FP32 朴素 → V1 FP32 shared tiled ──────────────┐
                                                   ├→ V4 协作访存
V2a/b BF16 语义与 WMMA → V2c/d MMA 数据布局 ────┐│
                                                ├┴→ V5 tile 解码 + Tensor Core
V3a/b 独立 packed/reference → V3c/d W4 基线 ────┘
                                                        ↓
                           V6 权重布局 → V7 异步流水线
                                                        ↓
                                   V8 小 M 归约 → V9 项目化
```

依赖是“证据必须先有”，不是所有编号必须串行：例如 V2 的 Tensor Core 基线和 V3 的 W4 基线可以按学习进度交错；V5 同时需要这两条路径。V4 是可调换顺序的性能探索，不应阻止先做正确的 V5。

### 6.3.1 目标校准：什么叫“复现 Marlin”

2026-11-23 重新审阅：**按原 V6～V9 列表逐项完成，不能直接宣称已经复现 Marlin。** 原先的主目标是做一个受 Marlin 启发、能够独立解释和评测的 W4A16 算子；而“复现”还需要指定参考实现、形状、功能与内部机制。现将目标分成三个可独立验收的层次：

1. **数学/接口等价**：对指定 Q/Z/S、输入、输出 dtype、bias 和允许的形状，与独立参考及冻结 Marlin 达到事先约定的误差；内部布局、算法和速度可以不同。
2. **固定实例机制复现**：以 [固定实例](12_Marlin固定实例_启动布局与反量化.md) 为目标，明确并逐层验证输入准备、packed Q/Z/S、block/warp/lane 映射、解码片段、MMA、流水线、跨 block 归约和输出。这不表示复现所有模板实例或所有量化格式。
3. **性能接近与范围扩展**：在指定 GPU、真实 shape 与可复现口径下测量并优化。性能是否接近必须实测，不因机制看起来相似就自动成立；也不承诺覆盖所有 Marlin 功能。

当前已验证的 V5f 是每 block 1 warp、一个 16×8 输出 tile、K=16 循环；每个 W 先逐元素解码到 BF16 shared，再由 `ldmatrix` 读取。V6b/c 的自定义 R 只是一个教学重排，已完成声明范围内的正确性验证，它不是 Marlin 格式。V5/V6 当前教学输出是 FP32、没有 bias；固定参考实例是 BF16 输出且带 bias。已有固定实例的 block/warp/lane 图是**源码推导**，已验证 repack 物理位序；不能将其误称为已逐 lane 捕获到实际运行寄存器。

固定实例：BF16 A `[16,256]`、逻辑 U4 Q `[256,128]`、group 128、整数 packed zero、无 act-order、BF16 bias 与输出；参考 trace 为 128 个 block、每 block 128 线程、4 warp、block N/K tile 为 64/128、4 stages，参考时记录实际 dispatch 和工作量；这只是一个历史实测实例，不假定其他设备/形状或每个 block 都按同一方式工作。输入/输出精度、零点、scale 和 bias 的实际执行顺序必须根据固定分支源码及实验单独核对，不能仅凭数学公式假定逐位相等。

要补上的核心桥梁：

```text
当前 V5/V6：每个逻辑 W 标量解码 → BF16 shared → ldmatrix → MMA
      ↓ 仍缺：参考布局、每 lane 寄存器片段、packed zero/scale、混合精度顺序
固定实例目标：特定重排 Q/Z/S → warp/lane 取 packed word
             → 寄存器级批量解码/反量化 → MMA fragment
             → 4-warp 协作、K/N tile、流水线、部分和归约、BF16 输出
```

`dequant.h` 中的 BF16 `kU4` 路径、`marlin_template.h` 的 `frag_b_quant`、MMA、流水线与归约分支是具体对照对象；不要假定所有格式都走同一反量化路径。后续仍优先实现自己能解释的版本；若为了兼容而复制或改编上游代码，要标记文件和具体来源、遵守许可证，不把移植代码称为独立从零实现。

**2026-11-24 学习方向调整**：用户要求后续以固定参考实例的 Q/Z/S 布局和实际 block/warp/lane 分工为主线，不再引入自定义物理布局作为练习目标。已完成的自定义 V6a～V6c 保留为历史对照。此后逐小节复现参考路径，简化只能裁剪输入规模或暂时关闭其他分支，必须明确哪些线程/计算路径还没实现；需要实际验证映射和舍入，不把源码推导写成 GPU 寄存器实测。复用、改编或移植参考代码时标明来源和许可。

#### V0：朴素 FP32 GEMM 与测试基座

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V0a：契约和调用 | 确定行优先 A/B/C、维度、stride 支持、输出初始化、构建与调用；助手准备可运行 host/测试模板。 | CPU 小例子可手算；编译、分配/复制、kernel 发射、错误检查、同步与释放的责任清楚。 |
| V0b：单线程单输出 | 自己写最简单 FP32 GEMM：每线程对应一个 C 元素，遍历 K。 | 矩形和非整除线程块的 M/N、不同 K 长度与独立 CPU 参考比较；越界工具可用时检查；暂不测优化收益。 |
| V0c：基准口径 | 固定输入、容差、warmup、重复次数、CUDA event 边界；区分一次 GPU kernel 时间与复制/构建/端到端。 | 先正确后计时；保存原始数据和机器、编译参数、GPU 占用，不用 CPU 异步发射时间冒充 GPU 时间。 |

#### V1：shared-memory FP32 tiled GEMM

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V1a：协作 tile | 使用 shared memory 合作加载 A/B tile、逐 K tile 累加并同步；参考基础练习但在新契约下自己完成。 | 手画每 block 的 C tile、线程加载及边界补零；不能让越界线程绕过 block 必需的屏障。 |
| V1b：边界和对照 | 覆盖 M/N/K 非整除、至少两轮 K tile；检查复用前后同步。 | 和 CPU 参考及 V0 比较；`compute-sanitizer` 的 memcheck/racecheck/synccheck 在工具和权限可用时运行，无法运行需如实记录。 |
| V1c：最小性能分析 | 比较 V0/V1 同一 case，记录资源与访问模式，提出一个可证伪假设。 | 本阶段只建立证据链；不把“使用 shared”自动等同于访存合并或一定加速。 |

#### V2：BF16/FP16 Tensor Core 基线（不引入量化）

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V2a：精度和最小 WMMA | 理解 FP16/BF16 的表示、FP32 累加与输出舍入；用 CUDA 支持的 WMMA 接口做一个 warp 的小矩阵乘。 | 明确当前 GPU、编译架构和工具支持；输入、累加及输出类型逐项记录，测试允许事先确定的浮点误差。 |
| V2b：多 tile Tensor Core GEMM | 将 WMMA 扩展到多个 warp/block 和有限的 K 循环；先支持明确对齐形状，再增加合理的尾部处理或主动拒绝。 | 每个输出 tile 的所有权、A/B layout、对齐、同步和边界明确；与 CPU FP32 参考及基线比较。 |
| V2c：MMA 片段与 `ldmatrix` | **可先做文档与定位实验**：从一条具体 `mma.sync` 的形状、lane、寄存器与布局建立模型，练习 shared 到 fragment 的地址；必要时给小模板。 | 不把 WMMA 的不透明 fragment 布局当成稳定逐 lane 契约；对具体指令/架构按源码与实验验证，未验证位置明确标注。 |
| V2d：最小低层 MMA | 在理解 V2c 后，先单独完成 `mma.sync` lane/寄存器实验，再用 [V2d 衔接实验](23_V2d_ldmatrix衔接MMA.md)把 shared→`ldmatrix`→`mma.sync` 接起来。可复用已有工具代码但注明来源。 | 独立小矩阵、多组可识别数据及每条 lane 的 fragment 位模式验证，不只依靠最终 C；检查编译后的 PTX/SASS 或 profiler 证据，区分源码、PTX 与实际机器指令。 |

WMMA 先建立 warp 协作模型；V2d 才为以后精确控制 fragment 布局提供基础。若工具链或时间不允许 V2d，可先以 WMMA 的 shared tile 路径完成正确的 V5a，再回来补低层 MMA；不得因此声称已经完成寄存器级融合或实现 Marlin 相同布局。

#### V3：独立量化参考与朴素 W4A16

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V3a：量化数据契约 | 固定自己定义的 packed 轴、每个 int32 的 8 个 nibble 位序、scale/zero 排列及对齐；画 Q 与物理地址映射。 | 用手算固定 bit pattern 与独立 CPU 参考分别验证 pack、定位和提取；覆盖 0、1、7、8、15、不同 zero、正负权重、两个 group。 |
| V3b：BF16 数值参考 | 建立逻辑 FP32 数学参考，并单独实现/说明“解码结果转 BF16、乘 scale 的实际舍入”参考，明确是否含 bias。 | 比较前登记容差；用精确可表示和非精确值检查两种参考的差异，禁止看到失败后随意放宽阈值。 |
| V3c：单元素解码 | 在小 CUDA 实验中从 packed Q 找一个逻辑权重，取对对应 group 的 Z/S 并得到约定中间值。 | 多个 word 边界、group 边界与手算对照；这可作为 V3d 的辅助检查，不单独夸成高性能算子。 |
| V3d：最简单 W4A16 GEMM | 每线程算输出、在 K 循环内按需解码权重并乘累加；支持先前声明的有限形状。 | 对 V3a/b 的独立参考测试；生产热路径不先完整展开整层权重；明确中间、累加、输出 dtype。 |
| V3e：健壮性与拒绝 | 处理或主动拒绝未对齐 K/N、非连续输入、非法 dtype/device/stride/zero、空输入和别名等情况。 | 不支持的输入有确定错误；支持的 case 覆盖重复调用、边界与工具检查，不能对未实现的并发/stream 功能作保证。 |

#### V4：朴素 W4 路径的协作与访存探索

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V4a：协作与复用 | 以 V3d 为基线，选一项改善：协作加载 A/Q、在 block 内复用、或减少重复解码。 | 先画访问和同步；与 V3 相同输入、输出和独立参考；一次实验只改变一个主要变量。 |
| V4b：向量化与对齐 | 选择适当的向量宽度，明确物理地址、对齐及越界限制；必要时保留标量尾部或明确拒绝。 | 手算相邻 lane 的地址，测试对齐及非法/边界输入；性能需靠实测而不是宽度推断。 |
| V4c：寄存器与性能定位 | 对可比较版本登记寄存器、shared、可用 profiler 指标和 GPU 时间。 | 分析访问、资源、spill/occupancy 与误差；工具/权限不足时记录，不将单项指标当成结论。 |

#### V5：tile 内解码与 Tensor Core 融合

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V5a：解码一个 tile | 只将当前 K/N tile 的 W4 解码为 BF16 临时片段或 shared tile，不构造完整 W；理解 scale/zero 在 tile 与 group 边界的对应。 | 用可识别 Q/Z/S 的逐元素独立参考检查 tile；注明 BF16 舍入以及临时数据的地址与生命周期。 |
| V5b：最小融合 | 从 V2b 的 WMMA 基线出发，将当前局部解码结果送入矩阵乘；先支持小范围对齐形状。 | 一次调用内完成局部解码与 GEMM；数学参考、量化参考、边界及精度均通过；shared 中一个 tile 的暂存是允许的。 |
| V5c：group 与尾部 | 覆盖至少两个 group、多个 K tile、非整除 M/N 和约定支持的 K 边界；逐步支持或拒绝 padding。 | 检查 scale/zero、越界补零和 shared 复用，先分清逻辑 K 与 padding 后 K；通过对应工具检查。 |
| V5d：低层 fragment 融合 | 在 V2d 的基础上，明确 `ldmatrix`/`mma.sync` 对应的 lane/寄存器与局部解码结果，再尝试寄存器级或目标片段融合。 | 独立验证 lane→逻辑 Q、Q/Z/S→fragment 及最终 C；确认目标指令存在，区分“局部融合”和“直接与上游 fragment 相同”。 |
| V5e：低层完整 K 循环 | 一个 warp 完成固定 M=16、N=8、K=256 的 W4A16 GEMM；循环填充 shared、ldmatrix、mma.sync，并在各 lane 的 FP32 寄存器中跨轮累加。 | 与独立 BF16 权重参考核对；覆盖两个 group、shared 复用和同步；检查目标机器指令。 |
| V5f：多 tile 与 M/N 尾部 | 将 V5e 扩展为每 block 一个 warp 计算 16×8 输出 tile；处理多个 block 及 M/N 非整除的 shared 补零和输出掩码。 | K 限定为 128 的正整数倍；覆盖多个 group、边缘 block 和 sanitizer，精度与 V5e 的契约一致。 |

#### V6：离线权重重排与布局

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V6a：设计布局 | 根据 V5 已知的读取模式设计一种自己能解释的权重重排；约定逻辑坐标、物理维度、bit 位序、zero/scale 与 padding。 | 画出固定 tile 中每个 lane 的读取，手算至少一个 word；离线布局与 kernel 的读取契约独立写清楚。 |
| V6b：独立映射验证 | 编写重排和不共享核心映射代码的 CPU reference；与原始逻辑 Q/Z/S 比较。 | 覆盖 K/N tile、group、pack 和 padding 边界；不能仅通过自身 pack→unpack 往返证明正确。 |
| V6c：集成 kernel | 使用重排结果替换 V5 的布局，保持数学语义不变。 | 同一逻辑输入对比正确性、GEMM 稳态时间；repack 耗时、显存及权重复用次数单列。 |
| V6d：对照 Marlin | 与工作区快照逐层比较 tile、warp/lane、Q、scale 和 zero 的实际位置，研究可借鉴设计。 | 明确独立设计、参考、改编或移植；复制上游代码时遵守许可并记录来源，不宣称布局兼容直至验证。 |
| V6e：冻结目标与捕获中间证据 | 对照固定实例，冻结上游源码/构建、输入、dispatch、位序、S/Z/bias 与输出；区分历史 trace、源码推导和本次实际捕获。 | 从同一逻辑 Q/Z/S 准备两侧数据；记录 padding、舍入与容差，找出当前 V5/V6 与固定实例的语义差异；暂不假定所有 lane 的寄存器已捕获。 |
| V6f：兼容 packed 权重 | 从固定实例的参考 repack 分支出发，独立实现或明确标记移植的离线映射，将教学布局与参考布局并列保留。 | 按逻辑 Q 的独立 CPU 映射、手算锚点、边界和参考输出逐 word 比较；区分 checkpoint/standard/目标布局，不靠 pack→unpack 往返。 |
| V6g：packed zero / scale 的布局 | 单独追踪并实现固定路径的 zero/scale 变换与读取；图示每个 group、列和物理 word 的对应。 | 精确验证 zero 的位序与 scale 的 BF16 位模式；覆盖 group 切换、N tile、padding；明确 act-order 不支持。 |
| V6h：寄存器解码基础 | 从一个 lane 的 packed word 出发，理解 BF16x2、位操作、偏置/zero 和 scale；先做单独的可观测解码实验。 | 覆盖 0/1/7/8/15、zero 0/7/15、正负值与非精确 scale；逐位或按预先规定舍入对照，区分参考分支与数学参考。 |
| V6i：Q/Z/S 到 MMA B 片段 | 按固定指令/布局设计 `packed → 各 lane 的 B 寄存器`；逐步替换 V5 中 BF16 shared B 暂存，保留可控的简单版本。 | 对 32 lane、每个寄存器的 BF16 位模式与独立逻辑坐标比较；用可识别数据覆盖 K/N/group 边界，再检查 MMA 输出和指令。不能因为看到 MMA 就声称和参考 fragment 相同。 |
| V6j：参考片段与 MMA | 将 V6i 的参考片段坐标和位序接入一个 warp 的两条 N8 MMA、两个 K step；测试阶段从逻辑 BF16 B 独立构造 fragment。 | 单 warp 16×16 输出与 CPU 参考对比；此时 Q/Z/S 的 GPU 解码与 MMA 尚未连成同一 kernel。 |
| V6k：四 warp 部分和 | 使用与参考一致的 K 分工和 B 片段坐标，在一个 block 中计算四个 warp 的 16×64 部分和。 | 逐 warp 验证部分和与合并结果；诊断缓冲与归约方式不冒称参考 shared 协议。 |
| V6l：参考写回数值顺序 | 按固定分支核对 FP32→BF16 与 BF16 bias 加法的顺序，并用精确 bit 测试证明与先加再舍入的差异；列出后续拼图。 | 覆盖舍入中点、正负值；只验证标量 epilogue 数值，不冒称完成参考的跨 block 归约与 shared 写回。 |
| V6m：参考布局融合四 warp | 把参考 repacked Q/Z/S 的逐 lane 解码和两条 MMA 直接连接在一个 kernel；固定 N tile 0、两个 K group、每 block 4 warp。 | 按逻辑输入独立验证两个 group × 4 warp 的每项部分和；明确不包含参考 `cp.async`、参考归约、bias 与完整输出。 |

V6a～V6d 保留为历史对照；V6c 已完成声明范围内的正确性验证及一次性能初测。V6f 对齐参考 repack 的 tile 内 warp/lane 输出映射；其简化 launch 和搬运过程不等同完整参考实现。V6g 的 S/Z 输出布局对照参考，独立 kernel 的线程分工不是 PyTorch 准备步骤的线程协议。V6h～V6m 的新文档和模板按固定参考输入与片段映射重写，先以独立测试逐层验证，再在 V6m 融合；参考流水线、线程块归约、全局归约和完整 epilogue 仍需 V7～V9。旧的简化模板已标记为历史草稿，不再作为主线。

#### V7：异步搬运、双缓冲与流水线

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V7a：同步单缓冲 | 将现有计算整理成“装入→等待/同步→计算→读取结束后复用”的明确单缓冲基线。 | 至少画两轮 K tile，检查尾部 predication；运行可用的 sanitizer。 |
| V7b：独立异步复制 | 不接完整 GEMM，先做一个小实验学习 `cp.async` 的发起、commit、wait 与 block 同步。 | 区分异步操作完成和线程间可见性；不会将未完成数据读出或覆盖，工具不可用时标注限制。 |
| V7c：双缓冲 GEMM | 将异步搬运和两个 shared 缓冲接入 V6/V5 的正确版本。 | 至少两个有效 K tile、多个 group 和尾部；对照单缓冲检查正确性与资源，不能仅因代码使用 `cp.async` 就宣称观察到重叠。 |
| V7d：多 stage 消融 | 只有 profile 或问题规模有依据时扩展 stage 数；比较单/双/多 stage。 | 记录流水线等待、shared/register、吞吐或时间线及同条件 GPU 时间；若无提升如实记录。 |
| V7e：固定实例流水线对照 | 在 V6k 已验证的多 warp 布局上，分步对照固定实例的 A/Q/S/Z 的 global→shared、stage 生命周期和下一轮数据依赖；确认 4 stages 是参考实例的配置，并根据实验决定是否实现。 | 跟踪至少两次 K tile 及 group 边界，验证各 buffer 写入/消费/复用次序；参考是否实际重叠与新实现是否重叠分别测量。 |

#### V8：小 M 的 split-K 和归约

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V8a：明确并行分工 | 对 M=1/2/7 的目标形状，画出不同 block 的 K 范围、部分和与输出所有权；先从一个最小例子开始。 | 确认每个 K 项恰好参与一次；记录是否需要中间 FP32 workspace、初始化和边界。 |
| V8b：两 kernel 归约 | 一个 kernel 写独立 FP32 部分和，另一个 kernel 合并并处理输出/bias；避免先引入锁协议。 | 两阶段都纳入正确性和计时；验证多轮重复、不同 shape 交替、workspace 初始化和尾部。 |
| V8c：stream/并发 | 显式传递正确的 CUDA stream，明确 workspace 的所有权与生命周期。 | 测同一 stream、有依赖的跨 stream 调用及各自独立 workspace；共享 workspace 如不支持并发则明确拒绝或规定外部同步，不能制造数据竞争。 |
| V8d：替代归约方案 | 有证据显示归约是瓶颈后，再探索 atomic 或锁/单 kernel 方案；可与参考实现的协议做对照。 | 覆盖死锁、初始化、数据可见性、顺序、精度和真实设备测试；性能比较包含全部归约与额外 launch 成本。 |
| V8e：固定实例调度与归约对照 | 对固定实例核实 stripe/slice 的工作分配和实际选中的 FP32 临时归约、bias 与输出路径；先画 block→K 范围→归约顺序，再实现可测试的对应协议。 | 覆盖空闲 block、多 K 分片、workspace/锁初始化、重复调用、并发和可见性；不将一般 split-K 默认等同于参考调度，也不假定所有输入走同一分支。 |

#### V9：配置、评测与可展示交付

| 检查点 | 目标与用户动手内容 | 验收与边界 |
|---|---|---|
| V9a：契约与 dispatch | 根据已验证的 shape、dtype、alignment 选择已有配置；未知情况明确拒绝或使用已验证的等价路径。 | 对支持/拒绝矩阵写测试；不同设备能力、stream、workspace、空输入/别名的行为可解释。 |
| V9b：可复现测量 | 比较自己的版本、冻结 Marlin 与可选 BF16 GEMM；固定同一逻辑 Q/Z/S、GPU、输入和正确性门槛。 | 分开预处理、稳态 kernel、端到端；记录 warmup、repeat、p50/p90、误差、原始日志、工具版本与源码/构建身份。 |
| V9c：真实 shape 与稳健性 | 从实际配置/权重读取 Qwen 投影与本地分片 shape，覆盖已支持的 M；逐步评估 CUDA Graph 等额外场景。 | 不能手写猜真实 shape，也不能用一个 shape 推广到所有 workload；不支持的功能明示。 |
| V9d：项目文档 | 将真实实现、架构图、消融、测量、局限、复现方式与出处整理到公开文档。 | 明确独立实现、参考和移植的边界；只展示完成且可追溯的数据，不预写加速倍数。 |
| V9e：固定实例复现验收 | 将 V6e～V8e 的布局、片段、数值、调度及结果证据与冻结参考逐层对照；逐项写已验证/未验证。 | 数学、布局、机制、接口和性能分别给出结论；记录编译及运行身份、误差、计时分布与工具限制，不把一个实例推广为整个 Marlin。 |

学习顺序：V6c 已完成，按 V6d→V6e/f/g→V6h/i/j→V6k/l/m 推进；然后进行 V7 的流水线、V8 的并行归约和 V9 的验收。可根据实际证据调整性能实验顺序，但不得跳过完成固定实例所必需的布局、数值和同步验证。

### 6.4 每个检查点的完成条件与下一步

1. **先约定**：记录输入/输出与不支持范围、布局与位序、精度及错误容差、测试和计时边界；固定例子按需扩展，但要标明逻辑轴、packed 轴、zero/scale 和 padding。
2. **再动手**：助手提供该节文档和必要模板；用户写核心机制，助手先提出针对性的 review 提示。纯阅读/推导不强制写 kernel。
3. **先正确、后性能**：手算→独立参考→边界与状态→可用的 sanitizer；只有通过约定正确性后，才在相同条件下用 CUDA event 和必要 profiler 做性能对照。工具不可用或 GPU 忙时明确记录，不抢占其他任务。
4. **留下可追溯证据**：记录实现版本、设备/驱动/工具链、输入、容差、输出误差、原始测试和计时数据、来源与许可。未做、失败和不适用的项目分别标注；不将计划写成实绩。
5. **小步决策**：一个版本的优化可以无收益；根据实际瓶颈决定是否进入、跳过或调整候选优化，记录原因。单次实验尽可能只改变一个主要变量。

下一节从 **V0a/V0b：朴素 FP32 GEMM 的契约、启动配置与单线程单输出**开始，文档放 `learning_notes/`，算子模板放 `kernels/marlin_from_scratch/`。


## 7. 正确性：每一版固定怎么验证

### 7.1 三层参考

```text
A. 位布局：手算小样本 + 独立 CPU 映射
   Q / zero / scale 的位置、打包、padding、重排
       ↓  整数和已约定 bit pattern：要求精确
B. 数学语义：从原始 Q/Z/S 独立构造 FP32 W，计算 FP32 A @ W
       ↓  根据 dtype 与累加顺序设定事前约定容差，报告误差
C. 同输入实现对照：本项目 Marlin / 冻结 vLLM Marlin
       ↓  必须保证相同逻辑权重、输入、输出 dtype 和配置
```

数学参考允许在**测试代码**中完整反量化，生产热路径不这么做。不能只与同一打包逻辑生成的另一个结果互相验证；错误的 pack 与 unpack 可能正好互相抵消。Marlin 差分有助于定位，但上游不是数学真理，疑似错误保留独立证据，不照搬错误。

### 7.2 固定覆盖

- 手算：nibble 值覆盖 0、1、7、8、15；zero point 取 0、7、15；正负结果都出现；有两组不同 scale。
- shape：`M=1,2,7,16,32,128`，覆盖 decode 小 M 和较大批次；固定例子 `K=256,N=128`；再覆盖真实投影尺寸、非整除 tile 和明确拒绝的 group/pack 不对齐输入。具体哪些非对齐由当前版本承诺处理必须写清楚。
- 比较：shape、dtype、device、finite；max/mean/p99 绝对误差、相对误差、容差外元素数。参考接近 0 时，相对误差要结合绝对误差解释；不把 NaN 当相等。
- 状态：初始化完整、workspace 独立、重复调用、不同 M 交替、stream、输入/输出别名、越界与并发；成熟版本再测试 CUDA Graph。
- 工具：基础与边界例子运行 compute-sanitizer；对无法支持的输入主动报错，不悄悄裁剪、补随机值或回退到不等价实现。

容差在每轮对比之前按输入/输出 dtype、FP32 参考及冻结上游重复噪声登记；若失败，定位累加/舍入或布局，不看见结果后随意调阈值。**不要要求 CUDA 与 FP32 所有元素 bitwise 相等，也不要拿误差均值掩盖少数越界大误差。**

## 8. 性能：怎样比较才公平

### 8.1 比较对象和四个时间

```text
冷启动总时间 = 构建/JIT + 加载/准备 + 请求执行
准备时间     = 加载后的格式转换、repack、scale/zero 重排、workspace 初始化
稳态算子时间 = 已准备权重，从一次 kernel 输入到输出的 GPU 时间
端到端时间   = 根据实际接口定义是否包含 Python、launch、排队、通信、后处理
```

分别记录，不把一种写成另一种。测稳态时，双方都使用已经准备好的、对应各自实现的布局；另外测 repack 时间和显存占用，并给出权重复用次数下的摊销。权重布局不相同也可以公平比较，但要从同一逻辑 Q/Z/S 得到各自布局，不能忽略仅一方的准备成本。比较端到端时，双方都纳入相同的边界；不要把 CUDA event 算子时间和 Python `wall clock` 服务延迟放在同一个分母。

比较基线至少包括：

1. 本项目冻结的 Marlin W4A16：相同 GPU、相同 Q/Z/S、M/N/K、输出 dtype、语义和计时边界。
2. 同源上游 vLLM 的 Marlin：作为实现/性能参考，单独标明安装版本及 source/build identity。
3. BF16 GEMM（例如 PyTorch/cuBLAS 路径）：展示未量化对照；它的输入表示和工作量不同，只能按明确标注的接口/数据准备边界作补充比较。
4. 自己的 V0、V1……各版本：在同一机器和配置下横向比较。

### 8.2 必备 case 和指标

- 第一批固定：合成 `K=256,N=128` 的调试例，以及实际 Qwen3-8B/14B 中 QKV、o_proj、gate/up、down_proj 的**真实逻辑与本地分片 shape**；用代码读取配置/权重并登记确切 shape，不手写猜测。本地投影取样可参考 `tests/step122/step122_real_projection.py`。
- 各投影测 `M=1,2,7,16,32,128`；先区分 decode 小 M 与较大 M。不要把优化某个单独 shape 的结果推广为整体加速。
- 每 case：正确性门槛、warmup 次数、重复次数、p50/p90 GPU 时间、最大/平均误差、显存、设备/时钟状态、软件版本、源码 commit/hash、编译参数、GPU UUID、其他占用和原始日志路径。
- 当问题很短时测量 launch 开销和计时误差；可把 CUDA Graph 重放作为**另一个明确标注的稳态计时方式**，但不能把它冒称本项目现有 AWQ Graph 推理支持。
- 判断高低时给出比值：`speedup = baseline_time / candidate_time`；当 candidate 更慢时如实给出小于 1 的数。只引用实测值和相同 case，注明重复波动；不要预先写“达到 Marlin 的 X%”。

### 8.3 从 profiler 提出假设，而不是堆术语

| 观察到的问题 | 先提出的假设 | 需要查什么 | 可能的下一实验 |
|---|---|---|---|
| 小 M 慢 | 可运行 block 太少或 launch/归约占比高 | 时间线、SM 活跃、block 数、kernel 数 | 尝试 K 切分，包含归约成本一起比较 |
| 加速比预期低 | 内存流量没有按压缩比下降 | DRAM/L2 流量、读写访问与复用 | 检查是否重复读 Q/scale，按实际布局优化访问 |
| 算力利用低 | 解码、访存或依赖使 MMA 吃不饱 | Tensor Core 指令、访存吞吐、warp stall | 分别测 decode 与搬运，改一项再复测 |
| 加大 tile 反而慢 | 寄存器/共享内存压力或尾部浪费 | 寄存器、spill、理论/实际占用、有效 tile | 改 tile，记录资源与时延 |
| 加流水线反而慢 | stage 太多或等待/同步开销大 | timeline、等待、shared 容量、占用 | 单缓冲/双缓冲/多 stage 消融 |
| pack 后快但端到端慢 | 准备成本高或复用次数少 | repack 耗时、预处理显存和调用次数 | 分别报告一次性成本及复用 N 次的摊销 |

指标的具体名称随 Nsight Compute / GPU 版本变化，以本机实际支持为准；不把“理论带宽/算力”直接当成已测实效，也不把低 occupancy 自动等同于性能差。必要时先用 roofline 做量纲估算，再用流量和时间证据检验；量化权重、激活、scale/zero 和输出流量均需计算。

## 9. 面向面试的项目组织

学习笔记随仓库同步；模型等被忽略的文件需另行准备。最终目录可按如下方式设计，等真正动手时再创建：

```text
quant-gemm-project/
├── README.md                  # 动机、支持范围、真实结果、复现方式、局限
├── kernels/                   # 自己逐版实现的 CUDA/C++ 代码
├── tests/                     # 独立量化参考、布局/边界/随机/并发测试
├── benchmarks/                # 可复现计时与基线对照脚本
├── profiling/                 # 可执行采集脚本；大型原始 trace 不提交
├── docs/                      # tile、warp/lane、流水线图与优化实验记录
└── third_party/ 或 NOTICE     # 若使用他人代码，保留许可和来源
```

每轮实验记录：**当时瓶颈 → 可证伪的假设 → 代码改动 → 正确性结果 → 指定 shape 的前后性能 → profiler 证据 → 不支持假设的结果/局限**。要标记自己的实现、受启发的设计和直接移植代码。本项目现有 Marlin CUDA 文件来自冻结 vLLM，同源许可见 `minivllm/csrc/quantization/marlin/PROVENANCE.md` 和 `LICENSE`；不能把它们或仅改名的内核描述成自己从零写的核心成果。最终简历只能填实际完成和实测的内容。

## 10. 我们之后每次怎么学

每一小节固定按下面的顺序：

1. **一句话目标**：这一节的代码想解决什么问题。
2. **画图**：画输入、输出、tile 和线程/warp/lane 的责任；明确逻辑索引与物理地址，不能混用。
3. **手算**：沿用第 2 节的 M/N/K；先手算一两个元素或一次 tile 搬运，再扩展到 kernel。
4. **读少量源码**：指明文件和本次只看的函数，逐个解释实际出现的 CUDA/C++ 语法；不跳过初学者看不懂的模板、指针或内联汇编。
5. **你自己写**：先提出小任务和预期输入输出，再由你尝试；我负责答疑、代码 review 和给出可以验证的提示。
6. **正确性与工具**：核对结果，必要时用 sanitizer 和 profiler；先解释看到的证据，再讨论优化。
7. **复盘问题**：用 2～4 个可验证问题检查是否能独立解释；未过关就回到该例子重画，不靠继续往下堆术语。

建议一节只解决一个机制；遇到陌生的 `reinterpret_cast`、模板特化、`constexpr`、lambda、`asm volatile` 等，先补当前需要的最小 C++ 知识再读代码。每个优化完成后，我们共同决定是否进入下一版。

## 11. CUDA 基础第一节的历史起点

本节保留阶段 A 当初的起点，基础练习现已移到 `kernels/base_learn/`。当前新的手写主线从第 6 节的 V0 开始，不需要重新从 vector add 开课。

先不碰复杂的 `marlin_template.h`。第一节是 **CUDA kernel 的调用、线程索引和边界检查**：

```text
任务：C[i] = A[i] + B[i]，N = 37，blockDim.x = 32

block 0: thread  0 ... 31 → i =  0 ... 31  全部有效
block 1: thread  0 ... 31 → i = 32 ... 63
                              其中 32 ... 36 有效
                                   37 ... 63 越界，不能访问

i = blockIdx.x * blockDim.x + threadIdx.x
```

学习时从零解释 `__global__`、`<<<...>>>`、host/device 指针、`const`、`if (i < N)`、异步发射和错误检查；然后你自己写出 kernel 和测试。通过后才进入 shared memory、tiled GEMM，再回到 Marlin。这是预定起点，**不是本次已经实现的练习**。

## 12. 现有代码与资料索引

| 内容 | 参考源码位置（除测试和环境脚本外，相对于 `third_party/vllm_marlin/`） | 学习用途 |
|---|---|---|
| AWQ 方法/参数 | `minivllm/layers/quantization/awq.py` | AWQ checkpoint 参数、后处理与 forward 接口 |
| Marlin 准备和调用 | `minivllm/layers/quantization/marlin.py` | padding、scale/zero、workspace、GEMM wrapper |
| 扩展调用与构建 | `minivllm/quant_ops/ops.py`、`build.py` | schema、动态编译、SM89 构建身份 |
| C++ binding 与 launch | `minivllm/csrc/quantization/marlin/bindings.cpp`、`marlin.cu` | Python 到 C++、检查、调度与发射 |
| kernel 声明/实例化 | `minivllm/csrc/quantization/marlin/kernel.h`、`generate_kernels.py` | 模板参数与实例化组合 |
| GEMM 主体 | `minivllm/csrc/quantization/marlin/marlin_template.h` | tile 映射、pipeline、fragment、计算和归约 |
| 底层机制 | `minivllm/csrc/quantization/marlin/marlin.cuh`、`dequant.h`、`marlin_mma.h` | 异步搬运、解码、MMA |
| 权重重排 | `minivllm/csrc/quantization/marlin/gptq_marlin_repack.cu` | 从标准 packed 到 Marlin 的离线重排 |
| 既有单层测试 | `tests/step121/`、`tests/step122/` | 对照冻结上游、真实投影和 FP32 参考；看清各自覆盖范围 |
| 本机环境 | `scripts/activate-4090.sh` | 仓库 Python 环境、CUDA 12.9 和 SM89 |
| 文件来源 | `minivllm/csrc/quantization/marlin/PROVENANCE.md` | 上游 commit 和许可；历史文字状态以最新测试报告为准 |

命令入口：参考仓库的环境脚本要求在参考仓库根目录执行；下面的 `cd` 与 `source` 均在当前 shell 运行，然后回到学习仓库：

```bash
cd ../vllm-from-scratch
source scripts/activate-4090.sh
cd ../quant-gemm-project
nvcc --version
nvidia-smi
ncu --version
nsys --version
compute-sanitizer --version
```

当前环境中能找到 `ncu`、`nsys`、`compute-sanitizer` 可执行文件；本次只确认可执行文件位置，没有验证 profiler 硬件计数器权限，也没有跑性能测试。如果某个 profiler 功能受权限限制，先记录具体现象，再用可用的 CUDA event、时间线或其他可访问证据推进，不能凭空声称拿到了计数器。
