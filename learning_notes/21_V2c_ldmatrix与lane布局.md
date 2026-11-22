# V2c：从零理解 ldmatrix、lane 和 PTX

## 1. 这节要解决什么问题？

V2a/V2b 中，我们用 `nvcuda::wmma` 加载矩阵并计算，WMMA 帮我们隐藏了矩阵片段如何分到各线程的寄存器里。接下来想研究更底层的计算，因此先单独学习一个问题：

> 已经放在一个 block 的 shared memory 里的矩阵，怎样由一个 warp 的 32 条 lane 合作读取，并分配到各 lane 的寄存器？

这节使用 `ldmatrix` 完成一次**数据搬运**。它不做矩阵乘法，也不做反量化；下一节再学习 MMA。练习模板是 [v2c_ldmatrix_template.cu](../kernels/marlin_from_scratch/v2c_ldmatrix_template.cu)。

## 2. 什么是 PTX 和内联汇编？

在 V2b，你写的是 CUDA C++ 接口：

```cpp
wmma::load_matrix_sync(...);
```

在 V2c，我们在 CUDA C++ 函数中写一条 PTX 指令：

```cpp
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...");
```

先把三层关系区分开：

```text
你写的 CUDA C++：变量、函数、循环、内联 asm
                   ↓ nvcc 等工具编译
PTX：NVIDIA GPU 使用的虚拟指令表示
                   ↓ 针对具体 GPU 的编译/加载过程
SASS：具体 GPU 架构的机器指令
```

`ldmatrix` 是这里写在内联汇编字符串中的 PTX 指令。你不需要先掌握整套 PTX；本节只需要理解这一条指令的输入、输出，以及它怎样和 C++ 变量连接。PTX 与最终执行的机器指令不是同一个层次；本节的重点是数据布局，不是性能测试。

## 3. 固定输入：4 个 8×8 矩阵

模板在一个 CUDA block 的 shared memory 中创建：

```cpp
__shared__ __align__(16) uint16_t tile[4][8][8];
```

逐段解释：

- `uint16_t`：无符号的 16-bit 整数，每个元素占 2 字节。本练习用它做**位置标记**；这里的数值不是 BF16，也不用解释 BF16 数学含义。
- `tile[4][8][8]`：4 个矩阵，每个矩阵 8 行、8 列。`tile[矩阵号][行号][列号]` 是一个元素。
- `__shared__`：这块数组在 block 的 shared memory 中，同一个 block 的线程可以访问。
- `__align__(16)`：要求数组起始地址按 16 字节对齐。每行 8 个 16-bit 元素，也就是 16 字节；因此每一行的行首也按 16 字节对齐，满足本练习中所用行地址的要求。

为了看清位置，模板把元素设置成：

```text
元素的值 = 1000 × 矩阵号 + 100 × 行号 + 列号
```

例如 `tile[0][1][2]=102`；`tile[1][1][2]=1102`。这只是用数字给每个位置贴标签，并不是实际的矩阵权重。

```text
矩阵 0 的一部分：
                列 0   列 1   列 2   列 3   ...  列 7
             ┌──────┬──────┬──────┬──────┬──────┐
行 0         │   0  │   1  │   2  │   3  │ ...  │
行 1         │ 100  │ 101  │ 102  │ 103  │ ...  │
行 2         │ 200  │ 201  │ 202  │ 203  │ ...  │
...          │      │      │      │      │      │
行 7         │ 700  │ 701  │ 702  │ 703  │ ...  │
             └──────┴──────┴──────┴──────┴──────┘
```

矩阵 1 的对应值在上图基础上加 1000；矩阵 2 加 2000；矩阵 3 加 3000。逻辑 `[矩阵,行,列]` 说的是元素坐标，不是字节地址。

## 4. 一个 block、一个 warp：先把数据放进 shared

本练习启动 `<<<1,32>>>`：1 个 block、32 个线程，恰好 1 个 warp。因为是一维 block，本例中 `threadIdx.x` 与 lane 编号都是 0～31。

```text
CPU 准备 input → 复制到 GPU global memory
                       │
                       ▼
      block 0：warp 0，lane 0～31 协作复制 256 个元素
                       │
                       ▼
           block 0 自己的 shared tile[4][8][8]
                       │
              __syncthreads()
                       │
          全 warp 协作执行 ldmatrix
                       ▼
             每条 lane 得到 4 个 32-bit 值
```

模板已经写好从 global 到 shared 的部分：

```cpp
uint16_t* flat = &tile[0][0][0];
for (int i = lane; i < 4*8*8; i += 32)
    flat[i] = input[i];
__syncthreads();
```

`&tile[0][0][0]` 是第一个元素的地址；`flat` 用一维方式依次访问同一块连续数组。共有 `4×8×8=256` 个元素，每个 lane 每次跨 32 个元素，因此各搬 8 个。`__syncthreads()` 让 block 中线程在进入后续读取前完成协作写入。**这行同步已经在模板中，不需要你补。**

## 5. 先看 ldmatrix 指令在说什么

本节用的指令形式是：

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

可以拆成下面几段理解：

| 部分 | 在本例中的意思 |
|---|---|
| `ldmatrix` | 协作加载矩阵片段到寄存器。 |
| `sync.aligned` | warp 级协作指令；参与的 lane 要按照该指令的要求一致执行。不是让整个 GPU 同步。 |
| `m8n8` | 每个被加载矩阵是 8 行、8 列。 |
| `x4` | 一次指令加载 4 个这样的矩阵。 |
| `shared` | 源数据在 shared memory 中。 |
| `b16` | 每个源元素按 16-bit 位模式处理。 |

它需要一个 warp 的 lane **合作提供地址**，执行后每条 lane 都得到 4 个 32-bit 结果。指令所要求的参与和布局可参考 [NVIDIA PTX ISA：ldmatrix](https://docs.nvidia.com/cuda/archive/11.5.2/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix)。

> 最容易混淆的地方：**每条 lane 提供的行地址**与**这条 lane 最终收到的数据**是两个不同的映射。下面分开画。

## 6. 第一张图：32 条 lane 各提供哪一行的地址？

4 个矩阵一共有 `4×8=32` 行，刚好让 32 条 lane 各提供一行的**行首地址**：

```text
提供地址的 lane        指向的 shared 行首

lane  0 ───────────→  &tile[0][0][0]    矩阵 0，第 0 行
lane  1 ───────────→  &tile[0][1][0]    矩阵 0，第 1 行
 ...
lane  7 ───────────→  &tile[0][7][0]    矩阵 0，第 7 行

lane  8 ───────────→  &tile[1][0][0]    矩阵 1，第 0 行
 ...
lane 15 ───────────→  &tile[1][7][0]    矩阵 1，第 7 行

lane 16～23 ───────→  矩阵 2 的第 0～7 行
lane 24～31 ───────→  矩阵 3 的第 0～7 行
```

C++ 中 `&tile[0][1][0]` 是取矩阵 0、第 1 行、第 0 列这个元素的地址，即这一行的开头；它**不是**数值 100。一个行首地址对应后面连续的 8 个 `uint16_t`：

```text
&tile[0][1][0]
       ↓
     [100][101][102][103][104][105][106][107]
      列 0                                    列 7
      ←────────────── 16 字节 ──────────────→
```

每条 lane 在 C++ 中有自己的局部 `address`，用自己负责的那一行的地址参与同一次指令。

## 7. 第二张图：每条 lane 最终收到什么？

对本节的 `m8n8.x4`、**不带 `.trans`** 的形式，每条 lane 收到 4 个 32-bit 结果，分别来自 4 个矩阵。先只看矩阵 0：

```text
矩阵 0 的一行含 8 个 16-bit 元素；每两列由一条 lane 接收。

                   逻辑列
              0   1 │ 2   3 │ 4   5 │ 6   7
             ───────┼───────┼───────┼───────
矩阵 0 行 0   lane 0│ lane 1│ lane 2│ lane 3
矩阵 0 行 1   lane 4│ lane 5│ lane 6│ lane 7
矩阵 0 行 2   lane 8│ lane 9│lane 10│lane 11
矩阵 0 行 3  lane 12│lane 13│lane 14│lane 15
矩阵 0 行 4  lane 16│lane 17│lane 18│lane 19
矩阵 0 行 5  lane 20│lane 21│lane 22│lane 23
矩阵 0 行 6  lane 24│lane 25│lane 26│lane 27
矩阵 0 行 7  lane 28│lane 29│lane 30│lane 31
```

每个格子表示一条 lane 收到该行的**两个相邻 16-bit 元素**，不是说这条 lane 为这行提供了行首地址。矩阵 1、2、3 的接收布局与上图相同，但送入同一条 lane 的另三个寄存器。

### 把 lane 5 的两种角色放在一起

```text
提供地址时：lane 5 提供 &tile[0][5][0]，即矩阵 0 的第 5 行行首。

接收数据时：lane 5 接收每个矩阵的第 1 行、第 2 和第 3 列：

     矩阵 0：tile[0][1][2]=102， tile[0][1][3]=103 → r0
     矩阵 1：tile[1][1][2]=1102，tile[1][1][3]=1103 → r1
     矩阵 2：tile[2][1][2]=2102，tile[2][1][3]=2103 → r2
     矩阵 3：tile[3][1][2]=3102，tile[3][1][3]=3103 → r3
```

所以，lane 5 虽然**提供**矩阵 0 第 5 行的地址，却**接收**四个矩阵各自第 1 行的一对元素。这不是冲突：32 条 lane 是一起提供地址、一起执行，再由指令按规定布局分发结果。

### 两个 16-bit 值怎样放进一个 32-bit 寄存器？

以 lane 5 的 `r0` 为例：

```text
r0：一个 32-bit 值
高 16 bit                        低 16 bit
┌─────────────────────────┬─────────────────────────┐
│ 103 = 0x0067            │ 102 = 0x0066            │
└─────────────────────────┴─────────────────────────┘

r0 的完整 bit pattern = 0x00670066
```

低/高指 bit 在 32-bit 值中的位置；不等于逻辑矩阵的上/下行。模板中的 CPU 参考正是分别取两个位置，拼出 32-bit 值后与输出比较。

需要手算任意 lane 时可以用：`接收的行 = lane / 4`，`该行第一列 = (lane % 4) * 2`；这里 `/`、`%` 都用于非负整数。比如 lane 5：行 `5/4=1`，第一列 `(5%4)*2=2`，所以收到列 2 和 3。这是**接收数据**的公式，不是第 6 节提供行地址的规则。

## 8. 再看模板里的内联汇编

```cpp
uint32_t r0, r1, r2, r3;
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
             "{%0,%1,%2,%3}, [%4];\n"
             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
             : "r"(address));
```

逐项看：

1. `uint32_t r0, r1, r2, r3;`：声明当前 lane 的四个 32-bit 结果变量。它们不是 shared 数组；每条线程有自己的结果。
2. `asm volatile(...)`：在 CUDA C++ 中插入汇编指令，`volatile` 表示这段汇编不应被当作可随意删除的普通无用计算。**它本身不是 `__syncthreads()` 的替代品**。
3. 引号中的文本是 PTX 指令；相邻的两个 C++ 字符串字面量会拼起来。`{%0,%1,%2,%3}` 是四个输出占位符；`[%4]` 是输入的 shared 地址占位符。
4. 第一个 `:` 之后的 `"=r"(r0)` 等声明输出：`=` 表示写入，`r` 表示通过合适的 32-bit 寄存器约束传递。依次对应 `%0`～`%3`。
5. 第二个 `:` 之后的 `"r"(address)` 声明输入，依次对应 `%4`。`address` 必须是本条 PTX 指令所需的 shared 地址表示。

你不需要在这节重写 asm 语句；模板已经提供。你要做的是让每条 lane 准备正确的行首地址。

## 9. C++ 指针为什么还要转换？

`&tile[...][...][0]` 是 CUDA C++ 中指向 shared 元素的指针。模板要求通过 CUDA 提供的 `__cvta_generic_to_shared`，把这个指针转换为适合这里 PTX shared 地址操作的表示，再放入 `uint32_t address`。

```text
C++ 的 shared 行首指针
        │ __cvta_generic_to_shared
        ▼
本条 PTX 所用的 shared 地址表示
        │ 作为 %4 输入
        ▼
ldmatrix 从 shared 中取数
```

具体转换和类型写法按模板、CUDA 工具链处理；不要把它当成从 global 复制到 shared，也不要把**一个矩阵元素的值**当成地址。地址空间及相关转换以 CUDA/PTX 文档和目标环境为准。

## 10. 你来完成和自查

复制 [模板](../kernels/marlin_from_scratch/v2c_ldmatrix_template.cu)到同目录 `v2c_ldmatrix.cu`，只完成以下内容：

1. 根据第 6 节的表，确定每条 lane 对应的矩阵号和行号，取该行第 0 列的地址。
2. 将该 shared 指针转换并写入 `address`。
3. 完成后，把 `ready` 改为全 warp 一致的 true。默认 false 时，所有 lane 在执行指令前一起退出，测试会保留哨兵值并报失败。

先在纸上核对：lane 14 给出哪个矩阵、哪一行的地址？再用第 7 节的表看：lane 14 最终收到的是每个矩阵的哪一行、哪两列？这两个答案不必相同。

编译、运行：

```bash
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
    kernels/marlin_from_scratch/v2c_ldmatrix.cu -o /tmp/quant_v2c
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v2c
```

GPU 1 只是示例，先确认当时空闲。写完后告诉我，我会先检查并用 CPU 参考、memcheck、racecheck 和 synccheck 验证。注意这节只验证 `ldmatrix` 的装载分布；它与下一节完整 MMA 的 fragment 衔接还需要单独推导。
