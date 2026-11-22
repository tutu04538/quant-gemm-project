# V1a：用 shared memory 合作计算 GEMM

V1a、V1b、V1c 共用 [V1 模板](../kernels/marlin_from_scratch/v1_gemm_template.cu)。本节先理解计算和数据流，然后由你填写 kernel；下一节再专门检查边界与同步。

## 1. 问题与输入

继续计算 `C = A × B`，其中 A `[M,K]`、B `[K,N]`、C `[M,N]` 均为 FP32、行优先连续，正尺寸；每个输出元素完全覆盖写入一次。无 bias、量化、Tensor Core，且 A/B/C 使用独立设备内存。模板是自包含练习程序，不是对任意 dtype、stride 或指针都具备检查的通用接口。

V0 中，同一 block 里不同输出线程可能重复读取 A 的同一值或 B 的同一值。V1 希望让一个 block 先合作把 A/B 的一小块加载到 shared memory，再由多个线程复用它们。本节建立正确的数据流；是否加速要在 V1c 同条件测量。

## 2. 输出 tile 与线程

设 tile 大小 `T=16`，block 为 `(16,16)`：每个 block 256 线程、8 个 warp，每个 warp 32 个 lane；threadIdx.x 先变化。例如 warp 0 包含 `(ty=0, tx=0..15)` 和 `(ty=1, tx=0..15)`。

```text
输出 C 按 16×16 分块：

            N →
         ┌─────────┬─────────┬─────┐
 M ↓     │ block   │ block   │ ... │
         │ (0,0)   │ (1,0)   │     │
         ├─────────┼─────────┼─────┤
         │ block   │ block   │ ... │
         │ (0,1)   │ (1,1)   │     │
         └─────────┴─────────┴─────┘

block (bx,by) 中：thread (tx,ty) 负责 C[by*16+ty, bx*16+tx]
```

每个线程拥有一个 FP32 局部累加器 `sum`；不同 block 各有自己的 shared 数组：

```cpp
__shared__ float as[16][16];
__shared__ float bs[16][16];
```

`as` 和 `bs` 是两个二维数组；`__shared__` 表示同一 block 内线程可共同访问。它们的索引是数组元素坐标，不是 global 内存字节地址。

## 3. 一轮 K tile 的数据流

假设当前 K tile 起点是 `base`：

```text
global A                             global B
A[row, base+tx]                      B[base+ty, col]
        │ thread (tx,ty) 各加载一个       │
        ▼                                  ▼
shared as[ty,tx]                    shared bs[ty,tx]
        └────────── 所有线程完成写入后同步 ──────────┘
                             │
thread (tx,ty) 循环 p=0..15，读取：as[ty,p] × bs[p,tx]
                             │
                         累加到 sum
                             │
          所有线程读完后同步，再覆盖 shared 做下一轮
```

每个线程写 A tile 的一个位置、B tile 的一个位置，整个 block 合作填满两块 shared 数组。输出线程读取 shared 中一整行 A 和一整列 B。A 的列索引用 `tx` 加 base；B 的行索引用 `ty` 加 base；不要把它们互换。

行优先 global 元素索引分别为 `A[r,p] → r*K+p`、`B[p,c] → p*N+c`、`C[r,c] → r*N+c`。计算的是元素索引，不是字节地址。

## 4. 手算：两个输出复用同一个 A 值

为方便手算，假设输入是 `M=2,K=3,N=2`，使用模板实际的 16×16 block 和 T=16；第一个 K tile 的其余 K 位置为边界填零：

```text
A = [1 2 3]        B = [1 0]       C = [4  5]
    [4 5 6]            [0 1]           [10 11]
                       [1 1]
```

thread `(tx=0,ty=0)` 加载 A[0,0] 到 `as[0,0]`；thread `(tx=1,ty=0)` 加载 A[0,1] 到 `as[0,1]`。同步后，计算 C[0,0] 和 C[0,1] 的两个输出线程都需要 `as[0,0]=1`：前者乘 `bs[0,0]=1`，后者乘 `bs[0,1]=0`。此处展示的是**逻辑复用**；具体访存和运行时间需要工具测量。

前 3 次 `p` 累加得到 C[0,0]=4；由于 K=3，`p=3..15` 对应越界 K，加载阶段填零，不影响结果。

## 5. 你来完成

建议复制模板为 `kernels/marlin_from_scratch/v1_gemm.cu`，保留原始模板，然后完成 5 个 TODO：

1. 每个线程填入一个 `as[ty][tx]` 和一个 `bs[ty][tx]`；A/B 的行列或 K 越界时写 0。
2. 等待整个 block 完成 tile 写入。
3. 用 shared 的对应行/列累加 `p=0..15`。
4. 等待整个 block 读完 tile，再覆盖以处理下一轮 K。
5. K 循环结束后，仅有效输出线程写回 C。

不要在循环前让输出越界线程提前返回；它们仍可能负责加载其他线程所需的输入，并必须参与同步。编译与测试命令见 V1b。代码可以直接写在模板或副本中；写完后告诉我，我会先 review 再运行检查。
