# 第 10 节：shared-memory tiled GEMM

## 目标与输入

实现 FP32 矩阵乘 `C = A × B`：`A` 为 `[M,K]`，`B` 为 `[K,N]`，`C` 为 `[M,N]`，三个矩阵均按行连续存储。每个线程计算一个输出元素，block 合作加载输入 tile。练习使用 `T = 4`，以便手算；这不是性能调优后的配置。

固定例子 `M=5, K=7, N=6`，包含 M、N、K 三个方向的尾部边界。输出：

```text
C[row,col] = Σ(k=0..K-1) A[row,k] * B[k,col]
```

## 矩阵、线程与内存流

```text
一个 block 算一个 4×4 的 C tile；block=(4,4)，共 16 个线程
lane 按 threadIdx.x 先变化排列：lane = threadIdx.y * 4 + threadIdx.x
该 block 只有一个部分填充的 warp；每个线程负责 C 中一个位置

       K 方向分两轮：0～3、4～7（k=7 补 0）
global A [M,K] ──合作加载──→ shared As[4,4] ─┐
                                               ├─ 每线程用寄存器 acc 累加 → global C[M,N]
global B [K,N] ──合作加载──→ shared Bs[4,4] ─┘

每轮：写 As/Bs → 全 block 同步 → 读 tile 并累加 → 全 block 同步 → 覆盖 tile
```

grid 的 x 方向覆盖输出列，y 方向覆盖输出行。线程 `(tx,ty)` 负责 `row = blockIdx.y * 4 + ty`、`col = blockIdx.x * 4 + tx`。超出 C 范围的线程不能提前返回：它们仍参与 tile 的填充和同步。

## 手算一项与边界

例如 `C[1,2]` 的一项 `k=3` 是 `A[1,3] × B[3,2]`。若两者分别是 2 和 -1，这项贡献为 -2；最终还需要加上其他 `k=0..6` 的贡献。

第 0 轮：线程 `(tx,ty)` 合作加载 `A[row, 0+tx]` 和 `B[0+ty, col]`。每个线程对 `p=0..3` 累加 `As[ty,p] * Bs[p,tx]`。第 1 轮把 K 起点换成 4；对应 `k=7` 的读无效，应填 0。行或列超界时相应输入也填 0。

行优先元素索引：`A[r,k] → r*K+k`，`B[k,c] → k*N+c`，`C[r,c] → r*N+c`。它们是元素索引，不是字节地址。

## 动手

打开 [10_template.cu](../kernels/10_template.cu)，补全 kernel 中的 TODO：

1. 用 `tx`、`ty` 和 block 坐标计算输出行列；初始化线程自己的 `acc`。
2. 每轮合作加载 `As[ty][tx]`、`Bs[ty][tx]`，对 M/N/K 边界分别检查，越界填 0。
3. 写入后同步；累加 4 项；读取完成后再同步，以便安全复用 shared 数组。
4. 循环结束后仅让有效输出线程写 `C[row,col]`。

模板测试多个形状，包括 `5×7` 乘 `7×6`，并用 CPU 三重循环计算参考值。输入为小整数，参考与结果可在 FP32 中精确表示，所以使用精确比较；它不是对一般浮点 GEMM 的通用容差要求。

思考：当 `row >= M` 时，为什么线程仍要参加同步？如果去掉读取后的第二个同步，会发生什么？
