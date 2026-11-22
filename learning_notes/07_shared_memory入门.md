# 第 7 节：用 shared memory 交换数据

## 要解决什么问题

让一个 block 内的线程先合作把输入搬到共享空间，再让每个线程读取邻居放进去的值。此例用于认识 shared memory 的作用域和数据流，不作为性能优化结论。

输入 `A` 长度为 37，输出 `B` 长度也为 37，每个 block 有 32 个线程。每个 block 中，线程 `t` 输出同一个 block 内线程 `(t + 1) % 32` 搬入的值；最后不足 32 个元素的位置按 0 填充。

## 线程和内存

```text
device global memory        block 0（一个 warp，lane 0～31）       device global memory
A[0]  ── lane 0 写 ──→ shared[0]  ── lane 31 读 ─────────────→ B[31]
A[1]  ── lane 1 写 ──→ shared[1]  ── lane 0 读 ──────────────→ B[0]
A[2]  ── lane 2 写 ──→ shared[2]  ── lane 1 读 ──────────────→ B[1]
...

block 1 拥有自己的 shared 数组：
A[32] → shared[0]；...；A[36] → shared[4]
越界线程不读 A，但仍将 0 写到各自的 shared[5]～shared[31]
```

shared memory 是 GPU 上按 block 分配的存储空间：同一个 block 的线程可以访问该 block 的 shared 数组，不同 block 各有一份，不能直接用它访问彼此的数据。它的生命周期随 block 执行结束。图中的索引是元素索引，不是字节地址。

```cpp
__shared__ float tile[32];
```

`__shared__` 声明 block 内共享的数组；`float tile[32]` 是 32 个 `float` 元素。`threadIdx.x` 是本 block 内线程号，`%` 是取余运算，因此 `(31 + 1) % 32 = 0`。

## 手算

假设 `A[i] = i`：

- block 0 的 lane 0 把 `A[0]` 放入 `tile[0]`；lane 1 把 `A[1]` 放入 `tile[1]`。等待所有线程写完后，lane 0 读 `tile[1]`，因此 `B[0] = 1`。
- block 0 的 lane 31 读 `tile[0]`，因此 `B[31] = 0`。
- block 1 的 lane 4 对应 `i = 36`，它的邻居 lane 5 对应无效输入索引 37，因此邻居将 0 写入 `tile[5]`，最终 `B[36] = 0`。

模板中的 `__syncthreads()` 放在写 shared 和读 shared 之间，等待同一 block 的线程到达并保证此前的相关写入对 block 内线程可见。**本节不要删除它，也不要让越界线程在到达它之前返回**：越界线程仍须参与填充和同步。我们下一节再仔细学习同步。

## 动手

打开 [07_template.cu](../kernels/07_template.cu)，填写两个 TODO：

1. 每个线程给 `tile[threadIdx.x]` 写值：全局索引有效时读取 `a[i]`，否则写 `0.0f`。
2. 同步后，仅有效线程从下一个线程负责的 shared 位置读取并写 `b[i]`。

模板已经准备输入、输出验证和 CUDA 错误检查。完成后想一想：为什么最后一个 block 中输入越界的线程仍需要写 shared？
