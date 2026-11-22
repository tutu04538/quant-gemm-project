# 第 1 节：vector add 的调用、线程索引与越界检查

## 目标与输入

让 GPU 的每个线程找到自己负责的元素，并避免访问数组末尾之外的内存。

输入 `A`、`B` 和输出 `C` 都是长度 `N = 37` 的 `float` 数组，目标是：

```text
C[i] = A[i] + B[i]，i = 0, 1, ..., 36
```

约定一个线程负责一个元素，每个 block 有 32 个线程。为了覆盖 37 个元素，启动 2 个 block。

## 线程与数据的关系

```text
CPU（host）发起 kernel：2 个 block，每个 block 32 个线程
  │
  ▼
GPU（device）

block 0，线程 0～31                 block 1，线程 0～31
└─ warp 0，lane 0～31               └─ warp 0，lane 0～31
   ├─ lane 0  → 元素 0                 ├─ lane 0 → 元素 32
   ├─ lane 1  → 元素 1                 ├─ lane 1 → 元素 33
   └─ lane 31 → 元素 31                ├─ lane 4 → 元素 36
                                       └─ lane 5～31 → 超出范围，不访问

GPU 显存：
A: [A[0], A[1], ... A[31], A[32], ... A[36]]
B: [B[0], B[1], ... B[31], B[32], ... B[36]]
      │  对有效的 i，线程分别读取 A[i] 和 B[i]
      ▼
C: [C[0], C[1], ... C[31], C[32], ... C[36]]
```

warp 是 GPU 执行线程时使用的 32 线程一组；lane 是线程在 warp 内的位置，编号为 0～31。本例每个 block 恰好是一个 warp，因此线程编号与 lane 编号相同。

`i` 是**元素索引**，不是字节地址。例如 `A[33]` 表示第 33 号 `float` 元素；C++ 会按元素大小计算对应地址。

## 线程索引与手算

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x`：当前 block 在一维 grid 中的编号，从 0 开始。
- `blockDim.x`：每个 block 的线程数，本例为 32。
- `threadIdx.x`：当前线程在自己 block 内的编号，从 0 开始。
- `int i = ...;`：声明一个整数变量 `i`，并用右侧结果初始化。
- `.x`：取 x 方向的编号或大小，本节只用一维配置。

```text
block 0，thread 7： i = 0 × 32 + 7 = 7
block 1，thread 1： i = 1 × 32 + 1 = 33
block 1，thread 5： i = 1 × 32 + 5 = 37
```

前两个索引有效。最后一个等于 37，而最后一个有效索引是 36，因此该线程不能访问 `A[37]`、`B[37]` 或 `C[37]`。例如 `A[33] = 2.0`、`B[33] = 3.5` 时，block 1 的 thread 1 写入 `C[33] = 5.5`。

## kernel 与调用语法

下面是待你完成的函数框架：

```cpp
__global__ void vector_add(
    const float* a, const float* b, float* c, int n) {
    // 计算当前线程负责的索引
    // 只让有效索引访问数组并计算
}
```

- `__global__` 声明一个 CUDA kernel：由 host 侧发起，在 GPU 上执行。
- `void` 表示没有普通函数返回值；结果通过 `c` 指向的数组写出。
- `float* c` 是指向 `float` 数据的指针，调用时需指向 GPU 可访问的输出内存。
- `const float* a` 表示本 kernel 通过 `a` 只读所指的 `float` 数据；`b` 同理。
- `int n` 是数组长度，避免将 37 写死。

host 侧调用的形状：

```cpp
vector_add<<<2, 32>>>(d_a, d_b, d_c, 37);
```

`<<<2, 32>>>` 是 CUDA 的 kernel 调用语法：前者是 grid 中的 block 数，后者是每个 block 的线程数。`d_a`、`d_b`、`d_c` 需指向 GPU 可访问内存，不能仅靠命名获得；显存分配和复制将在写测试时讲解。

kernel 发射通常是异步的：CPU 发出任务后可以继续执行，这不表示 GPU 已经算完。后续读取结果、检查错误时需要考虑同步和 CUDA 错误检查。

## 动手练习

只写 `vector_add` 的函数体，暂时不必写完整 `main`：

1. 计算当前线程的全局元素索引 `i`。
2. 用 `if (i < n)` 判断索引是否有效。
3. 仅在条件成立时，读取两个输入元素，并将和写入输出。

再回答：

1. 本例一共启动了多少个线程？其中多少个线程不会访问数组？
2. 若 `N = 64`、每个 block 仍有 32 个线程，需要几个 block？

完成后先 review 索引和边界，再一起补上内存准备、正确性测试及错误检查。
