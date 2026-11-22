# 第 8 节：block 内同步

## 目标与输入

让一个 block 的两个 warp 通过 shared memory 交换数据，理解为什么读之前需要等待其他线程的写入。输入 `A` 和输出 `B` 都有 64 个 `float`，使用 1 个 block、64 个线程：

```text
B[t] = A[(t + 32) % 64]，t = 0～63
```

## 线程、数据和内存流动

```text
1 个 block：64 个线程，共享 tile[64]

warp 0：lane 0～31，thread  0～31  ──写──→ tile[0～31]
warp 1：lane 0～31，thread 32～63  ──写──→ tile[32～63]
                                           │
                              等待整个 block 完成写入
                                           │
warp 0：thread  0～31              ←──读── tile[32～63] → global B[0～31]
warp 1：thread 32～63              ←──读── tile[0～31]  → global B[32～63]
```

每个 block 都有自己的一份 shared memory；本例仅有一个 block。每个 warp 内 lane 从 0 到 31，线程全局元素索引在本例中等于 `threadIdx.x`。图中的索引以 `float` 元素计数，不是字节地址。

不同 warp 的执行先后顺序没有由这段代码规定。如果 warp 0 已经开始读 `tile[32]`，warp 1 却尚未写入，那么值就不可靠。这是一种数据竞争，即使偶尔运行得到正确结果也不能证明没有问题。

## 手算

若 `A[i] = i`：thread 32 先写 `tile[32] = 32`；所有线程完成写入后，thread 0 读 `tile[(0 + 32) % 64]`，于是 `B[0] = 32`。thread 32 则读 `tile[0]`，得到 `B[32] = 0`。

## `__syncthreads()`

`__syncthreads()` 是 **block 内**的屏障：参与的线程在此等待，直到 block 中所有尚未退出的线程到达该屏障；屏障也保证此前相关的 shared/global 内存访问对 block 中线程可见。它不会同步其他 block，也不等同于 CPU 调用的 `cudaDeviceSynchronize()`。

本例没有线程提前退出，因此 64 个线程必须都执行同一个屏障；不要把它放在只由一个 warp 执行的 `if` 中。下一轮如果要覆盖 shared 数据，而其他线程可能还未读完，也需要在复用前另设同步；本例只读一次，不复用。

## 动手

打开 [08_template.cu](../kernels/08_template.cu)，在 TODO 处加入一个 block 内同步，让 shared 写入发生在跨 warp 读取之前。模板包含 host 端验证，运行通过只能说明该次结果正确；判断是否存在竞争还要靠同步依赖分析和工具。

想一想：如果把屏障放到读取之后，能够保证 `B[0]` 使用的是 thread 32 写入的值吗？为什么？
