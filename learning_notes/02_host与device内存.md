# 第 2 节：host 与 device 内存

## 要解决什么问题

上一节的 kernel 接收三个指针。现在要回答：CPU 上准备好的数组，怎样交给 GPU 上的 kernel 使用？我们先学习显存分配和数据复制，暂时不展开完整的错误处理和计时。

沿用 `N = 37`、每个 block 32 个线程的 vector add：`C[i] = A[i] + B[i]`。这里假设三个输入输出数组都使用 `float`。

## 线程和内存流动

```text
CPU / host 内存                         GPU / device 内存
h_a: 37 个 float  ── HostToDevice ──→  d_a: 37 个 float
h_b: 37 个 float  ── HostToDevice ──→  d_b: 37 个 float
                                         │
                               2 个 block，每个 32 线程
                               每个 block 1 个 warp，lane 0～31
                               有效线程 i 读 d_a[i]、d_b[i]
                                         │
                                         ▼
h_c: 37 个 float  ←─ DeviceToHost ──  d_c: 37 个 float
```

`h_` 和 `d_` 只是命名习惯，分别提醒我们指针通常指向 host 和 device 内存；名字本身不会分配或复制任何数据。这个例子使用普通 host 数组和 `cudaMalloc` 分配的 device 内存，不讨论其他内存模型。

上一节你写的 kernel 在 GPU 上运行，所以本例调用时要给它 `d_a`、`d_b`、`d_c`。CPU 不能把普通 host 数组的指针直接当作这里的 device 数组来用。

## 字节数与指针

假设 `float` 占 4 字节，那么 37 个元素需要 `37 × 4 = 148` 字节。代码中不用假定固定字节数，而是写：

```cpp
size_t bytes = n * sizeof(float);
```

`sizeof(float)` 给出一个 `float` 的字节数；`size_t` 是适合表示对象大小的无符号整数类型。这里的 `n` 约定非负且足够小。`a[i]` 的 `i` 仍然是元素索引，而 CUDA 的分配和复制函数接收的是**字节数**。

设备指针可以先声明：

```cpp
float* d_a = nullptr;
```

`float*` 表示指向 `float` 的指针；`nullptr` 表示目前不指向任何有效对象。分配 device 内存的调用形状是：

```cpp
cudaMalloc(&d_a, bytes);
```

`&d_a` 是变量 `d_a` 自己的地址，让 `cudaMalloc` 能把分配得到的设备指针写进 `d_a`。调用成功后，`d_a` 指向分配的 device 内存；分配失败时不能使用它，正式代码需要检查返回值。

## 复制方向与释放

```cpp
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
```

参数依次是**目标、来源、字节数、方向**：将 host 数组 `h_a` 的内容复制到 device 内存 `d_a`。计算结束后，把结果复制回来时方向是 `cudaMemcpyDeviceToHost`，目标和来源也要相应交换。

device 内存用完后调用 `cudaFree(d_a)` 释放。`cudaMalloc`、`cudaMemcpy`、`cudaFree` 都返回 CUDA 状态，本节先认识数据流；下一步写完整测试时要检查各次调用及 kernel 发射与执行错误，不能把示意片段当成完整可运行程序。

## 动手练习

先在纸上或回复中写出 host 侧的操作顺序，不需要写完整程序：

1. 分别为 `d_a`、`d_b`、`d_c` 分配多少字节？写出相应的三次 `cudaMalloc`。
2. 假设 host 数组叫 `h_a`、`h_b`、`h_c`，写出把两个输入复制到 GPU 的调用。
3. 写出 `vector_add` 的发射调用，再写出把结果复制回 `h_c` 的调用。
4. 最后写出释放三个 device 分配的调用。

想一想：`cudaMemcpy` 的目标和来源如果交换，数据会往哪个方向走？写完后先检查数据流，再一起补齐可运行程序和错误检查。
