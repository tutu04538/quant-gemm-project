# 第 11 节：用 CUDA event 测量 GPU 时间

## 目标

测量多个 kernel 在 GPU 上执行的总时间，并计算平均每次耗时。继续使用第 1 节的 vector add，输入为长度 4096 的 `float` 数组；这次重点是计时方法，不比较性能。

## CPU、GPU 与计时边界

```text
CPU / host                         GPU / 默认 stream 中的顺序
准备 device 输入 ───────────────→  A、B 在 global memory
发射若干 warmup ───────────────→  先执行预热 kernel
记录 start event ──────────────→  start 标记
循环发射 R 次 kernel ──────────→  vector_add 依次执行 R 次
记录 stop event ───────────────→  stop 标记
等待 stop 完成  ←──────────────── stop 已到达
计算 start 到 stop 的时间          每次 kernel：多个 block，每 block 256 线程
                                  每 warp 32 lane，lane 负责一个元素
```

本例 `4096 / 256 = 16` 个 block，每个 256 线程为 8 个 warp。`i` 是元素索引，每个有效线程读 `A[i]` 和 `B[i]`、写 `C[i]`。模板会先检查一次输出正确性，再进行计时；内存分配、输入复制、结果验证和预热都在计时边界外。

## 为什么需要 event 和预热

kernel 发射通常异步，CPU 从发射语句前后读到的时间可能主要反映提交耗时，而非 GPU 完成时间。CUDA event 可以记录同一 stream 中两个标记之间的 GPU 时间。预热有助于避免把首次初始化等开销混进稳态测量；是否足够稳定还需根据实际实验判断。

以 `R = 100` 为例，若 event 返回总计 `0.5 ms`，算术平均为 `0.5 / 100 = 0.005 ms`，即 `5 µs`。这些是手算示例，不是本机实测。循环多次得到的是这一批连续发射的平均值，不是单次延迟分布，也不自动排除 GPU 排队或其他活动的影响。

## event 的最小语法

模板已创建 `cudaEvent_t start, stop`，并在末尾销毁。你需要在 TODO 处使用：

- `cudaEventRecord(event)`：在当前调用指定的 stream 中记录标记；省略 stream 参数时使用默认 stream。
- `cudaEventSynchronize(stop)`：等待 stop 标记完成。
- `cudaEventElapsedTime(&ms, start, stop)`：计算两个 event 之间的毫秒数，写入 `float ms`。

以上 API 都返回 CUDA 状态，使用 `CUDA_CHECK(...)` 检查。发射循环后也要检查发射错误；等待 event 可以发现相关异步执行错误，但要检查其返回值。

## 动手

打开 [11_template.cu](../kernels/11_template.cu)，完成四个 TODO：记录开始、记录结束、等待结束、计算耗时。每次准备运行前查看 `nvidia-smi`，选择空闲 GPU；不要根据一次结果宣称加速或比较其他程序。记录 GPU、编译器、输入和计时边界后，未来才适合设计可复现的性能实验。

思考：如果只等待 start、没有等待 stop，为什么不能保证计时结果可用？如果把内存分配放在 start 和 stop 之间，测量口径会怎样改变？
