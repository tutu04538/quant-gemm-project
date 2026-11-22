# 第 3 节：kernel 发射与异步错误

## 要解决什么问题

CPU 发射 kernel 后，怎样检查发射是否成功，以及 GPU 执行过程中是否出错？本节继续使用长度 37 的 vector add，只练习两个检查点。

## 线程、数据和时间线

```text
输入：device 数组 d_a、d_b，各 37 个 float
输出：device 数组 d_c，37 个 float

host（CPU）                              device（GPU）
发射 <<<2, 32>>> ────────────────────→  block 0：warp 0，lane 0～31 → 元素 0～31
   │                                     block 1：warp 0，lane 0～4  → 元素 32～36
   │                                              lane 5～31 不访问
   │                                     有效线程读 d_a[i]、d_b[i]，写 d_c[i]
   ├─ 检查发射错误                             │
   ├─ 等待设备完成并检查执行错误 ←─────────────┘
   └─ 将 d_c 复制到 host 验证
```

这里的 lane 映射成立，是因为每个 block 恰好有 32 个线程。`i` 是数组元素索引，不是字节地址。

## 为什么要有两个检查点

kernel 发射通常是异步的：CPU 提交工作后可以继续往下执行。回到 host 的下一行时，GPU 可能还没有执行完。

1. `cudaGetLastError()` 返回并清除当前 host 线程最近的 CUDA runtime 错误状态。紧接发射调用它，可以检查发射配置等错误；它**不会等待 kernel 执行完成**。为了使判断清楚，之前的 CUDA 调用也应逐一检查。
2. `cudaDeviceSynchronize()` 等待当前设备上先前提交的工作完成，并返回状态；执行过程中的异步错误可能在这里报告。它的等待范围比单个 kernel 更广，不能把它简单理解为只等这一行发射的 kernel。

例如，本例中 block 1 的 lane 4 算 `i = 1 × 32 + 4 = 36`，读 `d_a[36]` 和 `d_b[36]`，写 `d_c[36]`。发射检查本身不证明这个读写已经完成；同步成功后，才知道此前设备工作已完成且同步调用未报告错误。之后仍需核对计算结果，API 成功不等于数学结果正确。

## 代码与练习

打开 [03_template.cu](../kernels/03_template.cu)。模板已准备输入、内存、kernel 发射、结果验证和释放；你只需在发射后填写两处 `CUDA_CHECK(...)`，先检查发射，再同步并检查执行。

`CUDA_CHECK(...)` 是模板定义的宏：执行括号内的 CUDA 调用，若返回状态不是 `cudaSuccess`，打印错误并结束程序。注意，`vector_add<<<...>>>(...)` 本身不是返回 `cudaError_t` 的普通 API 调用，因此不能简单把发射语句塞进这个宏。

可以用以下命令编译并运行；`CUDA_VISIBLE_DEVICES` 应按当时的空闲 GPU 选择，先用 `nvidia-smi` 查看，避免占用别人正在使用的 GPU：

```bash
nvcc -arch=compute_86 -code=compute_86 kernels/03_template.cu -o /tmp/quant_gemm_03
CUDA_VISIBLE_DEVICES=1 /tmp/quant_gemm_03
```

该编译参数适用于此前测试时的本机驱动和 GPU；其他环境需要根据实际工具链调整。模板未填 TODO 时也可能算对，所以看到数值通过不等于已完成错误检查。发射和同步检查要通过代码 review 确认。

思考：如果只检查发射配置，为什么仍不能断言 GPU 已经完成运算？如果同步成功，为什么还要检查 `C[i]` 的值？
