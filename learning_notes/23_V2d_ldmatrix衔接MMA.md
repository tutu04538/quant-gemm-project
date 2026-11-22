# V2d 衔接实验：把 ldmatrix 的结果直接送入 MMA

## 1. 为什么要补这个实验

此前我们完成了两个独立实验：

```text
V2c：shared → ldmatrix → 各 lane 的 32-bit 寄存器 → 输出验证
V2d：global → 各 lane 手动组装寄存器 → mma.sync → C
```

路线图里的 V2d 希望我们走通的是中间这一步：**用 ldmatrix 从 shared 装入 MMA 所要求的 A/B 寄存器，再执行 MMA**。这次仍只处理一个固定 tile，不引入 W4、scale、zero、多轮 K 或性能优化。

```text
global A、B
    ↓ 32 条线程合作加载
shared A [16,16]、shared B [16,8]
    ↓ ldmatrix：同一个 warp 的 32 条 lane 合作
各 lane 的 af[4]、bf[2] 寄存器
    ↓ mma.sync.aligned.m16n8k16
各 lane 的 4 个 FP32 输出 → global C [16,8]
```

输入是 BF16 A `[16,16]`、BF16 B `[16,8]`，输出是 FP32 C `[16,8]`。所有矩阵按行连续存储；launch 固定为一个 block、32 个线程，即一个 warp。模板是受限的独立实验，不是通用 GEMM，也不是 Marlin W4 解码。

## 2. 先看 A：为什么需要四个 8×8 子块

A 形状为 16×16，可切成四个 8×8 子块。这里用 `A0～A3` 仅表示逻辑子块，不表示四块分别分配的 shared 内存：

```text
shared as[16][16]（一个连续的二维数组）

                       列 0～7       列 8～15
                   ┌────────────┬────────────┐
       行 0～7     │ A0         │ A2         │
                   ├────────────┼────────────┤
       行 8～15    │ A1         │ A3         │
                   └────────────┴────────────┘

ldmatrix.m8n8.x4 的地址组顺序：A0 → A1 → A2 → A3
```

一次非转置 `ldmatrix.x4` 需要 32 条 lane 各提供一个行首地址：

| 提供地址的 lane | 对应子块 | 提供哪一行 |
|---|---|---|
| 0～7 | A0 | A 的行 0～7，列从 0 开始 |
| 8～15 | A1 | A 的行 8～15，列从 0 开始 |
| 16～23 | A2 | A 的行 0～7，列从 8 开始 |
| 24～31 | A3 | A 的行 8～15，列从 8 开始 |

例如 lane 0 提供 `&as[0][0]`，lane 8 提供 `&as[8][0]`，lane 16 提供 `&as[0][8]`，lane 24 提供 `&as[8][8]`。这些都是 shared 中的**地址**，不是矩阵元素的值。A 每行 16 个 BF16、32 字节；列 8 之后的 8 个 BF16 从该行起始地址偏移 16 字节，满足此例的行地址对齐要求。

## 3. 再看 B：为什么用 `.trans`

B 的 shared 形状是 16×8，按行连续。它沿 K 切成上、下两个 8×8 子块：

```text
shared bs[16][8]
                   列 0～7
                ┌────────────┐
     行 0～7    │ B0         │
                ├────────────┤
     行 8～15   │ B1         │
                └────────────┘

lane 0～7   提供 B0 的 8 个行首地址
lane 8～15  提供 B1 的 8 个行首地址
lane 16～31 也给合法的对齐地址；模板练习中可重复前 16 条 lane 的地址
```

此处使用 `ldmatrix.m8n8.x2.trans.shared.b16`。`.trans` 改变**加载后在 lane 寄存器中的片段分配**，使其适配本例 MMA 的 B 操作数；它不在 global/shared 内新建或原地转置整个 B 矩阵。

注意指令的地址提供规则：x2 使用前 16 条 lane 提供两组矩阵行地址；本练习让后 16 条 lane 也提供有效地址，避免依赖未初始化值。所有 32 条 lane 必须一致执行指令。

## 4. 一条 lane 收到什么？

令 `g=lane/4`、`t=lane%4`。最终每条 lane 的 A 寄存器 `af[0..3]` 与 B 寄存器 `bf[0..1]` 必须与 V2d 的 MMA 输入表一致：

```text
af[0] = (A[g,   2t],   A[g,   2t+1])
af[1] = (A[g+8, 2t],   A[g+8, 2t+1])
af[2] = (A[g,   2t+8], A[g,   2t+9])
af[3] = (A[g+8, 2t+8], A[g+8, 2t+9])

bf[0] = (B[2t,   g], B[2t+1, g])
bf[1] = (B[2t+8, g], B[2t+9, g])
```

每一对的前者位于 32-bit 寄存器低 16 位，后者位于高 16 位。注意“提供某行地址”和“最终收到哪些元素”是两种映射：lane 5 为 A0 提供第 5 行地址，但它收到的 `af[0]` 对应的是 A[1,2]、A[1,3]，因为 lane 5 的 `g=1,t=1`。

从 shared 写入到读取之前，模板使用 `__syncthreads()`；该同步保证协作加载完成，不能只依赖 `ldmatrix.sync` 代替。

## 5. 你来完成

复制 [v2d_ldmatrix_mma_template.cu](../kernels/marlin_from_scratch/v2d_ldmatrix_mma_template.cu) 为同目录 `v2d_ldmatrix_mma.cu`，只需完成四个 TODO：

1. 根据第 2 节的四组 lane，计算 A 对应的 8×8 子块行首指针。
2. 根据第 3 节的两组 lane，计算 B 的行首指针；后 16 条 lane 也提供合法对齐地址。
3. 用 `__cvta_generic_to_shared` 将 A/B 的行首指针分别转换为 `addr_a`、`addr_b`。
4. 填完后把 `ready` 设为全 warp 一致的 `true`。模板默认全 warp 在 `ldmatrix` 前返回，所以未填写时测试会失败但不会拿空地址执行指令。

`load_a`、`load_b` 两个包装和 `mma.sync` 指令已经提供，避免一节同时引入太多汇编语法；你需要理解它们输入的是 shared 地址、输出的是当前 lane 的寄存器。模板把这些实际寄存器输出写到 global 测试缓冲，再由 CPU 对照上面的逻辑坐标和原始 BF16 位模式，覆盖所有 32 条 lane 的 6 个寄存器。还用独立 FP32 参考验证 C；包括小整数、BF16 小数和逐位置可区分的数据。小数阈值事先固定为 `atol=0.002, rtol=0.0001`；其余两种数据在该固定形状下精确比较。

编译和运行（先检查 GPU 是否空闲，GPU 1 仅为示例）：

```bash
/usr/local/cuda-12.1/bin/nvcc -arch=sm_89 -O2 -lineinfo \
    kernels/marlin_from_scratch/v2d_ldmatrix_mma.cu -o /tmp/quant_v2d_ldmatrix
nvidia-smi
CUDA_VISIBLE_DEVICES=1 /tmp/quant_v2d_ldmatrix
```

我已用临时参考填充验证这套框架：三种输入的 fragment 和 C 均通过，memcheck、racecheck、synccheck 均未报告错误。**这不代表你尚未填写的模板已通过**；你完成后我会检查你的代码。

## 6. `asm volatile` 与 `"memory"` 是什么

以 `load_b` 为例：

```cpp
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(r[0]), "=r"(r[1])
    : "r"(address)
    : "memory");
```

这是 CUDA C++ 内联 PTX 汇编。基本结构是 `asm volatile(指令字符串 : 输出 : 输入 : 额外副作用)`；冒号分隔不同部分，不属于 PTX 指令本身。

- 指令字符串内 `%0`、`%1` 是输出占位符，绑定到 `r[0]`、`r[1]`；`%2` 绑定到输入 `address`。方括号 `[%2]` 表示 PTX 指令通过这个地址访问 shared memory。
- `"=r"` 中 `=` 表示写输出，`r` 是 32-bit 寄存器约束；`"r"(address)` 是放在寄存器中的输入。C++ 编译器负责按约束连接 C++ 变量与 PTX 占位符。
- `volatile` 提醒编译器这段汇编有需要保留的效果，不应把它当作普通的无副作用、可随意丢弃的计算；它不是 CUDA 线程同步指令。
- 最后 `"memory"` 位于第三个冒号之后，是 **memory clobber（内存副作用声明）**，不是传给 PTX 的参数。它告诉编译器汇编可能通过输入地址间接访问内存，不应跨该语句作不安全的内存优化；不是说整段 shared memory 被修改，也不是自动生成 GPU 的线程屏障或内存 fence。

本例先由线程写 shared 并执行 `__syncthreads()`，然后才做 `ldmatrix`：`__syncthreads()` 负责 block 内协作与可见性；`"memory"` 描述编译器看不见的间接内存访问；`ldmatrix.sync` 是该 warp 级指令本身的参与要求。三者作用不同，不能互相替代。[NVIDIA 的 CUDA 内联 PTX 说明](https://docs.nvidia.com/cuda/archive/12.4.0/inline-ptx-assembly/index.html#incorrect-optimization)解释了 `volatile` 和 `memory` clobber 的编译器约束。

**必须加 `"memory"` 吗？** 语法上不是每段内联汇编都必须加：例如只处理寄存器的加法不需要因为使用了 `asm` 就加它。这里的 `ldmatrix` 通过地址间接读取 shared memory，而 `"r"(address)` 只描述了地址数值输入，编译器前端不会从指令字符串自动理解这次读取；因此应保留 `"memory"` 来声明内存副作用。去掉后即使某次编译和测试通过，也不足以证明其他优化条件下同样可靠；也不能仅凭本次测试断言去掉后一定出错。`"memory"` 自身不插入 GPU 同步指令，不能替代 `__syncthreads()`。
