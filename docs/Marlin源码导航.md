# Marlin 源码导航与本机环境

本仓库承载自己的学习实现。当前机器上，学习对照来源是同级目录 `../vllm-from-scratch/`；下面的路径相对于那个仓库的根目录，**不是本仓库内已经复制的文件**。另一台机器上需要另行准备该参考仓库及其私有资料。

- `minivllm/layers/quantization/awq.py`：AWQ 权重、加载后处理和线性层调用。
- `minivllm/layers/quantization/marlin.py`：padding、重排、workspace 和 GEMM 包装。
- `minivllm/quant_ops/ops.py`、`build.py`：Torch 算子入口与扩展构建。
- `minivllm/csrc/quantization/marlin/marlin.cu`：C++ 检查、配置选择与 kernel 发射。
- `minivllm/csrc/quantization/marlin/marlin_template.h`：CUDA GEMM 主体。
- `minivllm/csrc/quantization/marlin/{marlin.cuh,dequant.h,marlin_mma.h}`：异步搬运、解码和 MMA 辅助实现。
- `minivllm/csrc/quantization/marlin/gptq_marlin_repack.cu`：加载时的权重重排。
- `tests/step121/`、`tests/step122/`：已有重排、GEMM 的测试与对照入口。

参考仓库的 Marlin CUDA 文件来自 vLLM commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`；具体来源及许可证以其 `minivllm/csrc/quantization/marlin/PROVENANCE.md` 和 `LICENSE` 为准。若以后复制或改编代码，需保留相应许可与出处，区分移植和独立工作。

本机参考仓库的环境脚本要求在它自己的根目录运行。若确实需要使用那套现有环境，可在**当前 shell**中执行：

```bash
cd ../vllm-from-scratch
source scripts/activate-4090.sh
cd ../quant-gemm-project
```

这会激活参考项目的 `.venv` 并配置其 CUDA 工具链；它不会为本仓库创建独立环境。是否继续复用或另建环境，由后续实际依赖决定。运行 profiler 前还需单独确认对应工具、GPU 空闲情况和硬件计数器权限。
