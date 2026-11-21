# Marlin 源码导航与本机环境

学习对照源码现已放在工作区 [third_party/vllm_marlin](../third_party/vllm_marlin/README.md)，可以直接在 VS Code 中阅读、搜索和跳转。这里是参考项目 `minivllm` 的源码快照与按其 SM89 W4A16 构建规则生成的 selector，不是本项目从零实现的算子；原始许可证、出处、生成文件和快照边界见目录内 README。

以下路径相对于 `third_party/vllm_marlin/`：

- `minivllm/layers/quantization/awq.py`：AWQ 权重、加载后处理和线性层调用。
- `minivllm/layers/quantization/marlin.py`：padding、重排、workspace 和 GEMM 包装。
- `minivllm/quant_ops/ops.py`、`build.py`：Torch 算子入口与扩展构建。
- `minivllm/csrc/quantization/marlin/marlin.cu`：C++ 检查、配置选择与 kernel 发射。
- `minivllm/csrc/quantization/marlin/marlin_template.h`：CUDA GEMM 主体。
- `minivllm/csrc/quantization/marlin/{marlin.cuh,dequant.h,marlin_mma.h}`：异步搬运、解码和 MMA 辅助实现。
- `minivllm/csrc/quantization/marlin/gptq_marlin_repack.cu`：加载时的权重重排。
- `generated/sm89_w4a16/kernel_selector.h`：参考构建规则生成的特化选择分支。

完整参考项目的其他模块、依赖和 `tests/step121/`、`tests/step122/` 仍在本机同级 `../vllm-from-scratch/`，没有全部复制到本仓库。快照中的 Python 文件采用原有相对导入，不能脱离完整项目直接运行。

## VS Code

仓库的 `.vscode/` 提供 CUDA/C++ 文件关联、扩展推荐和 IntelliSense 搜索路径，覆盖本仓库参考头文件、生成的 selector、同级 `cuda-12.9` 和 `vllm-from-scratch/.venv` 中的 PyTorch/CUBLAS 头文件；这些是当前机器上已确认存在的路径，不依赖 VS Code 启动时设置 `CUDA_HOME` 或 `VIRTUAL_ENV`。安装推荐的 Microsoft C/C++ 和 NVIDIA Nsight VS Code 插件后，直接打开本仓库。不同机器、Python 版本或工具链路径可能需要调整 `.vscode/c_cpp_properties.json`；编辑器索引不是编译验证。配置使用本机 `nvcc`，并在 IntelliSense 中指定 `__CUDA_ARCH__=890` 来查看 RTX 4090（SM89）的设备分支；这只影响编辑器分析，不会改变实际编译参数。真实 CUDA 编译包括 host 和 device 阶段，host 阶段可能未定义此宏，device 阶段的值由实际编译架构决定。

如需在本机当前 shell 中复用参考项目的 `.venv` 与 CUDA 工具链：

```bash
cd ../vllm-from-scratch
source scripts/activate-4090.sh
cd ../quant-gemm-project
```

这不会为本仓库创建独立环境。运行 profiler 前还需单独确认对应工具、GPU 空闲情况和硬件计数器权限。
