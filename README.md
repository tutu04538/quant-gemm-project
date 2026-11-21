# quant-gemm-project

从 CUDA 基础开始，逐步实现和优化一个 W4A16 量化 GEMM，并在明确的输入、硬件和计时口径下与 Marlin 比较。这个仓库用于学习与实验；当前只有项目骨架，尚无自行实现的 GEMM 或性能结论。

## 学习入口

本机的学习路线位于 `learning_notes/00_从CUDA基础到手写Marlin路线.md`。我们先复习线程、内存、同步和 tiled GEMM，再阅读 Marlin 的入口、权重布局和 kernel，最后逐版独立实现和测量。

`learning_notes/` 按约定被 Git 忽略，**不会同步到 GitHub**。在另一台机器克隆本仓库后，需要另行准备这些私有笔记。当前机器上的冻结参考实现位于同级的 `../vllm-from-scratch/`，参考范围和来源见 [Marlin 源码导航](docs/Marlin源码导航.md)。

## 目录

- `kernels/`：逐版 CUDA/C++ 算子实现。
- `tests/`：独立数学与布局参考、正确性和边界测试。
- `benchmarks/`：可复现的计时和基线对照脚本。
- `profiling/`：profiler 采集脚本和分析入口；大型原始结果保存在忽略目录。
- `docs/`：可以公开同步的设计、布局图和实验结论。
- `scripts/`：本机环境检查等辅助脚本。
- `learning_notes/`：不跟踪的学习路线与个人笔记。

## 环境检查

```bash
bash scripts/check_env.sh
```

该脚本只检查工具是否可见，不安装依赖、不配置系统环境，也不检查 profiler 硬件计数器权限。当前机器已有项目的 CUDA 12.9 工具链和 Python 环境；具体路径及如何在当前 shell 中启用，见 [环境与参考](docs/Marlin源码导航.md)。

先从文档中的 vector add 开始；学习过程中按版本保留实现、独立正确性参考和实验记录。性能只报告实际测得的结果，并标明设备、shape、数据类型、权重准备和计时边界。
