# Marlin 学习参考源码

这里是学习用快照，**不是本项目自行编写的 GEMM 算子**。`minivllm/` 下的文件逐字节复制自本机 `../vllm-from-scratch/` 的 `minivllm/` 对应文件，采集时该仓库 HEAD 为 `3ae5eefb0ab3eecdbc75c16b7d7e733c5116aec0`。

- `minivllm/csrc/quantization/marlin/`：GEMM、repack、相关 CUDA 头文件、生成器、绑定和原始 `LICENSE` / `PROVENANCE.md`。
- `minivllm/csrc/core/` 和 `minivllm/csrc/libtorch_stable/`：上面代码引用的项目内头文件。
- `minivllm/layers/quantization/`、`minivllm/quant_ops/`：阅读调用链所需的 Python 方法和构建入口；这些文件不是完整的 `minivllm` 包，不能直接把该快照当成可运行扩展。
- `generated/sm89_w4a16/`：按照参考项目 `minivllm/quant_ops/build.py` 的 W4A16 过滤规则和 SM89 架构生成的 `kernel_selector.h` 与 4 个模板实例，供阅读配置选择和 VS Code 跳转；不属于复制的原始源码，不承诺是独立构建产物。生成前在临时目录对生成器应用该构建文件所示过滤，原始 `generate_kernels.py` 未改动。

Marlin CUDA 来源是 vLLM commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`，许可、上游作者信息和本地 namespace 修改见 `minivllm/csrc/quantization/marlin/PROVENANCE.md`、`LICENSE` 及各源码头部；`bindings.cpp` 是参考项目自己的 schema 注册。`PROVENANCE.md` 最后有关“尚未编译/验收”的句子是早期历史状态，实际验收以参考仓库的 `docs/step121_alignment.md` 和 `docs/step122_alignment.md` 为准。

学习时可复制思路自行实现；若复制或改编已有代码，需遵守许可证并说明来源和个人贡献。参考实现的结构与第三方依赖不自动定义本学习项目的生产接口。
