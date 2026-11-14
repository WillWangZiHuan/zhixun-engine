# 来源与第三方组件

runtime/baselines 以比赛平台下载的官方基线为起点，包含项目修改。原始来源与历史核验保存在项目中。模型训练与预测使用 NumPy、scikit-learn、LightGBM；实际版本见 ENVIRONMENT.json。

当前逐轮入口只使用平台提供的 workspace_io.model；保留的历史研究模块曾使用 PyTRIO、Transformers 和 SwanLab，不属于本次逐轮模型调用路径。源码包不包含这些依赖二进制、分词器、模型权重或平台服务实现。

未为官方基线或第三方代码重新授予许可证，也未为整体指定统一开源许可证；原有权利与许可保持不变。
