# B/C 官方 CPU 续训代码

来源：官方 B 包代码，经 `runs/official-group-weight-v1` 与 `runs/official-indexed-selection-v1` 的训练组均衡改动及完整独立复算。此处仅整理已有实现，模型/训练/预测源码保持冻结 B 运行时原样；B/C 配置分别取各自官方包。原始文件、历史测试与结果未修改。

训练环境 Python3.11、NumPy1.26.4。入口在上级 `src/zhixun/joint.py`；用法与验收见 [官方续训工程入口](../OFFICIAL_ENGINEERING.md)。训练组元数据正式可用性仍待官方确认。

此目录不包含数据、检查点、大模型权重或密钥；未作为镜像上传。官方代码使用依据为赛事包提供的参赛修改用途，不自行赋予其他开源许可证。
