# 来源与第三方组件

项目实现：SDK 轮次衔接、数值程序校验、完整模型保存与恢复、状态和证据记录。
`runtime/baselines` 以比赛平台公开基线为起点，含项目修改。
官方 `contest_harness` 1.0.2 的 9 个 Python 文件保持原字节，来源为
https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.2.tar.gz 。
该归档 SHA-256 为 c676c8beeef69cece9cbec5a633d05dbb9832eb22b6f761913f2315bd1f85133。

数值训练与预测使用 NumPy、scikit-learn、LightGBM，版本见 ENVIRONMENT.json。
当前模型调用通过官方 SDK 的 host_gateway_rpc；没有自建外部模型客户端。
旧基础镜像可能保留历史依赖，不能据依赖存在推断本版调用路径。

未替官方 SDK、官方基线或第三方依赖重新授予许可证；各组件原有权利和许可保持不变。
未将整体代码标成自创的官方基线，也未为整个仓库添加未经确认的统一开源许可证。
