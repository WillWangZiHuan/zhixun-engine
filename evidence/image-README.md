# 第三版：官方 SDK 1.0.2 接入候选

从 `zhixun-workspace-qwen-v1` 的完整工业模型续训能力迁移至官方
`contest_harness` 1.0.2，协议为 `contest.harness.workspace.v2`。
旧版本、已上传镜像和初始模型保持不变。

平台入口：

```sh
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

每轮先校验父模型，读取当前公开数据、实际历史和上轮状态。
第一次 `rnd.model` 调用提出续训程序；执行 CPU 续训后，用 `rnd.submit_candidate`
提交完整检查点和指定预测命令。第二次 `rnd.model` 接收平台真实测量回执，
决定采用新候选还是保留父模型。随后保存状态、提交实际调用 ID 和历史 ID，
等待 `round_ack` 后结束。两次调用均走平台 SDK；不提供模型密钥、不自行计费估算、
不重复添加官方提示，不使用旧 `workspace_io` 或 PyTRIO 客户端。

预测只加载 `CONTEST_ARTIFACT_PATH` 指向的选中产物，从 `/model` 读取完整模型；
读到一行样本后立即输出并刷新，之后才读下一行。不等待输入流结束。
保留 Adam 状态、标准化参数、A 的附加头和读出表达式、C 的残差树。

`/app`、`/data`、`/previous`、`/state` 和 `/model` 只读。
所有新文件写入 `/workspace` 的本轮新目录；原生状态先复制再修改。
模型输出、候选测量、调用回执和失败环节分别记录。非法输入或程序直接报错，
不会转成固定规则训练或无声保留父模型。

源码目录不包含本地测试宿主、历史回答、数据、模型权重或凭据。
官方 SDK 的 Python 文件逐字保留，来源与哈希见 `PROVENANCE.json`。

## 本地证据的范围

验证见 `verification/sdk-integration-20261117/本地验收报告.md`。
本地用真实公开数据、真实 CPU 续训、独立预测进程和官方 SDK；
模型回答使用明确标识的历史文本回放及测试选择，不产生新的千问调用。
macOS 上仅将平台固定挂载路径映射到临时目录，测试映射代码不进入交付包。
本地镜像归档通过静态格式和文件核验，未在 Linux/Docker 中启动。
真实网关、平台资源隔离、隐藏数据及比赛成绩仍须平台验证。
此候选未上传、未启动平台试运行。
