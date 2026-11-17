# 智训引擎 SDK 1.0.2 参赛版本

本版本使用官方 `contest_harness` 1.0.2，通过平台模型网关调用基础模型
`Qwen/Qwen3.8-27B`，对 A/B/C 工业模型执行完整检查点续训。
未微调千问；未提交 Sampler 微调权重。

当前候选镜像 `zhixun-sdk-1.0.2-amd64-v1.tar` 已由队员上传，
2026-11-17 工作台显示镜像 5 可用。镜像就绪仅确认镜像处理状态，
本 SDK 版本尚未启动平台试运行。

## 运行入口

```sh
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

每轮第一次模型调用提出数值训练程序；续训并提交完整候选后，第二次模型调用
根据平台测量回执选择候选或父产物。历史、调用 ID、原生状态、候选预测命令和
结束确认均通过 SDK 交接。代码不使用旧 `workspace_io` 接口或自己的模型密钥。

完整作品说明见 [docs/作品说明.md](docs/作品说明.md)。
镜像对应关系见 [IMAGE_MANIFEST.json](IMAGE_MANIFEST.json)，
本地验证见 [evidence/completion-audit.json](evidence/completion-audit.json)。

本地完成 A/B/C 各 10 轮及 259,720 次逐样本预测核对；附加能力另验证 4 轮、
48,586 次预测。模型回答为明确标注的历史回放与测试回答，不代表真实网关验证，
不构成性能提升或隐藏测试成绩证据。Linux/Docker 运行尚未验证。

## 源码与复现

运行代码、官方 SDK、配置与 Dockerfile 和已上传镜像逐字一致。
42 个冻结文件中的 README 保存在 `evidence/image-README.md`，其余保持原路径；
本页及 docs/evidence 中的发布说明为镜像冻结后的文档，不进入已上传镜像。
`SHA256SUMS` 覆盖本仓库本次发布文件，`PROVENANCE.json` 保留镜像源码出处。

Dockerfile 依赖已冻结的上一版本地基础镜像 `zhixun-engine:workspace-qwen-amd64-v1`。
本仓库没有发布 162 MB 镜像、基础镜像层、数据集或模型权重；仅克隆仓库不能取得
这些构建输入。正式运行使用比赛平台已就绪的镜像 5。

旧版本保留在 Git 历史与原固定提交链接中。来源及许可说明见
[THIRD_PARTY.md](THIRD_PARTY.md)。本仓库不包含 API Key、登录凭据、模拟平台服务或测试回答。
