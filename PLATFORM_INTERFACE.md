# 新逐轮入口与平台接入状态

2026-11-15 00:13（北京时间）通过普通参赛账号只读刷新：公告、规则、基线目录、提交、镜像和前端与上一轮相同；现有运行报告仍为固定 rule/core 导出。试运行 1/10，正式 0/2，正式尚未开放。本次没有上传、提交或运行操作。

## 已确认与缺口

官方公告明确后续轮次使用平台父 checkpoint、保留真实历史、独立预测、checkpoint.npz 且模型产物合计不超过64MiB。三个本地官方训练包只提供训练/预测 CLI 示例，没有容器逐轮调度协议。公开前端没有 entrypoint/command 配置项；普通账户报告没有 argv、挂载、密钥环境变量或控制服务协议。

旧报告的 `One native rule experiment... official core format` 和 `per-row device meta` 说明均不在我们旧发布的 Python 源码里。这支持“平台调用链还存在外部适配”这一推断，不能据此断言适配器的位置或具体实现。仅替换镜像不会证明它停止绕过我们入口。

要用上本候选，平台调度器必须实际执行下列入口，并传入本轮文件；若平台覆盖镜像 ENTRYPOINT、强制调用旧固定 rule，必须在它的调度配置处改绑。提交表单当前没有可由我们填写的启动命令字段。不能通过改源码链接、上传说明或新增自定义镜像标签冒充已完成绑定。

## 本镜像提供的确切接口

默认进程：

```text
/opt/venv/bin/python -m zhixun.round_controller
```

镜像标签：`zhixun-engine:round-qwen-amd64-v1`。`CMD` 为空，缺参数会返回错误，不打印帮助后宣称训练完成。调度器须为一次调用提供：

| 参数/变量 | 真实来源与要求 |
|---|---|
| `--task A/B/C` | 当前数据集 |
| `--round 1..10` | 当前实际轮号 |
| `--parent FILE` | 平台本轮选定的父检查点；不是镜像自带模型 |
| `--train FILE`、`--validation FILE` | 本轮允许使用的数据，id/inputs/answer；meta 可缺 |
| `--history FILE` | 前 n-1 轮真实历史 JSONL；有序 round_index、decision=promote/keep/retain，其他真实字段保留 |
| `--output DIR` | 此轮新的输出目录，其父目录须存在且 UID10001 可写 |
| `--price-record FILE` | 可用随镜像 `/opt/zhixun/config/qwen-price-reference.json`；2026-11-12估算记录，非现价或账单 |
| `--tokenizer DIR` | 随镜像 `/opt/zhixun/assets/tokenizer`，本地哈希核验，无下载 |
| `PYTRIO_API_KEY` | 运行时传入；镜像/源码均不含密钥。平台当前具体注入方式未披露 |

这张表描述我们实际实现的接口，**不是已确认的平台挂载约定**。若平台原生历史字段不同，需要取得真实格式后明确映射；不从检查点步数补造轮次，不假设末次候选已采用。

成功返回 stdout JSON `status=candidate_ready`，退出0；候选 `OUTPUT/model/checkpoint.npz`，完整训练记录 `OUTPUT/training.json`；`OUTPUT/evidence/platform-history.jsonl` 保存收到的历史原字节。失败退出2，保留已生成的请求、错误和执行记录；没有固定规则替代、没有自动重抽。下一父模型由平台决定。

独立预测必须由平台直接执行以下进程，不能把完整 native NPZ 交给只支持原15数组的 stock loader：

```text
/opt/venv/bin/python -m zhixun.solution predict --task B --checkpoint ACTUAL_CHECKPOINT --training-runtime /opt/zhixun/runtime
```

stdin 为每行 `{"inputs":...}`，可带任务类型 `classification/regression`；不含标签/设备信息。输出沿用官方 JSONL 的 outputs 字段。Docker 直接运行预测时需显式覆盖默认训练 ENTRYPOINT。

## 需平台确认的最小三项

1. 本队 v1 当前真实启动 argv/ENTRYPOINT 覆盖规则；能否执行上面的新训练进程和独立预测进程，停止固定 rule/core-only 适配？
2. 每轮父 checkpoint、当前 train/validation、真实历史、输出目录的交付方式与历史字段实例（可脱敏）；平台怎样读取候选并返回下一轮采用结果？
3. 指定模型凭据和计量/网络访问如何注入；是否允许镜像内 PyTRIO 0.2.9 发起真实采样并保留其用量？

这些问题没有发送给管理员。本次只准备具体可核验接口与候选，不消耗试运行排查。

## 镜像与验证边界

新归档使用现有已导入镜像的相同底层环境，追加已本地验收的源码和明确入口；完整检查层哈希、配置、文件覆盖、tokenizer、Linux依赖及路径。无 Docker daemon、未使用本地 VM或已退租服务器，因此未执行真实 docker load/run/save。归档是工具构造并经 crane 验证的 Docker-save 格式候选；不把静态验证写成 Docker生命周期或平台验收。

标准 ENTRYPOINT 可被运行方覆盖，Docker 官方说明：https://docs.docker.com/reference/dockerfile/#entrypoint 。Docker-save 包含配置及所有层：https://docs.docker.com/reference/cli/docker/image/save/ 。如果平台要求提供实际 docker save 导出命令证据，应在获准的 Docker 环境 load/run 核验后再 save，不能虚构这一步已执行。
