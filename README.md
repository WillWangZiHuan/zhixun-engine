# 智训引擎：平台模型接口版

本候选将每轮决策改为一次 `workspace_io.model(messages)`，由平台选择提交表单中的 Qwen/Qwen3.8-27B 并记账。不直接访问 PyTRIO SDK，不需要向训练进程注入 API Key、服务地址、报价或 tokenizer。失败保留记录，不重抽、不用规则或未训练父模型替代。

## 平台交接

平台失败记录（2026-11-15，运行 422c86316d6c77f1cd51a4ae5866f81f）确认训练/预测入口可桥接，并指定 `workspace_io.model(messages)`。本地未取得 workspace_io 的完整部署实现；不声称它已自动出现在镜像中。

首选由持有真实 workspace_io 的平台适配器在同一 Python 进程调用：

```python
from argparse import Namespace
from pathlib import Path
from zhixun.round_controller import run
result = run(Namespace(
    task=current_task, round=current_round,
    parent=Path(actual_parent), train=Path(actual_train),
    validation=Path(actual_validation), history=Path(actual_history),
    output=Path(new_output_directory), cpu_seconds=20, check=False,
), workspace_io=workspace_io)
```

变量均应来自本轮真实平台数据，不是预设路径。调用器须提供同步 `model(messages)` 方法；本候选不构造私有网关、不注入 SDK 凭据、不伪造服务响应。

平台若将同一对象作为可导入 `workspace_io` 模块暴露，可以使用镜像默认入口：

```text
/opt/venv/bin/python -m zhixun.round_controller \
  --task A --round 2 --parent ACTUAL_PARENT.npz \
  --train ACTUAL_TRAIN.jsonl --validation ACTUAL_VALIDATION.jsonl \
  --history ACTUAL_HISTORY.jsonl --output NEW_OUTPUT_DIR
```

以上大写路径是接口占位符。CLI 只导入平台提供的 workspace_io，不打包同名假服务；缺失时明确失败。`--check` 只做输入预检，不表示模型服务可用。模型调用与整个作业的超时、令牌上限由平台网关/调度管理，CPU训练子进程沿用20秒CPU限制和40秒等待限制。

## 输入、响应和输出

- 父模型必须是平台本轮指定 checkpoint；同时支持官方原始检查点及本方案完整 native NPZ。恢复优化器、归一化、累计步数和附加模型；不自选上一候选作父模型。
- 数据使用 id、inputs、answer。训练/验证 ID 不重叠；不依赖或补造设备 meta。验证标签只用于评价与提供训练诊断，不用于拟合。
- 历史为前 n-1 轮真实 JSONL，有序 `round_index` 与真实 `decision`（promote/keep/retain），其他字段原样保留。保留原始历史文件及哈希，不用 checkpoint 步数重建历史。
- `model(messages)` 返回 JSON 对象，其中唯一 `choices[0].message.content` 是完整动作文本。只解析实际文本；不要求 token id、本地 token 数或 stop_reason。完整 JSON、动作 schema、安全数值程序和梯度检查仍必须通过。没有固定JSON前缀或重新采样。
- `usage` 可缺失；若返回，原样保留并核验已知计数字段的非负整数类型。不将缺失计数写成0，不从字符或本地 tokenizer 推算账单。真实费用以平台账本为准。
- `calls/call-01/` 保留消息、原始响应、动作、用量和结果。调用开始标记只证明进入方法，不冒称远端已收到请求；异常不自动重试。账单的 `estimated_cny=null` 表示未知。
- 成功输出 `model/checkpoint.npz`（完整模型产物<=64MiB）、`training.json`、本轮训练 `history.jsonl`；原始平台历史另存 `evidence/platform-history.jsonl`。候选是否采用由平台决定。
- 失败退出2（CLI），保留已生成证据，不把失败候选导出为成功模型。

独立预测：

```text
/opt/venv/bin/python -m zhixun.solution predict \
  --task A --checkpoint ACTUAL_CHECKPOINT.npz \
  --training-runtime /opt/zhixun/runtime
```

stdin逐行 `{"inputs":...}`，输出官方 JSONL `outputs` 格式。预测使用完整A分类头/读出、C森林，不以原15数组loader代替。Docker启动预测需要覆盖默认训练入口。

## 验证边界

本地测试覆盖实际训练/预测进程，模型接口由明确的测试宿主提供；历史千问文本回放不是新千问生成，不是平台已接入。未上传本候选、未新增试运行、未执行Docker load/run/save。镜像沿用已被平台导入的Linux/amd64底层，通过Docker-save格式静态核验；静态检查不能替代容器运行。

使用 `PYTHONPATH=/opt/zhixun/src:/opt/zhixun/runtime`。源码包不含密钥、数据集、模型权重或模拟 workspace_io 服务。
