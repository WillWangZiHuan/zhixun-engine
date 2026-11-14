# 智训引擎 · Native Checkpoint v1

本仓库提供参赛候选镜像 `zhixun-engine:native-checkpoint-candidate-amd64-v1` 对应的 Python 源码、运行配置和 A/B/C 参考检查点，供代码核查。

镜像归档 SHA-256：
```
8faf811bbe18cc170106db4d7a41baff43be801f5706020699169f7d103619bc
```

## 内容与核验

- `src/`：55 个源码及原始包元信息文件，对应镜像 `/opt/zhixun/src`。
- `runtime/`：9 个运行代码和配置文件，对应 `/opt/zhixun/runtime`。
- `models/`：3 个原生 NPZ 参考检查点及 2 个模型元信息文件，对应 `/opt/zhixun/models`。
- `PROVENANCE.json`：上述 69 个镜像组件的 SHA-256 与来源。
- `ENVIRONMENT.json`：已核验镜像环境的版本记录。
- `SHA256SUMS`：冻结源码包的文件校验清单。

69 个镜像组件保持冻结版本的原始字节。`README.txt`、`PROVENANCE.json` 等保留打包时的历史状态，其中“尚未发布”描述的是该快照生成时的状态；本次发布另记于 `PUBLICATION.json`。

## 运行入口

已核验环境为 Linux / Python 3.12，具体依赖版本见 `ENVIRONMENT.json`。使用源码路径加载，不依据历史 `egg-info` 元信息安装依赖。

```sh
export PYTHONPATH="$PWD/src:$PWD/runtime"
python -B -m zhixun.experiment_agent.unified_controller --help
python -B -m zhixun.solution --help
```

独立预测入口：
```sh
python -B -m zhixun.solution predict \
  --checkpoint "$PWD/models/A/checkpoint.npz" \
  --task A --training-runtime "$PWD/runtime"
```

完整命令与运行边界见 `README.txt`。实际预测输入须按命令帮助提供。Qwen 策略使用基础模型 `Qwen/Qwen3.8-27B`；运行时由环境变量 `PYTRIO_API_KEY` 提供密钥。仓库不包含密钥、原始任务数据、训练日志、分词器或大模型权重。

## 检查点与验证范围

原生 `checkpoint.npz` 保存实际模型及续训状态，支持独立预测。A/B/C 参考检查点来自已完成的历史规则策略实验，不是 Qwen 微调权重，也不能代替平台在后续轮次提供的父检查点。续训必须使用实际父检查点并保留真实历史。

该版本已通过本地及 Linux 进程级检查。平台镜像导入就绪不等于模型校验、续训协议验证或评测通过；本仓库不声明已完成平台试运行或正式评测。镜像与源码的对应关系可依上述哈希核查。

官方基线来源和第三方说明见 `THIRD_PARTY.md`。
