智训引擎 native checkpoint 源码审阅包 v1

本包与已验收工程镜像的 src、runtime、models 文件逐字一致。仅供发布前本地审阅，
尚未发布。PROVENANCE.json 绑定镜像 SHA 和独立审计；SHA256SUMS 列出包内文件。

在已有、已验收的 Linux Python 环境中运行。进入解包后的本目录：
  export PYTHONPATH="$PWD/src:$PWD/runtime"
  /opt/venv/bin/python -B -m zhixun.experiment_agent.unified_controller --help

依赖版本见 ENVIRONMENT.json，来自现有镜像元数据和 Linux 实测记录。此包不安装
环境，不包含依赖二进制、SDK 密钥或 tokenizer。冻结 src 中 egg-info 是历史元信息，
其中 NumPy 2.5.4 不代表已验收运行环境；运行环境使用 NumPy 1.26.4。
请直接用 PYTHONPATH，不按该历史 egg-info 重新安装。runtime/requirements.txt 是
官方原文件，也不是整个应用的依赖清单。

独立预测（stdin 为每行一个含 inputs、可选 task 的 JSON 对象）：
  /opt/venv/bin/python -B -m zhixun.solution predict --checkpoint "$PWD/models/A/checkpoint.npz" --task A --training-runtime "$PWD/runtime"
B/C 使用相应文件与 --task B/C。预测不读取训练集、标签或模型目录的其他附加文件。

三个 models/*/checkpoint.npz 是原镜像老规则模型的完整、无损 NPZ 转换，含实际
核心/Adam/归一化状态和已有分类头或残差森林。它们是参考模型，不是平台指定初始
模型或父模型，也没有换成千问/新规则的模型或得分。models.json、summary.json 保留
历史指标和来源；它们不代表本轮新训练结果。原版 baseline Model.load 不支持 native
格式，应使用本包实际预测入口；官方 15-array core 输入由 native loader 显式识别。

真实训练必须使用任务供给的 A/B/C bundle（含其 checkpoint、baseline-info 和 public
数据），不是自动使用这里的参考模型。已有 Linux 环境的本地控制器命令形态为：
  /opt/venv/bin/python -B -m zhixun.experiment_agent.unified_controller --bundles <供给bundle根目录> --output <新输出目录> --policy rule --cpu-budget 600
qwen 策略仍需显式 --policy qwen、--price-record、--tokenizer 和运行时
PYTRIO_API_KEY；本包不提供这些私密/外部输入，也不授权调用。可沿用已有环境的
/opt/zhixun/assets/tokenizer。导出已完成的本地三任务运行：
  /opt/venv/bin/python -B -m zhixun.solution export --runs <已完成run根目录> --output <新导出目录> --training-runtime "$PWD/runtime"

镜像默认 --help 仅展示帮助，不是正式平台执行入口。正式启动、挂载路径、平台父模型
与历史交付、采用权归属及 checkpoint schema 合同尚未确认。此包不声称平台接受或
Docker 生命周期已验证。独立审计已覆盖相同冻结组件；源码包本轮只做解包哈希及
现有本地环境的 import/help 检查，不新跑 Linux/训练/全量预测/API/平台动作。
