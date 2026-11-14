# 智训引擎：逐轮千问续训入口

此版本对应 round-qwen-amd64-v1 镜像候选。模型为平台指定的 Qwen/Qwen3.8-27B 基础模型，未宣称使用微调 Sampler。每轮读取真实父 checkpoint、当前数据与真实历史；千问提出数值训练程序，继续工业小模型并输出完整 native NPZ。缺少设备 meta 不阻止训练；API 失败不会切换固定规则。

默认入口：`/opt/venv/bin/python -m zhixun.round_controller`。本地使用 `PYTHONPATH=src:runtime python -m zhixun.round_controller --help` 查看完整参数。独立预测：`python -m zhixun.solution predict --task B --checkpoint ACTUAL_CHECKPOINT --training-runtime runtime`。

本次最终源码在官方公开数据完成三任务各两轮真实千问续训；第二轮仅去掉 meta，6/6 完成；独立预测核对25,972行。此前小合成十轮记录属于较早源码，不混称最终版三十轮通过。公开开发结果不是隐藏成绩或获奖证据。

完整输入输出见 PLATFORM_INTERFACE.md（构建时接口核查快照）。平台实际调用绑定仍待此次试运行证据确认。没有打包私钥、API Key、大模型权重或旧参考工业模型充当父模型。

镜像 SHA256：418d733a65901b44974b9e1ed98956708bf097fff78a48997d0766351df21003。镜像以既有已导入层为底层追加源码，经静态校验；未声称执行了真实 Docker load/run/save。

ENVIRONMENT.json 区分镜像依赖与本地验证环境。源码按 PYTHONPATH 使用；不使用旧 egg-info 作为安装要求。第三方声明见 THIRD_PARTY.md。
