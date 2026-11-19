# 实验工作区

这里集中管理 prompt 实验、验证记录和复用工具。日常拍照助手直接调用 DeepSeek API，见 [项目 README](../README.md) 和 [API 调用](../docs/api.md)；独立图文实验工具的用法见 [API 实验说明](../docs/api-experiments.md)。

| 目录 | 内容 |
|---|---|
| [records/](records/README.md) | 按日期记录的问题、条件、结果与限制 |
| [prompts/](prompts/) | 有版本的候选提示词，默认客户端不会自动加载 |
| [scripts/](scripts/) | 可维护的实验入口，包括独立 API 图文会话和模块示例 |
| [templates/](templates/report.md) | 新实验的记录模板 |
| [plans/](plans/human-evaluation.md) | 尚待执行的真人评估设计 |
| runtime/ | 本机原始日志、媒体、请求及历史脚本；被 Git 忽略 |

## 从哪里开始

所有命令在仓库根目录执行。独立 API 实验需要 Python 3.12，与拍照助手共用仓库根目录 `settings.json` 中的 `api_key`。也支持环境变量 `DEEPSEEK_API_KEY`。完整参数见[实验工具说明](../docs/api-experiments.md)。

```powershell
python -m pip install -r experiments/requirements.txt
python -m experiments.scripts.hum_api --interactive
python -m experiments.scripts.hum_api --prompt-file experiments/prompts/p2-20261108.txt --interactive
python -m experiments.scripts.api_module --image 'D:\photos\scene.jpg'
```

新一轮先复制 [记录模板](templates/report.md)，记录实际发送的 system 文本及哈希。候选提示词和历史成功率只对应记录中的条件，不能视为当前版本已通过。

## 2026-11-13 路径整理

| 原位置 | 现在的位置 |
|---|---|
| docs/validation/ | experiments/records/ |
| prompts/experiments/ | experiments/prompts/ |
| hum_api/ | experiments/scripts/hum_api/ |
| examples/api_module.py | experiments/scripts/api_module.py |
| tools/runtime_effect_probe.py | 旧录音重放脚本，已于 2026-11-18 从日常工作区移除 |
| requirements-api.txt | experiments/requirements.txt |
| docs/maintenance/experiment-template.md | experiments/templates/report.md |
| docs/maintenance/human-evaluation.md | experiments/plans/human-evaluation.md |
| runtime_results/ | experiments/runtime/ |

历史报告保留当时运行的命令、路径、版本和结论，Markdown 导航已更新。报告中提到的 `runtime_results/` 现在对应 `experiments/runtime/`，不是丢失的材料。

原始材料迁移前后逐文件核对相对路径、字节数和 SHA-256：1230 个文件、1,283,115,158 字节全部一致。本机核对清单位于被忽略的 `experiments/runtime-migration.local.json`。这只证明材料完整迁移，不新增模型质量结论。

`runtime/` 中的历史脚本保留原字节；其中的绝对机器路径、旧目录名和 `parents[n]` 根目录定位均可能不再适用，不能直接承诺复跑。需要复跑时复制为新实验脚本，明确修正路径并记录差异，不覆盖原始证据。
