# Flash 无思考模式的拍照提示词

2026-11-18，Asia/Shanghai。基于 `a9fafa9` 后包含立绘图层的本地工作区。本轮调整了 `prompt.py` 的两份模板、`direct_api.py` 发出的观察消息，以及观察消息的集成断言。

指导和合照均先区分用户消息与自动观察，再决定回应、动作或快门。合照模板用三种不同的用户意图说明图层、缩放及肩旁探头的构图。后续观察仅在存在待拍请求且条件满足时触发快门。

| 验证 | 结果 |
|---|---|
| 项目 Python 运行 `python -m unittest discover -s tests` | 117 项通过，涵盖实时提示词组装、自动观察标记、模式切换、动作解析及执行。 |
| 在本地传输层模拟 8 个纯文本用例 | 实时入口组装的模板进入请求，显式使用 `model=deepseek-flash`、`thinking.type=disabled`，请求未携带 `reasoning_effort`。问候、自动观察问候、邀请合照、到人前、回人后、靠近镜头、立即拍摄、已拍后的观察共 8 项，预置的模拟回复均解析成预期标签。 |
| 外部供应商 | 实际请求 0 次。自动审批未允许将项目提示词和示例状态发送给外部 DeepSeek 服务。未取得真实回复、服务端模型版本或效果评测数据。 |

用例中指导模板 SHA-256 为 `3e23862aebe49227d674fcb81fe14a08f2aabee706dfb3eabaaed30b5918775d`，合照模板为 `ff03145173b2d557e44c6c8d09add56bc2ea85a3a2e7b6e30e9040dae48a44bf`。本地模拟不涉及采样和摄像头取景。

先行模拟因测试脚本变量重名在第 2 个用例中止；修正后 8 项通过。两次结果分别保存在本机忽略目录 `experiments/runtime/flash-off-offline-20261118T115415Z.jsonl` 和 `experiments/runtime/flash-off-offline-20261118T115453Z.jsonl`。
