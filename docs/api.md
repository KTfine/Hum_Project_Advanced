# 拍照助手如何调用 API

`realtime_av_test.py` 自动读取同目录 `settings.json` 中的 `api_key`，通过 HTTPS 向 DeepSeek 的 Chat Completions 接口发送请求。调用 Python 接口时，显式传给 `DeepSeekClient(api_key=...)` 的密钥优先。省略此参数时，客户端先读配置文件，再读环境变量 `DEEPSEEK_API_KEY`。两者都未提供密钥时，实时客户端可以在交互终端隐藏输入。详见[项目启动说明](../README.md)。

客户端按当前模式组装 system 提示词。提交给模型的内容可以包含用户文字和当前画面的 JPEG。指导模式使用真人画面。合照模式按当前位置和透视大小叠加立绘，传送期间也使用到位后的清晰形象。模型通过流式响应返回文字，客户端拼接分块并执行已闭合的标签。例如 `<speak:抬高一点下巴。>` 会在终端显示建议，`<teleport:x=0.7,y=0.9,scale=1.1><capture>` 会按指定构图直接拍照。[模式切换](modes-and-prompts.md) · [标签协议](reference/command-protocol.md)

在实验中向模型发送本地图片或订阅增量文字和指令，可直接使用 Python 接口 `DeepSeekClient`、`PhotoSession`。示例见[独立图文实验](api-experiments.md)。

实时客户端收到键盘输入或语音转写时，优先提交用户的话和当时可用的画面。用户开始对话后，内置指导和合照模式每 2 秒有一次观察机会。会话空闲且有可用摄像头画面时，客户端编码最新画面的 JPEG 并发起观察，即使画面内容保持静止。模型忙或摄像头暂时没有画面时，本次机会直接跳过。下一次观察取当时的画面。无摄像头或使用自定义 `--prompt` 时，仍由用户输入发起请求。

观察请求在日志中记为 `api.observation`，流式事件带有 `turn_source=observation`。新的用户输入可以打断正在生成的观察回复。指导模式发送实景，合照模式把立绘与实景合成，并附上同帧的角色位置信息。[实时会话](../photo_assistant/api/session.py) · [HTTP 流式客户端](../photo_assistant/api/client.py)
