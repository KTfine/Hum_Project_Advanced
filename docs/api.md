# 拍照助手如何调用 API

`realtime_av_test.py` 自动读取同目录 `settings.json` 中的 `api_key`，通过 HTTPS 向 DeepSeek 的 Chat Completions 接口发送请求。调用 Python 接口时，显式传给 `DeepSeekClient(api_key=...)` 的密钥优先。省略此参数时，客户端先读配置文件，再读环境变量 `DEEPSEEK_API_KEY`。两者都未提供密钥时，实时客户端可以在交互终端隐藏输入。详见[项目启动说明](../README.md)。

客户端按当前模式组装 system 提示词。提交给模型的内容可以包含用户文字和当前画面的 JPEG。指导模式使用真人画面。合照模式按当前位置和透视大小叠加立绘，传送期间也使用到位后的清晰形象。模型通过流式响应返回文字，客户端拼接分块并执行已闭合的标签。例如 `<speak:抬高一点下巴。>` 会在终端显示建议，`<teleport:x=0.7,y=0.9,scale=1.1><capture>` 会按指定构图直接拍照。[模式切换](modes-and-prompts.md) · [标签协议](reference/command-protocol.md)

在实验中向模型发送本地图片或订阅增量文字和指令，可直接使用 Python 接口 `DeepSeekClient`、`PhotoSession`。示例见[独立图文实验](api-experiments.md)。

实时客户端首次收到拍摄目标后才向云端发送请求。成功回复后，它继续使用近期画面和对话判断是否需要指导。新文字输入会打断尚未完成的回复，优先处理新目标。从代码查看行为可读[实时会话](../photo_assistant/api/session.py)与[HTTP 流式客户端](../photo_assistant/api/client.py)。
