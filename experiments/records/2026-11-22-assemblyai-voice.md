# AssemblyAI 实时语音与 5 秒自动观察

2026-11-22，Asia/Shanghai。基线为 `015279e` 后的工作区，运行环境为 Windows 和 Python 3.12.5。验证覆盖语音输入输出、启动配置及会话调度。

| 验证项目 | 条件与结果 |
|---|---|
| 本机麦克风 | 默认输入设备以 44,100 Hz 单声道采样一秒，44,100 个采样中 38,392 个非零，没有溢出。只统计电平，未保存录音。 |
| 本机回环 | 使用真实默认麦克风和正式听写客户端，向本机 WebSocket 测试服务发出三个 100 毫秒音频块，每块 8,820 字节。客户端接收模拟的讲话开始和最终转写，交给文字回调，并在退出时发送 `Terminate`。 |
| 听写提交 | 模拟 AssemblyAI WebSocket 的讲话开始、临时转写和回合结束。中文与英文最终指令各提交一次。覆盖重复结果、外放静音、断线重连和退出时结束远程会话。 |
| 分段朗读 | 模拟 AssemblyAI HTTP 音频流及扬声器，核对第一批有效 PCM 可以在该段后续音频抵达前播放；核对含汉字段落使用 Edge，纯英文段落使用 Audrey。停顿切分回归覆盖跨分块的逗号、小数、时间和尾段。 |
| 视觉复查间隔 | 模拟会话时钟。经过 4.99 秒时不会提交观察，达到 5 秒后提交一次。 |

`python -m unittest discover -s tests` 共 231 项通过。本机没有配置 AssemblyAI 密钥，本轮未验证真实云端识别文字、声音或时延。官方协议见 [实时听写](https://www.assemblyai.com/docs/streaming/api-spec/streaming-websocket) 与 [流式合成](https://www.assemblyai.com/docs/tts/api-reference/http-api/overview)。
