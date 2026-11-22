# Edge TTS 双语流式朗读

2026-11-22，Asia/Shanghai。Advanced 主分支 `015279e` 后的工作区。变更涉及流式协议解析、客户端分段和播放工作线程。运行环境为 Windows、Python 3.12、edge-tts 7.3.1、miniaudio 1.73、sounddevice 0.5.6。

在 `DirectSession` 使用本地回复提供器，逐块送入 `<sp`、`eak:肩膀`、`放松，`。收到第一段有效音频并写入实际默认输出设备后，才继续送入英文和闭合符。英文尾段刻意省略标点，以检验标签闭合会提交剩余台词。合成和播放均使用运行时的实际实现。这次回放没有调用 DeepSeek、摄像头或麦克风。

| 输入 | Edge 端实际收到的语音请求 | 结果 |
|---|---|---|
| `<speak:肩膀放松，` | `zh-CN-XiaoyiNeural`，一次 | 首次请求在测试开始后 0.02 秒发出，2.11 秒时已向扬声器写入非静音采样。此时尚未发送闭合符。 |
| ` Now look at the camera><blank>` | `en-US-MichelleNeural`，一次 | 闭合后提交剩余英文，紧接着的静默指令没有触发朗读。 |

两段音频时长分别为 1.82 秒、2.18 秒。实际写向扬声器的采样保存在本机 `experiments/runtime/edge-bilingual-stream-preview.wav`。上述延迟为单次网络观测值。

可用 `python -m unittest tests.test_command_protocol tests.test_spoken_replies tests.test_speech_output` 复核跨分块协议、停顿分段、数字标点、语种选择及播放顺序。内置语言提示已改为跟随中英文输入，本次验证使用固定分块，真人对话效果未测。
