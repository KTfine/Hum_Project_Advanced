# 普通话麦克风输入验证

2026-11-22，Asia/Shanghai。主版工作区基于 `0ac8b21`。本轮将 `photo_assistant/voice_input.py` 接入客户端，与打字共用输入队列。测试使用 Windows x64、Python 3.12.5、sounddevice 0.5.6、Vosk 0.3.45 和 `vosk-model-small-cn-0.22`。模型压缩包与官方清单中的大小和 MD5 一致。[官方模型清单](https://alphacephei.com/vosk/models/model-list.json)

## 设备与录音回放

实际默认麦克风成功打开，程序收到就绪提示后结束识别并关闭输入流。另用项目本地 `experiments/runtime/` 中的三段 WAV 替代输入设备，按采样率分块，每段后加入两秒静音。通过实际 Vosk 模型、客户端队列和 `send_realtime_inputs`，在本地替身会话接收文字。每段恰好提交一次。

| 音频 | 最终提交的文字 | 对照标注，忽略标点 |
|---|---|---|
| `photo_move_left.wav` | 现在请把你自己像画面的左边移动一点 | “向”转写为“像” |
| `photo_enter.wav` | 我们一起来拍照吧请你进入画面和我合照 | 一致 |
| `photo_hold.wav` | 你先保持这个位置站在原地不要动 | 一致 |

本轮未采集真人现场口令，也未调用 DeepSeek 或实际拍摄。回放只检验这些样本的听写和会话投递。录音保存在本机忽略目录。

## 程序回归

在 `Hum_Project` 中运行 `python -m unittest tests.test_voice_input tests.test_voice_input_integration` 可复跑模拟输入测试，覆盖分句、清除中文分词空格、丢帧恢复、设备错误、退出释放、键盘与麦克风共用队列，以及命令行和配置开关。
