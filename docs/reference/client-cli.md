# 实时客户端参数

在 VS Code 中打开 `realtime_av_test.py` 并点右上角 ▶，或在项目目录运行 `python realtime_av_test.py`。客户端先读取脚本同目录的 `settings.json` 中的 `api_key`，没有配置密钥时读取环境变量 `DEEPSEEK_API_KEY`。两者都为空时，交互终端会提示输入 Key。终端输入拍摄目标后按 Enter，程序会直接向 DeepSeek API 发出图文请求，并在同一终端显示回复。

| 选项 | 默认值与用途 |
|---|---|
| `--camera-device` | OpenCV 摄像头编号，选择顺序见下文 |
| `--moving-camera` | 手持或移动镜头时持续更新家具及深度。默认按固定机位运行；PowerShell 启动脚本对应参数为 `-MovingCamera`。 |
| `--no-camera` | 关闭摄像头采集，指导模式使用空白帧 |
| `--no-preview` | 隐藏预览窗口 |
| `--microphone` / `--no-microphone` | 开启或关闭 AssemblyAI 实时听写，默认开启，命令行优先于设置文件 |
| `--microphone-device` | 输入设备编号或可唯一匹配的名称，默认使用系统的输入设备 |
| `--tts` / `--no-tts` | 开启或关闭分段朗读，默认开启，命令行优先于设置文件 |
| `--sprite-catalog` | 指定带五部位标签的 JSON 立绘库。实时客户端默认使用 `models/avatars/white_dragon/catalog.json`。[立绘目录](../../models/avatars/README.md) |
| `--canvas-width` / `--canvas-height` | 没有可用摄像头帧时的画布尺寸，默认 1280 × 720；有摄像头画面时沿用其分辨率 |
| `--jpeg-quality` | 发给模型的 JPEG 质量，默认 80，范围 1–100 |
| `--character-name` | 自定义角色在对话中的名字 |
| `--character-persona-file` | 指定 UTF-8 性格文本，默认读取[角色性格.txt](../../角色性格.txt)，指导与合照共用。启动脚本对应参数为 `-PersonaFile`。 |
| `--character-persona` | 本次启动直接使用指定的性格文字 |
| `--prompt` | 使用完整自定义提示词，停用内置模板自动切换 |
| `--log-file` | 指定 JSONL 日志，默认存入项目内的 `experiments/runtime/` |
| `--verbose` | 在终端额外显示 `[blank]` 和 API 元数据 |

使用 `python realtime_av_test.py --help` 查看当前版本完整参数。在预览窗口按 Q 或在终端按 Ctrl+C 退出。

终端把同一回复的说话片段接在一行 `[speak]` 后面。拍照结果和模式切换各占一行。查看模型实际返回的标签和分块时，可在 JSONL 中查找 `api.receive`。

客户端默认使用 `deepseek-flash` 和 `low` 思考强度。可在运行前用 `DEEPSEEK_MODEL` 选择支持图像输入的模型，用 `DEEPSEEK_THINKING` 设置思考强度。程序进入指导模式后，在终端提交第一条文字以发起请求。

在 PowerShell 中先执行 `$env:DEEPSEEK_THINKING = 'off'`，再从同一终端启动客户端，即会使用无思考模式。

在终端输入“现在拍一张”。模型返回 `<capture>` 后，客户端会将照片保存到脚本同目录的 `photos/` 文件夹。指导模式保存摄像头原图，合照模式会按快门时的构图叠加角色。两种模式都使用相机帧的尺寸保存 PNG。使用 `--no-camera` 或尚未取得相机帧时会跳过拍摄。

## 小白龙的声音

在项目目录用启动客户端的 Python 执行 `python -m pip install -r requirements-voice.txt` 即可安装语音依赖。客户端把英文台词分段发送给 AssemblyAI，以美式女声 Audrey（`aai-audrey`）朗读。AssemblyAI 的独立合成服务支持英语等六种语言，普通话仍使用 Edge 晓伊（`zh-CN-XiaoyiNeural`）。[AssemblyAI 音色与语言](https://www.assemblyai.com/docs/tts/voices) · [Edge TTS](https://github.com/rany2/edge-tts)

例如，`<speak:肩膀放松，Now look at the camera.>` 在中文逗号处就开始合成第一段，结尾的英文按顺序朗读。整段没有停顿符时，`>` 会提交尾段。纯数字倒数沿用上一段音色。

外放时暂停接收麦克风指令，播完自动恢复。语音服务不可用时仍可查看终端回复。在 `settings.json` 中设置 `"tts": false` 可默认静音，也可用 `python realtime_av_test.py --no-tts` 或 `.\scripts\start_client.ps1 -NoTts` 临时静音。

## 麦克风听写

正常启动 `python realtime_av_test.py` 或 `.\scripts\start_client.ps1` 就会启用麦克风。若本机配置中保留了 `"microphone": false`，可以改成 `true`，也可用 `--microphone` 或 PowerShell 的 `-Microphone` 开启本次听写。启用听写或朗读时，程序从环境变量 `ASSEMBLYAI_API_KEY` 或本机 `settings.json` 的 `assemblyai_api_key` 字段读取密钥。两处都为空时，在交互终端提示处输入密钥，输入内容隐藏，仅供本次运行使用。

客户端通过 sounddevice 采集麦克风音频并实时发送给 AssemblyAI。看到 `[mic]` 就绪提示后，用普通话说“现在拍一张”，或直接用英语说“Take a photo”。服务检测语音活动和回合结束，客户端只将完整转写提交一次，与键盘输入共用队列。DeepSeek 收到的是转写文字和当前画面。[AssemblyAI 多语种听写](https://www.assemblyai.com/docs/streaming/multilingual-transcription) · [回合检测](https://www.assemblyai.com/docs/streaming/turn-detection)

如果说话后终端一直没有转写，先看 `[mic]` 显示的设备名，再用运行客户端的 Python 执行 `python -m sounddevice`，查看有输入通道的设备编号。比如外接麦克风编号为 3，就用 `python realtime_av_test.py --microphone-device 3`，或 `.\scripts\start_client.ps1 -MicrophoneDevice 3` 启动。可以用 `settings.json` 的 `microphone_device` 保存选择。还应检查 Windows「设置 → 隐私和安全性 → 麦克风」是否允许桌面应用访问，以及输入设备是否静音。只用键盘和字幕时可运行 `.\scripts\start_client.ps1 -NoMicrophone -NoTts`。

## 选择摄像头

在 Windows 上，可把 `settings.json` 中的 `camera_name` 设为外接摄像头在设备管理器中显示的完整名称。本机的外接设备名为 `2K USB Camera`。客户端先用 DirectShow 打开该相机，失败后再用 Media Foundation 尝试同一台设备。

临时指定编号时，可运行 `python realtime_av_test.py --camera-device 0`，或使用 `.\scripts\start_client.ps1 -CameraDevice 0`。把 `0` 换成目标摄像头的 OpenCV 编号。

要为今后的 ▶ 启动保存编号，在 PowerShell 执行以下命令，然后完全退出并重新打开 VS Code：

```powershell
$env:HUM_CAMERA_DEVICE = '0'
[Environment]::SetEnvironmentVariable('HUM_CAMERA_DEVICE', $env:HUM_CAMERA_DEVICE, 'User')
```

选择顺序为命令行编号、环境变量 `HUM_CAMERA_DEVICE`、Windows 配置项 `camera_name`。这些选项均未设置时，默认编号为 `701`。如果之前保存过环境变量并希望改为按名称选择，请清除当前用户的 `HUM_CAMERA_DEVICE`，重新打开 VS Code。
