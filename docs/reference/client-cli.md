# 实时客户端参数

在 VS Code 中打开 `realtime_av_test.py` 并点右上角 ▶，或在项目目录运行 `python realtime_av_test.py`。客户端先读取脚本同目录的 `settings.json` 中的 `api_key`，没有配置密钥时读取环境变量 `DEEPSEEK_API_KEY`。两者都为空时，交互终端会提示输入 Key。终端输入拍摄目标后按 Enter，程序会直接向 DeepSeek API 发出图文请求，并在同一终端显示回复。

| 选项 | 默认值与用途 |
|---|---|
| `--camera-device` | OpenCV 摄像头编号，选择顺序见下文 |
| `--no-camera` | 关闭摄像头采集，指导模式使用空白帧 |
| `--no-preview` | 隐藏预览窗口 |
| `--sprite-catalog` | 指定带五部位标签的 JSON 立绘库。实时客户端默认使用 `models/avatars/white_dragon/catalog.json`。[立绘目录](../../models/avatars/README.md) |
| `--canvas-width` / `--canvas-height` | 没有可用摄像头帧时的画布尺寸，默认 1280 × 720；有摄像头画面时沿用其分辨率 |
| `--jpeg-quality` | 发给模型的 JPEG 质量，默认 80，范围 1–100 |
| `--character-name` / `--character-persona` | 自定义合照角色的名字和设定 |
| `--prompt` | 使用完整自定义提示词，停用内置模板自动切换 |
| `--log-file` | 指定 JSONL 日志，默认存入项目内的 `experiments/runtime/` |
| `--verbose` | 在终端额外显示静默观察的 `[blank]` 和 API 元数据 |

使用 `python realtime_av_test.py --help` 查看当前版本完整参数。在预览窗口按 Q 或在终端按 Ctrl+C 退出。

终端把同一回复的说话片段接在一行 `[speak]` 后面。拍照结果和模式切换各占一行，静默观察默认不刷屏。查看模型实际返回的标签和分块时，可在 JSONL 中查找 `api.receive`。

客户端默认使用 `deepseek-flash` 和 `low` 思考强度。可在运行前用 `DEEPSEEK_MODEL` 选择支持图像输入的模型，用 `DEEPSEEK_THINKING` 设置思考强度。程序进入指导模式后，在终端提交第一条文字以发起请求。

在 PowerShell 中先执行 `$env:DEEPSEEK_THINKING = 'off'`，再从同一终端启动客户端，即会使用无思考模式。

在终端输入“现在拍一张”。模型返回 `<capture>` 后，客户端会将照片保存到脚本同目录的 `photos/` 文件夹。指导模式保存摄像头原图，合照模式会按快门时的构图叠加角色。两种模式都使用相机帧的尺寸保存 PNG。使用 `--no-camera` 或尚未取得相机帧时会跳过拍摄。

## 选择摄像头

在 Windows 上，可把 `settings.json` 中的 `camera_name` 设为外接摄像头在设备管理器中显示的完整名称。本机的外接设备名为 `2K USB Camera`。客户端先用 DirectShow 打开该相机，失败后再用 Media Foundation 尝试同一台设备。

临时指定编号时，可运行 `python realtime_av_test.py --camera-device 0`，或使用 `.\scripts\start_client.ps1 -CameraDevice 0`。把 `0` 换成目标摄像头的 OpenCV 编号。

要为今后的 ▶ 启动保存编号，在 PowerShell 执行以下命令，然后完全退出并重新打开 VS Code：

```powershell
$env:HUM_CAMERA_DEVICE = '0'
[Environment]::SetEnvironmentVariable('HUM_CAMERA_DEVICE', $env:HUM_CAMERA_DEVICE, 'User')
```

选择顺序为命令行编号、环境变量 `HUM_CAMERA_DEVICE`、Windows 配置项 `camera_name`。这些选项均未设置时，默认编号为 `701`。如果之前保存过环境变量并希望改为按名称选择，请清除当前用户的 `HUM_CAMERA_DEVICE`，重新打开 VS Code。
