# Hum_Project 拍照助手

小白龙是一个通过摄像头看画面、在终端里和你对话的拍照助手。客户端直接调用 DeepSeek API，可以指导你拍照，也可以让小白龙入镜合照。

## 当前版本（2026-11-23）

- **拍照指导。** 可以问“我的手往哪儿放”，也可以告诉她“别让我挪位置”。你开口或打字开始对话后，指导和合照模式每 2 秒有一次机会把最新画面发给模型。即使你保持姿势不动，她也能再看一眼，自行决定要不要说话。上一轮还在回复时会跳过这次。编辑角色性格文件并重启，即可调整两个模式里的说话风格。[角色性格.txt](角色性格.txt)
- **双人合照。** 说“过来一起拍”可以邀请她入镜，问“咱俩怎么拍好看”可以一起商量构图。全身照默认并排站，半身照默认从身侧探头。你走开后，默认探身的小白龙会在原地站直。再说“跟过来”，她会按你当时的位置重新站位。也可以让她从肩后探头或原地露齿笑。[肩后转正与跟随实测](experiments/records/2026-11-21-departure-prompt.md)
- **选择坐位。** 说“找个离镜头远的地方坐”，她会从远到近寻找能坐的边沿。想指定物体，可以说“坐到右边那把椅子上”。[坐姿指令](docs/reference/command-protocol.md)
- **前后关系与大小。** 客户端可以区分画面里的多张桌子。地面深度估计可靠时，小白龙以和你并排时的身高为基准，按落脚位置近大远小。平视自习室样图中，真人能挡住她的身体，桌面能挡住她的腰。站到桌后时，白裙仍可能穿过细桌腿。实测使用固定机位的生成图。[实测结果与已知问题](experiments/records/2026-11-21-product-trial.md)
- **扶墙合照。** 让墙柱的竖边和柱脚一起入镜，再对她说“小白龙，去扶一下那根柱子”。她会选靠墙的一只手，把手掌贴到柱边。说“松手站好”就会在原地放下手。[扶墙实测与复跑方法](experiments/records/2026-11-21-wall-brace.md)

## 快速开始

固定相机合照时，摆好机位再让小白龙入镜。程序在入场时识别桌椅和背景深度，之后持续更新真人轮廓。移动相机或家具后，退出合照再进入即可重新识别。手持移动拍摄可用 `--moving-camera`。[摄像头参数](docs/reference/client-cli.md)

首次运行前，打开与 `realtime_av_test.py` 同目录的 `settings.json`。在 `api_key` 的引号内填入自己的 DeepSeek API Key，按 Ctrl+S 保存。此后每次点 ▶，程序都会自动读取该文件。Git 忽略本机的 `settings.json`。新检出的仓库可以复制 [settings.example.json](settings.example.json) 为 `settings.json` 后填写。

```json
{
  "api_key": "",
  "assemblyai_api_key": "",
  "camera_name": "2K USB Camera",
  "microphone": true,
  "tts": true
}
```

没有配置文件或 `api_key` 为空时，客户端读取环境变量 `DEEPSEEK_API_KEY`。两者都未设置时，实时客户端才会在交互终端提示输入。

麦克风听写和语音朗读默认开启。启动后按终端提示输入 **AssemblyAI API Key**，输入时隐藏，仅供本次运行使用。也可以提前设置环境变量 `ASSEMBLYAI_API_KEY` 或填写本机的 `assemblyai_api_key`。如果只用键盘和字幕，可加 `--no-microphone --no-tts`。[麦克风与密钥设置](docs/reference/client-cli.md#麦克风听写)

小白龙按中英文逗号、句号等停顿分段朗读。收到“肩膀放松，”时就开始合成这半句，后面的指导可以继续生成。英文段使用 AssemblyAI，中文段使用 Edge 晓伊（`zh-CN-XiaoyiNeural`）。[语音播放设置](docs/reference/client-cli.md#小白龙的声音)

从新检出的仓库启动时，在仓库目录运行以下命令，安装依赖并下载人体分割、物体分割和相对深度模型。

```powershell
python -m venv .venv-client
.\.venv-client\Scripts\python.exe -m pip install -r requirements-client.txt
.\.venv-client\Scripts\python.exe -m pip install -r requirements-voice.txt
.\.venv-client\Scripts\python.exe -m photo_assistant.vision.segmentation
.\.venv-client\Scripts\python.exe -m photo_assistant.vision.scene_objects
.\.venv-client\Scripts\python.exe realtime_av_test.py
```

也可以在 VS Code 中选择刚建好的 Python 解释器，打开 `realtime_av_test.py`，点右上角 **Run Python File ▶**。客户端会加载小白龙的立绘。[查看表情和姿势](models/avatars/README.md)

在终端输入“我想拍一张显得有活力的照片”并按 Enter，模型会结合当前画面回复。想进入合照可输入“和我一起合照吧”，返回时输入“继续指导我摆姿势”。客户端收到模型生成的模式标签后切换提示词和预览。在预览窗口按 Q 或在终端按 Ctrl+C 退出。

看到 `[mic]` 就绪提示后，说“现在拍一张”并稍作停顿。AssemblyAI 判定你说完后，助手会显示转写并自动提交。若收不到声音，可用 `python -m sounddevice` 查看输入设备，启动时加 `--microphone-device 设备编号` 选择实际使用的麦克风。[麦克风设置](docs/reference/client-cli.md#麦克风听写)

想拍半身合照，可以在合照模式输入“靠近镜头一点，和我拍张半身照”。小白龙会根据新的鞋底落点调整画面中的大小。脚点可以落在画面下方，让下半身自然出框。想让她回到你身旁，说“跟过来”即可。预览满意后，输入“现在帮我拍一张”并按 Enter。程序将照片保存为项目 `photos/` 目录下的 PNG。指导模式保存摄像头原图，合照模式将实景与角色一起保存。入场和换位光效仅在预览中显示。

Windows 会按 `settings.json` 中的 `camera_name` 选择相机。本机外接摄像头名为 `2K USB Camera`，使用其他摄像头时填写它在 Windows 中显示的完整名称。也可以用 `--camera-device 0` 临时指定编号。详见[摄像头设置](docs/reference/client-cli.md#选择摄像头)。

| 我想做什么 | 入口 |
|---|---|
| 调整启动参数，查找运行日志 | [客户端参数](docs/reference/client-cli.md) |
| 了解 API 调用、模型输入及流式指令 | [API 接口](docs/api.md) |
| 调整指导和合照提示词 | [模式说明](docs/modes-and-prompts.md) |
| 用指定图片做实验，查看历史记录 | [实验工作区](experiments/README.md) |
| 查看素材来源与项目变更 | [来源](docs/reference/upstream-sources.md) · [变更记录](CHANGELOG.md) |

## 项目目录

| 位置 | 内容 |
|---|---|
| [photo_assistant/](photo_assistant/) | 实时客户端、拍照提示词与流式指令协议 |
| [photo_assistant/avatar/](photo_assistant/avatar/) | 立绘标签、PNG 读取与角色状态 |
| [photo_assistant/vision/](photo_assistant/vision/) | 摄像头采集、人体分割与前景遮挡 |
| [photo_assistant/api/](photo_assistant/api/) | DeepSeek 请求与实时会话 |
| [models/](models/) | 立绘库、素材说明及本地人体分割权重 |
| [tests/](tests/) · [experiments/](experiments/) | 自动化回归、独立图文实验与验证记录 |
