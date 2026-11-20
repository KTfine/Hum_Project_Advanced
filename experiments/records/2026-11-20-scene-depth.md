# 平视室内桌椅遮挡与合照脚点投影

2026-11-20（Asia/Shanghai），在父提交 `a9fafa9` 后的本机工作区执行。改动涉及场景分割、逆深度、地面投影、合照合成、随帧提示及测试。主验收输入为参照用户原图生成的水平机位照片，保留原俯拍实景及历史人像作为对照。图片和逐次结果保存在本机忽略目录 `experiments/runtime/`。本轮使用本地模型和模拟对话传输，远程模型调用次数为零。

## 相机回放

使用 ImageGen 生成平视空景。原图包含左侧书柜、右侧玻璃隔断及白色拼桌。生成时给出的指令如下。

> Based on the supplied room photo, make a new photorealistic 4:3 landscape webcam test exposure of this same university study room. Move the camera back and lower it to ordinary seated-to-chest height, 1.2 meters, lens aimed perfectly horizontally: zero pitch, level horizon and room vanishing point at the exact center height of the picture. Straight verticals. Show the front edge, whole legs and floor contacts of the nearest white study table, a little of its top. A second white table clearly separate farther back, a few ordinary white chairs and one dark backpack on a table. Wooden and blue shelving on the left, glass partition with navy stripe on the right, ordinary even fluorescent light as in the reference. Leave an aisle with visible level floor alongside the two tables, including foreground floor. Realistic candid camera image for testing camera geometry. No people or characters, labels, diagrams, or text overlays.

以下命令由应用根目录运行，图片按比例缩放到 640 × 480。每个位置都通过实时客户端的传送、快门画面和 JPEG 入口。另渲染一次相同脚点、相同尺寸的最前层立绘，统计有多少角色像素因实景遮挡而改变。

```powershell
python -m experiments.scripts.scene_replay experiments/runtime/level-room-20261120/level-room.png experiments/runtime/level-room-20261120 --x .53 --far-y .797 --near-y .98
python -m experiments.scripts.scene_replay photos/photo-20261118-172627-827393-nqbamo01.png experiments/runtime/table-depth-20261120
```

| 平视图落脚位置 | 等效站高 | 遮挡影响的角色像素 | 立绘占据的像素 |
|---|---:|---:|---:|
| 前桌后方，两组桌子之间 | 212.21 px | 7,891 | 18,954 |
| 前桌前方 | 317.09 px | 0 | 42,146 |
| 回到最初脚点 | 212.21 px | 7,891 | 18,954 |

平视图检测为前后两个桌组和五把椅子。两个桌组掩膜内中位相对逆深度约为 0.780 与 0.555，数值越大表示越近。两次回到远处的成片逐像素一致。原俯拍实景将紧贴的桌面合为一个桌组，检出三把椅子、两个包状轮廓，其中后桌衣物也被归为包。其远、近站位等效高度分别为 173.04 和 368.74 px。平视图远桌上的小背包漏检，细桌腿局部缺失。

最初的 nano 权重在俯拍图漏掉了左前桌面，且重复报告同一个包。另对比 small 和 medium 导出，采用 small，补全同一桌面内的颜色连通区域并去除重复实例。第一版平视回放中人体分割误把物件当成了真人，导致近景站高仅约 96 px；现用同帧人物框与分割轮廓交叉验证。正式结果采用下载器核对过哈希的 YOLO11s-seg 和 MiDaS-small。

## 体型与对话验证

Windows 正常运行权限下，最终全量回归 162 项，156 项通过。4 项肩侧几何和 2 项旧入场预期在原版副本也未通过。新场景推理、地面、合成和模拟 API 检查通过。两次全量日志分别保存于 `experiments/runtime/scene-final-regression.log` 与 `experiments/runtime/scene-unittest-base.log`。沙箱内首次尝试在 Windows asyncio 初始化回环套接字时挂起。

同机位的白龙桌沿坐姿图与图库立绘匹配得到约 0.22413 的投影倍率，等效站高约 319.4 px，脚点约 (380, 472)。坐姿的鞋底可能离地，所以这组数只用于无人可供定尺时估算。真人入场优先沿用项目实测的脚点与投影尺寸。历史实拍近景也使用图库特征匹配复核。单目估计给出同帧远近关系，无法从这些照片独立测量真实身高或米制距离。

用原项目全身、半身和头肩三张底图回放。新旧流程入场等效站高分别同为约 284、306 和 718 px。角色按原入场观感定尺，传送以脚点相对于地平线的位置推算新高度。稳定预览、角色换姿和快门共用该次落点。

检查了原说话规则、动作时机、自动观察及模式切换；合照新增每帧场景列表、建议脚点和站高数据。模拟 API 收到同类多个实例，并能在新帧中清除旧实例。默认白龙和相机图层组装出的合照系统文本 SHA-256 为 `5b49647cd8b3c0eca29346f6d849cc7a16b20b099b2d2530fad0a4eec5c5479b`，指导为 `59b4493a3b477174bf4377328b33f2c1c728f7ca5f296cbd44f309679c3b0bff`。本次没有真人现场移动试验或远程模型交互效果统计。
