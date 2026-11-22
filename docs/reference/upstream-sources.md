# 来源与基线

本仓库首次导入基线来源于用户指定的 `DDRP674/AIAA3800`，不是与该仓库共享 Git 历史的 fork。后续修改以本仓库提交和 `CHANGELOG.md` 为准。

| 内容 | 来源及固定提交 |
|---|---|
| 场景实例分割 | YOLO11s-seg，采用 [unileon-robotics 固定版本 ONNX 导出](https://huggingface.co/unileon-robotics/YOLO11-ONNX/tree/0e2a3f195ae62a25bc038ee9665b8e4ac5b98793)，使用图像和原型掩膜两个输出。 |
| 场景相对逆深度 | [MiDaS v2.1 官方发布](https://github.com/isl-org/MiDaS/releases/tag/v2_1) 的 model-small.onnx。两份场景权重的长度、哈希与安装入口见 [scene_objects.py](../../photo_assistant/vision/scene_objects.py)。 |
| 小白龙立绘 | 随库 PNG、表情和姿势标签见[立绘目录](../../models/avatars/README.md)。 |
| 初始客户端、协议解析器、提示词及测试 | [AIAA3800 main](https://github.com/DDRP674/AIAA3800/tree/177d70afd6d8e6d95d619621c61117cef8cd59cd)，`177d70afd6d8e6d95d619621c61117cef8cd59cd` |
| 历史部署配置、脚本和客户端工具 | [AIAA3800 backend](https://github.com/DDRP674/AIAA3800/tree/2accaaff4e616caac397037e41fbcf9324339d63)，`2accaaff4e616caac397037e41fbcf9324339d63` |
| 历史实验使用的外部 Demo | [OpenBMB/MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/47709a9210dfd71afa76c058e017fc8c4db5c8d2)，`47709a9210dfd71afa76c058e017fc8c4db5c8d2` |
| 历史实验使用的模型权重 | [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/503e754207c94da6bb26850b4469f367c9ea3582)，revision `503e754207c94da6bb26850b4469f367c9ea3582` |
| 单人合照前景分割 | [OpenCV Zoo 的 PPHumanSeg，4.10.0 标签](https://github.com/opencv/opencv_zoo/tree/4.10.0/models/human_segmentation_pphumanseg)，[Apache 2.0 许可证](https://github.com/opencv/opencv_zoo/blob/4.10.0/models/human_segmentation_pphumanseg/LICENSE)。使用 2023mar 浮点 ONNX，SHA-256 为 `552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24`。 |

上述是来源定位，不代表当前源码仍与导入版本逐字一致。2026-11-07 起的运行与实验基线见 [模式说明](../modes-and-prompts.md) 和 [实验索引](../../experiments/records/README.md)；当前客户端直接调用 DeepSeek API。

导入时排除了上游历史会话。
