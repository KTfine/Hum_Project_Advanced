# 单人合照遮挡验证

2026-11-18（Asia/Shanghai），基于 `a9fafa9` 的未提交工作区。主要改动在 `realtime_av_test.py`、`person_segmenter.py`、`person_occlusion.py` 及合照提示词。此次验证运行本地推理和合成，不涉及远端模型请求、采样或会话记录。

- **回归**：使用项目 Python 3.12.5 运行 `python -m unittest discover -s tests`，112 项通过。Windows 沙箱内 asyncio 创建回环套接字时阻塞，全套结果取自在沙箱外的离线运行。
- **公开单人照片**：将 scikit-image 的 512×512 NASA 宇航员照片作为摄像头帧，使用项目立绘和 OpenCV Zoo PPHumanSeg 模型。保存的 PNG 和同帧预览逐像素一致，模型输入 JPEG 与直接编码该预览的结果一致。水平翻转原图后立即拍照，结果与新图的遮罩一致，指导模式显示原图。
- **延迟**：本机 CPU、OpenCV 4.14.0、640×480 样片。预热 8 次后测 40 次，从提交原始帧到取出匹配的遮罩，中位 27.08 ms，P95 为 29.27 ms。测量包含工作线程调度，不包含传感器、显示、PNG 编码或网络耗时。
- **范围**：未进行真人运动摄像头评估；细发丝、快速挥手和复杂背景下的轮廓质量仍需现场观察。没有验证模型根据新提示词主动选择站位的表现。

本机同帧对照图保存在忽略目录 `experiments/runtime/person-occlusion-20261118-190134-986336/preview-comparison.png`。
