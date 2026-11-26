# Penn Action + RAVDESS 正式评测设计

## 目标

在不训练、不使用人工判分、RTX 4060 Laptop 8GB 可运行的约束下，用 20–100 条公开可自动下载数据，比较 DeepSeek API 与专用模型在 activity、pose、emotion 三项任务上的表现。主张限定为：单一通用 API 能完成三项任务并取得可量化的非平凡结果；不预设 DeepSeek 必须超过专用模型。

## 数据与输入公平性

- Activity + pose：Penn Action，自然人体视频同时提供 15 类动作标签和逐帧 13 点姿态 GT。固定分层抽取 60 条，每类 4 条。
- Emotion：RAVDESS video-only speech，排除 calm，固定抽取 neutral/happy/sad/angry/fear/disgust/surprise 各 8 条，共 56 条。
- 同一任务内 DeepSeek 和专用模型读取同一源视频。DeepSeek 接收均匀抽取的 8 帧；专用模型按官方预处理读取相同视频，不允许额外人工裁剪或筛样。
- 抽样、标签映射和失败惩罚在推理前冻结。任何模型失败都进入完整分母。

## 模型与阶梯

- Activity：DeepSeek；X-CLIP large-patch14 → base-patch32 的公开权重。X-CLIP 直接对冻结的 Penn Action 15 个原始文本标签做零样本视频分类，不经过 Kinetics 类别合并。
- Pose：DeepSeek；YOLO11-pose x/l/m/s/n。自然视频消除旧骨架图域偏移。两者都按 Penn Action 可见 13 点计算。
- Emotion：DeepSeek；HSEmotion 已发布的 EfficientNet 系列权重，从较大到较小运行。仅使用与 RAVDESS 七类一致的输出，禁止主观合并 calm。
- 若专用模型分数高于 DeepSeek，沿同系列由大到小继续；为了形成完整曲线，在资源允许时运行全部公开兼容档位。

## 自动指标

- Activity：Top-1 accuracy、Macro-F1、MRR、每类召回率、配对 bootstrap 差值。
- Pose：PCK@0.05/0.10/0.20、NME、missing rate、左右交换率、配对 bootstrap 差值。
- Emotion：Top-1 accuracy、Macro-F1、balanced accuracy、NLL、混淆矩阵、配对 bootstrap 差值。
- 不使用人工复核、LLM judge 或事后剔除失败样本。

## 删除与保留

- 删除旧 HumanML3D/BoLD 原始运行目录、数据子集、BoLD 权重以及两份旧实验记录。
- 从索引、CHANGELOG 和评测状态中删除旧结果引用。
- 保留通用 DeepSeek 客户端、评分/报告基础设施和新实验仍使用的 YOLO 权重。
- 旧结果不进入主表或附录；只在 Git 工作区差异中保留可审计的代码演进，不恢复为结论。

## 交付物

- 可断点续跑的数据下载、抽样、推理、评分 CLI。
- 冻结 manifest、原始 JSONL、模型/数据哈希、环境版本、自动报告与图表。
- 一份明确区分 calibration 限制、模型胜负和 claim 边界的正式实验记录。
