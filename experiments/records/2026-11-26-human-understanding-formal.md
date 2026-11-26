# DeepSeek 与专用模型的人体理解正式对照

日期：2026-11-26（Asia/Shanghai）

## 结论

在不训练、不作人工判定、失败样本不移出分母的条件下，DeepSeek Flash 能够单独完成 activity、pose 和 emotion 三项任务。它在本次冻结样本上的 activity 明显高于 X-CLIP 两档，在 emotion 上点估计高于 HSEmotion 两档；pose 则低于 YOLO11-pose 的全部 x/l/m/s/n 档，缩到最小的 n 档后仍未出现交叉。

因此本次证据支持的 claim 是：DeepSeek 可以用同一个通用模型完成三类任务并取得可量化的结果；它不必、也没有在每项任务上都超过专用模型。

## 固定条件

- Penn Action：60 个自然视频，15 类 activity 各 4 个；每个视频固定抽取 8 帧，同时评估 activity 与 13 关节 pose。
- RAVDESS video-only speech：56 个视频，7 类 emotion 各 8 个；在推理前排除专用模型没有独立输出的 `calm`。
- 随机种子：3800。
- 所有模型使用同一冻结 manifest；不训练、不微调、不人工修正标签或预测。
- 失败记录仍在完整分母中计错；正式结果的所有模型均为 0 failure。
- 模型阶梯：activity 为 X-CLIP large → base；pose 为 YOLO11-pose x → l → m → s → n；emotion 为 HSEmotion EfficientNet-B2 → B0。
- 主要指标：activity/emotion 使用 accuracy；pose 使用 PCK@0.10。另报 Macro-F1、PCK@0.20、NME、MRR、NLL 等自动指标。
- 置信区间为按样本配对 bootstrap，10,000 次，seed 3800。

## 结果

### Activity（Penn Action，60 条）

| 模型 | Accuracy | Macro-F1 | MRR | NLL | Failure |
|---|---:|---:|---:|---:|---:|
| DeepSeek Flash | **0.9667** | **0.9661** | **0.9783** | **0.2194** | 0 |
| X-CLIP large-patch14 | 0.7833 | 0.7363 | 0.8861 | 0.6959 | 0 |
| X-CLIP base-patch32 | 0.7333 | 0.6908 | 0.8569 | 0.6925 | 0 |

X-CLIP large − DeepSeek 的 accuracy 差为 −0.1833，95% CI [−0.3000, −0.0833]；base − DeepSeek 为 −0.2333，95% CI [−0.3500, −0.1167]。DeepSeek 在第一档 large 已达到/超过专用模型；base 也已执行以保留完整曲线。

### Pose（Penn Action，60 条、5,004 个可见关节）

| 模型 | PCK@0.10 | PCK@0.20 | NME | Missing rate | Failure |
|---|---:|---:|---:|---:|---:|
| DeepSeek Flash | 0.2858 | 0.6019 | **0.2036** | **0.0000** | 0 |
| YOLO11x-pose | 0.3815 | **0.6896** | 0.2210 | 0.0268 | 0 |
| YOLO11l-pose | 0.3735 | 0.6785 | 0.2308 | 0.0290 | 0 |
| YOLO11m-pose | **0.3895** | 0.6892 | 0.2231 | 0.0256 | 0 |
| YOLO11s-pose | 0.3837 | 0.6671 | 0.2396 | 0.0186 | 0 |
| YOLO11n-pose | 0.3667 | 0.6459 | 0.2543 | 0.0174 | 0 |

最小的 YOLO11n-pose 仍比 DeepSeek 高 0.0907 PCK@0.10，95% CI [0.0251, 0.1544]，因此阶梯在 n 档耗尽。DeepSeek 的 PCK@0.20 为 0.6019、missing rate 为 0，说明它确实产生了完整且可评分的 pose，但精确定位不如专用模型。

Penn 标签的左右定义没有按结果调换。自动审计显示正面与背面视频的左右关节横坐标关系反转，符合人体解剖学左右；因此沿用 Penn 与 COCO 的同名左右映射。

### Emotion（RAVDESS，56 条）

| 模型 | Accuracy | Macro-F1 | Balanced accuracy | MRR | NLL | Failure |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek Flash | **0.6071** | **0.6113** | **0.6071** | **0.7459** | 1.2416 | 0 |
| HSEmotion enet-b2-8 | 0.5536 | 0.5014 | 0.5536 | 0.7396 | 1.2116 | 0 |
| HSEmotion enet-b0-8-afew | 0.5536 | 0.5117 | 0.5536 | 0.7408 | **1.1133** | 0 |

B2 − DeepSeek 的 accuracy 差为 −0.0536，95% CI [−0.1964, 0.0714]；B0 − DeepSeek 也是 −0.0536，95% CI [−0.1964, 0.1071]。点估计上 DeepSeek 在第一档 B2 已达到/超过专用模型，但区间跨 0，不能据此声称统计显著优于 HSEmotion。

首轮 emotion 运行曾有 26 条在本地末帧读取阶段失败，根因是容器声明帧数大于实际可解码帧数；这些请求尚未发送给模型。修复为顺序解码全部真实帧并统一等距抽样后，已删除无效首轮目录并从零重跑，最终三模型均为 56/56 有效。

## 证据与存储

- Penn 正式运行：`D:\AI_cache\hum_project_evaluation\runs\20261126-penn-formal`
- RAVDESS 正式运行：`D:\AI_cache\hum_project_evaluation\runs\20261126-ravdess-formal`
- Penn `metrics.json` SHA-256：`CE077F09FA59790FB897B613B098D7D18826AD8959BF220F4CC83267620701F6`
- RAVDESS `metrics.json` SHA-256：`DDA63ADD46658BDABC8EC4062D69F2F82AEB4C55272B22AA77CA33681C80F5B5`
- Penn 数据：6,695,999,375 bytes（6.236 GiB）；RAVDESS 数据：2,266,976,494 bytes（2.111 GiB）。
- 模型缓存：3,492,446,032 bytes（3.253 GiB）；两次正式运行输出合计约 39.2 MiB。

每个运行目录都包含逐样本 JSONL、`metrics.json`、`report.md` 和记录 manifest/预测哈希与环境版本的 `run_config.json`。数据与模型来源、复现命令见 [evaluation README](../../evaluation/README.md)。

## 验证

- `D:\Conda_Envs\aiaa2205\python.exe -m pytest evaluation\tests -q`
- 结果：82 passed in 4.03s。
- 正式 JSONL 完整性检查：Penn 8 个模型均为 60/60 `ok`；RAVDESS 3 个模型均为 56/56 `ok`，样本 ID 与各自 manifest 完全一致。
- `git diff --check`：通过。
- 仓库文档指定的 `.venv-client` 在当前工作区不存在；没有用缺少依赖或 Python 版本不符的其他环境结果冒充主项目全套测试通过。
