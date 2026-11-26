# Automatic human-understanding evaluation

This package contains only the current no-training, no-human-judgment comparison
between the project's DeepSeek API client and frozen specialist models. Every
failed sample remains in the denominator. The retired HumanML3D/BoLD pilot code
has been removed; its results are not part of the deliverable.

## Formal datasets

- Penn Action: 60 natural videos, stratified as four samples from each of 15
  activity classes. The same eight uniformly sampled frames are used for
  activity classification and 13-joint pose evaluation.
- RAVDESS video-only speech: 56 videos, stratified as eight samples from each
  of neutral, happy, sad, angry, fear, disgust, and surprise. `calm` is excluded
  before inference because it has no distinct HSEmotion output.

All datasets, model weights, and raw run outputs live below
`D:\AI_cache\hum_project_evaluation`.

## Frozen formal results (2026-11-26)

All values below are copied from the frozen `metrics.json` files. No failed
sample was removed from a denominator.

### Dataset summary

| Dataset | Tasks | Frozen samples | Sampling |
| --- | --- | ---: | --- |
| Penn Action | Activity, pose | 60 videos | 15 activity classes × 4; 8 uniformly sampled frames per video |
| RAVDESS video-only speech | Emotion | 56 videos | 7 emotion classes × 8; `calm` excluded before inference |
| Frozen command requests | Command-format comparison | 30 paired requests | Each request evaluated once with tags and once with JSON |

### Activity classification — Penn Action

Higher is better for accuracy, Macro-F1, and MRR; lower is better for NLL.

| Model | Accuracy | Macro-F1 | MRR | NLL | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek Flash | **0.9667** | **0.9661** | **0.9783** | **0.2194** | 0 |
| X-CLIP large-patch14 | 0.7833 | 0.7363 | 0.8861 | 0.6959 | 0 |
| X-CLIP base-patch32 | 0.7333 | 0.6908 | 0.8569 | 0.6925 | 0 |

### Pose estimation — Penn Action

Higher is better for PCK; lower is better for NME and missing rate. PCK is
computed over 5,004 visible Penn Action joints.

| Model | PCK@0.05 | PCK@0.10 | PCK@0.20 | NME | Missing rate | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek Flash | 0.0969 | 0.2858 | 0.6019 | **0.2036** | **0.0000** | 0 |
| YOLO11x-pose | 0.1445 | 0.3815 | **0.6896** | 0.2210 | 0.0268 | 0 |
| YOLO11l-pose | 0.1435 | 0.3735 | 0.6785 | 0.2308 | 0.0290 | 0 |
| YOLO11m-pose | **0.1497** | **0.3895** | 0.6892 | 0.2231 | 0.0256 | 0 |
| YOLO11s-pose | 0.1481 | 0.3837 | 0.6671 | 0.2396 | 0.0186 | 0 |
| YOLO11n-pose | 0.1407 | 0.3667 | 0.6459 | 0.2543 | 0.0174 | 0 |

### Emotion classification — RAVDESS

Higher is better for accuracy, Macro-F1, and MRR; lower is better for NLL.

| Model | Accuracy | Macro-F1 | MRR | NLL | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek Flash | **0.6071** | **0.6113** | **0.7459** | 1.2416 | 0 |
| HSEmotion EfficientNet-B2 | 0.5536 | 0.5014 | 0.7396 | 1.2116 | 0 |
| HSEmotion EfficientNet-B0 | 0.5536 | 0.5117 | 0.7408 | **1.1133** | 0 |

### Streaming tags versus JSON

Both formats used the same 30 frozen requests and DeepSeek Flash. Token counts
refer to completion tokens only.

| Format | Mean tokens | Median tokens | Strict parse | Exact semantics | Tokens/correct command |
| --- | ---: | ---: | ---: | ---: | ---: |
| Streaming tags (`<...>`) | **15.90** | **11.00** | 93.33% | 93.33% | **12.89** |
| JSON | 24.67 | 21.00 | **100.00%** | 93.33% | 20.00 |

Among the 27 pairs where both formats were semantically correct, streaming tags
used 14.67 mean completion tokens versus 23.70 for JSON, a 38.13% reduction.
The paired JSON-minus-tags difference was 9.04 tokens with a 95% bootstrap
confidence interval of [8.33, 9.81].

## Commands

Run from the repository root with `D:\Conda_Envs\aiaa2205\python.exe`:

```powershell
python -m evaluation --cache-root D:\AI_cache\hum_project_evaluation doctor
python -m evaluation --cache-root D:\AI_cache\hum_project_evaluation prepare-penn --count-per-class 4 --seed 3800
python -m evaluation --cache-root D:\AI_cache\hum_project_evaluation run-penn --manifest D:\AI_cache\hum_project_evaluation\datasets\penn-action\Penn_Action\manifest.json --run-id 20261126-penn-formal
python -m evaluation --cache-root D:\AI_cache\hum_project_evaluation prepare-ravdess --count-per-class 8 --seed 3800
python -m evaluation --cache-root D:\AI_cache\hum_project_evaluation run-ravdess --manifest D:\AI_cache\hum_project_evaluation\datasets\ravdess\videos\manifest.json --run-id 20261126-ravdess-formal
```

`run-penn` evaluates DeepSeek, X-CLIP large/base, and YOLO11-pose x/l/m/s/n.
`run-ravdess` evaluates DeepSeek and HSEmotion EfficientNet-B2/B0. Runs are
resumable: an existing JSONL is accepted only when its model and sample IDs
match the frozen manifest.

## Automatic metrics

- Activity: accuracy, Macro-F1, balanced accuracy, MRR, NLL, per-class recall,
  confusion matrix, and paired bootstrap accuracy differences.
- Pose: PCK@0.05/0.10/0.20, NME, missing rate, left/right swap rate, and paired
  bootstrap PCK@0.10 differences.
- Emotion: accuracy, Macro-F1, balanced accuracy, MRR, NLL, confusion matrix,
  and paired bootstrap accuracy differences.

Each run contains raw JSONL predictions, `metrics.json`, `report.md`, and a
`run_config.json` with manifest/prediction hashes and environment versions.

Generated datasets, model weights, raw predictions, and metrics remain under
`D:\AI_cache\hum_project_evaluation`; presentation examples remain under
`experiments/examples/2026-11-26-human-understanding`.

The completed 2026-11-26 results and bounded conclusion are recorded in
[`experiments/records/2026-11-26-human-understanding-formal.md`](../experiments/records/2026-11-26-human-understanding-formal.md).
