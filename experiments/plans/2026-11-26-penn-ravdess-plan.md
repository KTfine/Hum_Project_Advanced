# Penn Action + RAVDESS Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the invalid HumanML3D/BoLD pilot with a reproducible, no-training, automatically scored Penn Action and RAVDESS evaluation.

**Architecture:** Dataset adapters create frozen manifests below the external D-drive cache. Model adapters emit the existing normalized JSONL record schema, while task-specific scorers consume complete manifests and penalize every failed sample. A CLI orchestrates resumable inference, evidence hashing, reports, and plots.

**Tech Stack:** Python 3.10, PyTorch/CUDA, Transformers X-CLIP, Ultralytics, HSEmotion, OpenCV/Pillow, scikit-learn, DeepSeek API.

**Spec:** `experiments/plans/2026-11-26-penn-ravdess-design.md`

## Global Constraints

- No training, manual judging, LLM judging, or post-result sample removal.
- Use exactly 60 Penn Action and 56 RAVDESS samples unless a pre-inference compatibility check records why a class is impossible.
- Run on RTX 4060 Laptop 8GB with batch size one where necessary.
- Every model failure remains in the denominator; blocked upstream data produces no fabricated score.
- Preserve unrelated workspace changes, especially the pre-existing `.gitignore` modification.

## Review Focus

- Interrupted downloads must resume without accepting partial archives.
- Filename/annotation label parsing must reject unknown classes before inference.
- Missing Penn joints must not become zero-coordinate valid joints.
- A resumed run must reject records from another model or manifest.
- X-CLIP Penn text labels and HSEmotion/RAVDESS mappings must be frozen and written to run evidence before scores exist.

---

### Task 1: Remove obsolete experiment evidence safely

**Files:**
- Delete: `experiments/records/2026-11-25-human-understanding-pilot.md`
- Delete: `experiments/records/2026-11-25-pose-model-ladder.md`
- Modify: `experiments/records/README.md`, `CHANGELOG.md`, `evaluation/RUN_STATUS.md`

- [ ] Resolve and validate each external cache target below `D:\AI_cache\hum_project_evaluation`.
- [ ] Delete only old HumanML3D/BoLD runs, datasets, and the obsolete BoLD checkpoint.
- [ ] Remove obsolete repository records and references with `apply_patch`.
- [ ] Run `git diff --check` and verify retained YOLO weights still exist.

### Task 2: Penn Action acquisition and frozen subset

**Files:**
- Create: `evaluation/data/penn_action.py`
- Test: `evaluation/tests/test_penn_action.py`
- Modify: `evaluation/cli.py`

**Interfaces:**
- Produces: `prepare_penn_action(root: Path, count_per_class: int, seed: int) -> Path` returning a frozen manifest.

- [ ] Write failing tests for archive integrity, 15-class stratification, deterministic IDs, joint visibility, and partial-download rejection.
- [ ] Run focused tests and confirm failure.
- [ ] Implement resumable official download, safe extraction, annotation parsing, and deterministic 4-per-class sampling.
- [ ] Add `prepare-penn` CLI and manifest evidence hashes.
- [ ] Run focused tests and confirm pass.

### Task 3: Penn activity and pose adapters

**Files:**
- Create: `evaluation/models/xclip.py`, `evaluation/models/penn_deepseek.py`
- Modify: `evaluation/models/pose.py`, `evaluation/cli.py`
- Test: `evaluation/tests/test_penn_models.py`

**Interfaces:**
- Produces: normalized activity probability records and 13-joint frame pose records for each sample/model.

- [ ] Write failing tests for frozen label mapping, frame sampling, output shapes, missing-joint encoding, model naming, and resume mismatch.
- [ ] Run focused tests and confirm failure.
- [ ] Implement DeepSeek Penn prompt/parser, X-CLIP large/base zero-shot inference, and Penn-compatible YOLO conversion.
- [ ] Add resumable `run-penn` CLI with batch-size-one model unloading.
- [ ] Run focused tests and confirm pass.

### Task 4: RAVDESS acquisition and emotion adapters

**Files:**
- Create: `evaluation/data/ravdess.py`, `evaluation/models/emotion.py`
- Modify: `evaluation/cli.py`
- Test: `evaluation/tests/test_ravdess.py`, `evaluation/tests/test_emotion_models.py`

**Interfaces:**
- Produces: a deterministic seven-class 56-video manifest and normalized seven-class probability records.

- [ ] Write failing tests for filename parsing, calm exclusion, 8-per-class stratification, contact-sheet sampling, label order, and failed-face detection.
- [ ] Run focused tests and confirm failure.
- [ ] Implement official Zenodo download/extraction and frozen manifest creation.
- [ ] Implement DeepSeek seven-class parser and HSEmotion EfficientNet inference with frame-probability aggregation.
- [ ] Add resumable `prepare-ravdess` and `run-ravdess` CLI commands.
- [ ] Run focused tests and confirm pass.

### Task 5: Fair automatic scoring and reports

**Files:**
- Create: `evaluation/metrics/classification.py`
- Modify: `evaluation/pipeline.py`, `evaluation/report.py`, `evaluation/visualize.py`, `evaluation/cli.py`
- Test: `evaluation/tests/test_classification_metrics.py`, `evaluation/tests/test_formal_reports.py`

**Interfaces:**
- Consumes: frozen manifests and normalized records from Tasks 2–4.
- Produces: task metrics, paired bootstrap comparisons, confusion matrices, Markdown reports, and plots.

- [ ] Write failing tests for full-denominator accuracy/F1/MRR/NLL, bootstrap pairing, label-map freezing, and all-failure behavior.
- [ ] Run focused tests and confirm failure.
- [ ] Implement shared classification scoring plus Penn/RAVDESS report generation.
- [ ] Add `score-penn` and `score-ravdess` CLI commands with source hashes.
- [ ] Run focused tests and confirm pass.

### Task 6: Execute formal experiments and publish evidence

**Files:**
- Create: `experiments/records/2026-11-26-human-understanding-formal.md`
- Modify: `experiments/records/README.md`, `CHANGELOG.md`, `evaluation/RUN_STATUS.md`, `evaluation/README.md`

- [ ] Download and freeze Penn Action 60 and RAVDESS 56 manifests before inference.
- [ ] Run DeepSeek and all compatible specialist tiers, preserving raw per-sample output.
- [ ] Score both datasets and inspect only for pipeline errors, never for sample removal.
- [ ] Record exact metrics, hashes, versions, timing, failures, and claim boundaries.
- [ ] Run the full evaluation and main-project test suites plus `git diff --check`.
