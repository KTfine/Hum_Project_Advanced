# Evaluation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep only the reproducible Penn Action and RAVDESS formal evaluation path while preserving all generated images and external cached data.

**Architecture:** Reduce `evaluation/` to its formal dataset preparation, inference, automatic scoring, and tests. Remove the retired HumanML3D/BoLD caption-retrieval pipeline and generated Python caches, then update the CLI and documentation to expose only supported commands.

**Tech Stack:** Python 3.10, pytest, PyTorch/Transformers, Ultralytics, ONNX Runtime.

**Spec:** User request in the 2026-11-26 session and repository maintenance rules.

## Global Constraints

- Preserve `experiments/examples/2026-11-26-human-understanding/`.
- Preserve everything below `D:\AI_cache\hum_project_evaluation`.
- Preserve the Penn Action and RAVDESS formal runs and their reproducibility path.
- Do not modify runtime materials or historical evidence.

## Review Focus

- `python -m evaluation --help` lists only supported formal commands.
- Penn Action and RAVDESS preparation commands remain parseable.
- Formal run commands still dispatch through `evaluation.formal`.
- The example exporter can still import `evaluation.video`.
- No retained module imports a deleted legacy module.

---

### Task 1: Freeze the formal-only CLI

**Files:**
- Modify: `evaluation/tests/test_cli.py`
- Modify: `evaluation/cli.py`

- [x] Add a parser test asserting the supported command set.
- [x] Run it and observe failure while legacy commands remain.
- [x] Remove HumanML3D/BoLD commands and imports from the CLI.
- [x] Run the parser and formal evaluation tests.

### Task 2: Remove retired modules and generated caches

**Files:**
- Delete: retired HumanML3D/BoLD data, model, rendering, pipeline, reporting, and visualization modules and their tests.
- Delete: all `evaluation/**/__pycache__/` directories.
- Modify: `evaluation/requirements.txt`, `evaluation/README.md`, `evaluation/RUN_STATUS.md` as needed.

- [x] Verify no retained source imports a retired module.
- [x] Delete the exact retired files and caches.
- [x] Update dependency and usage documentation.
- [x] Run all retained evaluation and export tests.

### Task 3: Verify preserved artifacts

**Files:**
- No data or image changes.

- [x] Confirm the 12 example images and manifest still exist.
- [x] Confirm both frozen formal metrics files remain in the external cache.
- [x] Run `git diff --check` and report deleted files and retained artifacts.
