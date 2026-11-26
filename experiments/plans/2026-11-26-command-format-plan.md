# Command Format Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a reproducible DeepSeek comparison of the existing angle-bracket protocol and an equivalent JSON protocol, then preserve automatically selected prior-experiment example images.

**Architecture:** A focused experiment module owns frozen scenarios, strict parsers, paired metrics, API execution, and reporting. Existing project parsers and API client are reused without changing runtime behavior. A separate exporter reads frozen prediction files and creates score-selected contact sheets.

**Tech Stack:** Python, existing DeepSeek client, pytest, Pillow/OpenCV, JSONL.

**Spec:** `experiments/plans/2026-11-26-command-format-design.md`

## Global Constraints

- No human or LLM judging.
- Use API-reported completion tokens.
- Keep all failures in the reliability denominator.
- Do not modify the production command protocol.
- Store experiment material below `experiments/` and large generated runtime material below the ignored external evaluation cache.

## Review Focus

- Reject prose or Markdown around an otherwise valid response.
- Reject missing, extra, duplicated, non-finite, or out-of-range parameters.
- Preserve command order and Unicode speech exactly.
- Do not treat malformed shorter outputs as token wins.
- Resume only when scenario, format, model, and expected semantics match.

---

### Task 1: Strict paired benchmark core

**Files:**
- Create: `experiments/scripts/command_format_comparison.py`
- Test: `tests/test_command_format_comparison.py`

**Interfaces:**
- Produces `scenario_manifest()`, `parse_angle_output()`, `parse_json_output()`, and `score_records()`.

- [ ] Write failing parser, manifest balance, and metric tests.
- [ ] Run the focused tests and verify expected missing-interface failures.
- [ ] Implement the minimum frozen scenarios, strict normalization, and paired metrics.
- [ ] Run focused tests to green.

### Task 2: Real API runner and report

**Files:**
- Modify: `experiments/scripts/command_format_comparison.py`
- Test: `tests/test_command_format_comparison.py`
- Create at runtime: external run JSONL, `metrics.json`, `report.md`, `run_config.json`

**Interfaces:**
- Consumes Task 1 parsers and scenarios.
- Produces resumable raw responses and automatic summary artifacts.

- [ ] Write failing tests for alternated order, usage extraction, and resumable identity checks.
- [ ] Run focused tests red.
- [ ] Implement the runner and artifact writer.
- [ ] Run 60 real paired calls and score them.

### Task 3: Preserve automatic prior-experiment examples

**Files:**
- Create: `experiments/scripts/export_human_examples.py`
- Test: `tests/test_export_human_examples.py`
- Create: `experiments/examples/2026-11-26-human-understanding/README.md`
- Create: selected contact-sheet images and machine-readable selection manifest.

**Interfaces:**
- Consumes frozen Penn/RAVDESS manifests and prediction JSONL files.
- Produces automatically selected best/worst example sheets without affecting scores.

- [ ] Write failing selection and rendering tests.
- [ ] Run focused tests red.
- [ ] Implement deterministic score selection and contact-sheet export.
- [ ] Export and verify the example bundle.

### Task 4: Record and verification

**Files:**
- Create: `experiments/records/2026-11-26-command-format-comparison.md`
- Modify: `experiments/records/README.md`

- [ ] Record exact conditions, scores, failures, hashes, and bounded claims.
- [ ] Run focused and evaluation regression tests.
- [ ] Run JSONL integrity and `git diff --check` verification.
