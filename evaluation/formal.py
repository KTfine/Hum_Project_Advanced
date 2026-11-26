from __future__ import annotations

import asyncio
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys

import numpy as np

from evaluation.classification_metrics import paired_bootstrap_accuracy, score_classification
from evaluation.io import append_jsonl, atomic_write_json, read_jsonl
from evaluation.metrics import paired_bootstrap_difference
from evaluation.penn_metrics import score_penn_pose


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _group_records(records: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    seen = set()
    for record in records:
        key = (record.get("model"), record.get("sample_id"))
        if key in seen:
            raise ValueError(f"duplicate prediction for model={key[0]} sample_id={key[1]}")
        seen.add(key)
        groups.setdefault(record["model"], []).append(record)
    return groups


def score_penn_predictions(
    manifest: dict, records: list[dict], *, bootstrap_iterations: int = 10_000,
) -> dict:
    models = {}
    for model, current in _group_records(records).items():
        tasks = {task for record in current for task in record.get("tasks", [])}
        result = {"declared_tasks": sorted(tasks)}
        if "activity" in tasks:
            result["activity"] = score_classification(
                manifest, current, truth_key="action",
                probability_key="activity_probabilities",
            )
        if "pose" in tasks:
            result["pose"] = score_penn_pose(manifest, current)
        models[model] = result

    activity_comparisons = {}
    pose_comparisons = {}
    baseline = models.get("deepseek-flash", {})
    if "activity" in baseline:
        right = baseline["activity"]["per_sample_correct"]
        for model, values in models.items():
            if model != "deepseek-flash" and "activity" in values:
                activity_comparisons[f"{model}-minus-deepseek-flash"] = paired_bootstrap_accuracy(
                    values["activity"]["per_sample_correct"], right,
                    iterations=bootstrap_iterations, seed=3800,
                )
    if "pose" in baseline:
        ids = [sample["sample_id"] for sample in manifest["samples"]]
        right_map = baseline["pose"]["per_sample_pck_at_0_10"]
        right = np.asarray([right_map[sample_id] for sample_id in ids])
        for model, values in models.items():
            if model != "deepseek-flash" and "pose" in values:
                left_map = values["pose"]["per_sample_pck_at_0_10"]
                left = np.asarray([left_map[sample_id] for sample_id in ids])
                pose_comparisons[f"{model}-minus-deepseek-flash"] = paired_bootstrap_difference(
                    left, right, iterations=bootstrap_iterations, seed=3800,
                )
    activity_ladder = _ladder_summary(
        models, "activity", "accuracy",
        ["xclip-large-patch14", "xclip-base-patch32"],
    )
    pose_ladder = _ladder_summary(
        models, "pose", "pck_at_0_10",
        [f"yolo11{variant}-pose" for variant in "xlmsn"],
    )
    return {
        "schema_version": 1, "dataset": manifest.get("dataset"),
        "scoring_policy": "all manifest samples remain in denominator; no human or LLM judge",
        "models": models, "activity_comparisons": activity_comparisons,
        "pose_comparisons": pose_comparisons,
        "activity_ladder": activity_ladder, "pose_ladder": pose_ladder,
    }


def score_ravdess_predictions(
    manifest: dict, records: list[dict], *, bootstrap_iterations: int = 10_000,
) -> dict:
    models = {}
    for model, current in _group_records(records).items():
        tasks = {task for record in current for task in record.get("tasks", [])}
        if "emotion" in tasks:
            models[model] = score_classification(
                manifest, current, truth_key="emotion",
                probability_key="emotion_probabilities",
            )
    comparisons = {}
    if "deepseek-flash" in models:
        right = models["deepseek-flash"]["per_sample_correct"]
        for model, values in models.items():
            if model != "deepseek-flash":
                comparisons[f"{model}-minus-deepseek-flash"] = paired_bootstrap_accuracy(
                    values["per_sample_correct"], right,
                    iterations=bootstrap_iterations, seed=3800,
                )
    emotion_ladder = _ladder_summary(
        models, None, "accuracy", ["hsemotion-enet-b2-8", "hsemotion-enet-b0-8-afew"],
    )
    return {
        "schema_version": 1, "dataset": manifest.get("dataset"),
        "scoring_policy": "all manifest samples remain in denominator; no human or LLM judge",
        "models": models, "emotion_comparisons": comparisons,
        "emotion_ladder": emotion_ladder,
    }


def _ladder_summary(
    models: dict, section: str | None, metric: str, specialist_order: list[str],
) -> dict:
    baseline = models.get("deepseek-flash")
    baseline_values = baseline.get(section) if baseline and section else baseline
    baseline_score = baseline_values.get(metric) if baseline_values else None
    tiers = []
    crossing = None
    for name in specialist_order:
        values = models.get(name)
        values = values.get(section) if values and section else values
        if not values or metric not in values:
            continue
        score = values[metric]
        tiers.append({"model": name, "score": score})
        if crossing is None and baseline_score is not None and baseline_score >= score:
            crossing = name
    return {
        "metric": metric, "deepseek_score": baseline_score, "specialist_tiers": tiers,
        "first_tier_deepseek_matches_or_exceeds": crossing,
        "smallest_available_tier_exhausted": bool(tiers) and crossing is None,
    }


def _resumable_pending(path: Path, model: str, sample_ids: list[str]) -> list[str]:
    records = read_jsonl(path) if path.is_file() else []
    expected = set(sample_ids)
    completed = set()
    for record in records:
        if record.get("model") != model or record.get("sample_id") not in expected:
            raise ValueError(f"existing {path} does not match model or manifest")
        if record["sample_id"] in completed:
            raise ValueError(f"duplicate prediction in {path}: {record['sample_id']}")
        completed.add(record["sample_id"])
    return [sample_id for sample_id in sample_ids if sample_id not in completed]


def _release_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _all_records(files: list[Path]) -> list[dict]:
    return [record for path in files for record in read_jsonl(path)]


def _write_report(metrics: dict, output: Path, title: str) -> Path:
    lines = [
        f"# {title}", "",
        "All scores are automatic. Every frozen manifest sample remains in the denominator; no human or LLM judging was used.", "",
        "| Model | Task | Accuracy/PCK@0.10 | Macro-F1 | Failures |", "|---|---|---:|---:|---:|",
    ]
    for model, values in metrics["models"].items():
        if "activity" in values:
            item = values["activity"]
            lines.append(f"| {model} | activity | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | {item['failure_count']} |")
        if "pose" in values:
            item = values["pose"]
            lines.append(f"| {model} | pose | {item['pck_at_0_10']:.4f} | — | {item['failure_sample_count']} |")
        if "accuracy" in values:
            lines.append(f"| {model} | emotion | {values['accuracy']:.4f} | {values['macro_f1']:.4f} | {values['failure_count']} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run_penn_formal(paths, manifest_path: Path, *, run_id: str, use_deepseek: bool = True) -> Path:
    from evaluation.models.penn_deepseek import run_deepseek_penn
    from evaluation.models.pose import (
        POSE_VARIANTS, YoloPoseEstimator, download_pose_weights, pose_model_name,
        run_penn_pose_sample,
    )
    from evaluation.models.xclip import (
        XCLIP_VARIANTS, XClipPennClassifier, run_xclip_sample, xclip_model_name,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = [sample["sample_id"] for sample in manifest["samples"]]
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    if use_deepseek:
        model_name = "deepseek-flash"
        output = run_dir / f"{model_name}.jsonl"
        for sample_id in _resumable_pending(output, model_name, sample_ids):
            sample = samples[sample_id]
            try:
                record = asyncio.run(run_deepseek_penn(
                    sample_id=sample_id, frame_paths=sample["selected_frame_paths"],
                ))
            except Exception as exc:
                record = {"sample_id": sample_id, "model": model_name, "status": "error",
                          "reason": f"{type(exc).__name__}: {exc}", "tasks": ["activity", "pose"]}
            append_jsonl(output, record)
        outputs.append(output)

    device = "cuda" if _cuda_available() else "cpu"
    for variant, model_id in XCLIP_VARIANTS:
        model_name = xclip_model_name(variant)
        output = run_dir / f"{model_name}.jsonl"
        pending = _resumable_pending(output, model_name, sample_ids)
        classifier = None
        initialization_error = None
        if pending:
            try:
                classifier = XClipPennClassifier(
                    model_id, cache_dir=paths.models / "xclip", device=device,
                )
            except Exception as exc:
                initialization_error = f"{type(exc).__name__}: {exc}"
        for sample_id in pending:
            record = run_xclip_sample(samples[sample_id], classifier, model_name=model_name)
            if initialization_error and record["status"] == "blocked":
                record["reason"] = initialization_error
            append_jsonl(output, record)
        outputs.append(output)
        _release_model(classifier)

    for variant in POSE_VARIANTS:
        model_name = pose_model_name(variant)
        output = run_dir / f"{model_name}.jsonl"
        pending = _resumable_pending(output, model_name, sample_ids)
        estimator = None
        initialization_error = None
        if pending:
            try:
                weights = download_pose_weights(paths.models / f"{model_name}.pt", variant=variant)
                estimator = YoloPoseEstimator(weights, device=0 if device == "cuda" else "cpu")
            except Exception as exc:
                initialization_error = f"{type(exc).__name__}: {exc}"
        for sample_id in pending:
            record = run_penn_pose_sample(samples[sample_id], estimator, model_name=model_name)
            if initialization_error and record["status"] == "blocked":
                record["reason"] = initialization_error
            append_jsonl(output, record)
        outputs.append(output)
        _release_model(estimator)

    metrics = score_penn_predictions(manifest, _all_records(outputs))
    metrics.update(manifest=str(manifest_path.resolve()), manifest_sha256=_sha256_file(manifest_path))
    metrics_path = atomic_write_json(run_dir / "metrics.json", metrics)
    _write_report(metrics, run_dir / "report.md", "Penn Action formal evaluation")
    _write_run_config(run_dir, manifest_path, outputs, "penn-formal")
    return metrics_path


def run_ravdess_formal(paths, manifest_path: Path, *, run_id: str, use_deepseek: bool = True) -> Path:
    from evaluation.models.emotion import (
        EMOTION_VARIANTS, HSEmotionVideoClassifier, download_emotion_weights,
        emotion_model_name, run_deepseek_ravdess, run_hsemotion_sample,
    )
    from evaluation.video import extract_uniform_video_frames

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = [sample["sample_id"] for sample in manifest["samples"]]
    samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if use_deepseek:
        model_name = "deepseek-flash"
        output = run_dir / f"{model_name}.jsonl"
        for sample_id in _resumable_pending(output, model_name, sample_ids):
            sample = samples[sample_id]
            try:
                frames = extract_uniform_video_frames(
                    sample["video_path"], run_dir / "frames" / sample_id,
                )
                record = asyncio.run(run_deepseek_ravdess(sample_id=sample_id, frame_paths=frames))
            except Exception as exc:
                record = {"sample_id": sample_id, "model": model_name, "status": "error",
                          "reason": f"{type(exc).__name__}: {exc}", "tasks": ["emotion"]}
            append_jsonl(output, record)
        outputs.append(output)

    for variant, checkpoint in EMOTION_VARIANTS:
        model_name = emotion_model_name(variant)
        output = run_dir / f"{model_name}.jsonl"
        pending = _resumable_pending(output, model_name, sample_ids)
        classifier = None
        initialization_error = None
        if pending:
            try:
                weights = download_emotion_weights(
                    paths.models / f"{checkpoint}.onnx", checkpoint=checkpoint,
                )
                classifier = HSEmotionVideoClassifier(checkpoint, weights)
            except Exception as exc:
                initialization_error = f"{type(exc).__name__}: {exc}"
        for sample_id in pending:
            record = run_hsemotion_sample(samples[sample_id], classifier, model_name=model_name)
            if initialization_error and record["status"] == "blocked":
                record["reason"] = initialization_error
            append_jsonl(output, record)
        outputs.append(output)
        _release_model(classifier)

    metrics = score_ravdess_predictions(manifest, _all_records(outputs))
    metrics.update(manifest=str(manifest_path.resolve()), manifest_sha256=_sha256_file(manifest_path))
    metrics_path = atomic_write_json(run_dir / "metrics.json", metrics)
    _write_report(metrics, run_dir / "report.md", "RAVDESS formal evaluation")
    _write_run_config(run_dir, manifest_path, outputs, "ravdess-formal")
    return metrics_path


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _write_run_config(run_dir: Path, manifest: Path, outputs: list[Path], kind: str) -> Path:
    packages = {}
    for name in ("torch", "transformers", "ultralytics", "emotiefflib", "onnxruntime", "opencv-python"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu = {"name": properties.name, "memory_bytes": properties.total_memory}
    except ImportError:
        pass
    return atomic_write_json(run_dir / "run_config.json", {
        "kind": kind, "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256_file(manifest),
        "prediction_sources": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in outputs
        ],
        "python": sys.version, "platform": platform.platform(),
        "cuda_available": _cuda_available(), "gpu": gpu, "package_versions": packages,
    })
