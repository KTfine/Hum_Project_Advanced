from __future__ import annotations

import numpy as np
from PIL import Image


PENN_LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))


def _valid_point(point) -> bool:
    return (
        isinstance(point, (list, tuple)) and len(point) == 2
        and point[0] is not None and point[1] is not None
        and np.isfinite(np.asarray(point, dtype=float)).all()
    )


def score_penn_pose(manifest: dict, records: list[dict]) -> dict:
    record_by_id = {}
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id in record_by_id:
            raise ValueError(f"duplicate prediction for sample_id={sample_id}")
        record_by_id[sample_id] = record

    all_errors = []
    missing_count = 0
    valid_samples = 0
    compared_pairs = 0
    swapped_pairs = 0
    per_sample = {}
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        record = record_by_id.get(sample_id)
        predicted_frames = None
        if record is not None and record.get("status") == "ok":
            predicted_frames = record.get("prediction", {}).get("frames")
            if not isinstance(predicted_frames, list) or len(predicted_frames) != len(sample["selected_frame_paths"]):
                raise ValueError(f"sample {sample_id} has invalid predicted frame count")
            valid_samples += 1
        sample_errors = []
        for output_index, (frame_path, source_index) in enumerate(zip(
            sample["selected_frame_paths"], sample["selected_frame_indices"]
        )):
            with Image.open(frame_path) as image:
                width, height = image.size
            gt = np.asarray(sample["joints_xy"][source_index], dtype=float)
            gt[:, 0] /= width
            gt[:, 1] /= height
            visible = np.asarray(sample["visibility"][source_index], dtype=bool)
            visible_points = gt[visible]
            if visible_points.size == 0:
                continue
            span = visible_points.max(axis=0) - visible_points.min(axis=0)
            scale = max(float(np.linalg.norm(span)), 1e-8)
            pred = None
            if predicted_frames is not None:
                frame = predicted_frames[output_index]
                pred = frame.get("joints") if isinstance(frame, dict) else None
            for joint_index in np.flatnonzero(visible):
                point = pred[joint_index] if isinstance(pred, list) and joint_index < len(pred) else None
                if not _valid_point(point):
                    error = 1.0
                    missing_count += 1
                else:
                    error = float(np.linalg.norm(np.asarray(point, dtype=float) - gt[joint_index]) / scale)
                sample_errors.append(error)
                all_errors.append(error)
            if not isinstance(pred, list):
                continue
            for left, right in PENN_LEFT_RIGHT_PAIRS:
                if not (visible[left] and visible[right]) or not (_valid_point(pred[left]) and _valid_point(pred[right])):
                    continue
                correct = np.linalg.norm(np.asarray(pred[left]) - gt[left]) + np.linalg.norm(np.asarray(pred[right]) - gt[right])
                swapped = np.linalg.norm(np.asarray(pred[left]) - gt[right]) + np.linalg.norm(np.asarray(pred[right]) - gt[left])
                compared_pairs += 1
                swapped_pairs += int(swapped < correct)
        per_sample[sample_id] = float(np.mean(np.asarray(sample_errors) <= 0.10)) if sample_errors else 0.0

    values = np.asarray(all_errors, dtype=float)
    return {
        "sample_count": len(manifest["samples"]),
        "valid_sample_count": valid_samples,
        "failure_sample_count": len(manifest["samples"]) - valid_samples,
        "joint_count": int(values.size),
        "missing_count": missing_count,
        "missing_rate": float(missing_count / values.size) if values.size else 0.0,
        "pck_at_0_05": float(np.mean(values <= 0.05)) if values.size else 0.0,
        "pck_at_0_10": float(np.mean(values <= 0.10)) if values.size else 0.0,
        "pck_at_0_20": float(np.mean(values <= 0.20)) if values.size else 0.0,
        "nme": float(np.mean(values)) if values.size else 0.0,
        "left_right_pair_count": compared_pairs,
        "left_right_swap_rate": float(swapped_pairs / compared_pairs) if compared_pairs else None,
        "per_sample_pck_at_0_10": per_sample,
    }
