from __future__ import annotations

import math

import numpy as np


def score_classification(
    manifest: dict, records: list[dict], *, truth_key: str, probability_key: str,
    failure_epsilon: float = 1e-12,
) -> dict:
    labels = list(manifest["label_order"])
    label_to_index = {label: index for index, label in enumerate(labels)}
    if len(label_to_index) != len(labels):
        raise ValueError("label_order contains duplicates")
    record_by_id = {}
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id in record_by_id:
            raise ValueError(f"duplicate prediction for sample_id={sample_id}")
        record_by_id[sample_id] = record

    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    correct = []
    reciprocal_ranks = []
    losses = []
    valid_count = 0
    failures = []
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        truth = sample[truth_key]
        if truth not in label_to_index:
            raise ValueError(f"unknown truth label {truth} for sample {sample_id}")
        truth_index = label_to_index[truth]
        record = record_by_id.get(sample_id)
        probabilities = None
        if record is not None and record.get("status") == "ok":
            probabilities = record.get("prediction", {}).get(probability_key)
            if not isinstance(probabilities, list) or len(probabilities) != len(labels):
                raise ValueError(f"sample {sample_id} must contain {len(labels)} probabilities")
            values = np.asarray(probabilities, dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
                raise ValueError(f"sample {sample_id} has invalid probabilities")
            probabilities = values / values.sum()
        if probabilities is None:
            correct.append(0)
            reciprocal_ranks.append(0.0)
            losses.append(-math.log(failure_epsilon))
            failures.append(sample_id)
            continue

        valid_count += 1
        predicted_index = int(np.argmax(probabilities))
        confusion[truth_index, predicted_index] += 1
        correct.append(int(predicted_index == truth_index))
        order = np.argsort(-probabilities, kind="stable")
        rank = int(np.flatnonzero(order == truth_index)[0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        losses.append(-math.log(max(float(probabilities[truth_index]), failure_epsilon)))

    recalls = []
    f1_values = []
    for index in range(len(labels)):
        true_total = sum(sample[truth_key] == labels[index] for sample in manifest["samples"])
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        recall = true_positive / true_total if true_total else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
    count = len(manifest["samples"])
    return {
        "sample_count": count,
        "valid_count": valid_count,
        "failure_count": count - valid_count,
        "failed_sample_ids": failures,
        "accuracy": float(np.mean(correct)) if count else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if count else 0.0,
        "nll": float(np.mean(losses)) if count else 0.0,
        "per_class_recall": dict(zip(labels, recalls)),
        "confusion_matrix": confusion.tolist(),
        "per_sample_correct": correct,
    }


def paired_bootstrap_accuracy(
    correct_a: list[int], correct_b: list[int], *, iterations: int = 10_000,
    seed: int = 3800,
) -> dict:
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("paired correctness arrays must be non-empty and equal length")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, a.size, size=(iterations, a.size))
    differences = (a[indices] - b[indices]).mean(axis=1)
    return {
        "mean_difference": float((a - b).mean()),
        "ci95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "win_probability": float(np.mean(differences > 0)),
        "tie_probability": float(np.mean(differences == 0)),
        "iterations": iterations,
        "seed": seed,
    }
