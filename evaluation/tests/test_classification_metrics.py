import math

import pytest

from evaluation.classification_metrics import paired_bootstrap_accuracy, score_classification


def _manifest():
    return {
        "label_order": ["a", "b", "c"],
        "samples": [
            {"sample_id": "1", "action": "a"},
            {"sample_id": "2", "action": "b"},
            {"sample_id": "3", "action": "c"},
        ],
    }


def test_classification_metrics_keep_failures_in_full_denominator():
    records = [
        {"sample_id": "1", "model": "m", "status": "ok",
         "prediction": {"activity_probabilities": [0.8, 0.1, 0.1]}},
        {"sample_id": "2", "model": "m", "status": "ok",
         "prediction": {"activity_probabilities": [0.6, 0.3, 0.1]}},
        {"sample_id": "3", "model": "m", "status": "error"},
    ]
    result = score_classification(
        _manifest(), records, truth_key="action", probability_key="activity_probabilities",
    )
    assert result["sample_count"] == 3
    assert result["valid_count"] == 2
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert result["macro_f1"] == pytest.approx(2 / 9)
    assert result["balanced_accuracy"] == pytest.approx(1 / 3)
    assert math.isfinite(result["nll"])
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 0]]


def test_classification_rejects_duplicate_or_wrong_length_predictions():
    duplicate = [
        {"sample_id": "1", "model": "m", "status": "error"},
        {"sample_id": "1", "model": "m", "status": "error"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        score_classification(_manifest(), duplicate, truth_key="action",
                             probability_key="activity_probabilities")
    invalid = [{"sample_id": "1", "model": "m", "status": "ok",
                "prediction": {"activity_probabilities": [1, 0]}}]
    with pytest.raises(ValueError, match="3 probabilities"):
        score_classification(_manifest(), invalid, truth_key="action",
                             probability_key="activity_probabilities")


def test_paired_bootstrap_is_deterministic_and_paired():
    first = paired_bootstrap_accuracy([1, 0, 1, 0], [0, 0, 1, 0], iterations=2000, seed=3800)
    second = paired_bootstrap_accuracy([1, 0, 1, 0], [0, 0, 1, 0], iterations=2000, seed=3800)
    assert first == second
    assert first["mean_difference"] == pytest.approx(0.25)
    assert first["win_probability"] > 0.5
