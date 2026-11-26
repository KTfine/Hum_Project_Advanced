import numpy as np
import pytest

from evaluation.data.ravdess import RAVDESS_LABELS
from evaluation.models.emotion import (
    EMOTION_VARIANTS,
    aggregate_hsemotion_scores,
    detect_largest_face,
    validate_ravdess_prediction,
)


def test_hsemotion_ladder_is_large_then_small():
    assert EMOTION_VARIANTS == (
        ("enet-b2-8", "enet_b2_8"),
        ("enet-b0-8-afew", "enet_b0_8_best_afew"),
    )


def test_hsemotion_mapping_drops_contempt_then_renormalizes():
    # HSE order: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise.
    scores = np.asarray([[0.10, 0.30, 0.05, 0.10, 0.20, 0.15, 0.05, 0.05]])
    result = aggregate_hsemotion_scores(scores)
    assert list(result) == list(RAVDESS_LABELS)
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["happy"] == pytest.approx(0.20 / 0.70)
    assert result["angry"] == pytest.approx(0.10 / 0.70)


def test_validate_ravdess_prediction_requires_exact_seven_classes():
    value = {"emotion_probabilities": {label: 1 for label in RAVDESS_LABELS}}
    parsed = validate_ravdess_prediction(value)
    assert sum(parsed["emotion_probabilities"]) == pytest.approx(1.0)
    del value["emotion_probabilities"]["neutral"]
    with pytest.raises(ValueError, match="exactly the 7"):
        validate_ravdess_prediction(value)


def test_face_detection_failure_is_explicit():
    class EmptyDetector:
        def detectMultiScale(self, *_args, **_kwargs):
            return ()

    assert detect_largest_face(np.zeros((100, 100, 3), dtype=np.uint8), EmptyDetector()) is None


def test_face_detection_accepts_yunet_style_detector():
    class Detector:
        def detect_faces(self, _frame):
            return np.asarray([[20.2, 19.8, 40.0, 40.0], [10.0, 10.0, 10.0, 10.0]])

    face = detect_largest_face(np.zeros((100, 100, 3), dtype=np.uint8), Detector())
    assert face is not None
    assert face.shape[:2] == (56, 56)
