import numpy as np
import pytest

from evaluation.models.common import blocked_record, normalize_xy
from evaluation.models.deepseek import ensure_asyncio_timeout, parse_deepseek_json
from evaluation.models.pose import POSE_VARIANTS, pose_model_name, pose_weights_url


def test_parse_deepseek_json_accepts_fenced_object_and_rejects_trailing_text():
    assert parse_deepseek_json('```json\n{"value": 1}\n```') == {"value": 1}
    with pytest.raises(ValueError, match="single JSON object"):
        parse_deepseek_json('{"value": 1} explanation')


def test_normalize_xy_uses_image_dimensions():
    points = np.array([[50.0, 25.0], [100.0, 50.0]], dtype=np.float32)
    result = normalize_xy(points, width=100, height=50)
    np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])


def test_yolo11_pose_ladder_runs_largest_to_smallest():
    assert POSE_VARIANTS == ("x", "l", "m", "s", "n")
    assert [pose_model_name(size) for size in POSE_VARIANTS] == [
        "yolo11x-pose", "yolo11l-pose", "yolo11m-pose", "yolo11s-pose", "yolo11n-pose",
    ]
    assert pose_weights_url("n").endswith("/yolo11n-pose.pt")
    with pytest.raises(ValueError, match="unsupported YOLO11 pose variant"):
        pose_weights_url("tiny")


def test_blocked_record_keeps_declared_task():
    record = blocked_record("sample-1", "yolo11n-pose", "unavailable", tasks=["pose"])
    assert record["status"] == "blocked"
    assert record["tasks"] == ["pose"]


def test_python310_asyncio_timeout_compatibility(monkeypatch):
    import asyncio
    monkeypatch.delattr(asyncio, "timeout", raising=False)
    ensure_asyncio_timeout()
    assert callable(asyncio.timeout)
