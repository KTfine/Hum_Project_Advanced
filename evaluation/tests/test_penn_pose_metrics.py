from PIL import Image
import pytest

from evaluation.penn_metrics import score_penn_pose


def test_penn_pose_metrics_normalize_pixels_and_penalize_failed_sample(tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 200), "black").save(frame)
    joints = [[50.0, 100.0] for _ in range(13)]
    visibility = [True for _ in range(13)]
    manifest = {"samples": [
        {"sample_id": "ok", "selected_frame_paths": [str(frame)],
         "selected_frame_indices": [0], "joints_xy": [joints],
         "visibility": [visibility]},
        {"sample_id": "failed", "selected_frame_paths": [str(frame)],
         "selected_frame_indices": [0], "joints_xy": [joints],
         "visibility": [visibility]},
    ]}
    records = [
        {"sample_id": "ok", "status": "ok", "model": "m",
         "prediction": {"frames": [{"joints": [[0.5, 0.5] for _ in range(13)]}]}},
        {"sample_id": "failed", "status": "error", "model": "m"},
    ]
    result = score_penn_pose(manifest, records)
    assert result["sample_count"] == 2
    assert result["valid_sample_count"] == 1
    assert result["joint_count"] == 26
    assert result["missing_rate"] == pytest.approx(0.5)
    assert result["pck_at_0_10"] == pytest.approx(0.5)
    assert result["nme"] == pytest.approx(0.5)
    assert result["per_sample_pck_at_0_10"] == {"ok": 1.0, "failed": 0.0}


def test_penn_pose_metrics_reject_duplicate_prediction(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        score_penn_pose({"samples": []}, [
            {"sample_id": "x", "status": "error"},
            {"sample_id": "x", "status": "error"},
        ])
