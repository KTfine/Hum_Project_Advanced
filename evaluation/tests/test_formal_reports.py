import json

from PIL import Image

from evaluation.formal import score_penn_predictions, score_ravdess_predictions


def test_formal_penn_score_separates_activity_and_pose_models(tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100), "black").save(frame)
    sample = {
        "sample_id": "p1", "action": "a", "selected_frame_paths": [str(frame)],
        "selected_frame_indices": [0], "joints_xy": [[[50, 50]] * 13],
        "visibility": [[True] * 13],
    }
    manifest = {"dataset": "Penn", "label_order": ["a", "b"], "samples": [sample]}
    records = [
        {"sample_id": "p1", "model": "deepseek-flash", "status": "ok",
         "tasks": ["activity", "pose"], "prediction": {
             "activity_probabilities": [0.9, 0.1],
             "frames": [{"joints": [[0.5, 0.5]] * 13}],
         }},
        {"sample_id": "p1", "model": "activity-only", "status": "ok",
         "tasks": ["activity"], "prediction": {"activity_probabilities": [0.4, 0.6]}},
    ]
    metrics = score_penn_predictions(manifest, records, bootstrap_iterations=100)
    assert "activity" in metrics["models"]["activity-only"]
    assert "pose" not in metrics["models"]["activity-only"]
    assert metrics["models"]["deepseek-flash"]["pose"]["pck_at_0_10"] == 1.0
    assert "activity-only-minus-deepseek-flash" in metrics["activity_comparisons"]
    assert metrics["activity_ladder"]["specialist_tiers"] == []


def test_formal_ravdess_score_includes_failures(tmp_path):
    manifest = {"dataset": "RAVDESS", "label_order": ["neutral", "happy"], "samples": [
        {"sample_id": "r1", "emotion": "neutral"},
        {"sample_id": "r2", "emotion": "happy"},
    ]}
    records = [
        {"sample_id": "r1", "model": "m", "status": "ok", "tasks": ["emotion"],
         "prediction": {"emotion_probabilities": [0.8, 0.2]}},
        {"sample_id": "r2", "model": "m", "status": "error", "tasks": ["emotion"]},
    ]
    metrics = score_ravdess_predictions(manifest, records, bootstrap_iterations=100)
    assert metrics["models"]["m"]["sample_count"] == 2
    assert metrics["models"]["m"]["accuracy"] == 0.5
    assert metrics["models"]["m"]["failure_count"] == 1
    assert metrics["emotion_ladder"]["deepseek_score"] is None
