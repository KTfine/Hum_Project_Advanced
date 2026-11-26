import json
from pathlib import Path

import pytest

from evaluation.data.ravdess import (
    RAVDESS_LABELS,
    build_ravdess_manifest,
    parse_ravdess_filename,
    select_ravdess_samples,
)


def test_parse_ravdess_video_speech_filename():
    item = parse_ravdess_filename("02-01-03-02-01-02-01.mp4")
    assert item == {
        "modality": "video_only", "vocal_channel": "speech",
        "emotion": "happy", "intensity": "strong", "statement": 1,
        "repetition": 2, "actor": 1,
    }


def test_parse_ravdess_rejects_unknown_or_non_video_speech():
    with pytest.raises(ValueError, match="video-only speech"):
        parse_ravdess_filename("01-01-03-01-01-01-01.mp4")
    with pytest.raises(ValueError, match="unknown emotion"):
        parse_ravdess_filename("02-01-99-01-01-01-01.mp4")


def test_ravdess_selection_excludes_calm_and_is_balanced():
    entries = [
        {"sample_id": f"{label}-{index}", "emotion": label}
        for label in (*RAVDESS_LABELS, "calm")
        for index in range(10)
    ]
    selected = select_ravdess_samples(entries, count_per_class=8, seed=3800)
    assert len(selected) == 56
    assert "calm" not in {item["emotion"] for item in selected}
    assert all(sum(item["emotion"] == label for item in selected) == 8 for label in RAVDESS_LABELS)


def test_build_ravdess_manifest_freezes_56_video_paths(tmp_path):
    for label_index, label in enumerate(RAVDESS_LABELS):
        for index in range(8):
            emotion_code = {"neutral": 1, "happy": 3, "sad": 4, "angry": 5,
                            "fear": 6, "disgust": 7, "surprise": 8}[label]
            name = (
                f"02-01-{emotion_code:02d}-01-{index % 2 + 1:02d}-"
                f"{(index // 2) % 2 + 1:02d}-{index // 4 + 1:02d}.mp4"
            )
            (tmp_path / name).write_bytes(b"video")
    path = build_ravdess_manifest(tmp_path, count_per_class=8, seed=3800)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["count"] == 56
    assert manifest["label_order"] == list(RAVDESS_LABELS)
    assert all(Path(sample["video_path"]).is_absolute() for sample in manifest["samples"])
