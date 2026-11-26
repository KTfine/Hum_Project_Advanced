import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
import evaluation.data.penn_action as penn_action

from evaluation.data.penn_action import (
    PENN_ACTIONS,
    build_penn_manifest,
    load_penn_annotation,
    select_penn_samples,
    validate_archive_member,
)


def _write_sequence(root: Path, sequence_id: str, action: str, *, visible: int = 13) -> None:
    frames = root / "frames" / sequence_id
    frames.mkdir(parents=True)
    for frame_index in range(1, 3):
        (frames / f"{frame_index:06d}.jpg").write_bytes(b"jpeg-fixture")
    x = np.arange(26, dtype=np.float32).reshape(2, 13)
    y = x + 10
    visibility = np.zeros((2, 13), dtype=np.uint8)
    visibility[:, :visible] = 1
    annotations = root / "labels"
    annotations.mkdir(exist_ok=True)
    savemat(annotations / f"{sequence_id}.mat", {
        "action": action,
        "x": x,
        "y": y,
        "visibility": visibility,
        "train": np.asarray([[1]], dtype=np.uint8),
    })


def test_load_penn_annotation_preserves_visibility_and_joint_shape(tmp_path):
    _write_sequence(tmp_path, "0001", "baseball_pitch", visible=7)

    item = load_penn_annotation(tmp_path / "labels" / "0001.mat")

    assert item["action"] == "baseball_pitch"
    assert item["joints_xy"].shape == (2, 13, 2)
    assert item["visibility"].shape == (2, 13)
    assert int(item["visibility"].sum()) == 14


def test_penn_action_labels_match_official_annotation_names():
    assert PENN_ACTIONS == (
        "baseball_pitch", "baseball_swing", "bench_press", "bowl",
        "clean_and_jerk", "golf_swing", "jumping_jacks", "jump_rope",
        "pullup", "pushup", "situp", "squat", "strum_guitar",
        "tennis_forehand", "tennis_serve",
    )


def test_select_penn_samples_is_deterministic_and_stratified():
    entries = [
        {"sample_id": f"{action}-{index}", "action": action}
        for action in PENN_ACTIONS
        for index in range(6)
    ]

    first = select_penn_samples(entries, count_per_class=4, seed=3800)
    second = select_penn_samples(entries, count_per_class=4, seed=3800)

    assert first == second
    assert len(first) == 60
    assert {action: sum(item["action"] == action for item in first) for action in PENN_ACTIONS} == {
        action: 4 for action in PENN_ACTIONS
    }


def test_select_penn_samples_rejects_underfilled_class():
    entries = [{"sample_id": "only-one", "action": PENN_ACTIONS[0]}]
    with pytest.raises(ValueError, match=PENN_ACTIONS[0]):
        select_penn_samples(entries, count_per_class=4, seed=3800)


def test_build_penn_manifest_writes_60_samples_with_visible_joint_gt(tmp_path):
    for action_index, action in enumerate(PENN_ACTIONS):
        for item_index in range(4):
            _write_sequence(tmp_path, f"{action_index:02d}{item_index:02d}", action)

    path = build_penn_manifest(tmp_path, count_per_class=4, seed=3800)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["dataset"] == "Penn Action"
    assert manifest["count"] == 60
    assert manifest["label_order"] == list(PENN_ACTIONS)
    assert len(manifest["samples"]) == 60
    first = manifest["samples"][0]
    assert len(first["frame_paths"]) == 2
    assert len(first["joints_xy"]) == 2
    assert len(first["visibility"][0]) == 13
    assert len(first["selected_frame_paths"]) == 8
    assert first["selected_frame_indices"] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_validate_archive_member_rejects_path_traversal(tmp_path):
    assert validate_archive_member("Penn_Action/frames/0001/000001.jpg")
    with pytest.raises(ValueError, match="unsafe archive member"):
        validate_archive_member("../../escape.txt")


def test_download_penn_archive_uses_range_from_first_byte(tmp_path, monkeypatch):
    requests = []

    class Response:
        status = 206

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if getattr(self, "served", False):
                return b""
            self.served = True
            return b"data"

    def open_request(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(penn_action, "urlopen", open_request)
    monkeypatch.setattr(penn_action, "PENN_ACTION_ARCHIVE_BYTES", 4)

    output = penn_action.download_penn_archive(tmp_path / "Penn_Action.tar.gz")

    assert output.read_bytes() == b"data"
    assert requests[0][0].get_header("Range") == "bytes=0-"


def test_download_lock_rejects_a_second_writer(tmp_path):
    lock_path = tmp_path / "Penn_Action.tar.gz.lock"
    with penn_action.exclusive_download_lock(lock_path):
        with pytest.raises(RuntimeError, match="already active"):
            with penn_action.exclusive_download_lock(lock_path):
                pass
