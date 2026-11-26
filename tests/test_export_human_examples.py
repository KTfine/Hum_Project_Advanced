from pathlib import Path

from PIL import Image

from experiments.scripts.export_human_examples import (
    render_contact_sheet,
    render_pose_frame,
    select_binary_examples,
    select_extreme_examples,
)


def test_binary_selection_is_deterministic_and_balanced():
    values = {"d": False, "b": True, "a": True, "c": False, "e": True}
    assert select_binary_examples(values, count=2) == {
        "success": ["a", "b"], "failure": ["c", "d"],
    }


def test_extreme_selection_uses_scores_and_never_duplicates(tmp_path):
    scores = {"a": 0.1, "b": 0.9, "c": 0.4, "d": 0.7}
    assert select_extreme_examples(scores, count=2) == {
        "success": ["b", "d"], "failure": ["a", "c"],
    }


def test_contact_sheet_preserves_eight_frames_with_title(tmp_path):
    paths = []
    for index in range(8):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (64, 48), (index * 20, 10, 10)).save(path)
        paths.append(path)
    output = tmp_path / "sheet.jpg"
    render_contact_sheet(paths, output, title="sample | success", panel_size=(160, 120))
    with Image.open(output) as image:
        assert image.size == (640, 280)
    assert output.stat().st_size > 1000


def test_pose_frame_draws_both_skeletons_on_one_frame(tmp_path):
    source = tmp_path / "pose.png"
    Image.new("RGB", (100, 100), "black").save(source)
    ground_truth = [[20 + joint, 20 + joint] for joint in range(13)]
    visibility = [True] * 13
    prediction = [[0.6 + joint / 100, 0.2 + joint / 100] for joint in range(13)]
    output = tmp_path / "pose-frame.png"

    render_pose_frame(
        source, output, title="pose example", ground_truth_xy=ground_truth,
        ground_truth_visibility=visibility, prediction_normalized=prediction,
        panel_size=(200, 150),
    )

    with Image.open(output) as image:
        colors = list(image.convert("RGB").getdata())
        assert image.size == (200, 190)
    assert any(g > 180 and r < 100 for r, g, _b in colors)
    assert any(r > 180 and b > 120 and g < 150 for r, g, b in colors)
