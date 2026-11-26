"""Export deterministic best/worst visual examples from the frozen formal runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageFont

from evaluation.video import extract_uniform_video_frames


PENN_SKELETON = (
    (0, 1), (0, 2), (1, 2),
    (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11), (8, 10), (10, 12),
)
GT_COLOR = (40, 235, 90)
DEEPSEEK_COLOR = (255, 45, 200)
POSE_DISPLAY_FRAME_SLOT = 4


def select_binary_examples(values: dict[str, bool], *, count: int = 2) -> dict[str, list[str]]:
    return {
        "success": sorted(key for key, value in values.items() if value)[:count],
        "failure": sorted(key for key, value in values.items() if not value)[:count],
    }


def select_extreme_examples(scores: dict[str, float], *, count: int = 2) -> dict[str, list[str]]:
    success = [key for key, _value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:count]]
    failure = []
    for key, _value in sorted(scores.items(), key=lambda item: (item[1], item[0])):
        if key not in success:
            failure.append(key)
        if len(failure) == count:
            break
    return {"success": success, "failure": failure}


def render_contact_sheet(
    frame_paths: list[str | Path], output_path: str | Path, *, title: str,
    panel_size: tuple[int, int] = (240, 180),
) -> Path:
    if len(frame_paths) != 8:
        raise ValueError("contact sheets require exactly eight frames")
    width, height = panel_size
    header = 40
    sheet = Image.new("RGB", (width * 4, height * 2 + header), "#111111")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="white", font=ImageFont.load_default())
    for index, path in enumerate(frame_paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(panel_size, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", panel_size, "black")
        panel.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        sheet.paste(panel, ((index % 4) * width, header + (index // 4) * height))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=88, optimize=True)
    return output


def _panel_points(
    points: list[list[float]], *, source_size: tuple[int, int],
    rendered_size: tuple[int, int], offset: tuple[int, int], normalized: bool,
) -> list[tuple[float, float]]:
    source_width, source_height = source_size
    rendered_width, rendered_height = rendered_size
    scale = min(rendered_width / source_width, rendered_height / source_height)
    left, top = offset
    result = []
    for x, y in points:
        if normalized:
            x, y = x * source_width, y * source_height
        result.append((left + x * scale, top + y * scale))
    return result


def _draw_skeleton(
    draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], visible: list[bool],
    *, color: tuple[int, int, int], line_width: int, radius: int, filled: bool,
) -> None:
    for start, end in PENN_SKELETON:
        if visible[start] and visible[end]:
            draw.line((points[start], points[end]), fill=color, width=line_width)
    for point, is_visible in zip(points, visible):
        if not is_visible:
            continue
        x, y = point
        box = (x - radius, y - radius, x + radius, y + radius)
        if filled:
            draw.ellipse(box, fill=color, outline=(255, 255, 255), width=1)
        else:
            draw.ellipse(box, outline=color, width=2)


def render_pose_frame(
    frame_path: str | Path, output_path: str | Path, *, title: str,
    ground_truth_xy: list[list[float]], ground_truth_visibility: list[bool],
    prediction_normalized: list[list[float]],
    panel_size: tuple[int, int] = (640, 480),
) -> Path:
    width, height = panel_size
    header = 40
    canvas = Image.new("RGB", (width, height + header), "#111111")
    header_draw = ImageDraw.Draw(canvas)
    header_draw.text((12, 12), title, fill="white", font=ImageFont.load_default())
    header_draw.line((width - 250, 20, width - 225, 20), fill=GT_COLOR, width=3)
    header_draw.text((width - 220, 12), "GT", fill=GT_COLOR, font=ImageFont.load_default())
    header_draw.line((width - 165, 20, width - 140, 20), fill=DEEPSEEK_COLOR, width=3)
    header_draw.text((width - 135, 12), "DeepSeek", fill=DEEPSEEK_COLOR, font=ImageFont.load_default())

    with Image.open(frame_path) as source:
        source_rgb = source.convert("RGB")
        source_size = source_rgb.size
        source_rgb.thumbnail(panel_size, Image.Resampling.LANCZOS)
        rendered_size = source_rgb.size
    panel = Image.new("RGB", panel_size, "black")
    offset = ((width - rendered_size[0]) // 2, (height - rendered_size[1]) // 2)
    panel.paste(source_rgb, offset)
    draw = ImageDraw.Draw(panel)
    ground_truth = _panel_points(
        ground_truth_xy, source_size=source_size, rendered_size=rendered_size,
        offset=offset, normalized=False,
    )
    prediction = _panel_points(
        prediction_normalized, source_size=source_size, rendered_size=rendered_size,
        offset=offset, normalized=True,
    )
    _draw_skeleton(
        draw, ground_truth, ground_truth_visibility, color=GT_COLOR,
        line_width=4, radius=6, filled=False,
    )
    _draw_skeleton(
        draw, prediction, [True] * 13, color=DEEPSEEK_COLOR,
        line_width=3, radius=5, filled=True,
    )
    canvas.paste(panel, (0, header))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".png":
        canvas.save(output, format="PNG", optimize=True)
    else:
        canvas.save(output, format="JPEG", quality=92, optimize=True)
    return output


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> dict[str, dict]:
    return {
        str(item["sample_id"]): item
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def export_examples(cache_root: Path, output_dir: Path) -> Path:
    penn_run = cache_root / "runs" / "20261126-penn-formal"
    ravdess_run = cache_root / "runs" / "20261126-ravdess-formal"
    penn_manifest = _read_json(cache_root / "datasets" / "penn-action" / "Penn_Action" / "manifest.json")
    ravdess_manifest = _read_json(cache_root / "datasets" / "ravdess" / "videos" / "manifest.json")
    penn_metrics = _read_json(penn_run / "metrics.json")
    ravdess_metrics = _read_json(ravdess_run / "metrics.json")
    penn_predictions = _read_jsonl(penn_run / "deepseek-flash.jsonl")
    emotion_predictions = _read_jsonl(ravdess_run / "deepseek-flash.jsonl")
    penn_samples = {str(item["sample_id"]): item for item in penn_manifest["samples"]}
    ravdess_samples = {str(item["sample_id"]): item for item in ravdess_manifest["samples"]}

    activity_correct = dict(zip(
        [str(item["sample_id"]) for item in penn_manifest["samples"]],
        [bool(value) for value in penn_metrics["models"]["deepseek-flash"]["activity"]["per_sample_correct"]],
    ))
    pose_scores = {
        str(key): float(value)
        for key, value in penn_metrics["models"]["deepseek-flash"]["pose"]["per_sample_pck_at_0_10"].items()
    }
    emotion_correct = dict(zip(
        [str(item["sample_id"]) for item in ravdess_manifest["samples"]],
        [bool(value) for value in ravdess_metrics["models"]["deepseek-flash"]["per_sample_correct"]],
    ))
    selections = {
        "activity": select_binary_examples(activity_correct),
        "pose": select_extreme_examples(pose_scores),
        "emotion": select_binary_examples(emotion_correct),
    }
    label_order = penn_manifest["label_order"]
    emotion_order = ravdess_manifest["label_order"]
    exported = []
    for task in ("activity", "pose"):
        for outcome, sample_ids in selections[task].items():
            for sample_id in sample_ids:
                sample = penn_samples[sample_id]
                if task == "activity":
                    probabilities = penn_predictions[sample_id]["prediction"]["activity_probabilities"]
                    predicted = label_order[max(range(len(probabilities)), key=probabilities.__getitem__)]
                    detail = f"truth={sample['action']} | pred={predicted}"
                else:
                    detail = f"PCK@0.10={pose_scores[sample_id]:.3f}"
                filename = f"{task}-{outcome}-{sample_id}.jpg"
                if task == "pose":
                    selected_indices = sample["selected_frame_indices"]
                    prediction = penn_predictions[sample_id]["prediction"]["frames"]
                    display_index = selected_indices[POSE_DISPLAY_FRAME_SLOT]
                    render_pose_frame(
                        sample["selected_frame_paths"][POSE_DISPLAY_FRAME_SLOT], output_dir / filename,
                        title=f"{task} | {outcome} | {sample_id} | {detail}",
                        ground_truth_xy=sample["joints_xy"][display_index],
                        ground_truth_visibility=sample["visibility"][display_index],
                        prediction_normalized=prediction[POSE_DISPLAY_FRAME_SLOT]["joints"],
                    )
                else:
                    render_contact_sheet(
                        sample["selected_frame_paths"], output_dir / filename,
                        title=f"{task} | {outcome} | {sample_id} | {detail}",
                    )
                item = {"task": task, "outcome": outcome, "sample_id": sample_id,
                        "file": filename, "detail": detail, "selection": "automatic"}
                if task == "pose":
                    item["overlay"] = "ground truth green; DeepSeek prediction magenta"
                    item["display_frame_slot"] = POSE_DISPLAY_FRAME_SLOT
                    item["source_frame_index"] = display_index
                exported.append(item)
    with tempfile.TemporaryDirectory(prefix="ravdess-example-") as temporary:
        temp_root = Path(temporary)
        for outcome, sample_ids in selections["emotion"].items():
            for sample_id in sample_ids:
                sample = ravdess_samples[sample_id]
                probabilities = emotion_predictions[sample_id]["prediction"]["emotion_probabilities"]
                predicted = emotion_order[max(range(len(probabilities)), key=probabilities.__getitem__)]
                frames = extract_uniform_video_frames(sample["video_path"], temp_root / sample_id)
                detail = f"truth={sample['emotion']} | pred={predicted}"
                filename = f"emotion-{outcome}-{sample_id}.jpg"
                render_contact_sheet(
                    frames, output_dir / filename,
                    title=f"emotion | {outcome} | {sample_id} | {detail}",
                )
                exported.append({"task": "emotion", "outcome": outcome, "sample_id": sample_id,
                                 "file": filename, "detail": detail, "selection": "automatic"})
    manifest = {
        "schema_version": 1,
        "selection_policy": "two lexicographically first correct/incorrect cases for classification; two highest/lowest non-overlapping PCK cases for pose",
        "source_runs": {
            "penn": str(penn_run.resolve()), "ravdess": str(ravdess_run.resolve()),
        },
        "examples": exported,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    lines = [
        "# Human-understanding examples", "",
        "These examples were selected automatically from the frozen formal runs; no image was hand-picked.", "",
        "Pose examples use the fixed fifth uniformly sampled frame (slot 4): Penn ground truth is green and DeepSeek prediction is magenta.", "",
    ]
    for item in exported:
        lines.append(f"- [{item['task']} {item['outcome']} {item['sample_id']}]({item['file']}): {item['detail']}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path(r"D:\AI_cache\hum_project_evaluation"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/examples/2026-11-26-human-understanding"),
    )
    args = parser.parse_args(argv)
    print(export_examples(args.cache_root, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
