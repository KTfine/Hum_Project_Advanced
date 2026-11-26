from __future__ import annotations

from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from evaluation.models.common import blocked_record, normalize_xy


POSE_VARIANTS = ("x", "l", "m", "s", "n")
WEIGHTS_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
COCO17_TO_PENN13 = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def coco17_to_penn13(
    xy: np.ndarray, confidence: np.ndarray, *, width: int, height: int,
    confidence_threshold: float = 0.05,
) -> tuple[list[list[float] | None], list[float]]:
    points = np.asarray(xy, dtype=float)
    scores = np.asarray(confidence, dtype=float)
    if points.shape != (17, 2) or scores.shape != (17,):
        raise ValueError("expected COCO arrays with shapes (17,2) and (17,)")
    normalized = normalize_xy(points, width=width, height=height)
    penn_points = []
    penn_scores = []
    for index in COCO17_TO_PENN13:
        score = float(scores[index])
        penn_scores.append(score)
        penn_points.append(normalized[index].astype(float).tolist() if score >= confidence_threshold else None)
    return penn_points, penn_scores


def pose_model_name(variant: str) -> str:
    if variant not in POSE_VARIANTS:
        raise ValueError(f"unsupported YOLO11 pose variant: {variant}")
    return f"yolo11{variant}-pose"


def pose_weights_url(variant: str) -> str:
    return f"{WEIGHTS_BASE_URL}/{pose_model_name(variant)}.pt"


def download_pose_weights(path: str | Path, *, variant: str = "x") -> Path:
    output = Path(path)
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = Request(pose_weights_url(variant), headers={"User-Agent": "HumProjectEvaluation/1.0"})
    with urlopen(request, timeout=300) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temporary.replace(output)
    return output


class YoloPoseEstimator:
    def __init__(self, weights: str | Path, *, device: int | str = 0):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc
        self.model = YOLO(str(weights))
        self.device = device

    def predict_frames(self, paths: list[str | Path]) -> list[dict]:
        images = [Image.open(path).convert("RGB") for path in paths]
        arrays = [np.asarray(image) for image in images]
        results = self.model.predict(arrays, device=self.device, verbose=False, conf=0.05)
        frames = []
        for image, result in zip(images, results):
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                frames.append({"joints": [None] * 13, "confidence": [0.0] * 13})
                continue
            selected = int(np.argmax(result.boxes.conf.detach().cpu().numpy()))
            xy = result.keypoints.xy[selected].detach().cpu().numpy()
            confidence = result.keypoints.conf[selected].detach().cpu().numpy()
            joints, scores = coco17_to_penn13(
                xy, confidence, width=image.width, height=image.height,
            )
            frames.append({"joints": joints, "confidence": scores})
        return frames

def run_penn_pose_sample(
    sample: dict, estimator: YoloPoseEstimator | None, *, model_name: str,
) -> dict:
    if estimator is None:
        return blocked_record(
            sample["sample_id"], model_name, "pose model failed to initialize", tasks=["pose"]
        )
    started = time.perf_counter()
    try:
        frames = estimator.predict_frames(sample["selected_frame_paths"])
        return {
            "sample_id": sample["sample_id"], "model": model_name, "status": "ok",
            "prediction": {"frames": frames}, "elapsed_s": time.perf_counter() - started,
            "input_modality": "same eight natural Penn Action frames",
            "tasks": ["pose"],
        }
    except Exception as exc:
        return {
            "sample_id": sample["sample_id"], "model": model_name, "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started, "tasks": ["pose"],
        }
