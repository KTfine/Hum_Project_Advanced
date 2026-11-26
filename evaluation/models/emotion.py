from __future__ import annotations

import asyncio
from pathlib import Path
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

from evaluation.data.penn_action import exclusive_download_lock
from evaluation.data.ravdess import RAVDESS_LABELS
from evaluation.models.deepseek import ensure_asyncio_timeout, parse_deepseek_json
from evaluation.video import read_decodable_video_frames, uniform_frame_indices
from photo_assistant.api.client import ApiError, DeepSeekClient, Image


EMOTION_VARIANTS = (
    ("enet-b2-8", "enet_b2_8"),
    ("enet-b0-8-afew", "enet_b0_8_best_afew"),
)
HSEMOTION_ORDER = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")
HSEMOTION_BASE_URL = (
    "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/"
    "models/affectnet_emotions/onnx"
)
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_detection_yunet/{YUNET_FILENAME}"
)

RAVDESS_SYSTEM_PROMPT = (
    "You are an evaluation model. Analyze only the eight separately supplied, "
    "time-ordered frames from one video. Return one JSON object only, without Markdown."
)
RAVDESS_USER_PROMPT = """Estimate the speaker's facial emotion across the complete video.
Return exactly {"emotion_probabilities": {every allowed label: probability from 0 to 1}}.
Allowed labels: neutral, happy, sad, angry, fear, disgust, surprise.
Use the full temporal sequence. Probabilities need not be pre-normalized but must have positive sum."""


def emotion_model_name(variant: str) -> str:
    for name, _checkpoint in EMOTION_VARIANTS:
        if name == variant:
            return f"hsemotion-{name}"
    raise ValueError(f"unsupported HSEmotion variant: {variant}")


def validate_ravdess_prediction(value: dict) -> dict:
    raw = value.get("emotion_probabilities")
    if not isinstance(raw, dict) or set(raw) != set(RAVDESS_LABELS):
        raise ValueError("emotion_probabilities must contain exactly the 7 RAVDESS labels")
    scores = []
    for label in RAVDESS_LABELS:
        score = raw[label]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score < 0:
            raise ValueError(f"invalid emotion probability for {label}")
        scores.append(float(score))
    total = sum(scores)
    if total <= 0:
        raise ValueError("emotion probabilities must have positive sum")
    return {"emotion_probabilities": [score / total for score in scores]}


def aggregate_hsemotion_scores(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or values.shape[0] < 1:
        raise ValueError("HSEmotion scores must have shape (frames, 8)")
    mean = values.mean(axis=0)
    mapped = {label: float(mean[index]) for index, label in enumerate(HSEMOTION_ORDER)}
    mapped.pop("contempt")
    total = sum(mapped.values())
    if total <= 0:
        raise ValueError("HSEmotion probabilities have non-positive sum")
    return {label: mapped[label] / total for label in RAVDESS_LABELS}


def detect_largest_face(frame_rgb: np.ndarray, detector) -> np.ndarray | None:
    if hasattr(detector, "detect_faces"):
        faces = detector.detect_faces(frame_rgb)
    else:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(max(20, frame_rgb.shape[1] // 12), max(20, frame_rgb.shape[0] // 12)),
        )
    if len(faces) == 0:
        return None
    x, y, width, height = [
        int(round(float(value)))
        for value in max(faces, key=lambda item: float(item[2]) * float(item[3]))[:4]
    ]
    margin = int(round(max(width, height) * 0.20))
    left, top = max(0, x - margin), max(0, y - margin)
    right = min(frame_rgb.shape[1], x + width + margin)
    bottom = min(frame_rgb.shape[0], y + height + margin)
    return frame_rgb[top:bottom, left:right]


def download_face_detector_weights(path: str | Path) -> Path:
    output = Path(path)
    if output.is_file() and output.stat().st_size > 100_000:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = Request(YUNET_URL, headers={"User-Agent": "HumProjectEvaluation/2.0"})
    with urlopen(request, timeout=300) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    if temporary.stat().st_size <= 100_000:
        raise RuntimeError("downloaded YuNet checkpoint is unexpectedly small")
    temporary.replace(output)
    return output


class YuNetFaceDetector:
    def __init__(self, weights: str | Path):
        self.model = cv2.FaceDetectorYN.create(
            str(weights), "", (320, 320), score_threshold=0.6,
            nms_threshold=0.3, top_k=5000,
        )

    def detect_faces(self, frame_rgb: np.ndarray) -> np.ndarray:
        height, width = frame_rgb.shape[:2]
        self.model.setInputSize((width, height))
        _status, faces = self.model.detect(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if faces is None:
            return np.empty((0, 4), dtype=float)
        return np.asarray(faces[:, :4], dtype=float)


def download_emotion_weights(path: str | Path, *, checkpoint: str) -> Path:
    output = Path(path)
    with exclusive_download_lock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file() and output.stat().st_size > 1_000_000:
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".part")
        request = Request(
            f"{HSEMOTION_BASE_URL}/{checkpoint}.onnx",
            headers={"User-Agent": "HumProjectEvaluation/2.0"},
        )
        with urlopen(request, timeout=300) as response, temporary.open("wb") as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
        if temporary.stat().st_size <= 1_000_000:
            raise RuntimeError("downloaded HSEmotion checkpoint is unexpectedly small")
        temporary.replace(output)
        return output


class HSEmotionVideoClassifier:
    def __init__(
        self, checkpoint: str, weights: str | Path,
        face_weights: str | Path | None = None,
    ):
        import emotiefflib.facial_analysis as facial_analysis

        weights_path = str(Path(weights).resolve())
        original = facial_analysis.get_model_path_onnx
        facial_analysis.get_model_path_onnx = lambda _model_name: weights_path
        try:
            self.recognizer = facial_analysis.EmotiEffLibRecognizer(
                engine="onnx", model_name=checkpoint,
            )
        finally:
            facial_analysis.get_model_path_onnx = original
        face_path = Path(face_weights) if face_weights else Path(weights).parent / YUNET_FILENAME
        self.detector = YuNetFaceDetector(download_face_detector_weights(face_path))

    def predict_video(self, path: str | Path, *, count: int = 8) -> tuple[list[float], int]:
        frames = read_decodable_video_frames(path)
        faces = []
        for index in uniform_frame_indices(len(frames), count):
            rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
            face = detect_largest_face(rgb, self.detector)
            if face is not None:
                faces.append(face)
        if not faces:
            raise ValueError("no face detected in any sampled frame")
        _labels, scores = self.recognizer.predict_emotions(faces, logits=False)
        probabilities = aggregate_hsemotion_scores(scores)
        return [probabilities[label] for label in RAVDESS_LABELS], len(faces)


def run_hsemotion_sample(
    sample: dict, classifier: HSEmotionVideoClassifier | None, *, model_name: str,
) -> dict:
    if classifier is None:
        return {
            "sample_id": sample["sample_id"], "model": model_name, "status": "blocked",
            "reason": "HSEmotion failed to initialize", "tasks": ["emotion"],
        }
    started = time.perf_counter()
    try:
        probabilities, detected_frames = classifier.predict_video(sample["video_path"])
        return {
            "sample_id": sample["sample_id"], "model": model_name, "status": "ok",
            "prediction": {"emotion_probabilities": probabilities},
            "detected_face_frames": detected_frames, "sampled_frames": 8,
            "elapsed_s": time.perf_counter() - started, "tasks": ["emotion"],
        }
    except Exception as exc:
        return {
            "sample_id": sample["sample_id"], "model": model_name, "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started, "tasks": ["emotion"],
        }


def build_ravdess_messages(frame_paths: list[str | Path]) -> list[dict]:
    if len(frame_paths) != 8:
        raise ValueError("RAVDESS evaluation requires exactly eight frames")
    content = [{"type": "text", "text": RAVDESS_USER_PROMPT}]
    content.extend(Image.from_path(path).block() for path in frame_paths)
    return [
        {"role": "system", "content": RAVDESS_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


async def run_deepseek_ravdess(
    *, sample_id: str, frame_paths: list[str | Path], retries: int = 2,
    model: str = "deepseek-flash", thinking: str = "off",
) -> dict:
    ensure_asyncio_timeout()
    messages = build_ravdess_messages(frame_paths)
    attempts = []
    started = time.perf_counter()
    async with DeepSeekClient(model=model, thinking=thinking, timeout=180, max_tokens=2048) as client:
        result = None
        for attempt in range(retries + 1):
            try:
                result = await client.complete(messages)
                break
            except ApiError as exc:
                attempts.append(exc.result.metadata)
                if attempt == retries:
                    return {
                        "sample_id": sample_id, "model": model, "status": "error",
                        "reason": str(exc), "attempts": attempts,
                        "elapsed_s": time.perf_counter() - started, "tasks": ["emotion"],
                    }
                await asyncio.sleep(0.5 * (attempt + 1))
    assert result is not None
    record = {
        "sample_id": sample_id, "model": model, "raw_text": result.text,
        "metadata": result.metadata, "attempts": attempts,
        "elapsed_s": time.perf_counter() - started, "tasks": ["emotion"],
    }
    try:
        record["prediction"] = validate_ravdess_prediction(parse_deepseek_json(result.text))
        record["status"] = "ok"
    except ValueError as exc:
        record.update(status="invalid", reason=str(exc))
    return record
