from __future__ import annotations

import asyncio
import time
from pathlib import Path

from evaluation.data.penn_action import PENN_ACTIONS
from evaluation.models.deepseek import ensure_asyncio_timeout, parse_deepseek_json
from photo_assistant.api.client import ApiError, DeepSeekClient, Image


PENN_JOINTS = (
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

PENN_SYSTEM_PROMPT = (
    "You are an evaluation model. Analyze only the eight separately supplied, "
    "time-ordered natural video frames. Return one JSON object only, without Markdown."
)

PENN_USER_PROMPT = """Classify the complete action and estimate pose in every frame.
Return exactly:
{
  "activity_probabilities": {every allowed label: probability from 0 to 1},
  "frames": [{"joints": [[x,y], ... exactly 13 joints ...]}, ... exactly 8]
}
Allowed labels: %s
Joint order: %s
Coordinates are normalized independently within each supplied image, top-left [0,0],
bottom-right [1,1]. Probabilities need not be pre-normalized but must have positive sum.
Infer occluded joints; do not omit entries.""" % (", ".join(PENN_ACTIONS), ", ".join(PENN_JOINTS))


def validate_penn_prediction(value: dict) -> dict:
    raw_probabilities = value.get("activity_probabilities")
    if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(PENN_ACTIONS):
        raise ValueError("activity_probabilities must contain exactly the 15 Penn labels")
    probabilities = []
    for label in PENN_ACTIONS:
        score = raw_probabilities[label]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score < 0:
            raise ValueError(f"invalid activity probability for {label}")
        probabilities.append(float(score))
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("activity probabilities must have positive sum")
    probabilities = [score / total for score in probabilities]

    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        raise ValueError("frames must contain exactly 8 entries")
    normalized_frames = []
    for frame in frames:
        joints = frame.get("joints") if isinstance(frame, dict) else None
        if not isinstance(joints, list) or len(joints) != 13:
            raise ValueError("each frame must contain exactly 13 joints")
        normalized_joints = []
        for point in joints:
            if (
                not isinstance(point, list) or len(point) != 2
                or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in point)
            ):
                raise ValueError("every joint must be a numeric [x, y] pair")
            x, y = float(point[0]), float(point[1])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("joint coordinates must be in [0, 1]")
            normalized_joints.append([x, y])
        normalized_frames.append({"joints": normalized_joints})
    return {"activity_probabilities": probabilities, "frames": normalized_frames}


def build_penn_messages(frame_paths: list[str | Path]) -> list[dict]:
    if len(frame_paths) != 8:
        raise ValueError("Penn evaluation requires exactly eight frames")
    content = [{"type": "text", "text": PENN_USER_PROMPT}]
    content.extend(Image.from_path(path).block() for path in frame_paths)
    return [
        {"role": "system", "content": PENN_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


async def run_deepseek_penn(
    *, sample_id: str, frame_paths: list[str | Path], retries: int = 2,
    model: str = "deepseek-flash", thinking: str = "off",
) -> dict:
    ensure_asyncio_timeout()
    messages = build_penn_messages(frame_paths)
    attempts = []
    started = time.perf_counter()
    async with DeepSeekClient(model=model, thinking=thinking, timeout=180, max_tokens=4096) as client:
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
                        "elapsed_s": time.perf_counter() - started,
                        "tasks": ["activity", "pose"],
                    }
                await asyncio.sleep(0.5 * (attempt + 1))
    assert result is not None
    record = {
        "sample_id": sample_id, "model": model, "raw_text": result.text,
        "metadata": result.metadata, "attempts": attempts,
        "elapsed_s": time.perf_counter() - started,
        "tasks": ["activity", "pose"],
    }
    try:
        record["prediction"] = validate_penn_prediction(parse_deepseek_json(result.text))
        record["status"] = "ok"
    except ValueError as exc:
        record.update(status="invalid", reason=str(exc))
    return record
