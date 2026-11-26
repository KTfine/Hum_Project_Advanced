from __future__ import annotations

import numpy as np


def normalize_xy(points: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    value = np.asarray(points, dtype=np.float32).copy()
    if value.shape[-1] != 2:
        raise ValueError("points must end in x,y")
    value[..., 0] /= float(width)
    value[..., 1] /= float(height)
    return np.clip(value, 0.0, 1.0)


def blocked_record(
    sample_id: str, model: str, reason: str, *, tasks: list[str] | None = None
) -> dict:
    record = {
        "sample_id": sample_id,
        "model": model,
        "status": "blocked",
        "reason": reason,
    }
    if tasks:
        record["tasks"] = tasks
    return record
