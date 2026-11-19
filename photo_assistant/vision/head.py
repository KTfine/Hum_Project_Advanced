"""Measure a person's face from the head, neck and shoulders in their camera mask."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceReference:
    """Face center and estimated cheek-to-cheek width in source-frame pixels."""

    center_x: float
    center_y: float
    width: float


def measure_face(frame_bgr, person_mask: np.ndarray | None) -> FaceReference | None:
    """Return the largest person's measurable face in a half- or full-body shot."""
    if person_mask is None or person_mask.ndim != 2 or not person_mask.size:
        return None
    if frame_bgr is not None and frame_bgr.shape[:2] != person_mask.shape:
        return None
    step = max(1, (max(person_mask.shape) + 383) // 384)
    binary = (person_mask[::step, ::step] >= 0.5).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for label in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        left, top, width, height, area = map(int, stats[label])
        if height < max(12, 0.15 * binary.shape[0]) or area < 0.003 * binary.size:
            continue
        body = labels[top:top + height, left:left + width] == label
        # A narrower neck must follow the widest part of the head and give
        # way to broader shoulders. Reject a plain rectangle or an isolated head.
        widths = np.convolve(np.pad(body.sum(axis=1), 1, mode="edge"), [1 / 3] * 3, mode="valid")
        head_width = np.maximum.accumulate(widths)
        later_width = np.maximum.accumulate(widths[::-1])[::-1]
        depth = np.arange(height) / np.maximum(head_width, 1)
        valid = ((head_width >= 4) & (widths <= 0.88 * head_width)
                 & (later_width >= 1.20 * head_width)
                 & (depth >= (0.68 if top == 0 else 1.08)) & (depth <= 1.80)
                 & (np.arange(height) < 0.90 * height))
        if not valid.any():
            continue
        score = 1 - widths / np.maximum(head_width, 1) - 0.10 * abs(depth - 1.38)
        neck = int(np.argmax(np.where(valid, score, -np.inf)))
        span = head_width[neck]
        face_y = max(0, round(neck - 0.46 * span))
        row = np.any(body[max(0, face_y - 1):face_y + 2], axis=0)
        edges = np.flatnonzero(np.diff(np.pad(row, 1))).reshape(-1, 2)
        if not len(edges):
            continue
        # An arm raised alongside the face can belong to the same silhouette.
        # The face must occupy a coherent horizontal run at the expected width.
        start, end = max(edges, key=lambda pair: pair[1] - pair[0])
        if end - start < 0.73 * span or left + start == 0 or left + end >= binary.shape[1]:
            continue
        # Cheeks span about 79% of the hair-and-ears silhouette in our
        # full-length, waist-up and close camera reference frames.
        return FaceReference(float((left + (start + end - 1) / 2) * step),
                             float((top + face_y) * step), float(0.79 * span * step))
    return None
