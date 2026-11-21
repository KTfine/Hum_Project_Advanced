"""Estimate sprite standing height on the camera's ground plane from a person mask."""

from dataclasses import dataclass

import cv2
import numpy as np


# Half-body reference portrait: face width and vertical offset in human cheek widths.
PORTRAIT_CHEEK_RATIO = 0.71
PORTRAIT_FACE_DROP = 0.84


@dataclass(frozen=True)
class PersonReference:
    """Foot ordinate and crown-to-sole height, both in units of frame height."""

    foot_y: float
    height: float
    estimated: bool = False


def silhouette_reference(labels, label, stats, frame_height):
    """Estimate a standing person's depth, including the unseen legs in a portrait."""
    left, top, width, height, area = map(int, stats)
    rows, cols = labels.shape
    if height < max(6, 0.15 * rows) or area < 0.2 * width * height:
        return None
    bottom_cut = top + height >= rows
    if not bottom_cut:
        if not 0.12 <= width / height <= 0.78:
            return None
        if left == 0 or top == 0 or left + width >= cols:
            # A tall person can lose the crown or a sleeve at the border.
            # Require a narrower head and feet around a broader torso; a
            # dangling hand entering from above is not a standing person.
            if height < 0.72 * rows:
                return None
            widths = (labels[top:top + height, left:left + width] == label).sum(axis=1)
            n = max(1, round(0.08 * height))
            shoulders = np.percentile(widths[n:height // 2], 90)
            if max(np.median(widths[:n]), np.median(widths[-n:])) >= 0.80 * shoulders:
                return None
        return PersonReference((top + height) / frame_height, height / frame_height)
    if height < 0.28 * rows or width < max(6, 0.12 * frame_height):
        return None
    widths = (labels[top:top + height, left:left + width] == label).sum(axis=1)
    # A raised hand extends the bounding box without widening most torso rows.
    torso_width = np.percentile(widths, 90)
    if torso_width < 0.19 * height or np.percentile(widths, 20) > 0.88 * torso_width:
        # A narrow or uniformly wide strip contains no discernible head/shoulder profile.
        return None
    full_height = float(np.clip(torso_width / 0.32, 1.05 * height, 2.8 * height))
    return PersonReference((top + full_height) / frame_height, full_height / frame_height, True)


def person_reference(mask: np.ndarray | None, x: float, y: float) -> PersonReference | None:
    """Choose an upright full-body or cropped torso reference near the sprite."""
    if mask is None or mask.ndim != 2 or not mask.size:
        return None
    # The live segmentation model itself operates at 192 px. Sampling a large
    # interpolated preview at bounded resolution keeps compositing inexpensive.
    stride = max(1, (max(mask.shape) + 383) // 384)
    binary = (mask[::stride, ::stride] >= 0.5).astype(np.uint8)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for label in range(1, len(stats)):
        ref = silhouette_reference(labels, label, stats[label], mask.shape[0] / stride)
        if ref is not None:
            center_x = centers[label, 0] * stride / mask.shape[1]
            candidates.append((int(stats[label, cv2.CC_STAT_AREA]), abs(center_x - x), abs(ref.foot_y - y), ref))
    if not candidates:
        return None
    # Body fragments beside a large clipped portrait are poor depth references.
    minimum_area = 0.22 * max((c[0] for c in candidates if c[3].estimated), default=0)
    return min((c for c in candidates if c[0] >= minimum_area), key=lambda c: c[1:3])[3]


def matched_standing_height_px(face_width: float, sprite_face_width: float, sprite_height: float,
                               *, cheek_ratio: float = 0.92) -> float:
    """Convert a companion-to-person cheek width ratio into projected standing pixels."""
    return cheek_ratio * face_width * sprite_height / sprite_face_width


def standing_height_px(
    reference: PersonReference | None, frame_height: int, y: float, *, fallback_height: float,
    matched_height_px: float | None = None,
) -> float:
    """Project an upright sprite at foot ordinate y; sitting/raised arms use art metadata."""
    # Out-of-frame shoes do not give a reliable depth: use the measured face
    # even when a teleport changes the portrait crop or the person leans closer.
    if matched_height_px is not None and (reference is None or reference.estimated):
        return max(1.0, matched_height_px)
    if reference is not None:
        # Estimate half a head from 7.5 heads per standing adult. On a flat
        # floor, image height is proportional to foot distance below the horizon.
        # A roughly level, chest-height camera puts the horizon 3/4 of the
        # reference person's visible height above their feet.
        foot_y = reference.foot_y
        horizon_y = foot_y - 0.75 * reference.height
        base_height = frame_height * reference.height * (1 - 0.5 / 7.5)
        if matched_height_px is not None:
            # The usual automatic pose stands one tenth of a person's height
            # behind them on screen. Convert that measurement to the foot plane.
            base_height = matched_height_px * (0.75 / (0.75 - 0.10))
    else:
        # Uncalibrated composition: keep the caller's nominal size at the
        # default foot point; use a fixed horizon until a full person is seen.
        foot_y, horizon_y = 0.95, 0.4
        base_height = fallback_height
    perspective = (y - horizon_y) / (foot_y - horizon_y)
    # Model moves may take the foot beyond any image edge. Bound extrapolation
    # so it cannot invert the character or allocate enormous temporary images.
    return max(1.0, base_height * min(2.5, max(0.2, perspective)))
