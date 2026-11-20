"""Project an upright sprite on a flat floor using its settled image-space body size."""
from dataclasses import dataclass

import cv2
import numpy as np


def camera_reference() -> tuple[float, float, float]:
    """Foot x, foot y, upright-equivalent height / image height for the reference camera."""
    # Sprite-to-photo registration on the seated dragon from this camera:
    # (380, 472) in 640x480; scale 0.22413 times 1425 standing pixels.
    # Hanging shoes approximate the ground anchor. A visible companion supersedes this prior.
    return 0.594, 0.984, 0.6654


def projected_height(anchor_y, anchor_height, y, horizon_y):
    separation = max(0.03, anchor_y - horizon_y)
    factor = 1.0 + (y - anchor_y) / separation
    return anchor_height * float(np.clip(factor, 0.12, 3.5))


@dataclass(frozen=True)
class GroundProjection:
    depth_slope: float
    depth_intercept: float
    horizon_y: float

    def depth_at(self, x: float, y: float) -> float:
        """Unclipped inverse depth of the ground plane, in this exposure's units."""
        return self.depth_slope * y + self.depth_intercept

    def height_ratio(self, anchor_x: float, anchor_y: float, anchor_height: float,
                     x: float, y: float) -> float:
        """Scale from one fixed body reference. Horizontal travel preserves height."""
        # Apparent height on a level floor grows with distance in the image
        # from the horizon. The monocular network has an arbitrary disparity
        # offset, so dividing its predictions would change the body's scale.
        return projected_height(anchor_y, anchor_height, y, self.horizon_y)


def _horizon(frame: np.ndarray) -> float:
    """Estimate the horizon from a vanishing point of horizontal room lines."""
    height, width = frame.shape[:2]
    scale = min(1.0, 384 / max(height, width))
    small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    h, w = small.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 50, 125)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(12, w // 19),
                            minLineLength=max(10, w // 9), maxLineGap=max(2, w // 70))
    # The ordinary standing camera fallback matches perspective.standing_height_px.
    fallback = 0.4
    if lines is None:
        return fallback
    x1, y1, x2, y2 = lines[:, 0].astype(np.float64).T
    slopes = (y2 - y1) / np.where(x2 != x1, x2 - x1, 1e-5)
    usable = (abs(slopes) > 0.14) & (abs(slopes) < 2.5)
    if usable.sum() < 6:
        return fallback
    slopes = slopes[usable]
    offsets = y1[usable] - slopes * x1[usable]
    lengths = np.hypot(x2 - x1, y2 - y1)[usable]
    # Opposite inclinations and long independent segments must agree about
    # both coordinates. This rejects a set of parallel shelf fronts.
    xs, ys = np.meshgrid(np.linspace(.16 * w, .84 * w, 45), np.linspace(.04 * h, .62 * h, 64))
    error = np.abs(xs[..., None] * slopes + offsets - ys[..., None])
    support = np.maximum(0.0, 1 - error / max(2.0, .014 * h))
    positive = slopes > 0
    score_pos = (support[..., positive] * lengths[positive]).sum(axis=-1)
    score_neg = (support[..., ~positive] * lengths[~positive]).sum(axis=-1)
    score = np.sqrt(score_pos * score_neg)
    row, col = np.unravel_index(np.argmax(score), score.shape)
    if min(score_pos[row, col], score_neg[row, col]) < 0.33 * w:
        return fallback
    return float(ys[row, col] / h)


def fit_ground(frame: np.ndarray, inverse_depth: np.ndarray, excluded_masks=()) -> GroundProjection | None:
    """Fit the farther depth envelope of the lower image, excluding segmented objects.

    For an upright camera without substantial roll, ground disparity increases
    approximately linearly down the image. Furniture protrudes *in front* of
    that plane and cannot serve as the sample beneath the avatar's shoes.
    """
    h, w = frame.shape[:2]
    if inverse_depth.shape != (h, w) or min(h, w) < 24:
        return None
    row_ids = np.linspace(.43 * (h - 1), .98 * (h - 1), 26).astype(int)
    col_ids = np.linspace(.04 * (w - 1), .96 * (w - 1), min(w, 120)).astype(int)
    positions = np.ix_(row_ids, col_ids)
    samples = inverse_depth[positions]
    valid = np.isfinite(samples)
    for mask in excluded_masks:
        if mask is not None:
            valid &= mask[positions] < 0.15

    rows, depths = [], []
    for row, values, available in zip(row_ids, samples, valid):
        if available.sum() >= max(6, len(col_ids) // 7):
            rows.append(row / (h - 1))
            depths.append(np.percentile(values[available], 10))
    if len(rows) < 9 or rows[-1] - rows[0] < 0.32:
        return None
    y, d = np.array(rows), np.array(depths)
    gaps = y[:, None] - y[None, :]
    separated = gaps > 0.16
    slope = np.median((d[:, None] - d[None, :])[separated] / gaps[separated])
    intercept = np.median(d - slope * y)
    residual = np.abs(d - (slope * y + intercept))
    # Require support across both the lower and upper parts of the fitted band.
    accepted = residual <= max(.03, min(.065, slope * .04))
    if slope < 0.12 or accepted.sum() < max(8, int(.65 * len(rows))) or np.ptp(y[accepted]) < .32:
        return None
    slope, intercept = np.polyfit(y[accepted], d[accepted], deg=1)
    # In a headshot the bottom corners can contain a distant wall throughout.
    # Only accept a floor when the sampled rows reach into the nearer depths.
    if slope * y[accepted].max() + intercept < .28:
        return None
    return GroundProjection(float(slope), float(intercept), _horizon(frame))
