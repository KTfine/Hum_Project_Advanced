"""Camera-space geometry shared by the preview, encoder and shutter."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Placement:
    x: float
    y: float
    scale: float
    # Projected upright height / frame height. Each PNG has its own pixel calibration.
    standing_height_ratio: float
    framing: str = ""
    source: str = ""

    def already_beside(self, other, entry, face, width, height):
        """Allow ordinary sway before spending a follow request on a teleport."""
        ratio = self.standing_height_ratio / other.standing_height_ratio
        if not 0.80 <= ratio <= 1.25:
            return False
        dx, dy = (self.x - other.x) * width, (self.y - other.y) * height
        head = face.width if face else max(1, other.standing_height_ratio * height * 0.15)
        if other.framing == "全身":
            return abs(dx) <= 0.9 * head and abs(dy) <= 0.45 * head
        if entry.face_box_px:
            # Portrait feet can lie far below the frame. Compare the visible faces.
            left, top, right, bottom = entry.face_box_px
            dz = (self.standing_height_ratio - other.standing_height_ratio) * height / entry.standing_height_px
            dx += ((left + right) / 2 - entry.anchor_px[0]) * dz
            dy += ((top + bottom) / 2 - entry.anchor_px[1]) * dz
        return abs(dx) <= 0.75 * head and abs(dy) <= 0.65 * head

    @staticmethod
    def portrait_foot(y, state, face, entry, sprite_scale, height):
        """Use the visible faces if the requested foot would crop the companion's face."""
        left, top, right, bottom = entry.face_box_px
        ax, ay = entry.anchor_px
        angle = math.radians(state.rotation)
        sine, cosine = math.sin(angle), math.cos(angle)
        offset_y = sprite_scale * (sine * ((left + right) / 2 - ax) + cosine * ((top + bottom) / 2 - ay))
        radius = sprite_scale * (abs(sine) * (right - left) + abs(cosine) * (bottom - top)) / 2
        center = y * height + offset_y
        if radius <= center <= height - radius:
            return y
        padding = min(radius, height / 2)
        beside = np.clip(face.center_y + 0.50 * face.width, padding, height - padding)
        return float(np.clip((beside - offset_y) / height, -2, 3))


@dataclass
class DepartureWatch:
    """Watch the original subject's occupied head/torso area until it clears."""

    region: tuple[float, float, float, float]
    initial_coverage: float
    missing_frames: int = 0
    last_frame: np.ndarray | None = None

    @staticmethod
    def _coverage(mask, region):
        h, w = mask.shape
        x0, y0, x1, y1 = np.clip(region, 0, 1) * (w, h, w, h)
        patch = mask[int(y0):max(int(y0) + 1, round(y1)), int(x0):max(int(x0) + 1, round(x1))]
        return float(np.mean(patch[::3, ::3] >= 0.5)) if patch.size else 0.0

    @classmethod
    def at_entry(cls, mask, face, frame):
        if mask is None:
            return None
        h, w = mask.shape
        if face is not None:
            cx, cy, span = face.center_x, face.center_y, face.width
            region = ((cx - .32 * span) / w, (cy - .15 * span) / h,
                      (cx + .32 * span) / w, (cy + 1.1 * span) / h)
        else:
            # A cropped or turned face may have no landmark. Use the middle of
            # the upper silhouette, avoiding the shoulders' moving outer edges.
            ys, xs = np.nonzero(mask[::4, ::4] >= 0.5)
            if not len(xs):
                return None
            top, bottom = np.percentile(ys, (3, 85))
            upper = xs[(ys >= top) & (ys <= (top + bottom) / 2)]
            if not len(upper):
                return None
            left, right = np.percentile(upper, (30, 70))
            region = (4 * left / w, 4 * top / h, 4 * (right + 1) / w, 2 * (top + bottom) / h)
        coverage = cls._coverage(mask, region)
        return cls(region, coverage, last_frame=frame) if coverage >= 0.35 else None

    def departed(self, frame, mask):
        if mask is None or frame is self.last_frame:
            return False
        self.last_frame = frame
        empty = self._coverage(mask, self.region) < self.initial_coverage * 0.25
        self.missing_frames = self.missing_frames + 1 if empty else 0
        return self.missing_frames >= 3
