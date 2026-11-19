"""Load and render straight-alpha BGRA character illustrations."""

from pathlib import Path
import threading

import cv2
import numpy as np

from .pose import AvatarPose, load_pose_catalog


def apply_teleport_effect(sprite_bgra: np.ndarray, progress: float) -> np.ndarray:
    """Reveal the projected sprite from top to bottom with a luminous scan line."""
    if progress >= 1.0:
        return sprite_bgra

    height = sprite_bgra.shape[0]
    row = (np.arange(height, dtype=np.float32) + 0.5) / height
    # Run the band past both edges so it enters and leaves cleanly.
    band_radius = 0.075
    center = -band_radius + (1 + 2 * band_radius) * max(0.0, progress)
    distance = (center - row) / band_radius
    opacity = 0.25 + 0.75 * np.clip((distance + 1) / 2, 0, 1)
    result = sprite_bgra.copy()
    result[:, :, 3] = np.rint(sprite_bgra[:, :, 3] * opacity[:, None])

    # Tint only the narrow band. Keep straight RGB and the original alpha edges.
    band_rows = np.flatnonzero(np.abs(distance) < 1)
    if band_rows.size:
        band = slice(band_rows[0], band_rows[-1] + 1)
        strength = (0.85 * (1 - np.abs(distance[band])))[:, None, None]
        color = np.array((255, 238, 150), dtype=np.float32)
        original = sprite_bgra[band, :, :3]
        tinted = np.rint(original + strength * (color - original))
        np.copyto(result[band, :, :3], tinted, where=sprite_bgra[band, :, 3:4] > 0,
                  casting="unsafe")
    return result


class AvatarFrames:
    def __init__(self, path: Path):
        self.catalog = load_pose_catalog(path)
        self._lock = threading.Lock()
        self._cached_pose: AvatarPose | None = None
        self._cached_frame: np.ndarray | None = None

    def frame(self, pose: AvatarPose) -> np.ndarray:
        with self._lock:
            if pose != self._cached_pose or self._cached_frame is None:
                path = self.catalog.entry(pose).path
                encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
                try:
                    pixels = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                except cv2.error as exc:
                    raise ValueError(f"无法读取立绘：{path}") from exc
                if (pixels is None or pixels.ndim != 3 or pixels.shape[2] != 4
                        or pixels.dtype != np.uint8 or not np.any(pixels[:, :, 3])):
                    raise ValueError(f"立绘须包含可见像素和透明通道：{path}")
                self._cached_pose, self._cached_frame = pose, pixels
            return self._cached_frame

    def close(self) -> None:
        with self._lock:
            self._cached_pose = self._cached_frame = None
