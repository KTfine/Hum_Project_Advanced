from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def uniform_frame_indices(total_frames: int, count: int = 8) -> list[int]:
    if total_frames < 1 or count < 1:
        raise ValueError("total_frames and count must be positive")
    return np.rint(np.linspace(0, total_frames - 1, count)).astype(int).tolist()


def select_uniform_paths(paths: list[str | Path], count: int = 8) -> list[str]:
    if not paths:
        raise ValueError("paths must not be empty")
    return [str(paths[index]) for index in uniform_frame_indices(len(paths), count)]


def read_decodable_video_frames(video_path: str | Path) -> list[np.ndarray]:
    """Decode sequentially because container frame counts can exceed readable frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot read video: {video_path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video has no decodable frames: {video_path}")
    return frames


def extract_uniform_video_frames(
    video_path: str | Path, output_dir: str | Path, *, count: int = 8,
) -> list[str]:
    frames = read_decodable_video_frames(video_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    outputs = []
    for position, frame_index in enumerate(uniform_frame_indices(len(frames), count)):
        output = target / f"{position:02d}.jpg"
        if not output.is_file() and not cv2.imwrite(str(output), frames[frame_index]):
            raise ValueError(f"failed to write frame {frame_index} from {video_path}")
        outputs.append(str(output.resolve()))
    return outputs
