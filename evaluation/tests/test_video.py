import numpy as np

import numpy as np

from evaluation.video import (
    extract_uniform_video_frames,
    read_decodable_video_frames,
    select_uniform_paths,
    uniform_frame_indices,
)


def test_uniform_frame_indices_include_endpoints_and_repeat_short_video():
    assert uniform_frame_indices(10, 4) == [0, 3, 6, 9]
    assert uniform_frame_indices(2, 4) == [0, 0, 1, 1]


def test_select_uniform_paths_uses_same_eight_frozen_frames():
    paths = [f"frame-{index}.jpg" for index in range(10)]
    assert select_uniform_paths(paths, count=4) == [
        "frame-0.jpg", "frame-3.jpg", "frame-6.jpg", "frame-9.jpg",
    ]


def test_video_sampling_uses_decodable_frames_not_reported_frame_count(tmp_path, monkeypatch):
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(9)]

    class FakeCapture:
        def __init__(self, _path):
            self.position = 0

        def isOpened(self):
            return True

        def read(self):
            if self.position >= len(frames):
                return False, None
            frame = frames[self.position]
            self.position += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr("evaluation.video.cv2.VideoCapture", FakeCapture)
    written = []
    monkeypatch.setattr(
        "evaluation.video.cv2.imwrite",
        lambda path, frame: written.append((path, int(frame[0, 0, 0]))) or True,
    )

    assert len(read_decodable_video_frames("reported-as-10-frames.mp4")) == 9
    outputs = extract_uniform_video_frames("reported-as-10-frames.mp4", tmp_path, count=4)

    assert len(outputs) == 4
    assert [value for _path, value in written] == [0, 3, 5, 8]
