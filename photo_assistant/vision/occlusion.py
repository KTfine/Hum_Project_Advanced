"""Schedule matched scene inference for previews and exact-exposure photos."""
from __future__ import annotations
from concurrent.futures import Future
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scene import SceneAnalysis


class PersonOcclusion:
    """One model owner, a replaceable preview frame and exact-frame shutter requests.

    The camera must hand off a new, stable array for each frame. Callers must
    treat both arrays returned by latest() as read-only.
    """

    def __init__(self, segmenter_factory, *, max_frame_age: float = 2.5):
        self._factory = segmenter_factory
        self._max_frame_age = max_frame_age
        self._condition = threading.Condition()
        self._thread = None
        self._enabled = False
        self._closed = False
        self._generation = 0
        self._pending: tuple[np.ndarray, float] | None = None
        self._captures: dict[int, tuple[np.ndarray, Future]] = {}
        self._latest: tuple[np.ndarray, np.ndarray | SceneAnalysis] | None = None
        self._latest_time = 0.0
        self._error: str | None = None

    def _clear(self, reason: str) -> None:
        """Invalidate queued and in-flight work while holding the condition."""
        self._generation += 1
        self._pending = self._latest = None
        for _, future in self._captures.values():
            future.set_exception(RuntimeError(reason))
        self._captures.clear()
        self._condition.notify_all()

    def set_enabled(self, enabled: bool) -> None:
        with self._condition:
            if self._closed or self._enabled == enabled:
                return
            self._enabled = enabled
            self._error = None
            self._clear("合照模式已切换，已取消人物分割。")
            if enabled and self._thread is None:
                self._thread = threading.Thread(target=self._run, name="person-segmentation", daemon=True)
                self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._condition:
            if self._enabled and self._error is None:
                self._pending = frame, time.monotonic()
                self._condition.notify()

    def latest(self) -> tuple[np.ndarray, np.ndarray | SceneAnalysis] | None:
        with self._condition:
            if self._error:
                raise RuntimeError(self._error)
            if time.monotonic() - self._latest_time > self._max_frame_age:
                return None
            return self._latest

    def mask_for_capture(self, frame: np.ndarray) -> np.ndarray | SceneAnalysis:
        with self._condition:
            if self._error:
                raise RuntimeError(self._error)
            if not self._enabled:
                raise RuntimeError("人物遮挡未启用。")
            if self._latest is not None and self._latest[0] is frame:
                return self._latest[1]
            key = id(frame)
            if key not in self._captures:
                self._captures[key] = frame, Future()
                self._condition.notify()
            _, future = self._captures[key]
        return future.result()

    def _run(self) -> None:
        segmenter = None
        model_generation = None
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._captures or self._pending)
                    if self._closed:
                        return
                    generation = self._generation
                    # A shutter must retain its exact frame even as newer frames arrive.
                    frame = (next(iter(self._captures.values()))[0]
                             if self._captures else self._pending[0])
                    submitted = None
                    if self._pending is not None and self._pending[0] is frame:
                        _, submitted = self._pending
                        self._pending = None
                try:
                    if segmenter is None:
                        segmenter = self._factory()
                    elif generation != model_generation:
                        reset = getattr(segmenter, "reset_scene", None)
                        if reset is not None:
                            reset()
                    model_generation = generation
                    mask = segmenter.segment(frame)
                except Exception as exc:
                    with self._condition:
                        if generation == self._generation:
                            self._error = f"合照场景分析失败：{exc}"
                            self._clear(self._error)
                    continue
                with self._condition:
                    if generation != self._generation:
                        continue
                    if submitted is not None:
                        self._latest = frame, mask
                        self._latest_time = submitted
                    capture = self._captures.pop(id(frame), None)
                    if capture is not None:
                        capture[1].set_result(mask)
        finally:
            if segmenter is not None:
                segmenter.close()

    def close(self) -> None:
        with self._condition:
            self._enabled = False
            self._closed = True
            self._error = None
            self._clear("人物分割已关闭。")
        if self._thread is not None:
            self._thread.join()
