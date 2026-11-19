"""Resolve foreground masks above the avatar in a fixed camera layer order."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from .obstacles import panel_mask


class LiveMaskSource(Protocol):
    def set_enabled(self, enabled: bool) -> None: ...
    def latest(self) -> tuple[np.ndarray, np.ndarray] | None: ...
    def mask_for_capture(self, frame: np.ndarray) -> np.ndarray: ...


class SceneLayers:
    """Back-to-front scene layers, with fixed masks and one optional live source."""

    def __init__(self, layers: Mapping[str, np.ndarray | LiveMaskSource], *, infer_obstacle: bool = False):
        # Explicit scene masks take precedence over camera-inferred panels.
        self._infer_obstacle = infer_obstacle and not any(name != "person" for name in layers)
        self.names = (("obstacle",) if self._infer_obstacle else ()) + tuple(layers)
        self._panel_cache: tuple[np.ndarray, np.ndarray | None] | None = None
        self._person_index = self.names.index("person") if "person" in self.names else None
        self._static: list[tuple[int, np.ndarray]] = []
        self.live: LiveMaskSource | None = None
        self._live_index = -1
        for index, (name, source) in enumerate(layers.items(), start=int(self._infer_obstacle)):
            if isinstance(source, np.ndarray):
                mask = np.array(source, dtype=np.float32, copy=True)
                if mask.ndim != 2 or not mask.size or not np.all((mask >= 0) & (mask <= 1)):
                    raise ValueError(f"invalid foreground mask for scene layer {name!r}")
                mask.setflags(write=False)
                self._static.append((index, mask))
            else:
                if self.live is not None:
                    raise ValueError("only one live mask source is supported")
                self.live = source
                self._live_index = index

    def foreground(
        self, frame: np.ndarray, avatar_index: int, *, capture: bool
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None, np.ndarray | None]:
        """Return a matched frame, occluders, and a person mask for ground calibration."""
        masks = [mask for index, mask in self._static if index >= avatar_index]
        person = next((mask for index, mask in self._static if index == self._person_index), None)
        occludes = self._live_index >= avatar_index
        if self.live is not None and (occludes or self._live_index == self._person_index):
            # An optional sizing reference may still be warming up in front-layer shots.
            try:
                pair = (frame, self.live.mask_for_capture(frame)) if capture else self.live.latest()
            except RuntimeError:
                if occludes:
                    raise
                pair = None
            if pair is None:
                if occludes:
                    return frame, None, None
            else:
                frame, live_mask = pair
                if occludes:
                    masks.append(live_mask)
                if self._live_index == self._person_index:
                    person = live_mask
        if (person is not None and person.shape != frame.shape[:2]) or any(
            mask.shape != frame.shape[:2] for mask in masks
        ):
            raise RuntimeError("场景遮罩与相机画面尺寸不一致，请重新设置场景图层。")
        if self._infer_obstacle and avatar_index == 0:
            # Preview, API and shutter share a snapshot when the camera has not
            # advanced. Only immutable frames can safely reuse a detected mask.
            cached = self._panel_cache
            if cached is not None and cached[0] is frame:
                obstacle = cached[1]
            else:
                obstacle = panel_mask(frame, person)
                if not frame.flags.writeable:
                    self._panel_cache = frame, obstacle
            if obstacle is not None:
                masks.append(obstacle)
        return frame, tuple(masks), person
