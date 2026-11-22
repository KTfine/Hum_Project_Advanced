"""Announce a completed co-photo placement once, from the rendered camera frame."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from .protocol import CHARACTER_ACTIONS, StreamingCommandParser


def _ready_message(placement: dict) -> str:
    source = placement.get("placement_source")
    commands = StreamingCommandParser().feed("render", placement["current_tachie"])
    pose = commands[0].params if commands and commands[0].kind == "tachie" else {}
    peace = pose.get("hand") == "比耶"
    if source == "桌沿坐姿" and pose.get("legs") == "坐着":
        return "我在桌沿坐好了" + ("，正比着剪刀手。" if peace else "。")
    if source == "扶墙站位" and placement.get("wall_contact_xy"):
        return "我已经扶稳了，姿势摆好了。"
    if source == "肩后探身" and pose.get("torso") == "半身歪斜向右":
        return "我从你肩后探出身了" + ("，正比着剪刀手。" if peace else "。")
    location = {"并排站位": "我在你旁边", "桌后站位": "我在桌子后面"}.get(source)
    if peace:
        return (location or "我到位了") + "，剪刀手摆好了。"
    if location:
        return location + "，姿势摆好了。"
    return "我到位了，姿势也摆好了。"


@dataclass(frozen=True)
class _Composition:
    x: float
    y: float
    scale: float
    pose: str
    rotation: float
    layer: int
    depth: str

    def matches(self, other: _Composition | None) -> bool:
        if other is None:
            return False
        # Position is normalized; subpixel changes from a fresh solve do not
        # constitute a new placement. Rotation is measured in degrees.
        return (
            self.pose == other.pose and self.layer == other.layer and self.depth == other.depth
            and isclose(self.x, other.x, abs_tol=0.001)
            and isclose(self.y, other.y, abs_tol=0.001)
            and isclose(self.scale, other.scale, abs_tol=0.001)
            and isclose(self.rotation, other.rotation, abs_tol=0.1)
        )


class CoPhotoReadiness:
    """One camera-backed co-photo session. Frame and turn events share the event loop."""

    def __init__(self) -> None:
        self.last_action_succeeded: bool | None = None
        self._pending: str | None = "entry"
        self._turn_open = False
        self._acted = False
        self._failed = False
        self._latest: _Composition | None = None
        self._before_turn: _Composition | None = None

    def begin_turn(self) -> None:
        if self._turn_open:
            return
        self._turn_open = True
        self._acted = self._failed = False
        self._before_turn = self._latest
        if self._pending == "action":
            self._pending = None

    def action_result(self, kind: str, *, applied: bool) -> None:
        if kind in CHARACTER_ACTIONS:
            self.begin_turn()
            self.last_action_succeeded = None
            self._acted = True
            if not applied:
                self._failed = True

    def finish_turn(self) -> None:
        if not self._turn_open:
            return
        self._turn_open = False
        if self._acted:
            self.last_action_succeeded = not self._failed
        if self._failed:
            # A partial multi-command pose is never the requested final result.
            self._pending = None
        elif self._acted:
            if self._pending == "entry":
                # Entry and its requested opening pose share the first cue.
                self._before_turn = None
            self._pending = "action"

    def cancel_turn(self) -> None:
        self._turn_open = False
        self._pending = None

    def observe(self, placement: dict | None) -> str | None:
        """Consume the next real composition; missing or obscured frames do not re-arm entry."""
        if (not placement or not placement.get("avatar_visible")
                or not placement.get("pose_settled") or not placement.get("current_tachie")):
            return None
        composed = _Composition(
            placement["x"], placement["y"], placement["scale"], placement["current_tachie"],
            placement.get("rotation", 0.0), placement.get("layer_index", 0),
            placement.get("camera_depth_mode", "alongside"),
        )
        self._latest = composed
        if self._turn_open or self._pending is None:
            return None
        pending, self._pending = self._pending, None
        if pending == "action" and composed.matches(self._before_turn):
            return None
        return _ready_message(placement)
