"""Thread-safe avatar placement, arrival effect timing and rotation."""
from dataclasses import dataclass, replace
import threading
import time

from ..protocol import DEFAULT_TRANSITION_MS, teleport_destination
from .pose import AvatarPose, PoseCatalog


TELEPORT_EFFECT_SECONDS = 0.35
_DEPTH_SCALE = {"alongside": 1.0, "near": 1.35, "far": 0.70}


@dataclass(frozen=True)
class AvatarState:
    x: float
    y: float
    scale: float
    rotation: float
    layer_index: int = 0
    pose: AvatarPose = AvatarPose()
    camera_depth_mode: str = "alongside"


class AvatarController:
    def __init__(
        self, initial_state: AvatarState, *, scene_layer_count: int = 0,
        pose_catalog: PoseCatalog | None = None, follow_person_ground: bool = True,
        arrival_at: float | None = None,
    ) -> None:
        self.pose_catalog = pose_catalog
        self._scene_layer_count = scene_layer_count
        self._target = initial_state
        self._follow_person_ground = follow_person_ground
        self._last_ground_observation = float("-inf")
        self._teleported_at = float("-inf") if arrival_at is None else arrival_at
        self._rotation_transition: tuple[float, float, float] | None = None
        self._lock = threading.Lock()

    def _state_at_unlocked(self, now: float) -> AvatarState:
        if self._rotation_transition is None:
            return self._target
        angle, started_at, duration = self._rotation_transition
        if now >= started_at + duration:
            self._rotation_transition = None
            return self._target
        progress = max(0.0, (now - started_at) / duration)
        return replace(self._target, rotation=angle + (self._target.rotation - angle) * progress)

    def state_at(self, now: float | None = None) -> AvatarState:
        with self._lock:
            return self._state_at_unlocked(time.monotonic() if now is None else now)

    def rotation_at_target(self, rendered_angle: float) -> bool:
        with self._lock:
            return rendered_angle == self._target.rotation

    def render_state(self, now: float | None = None) -> tuple[AvatarState, bool, float]:
        """Snapshot geometry including camera depth, automatic grounding and arrival progress."""
        with self._lock:
            timestamp = time.monotonic() if now is None else now
            progress = min(1.0, max(0.0, (timestamp - self._teleported_at) / TELEPORT_EFFECT_SECONDS))
            return self._state_at_unlocked(timestamp), self._follow_person_ground, progress

    def observe_ground(self, foot_y: float, observed_at: float) -> None:
        with self._lock:
            if self._follow_person_ground and observed_at >= self._last_ground_observation:
                self._last_ground_observation = observed_at
                self._target = replace(self._target, y=foot_y)

    def update_side_placement(
        self, x: float, y: float, scale: float, *, arrival_at: float | None = None,
    ) -> None:
        """Set a computed landing point and optionally start its arrival effect."""
        with self._lock:
            self._target = replace(self._target, x=x, y=y, scale=scale)
            self._follow_person_ground = False
            if arrival_at is not None:
                self._teleported_at = arrival_at

    def apply_command(self, command, now: float | None = None) -> AvatarState | None:
        with self._lock:
            timestamp = time.monotonic() if now is None else now
            params = command.params
            if command.kind == "teleport":
                destination = teleport_destination(params)
                if destination is None:
                    raise ValueError("传送需要绝对脚点 x、y（-2～3），可选倍率 scale（0.1～3）和 depth（near/far/alongside）")
                depth = destination.pop("depth", None)
                if depth is not None:
                    destination["camera_depth_mode"] = depth
                    destination.setdefault("scale", _DEPTH_SCALE[depth])
                self._target = replace(self._target, **destination)
                self._follow_person_ground = False
                self._teleported_at = timestamp
            elif command.kind == "layer":
                index = params.get("index")
                if type(index) is not int or not 0 <= index <= self._scene_layer_count:
                    raise ValueError(f"角色图层须在 0 到 {self._scene_layer_count} 之间")
                self._target = replace(self._target, layer_index=index)
            elif command.kind == "tachie" and self.pose_catalog is not None:
                self._target = replace(self._target, pose=self.pose_catalog.select(params))
            elif command.kind == "rotate":
                current = self._state_at_unlocked(timestamp)
                angle = float(params["value"])
                if params["mode"] == "by":
                    angle += self._target.rotation
                duration = params.get("ms", DEFAULT_TRANSITION_MS) / 1000
                self._rotation_transition = (current.rotation, timestamp, duration) if duration > 0 else None
                self._target = replace(self._target, rotation=min(180.0, max(-180.0, angle)))
            else:
                return None
            return self._target
