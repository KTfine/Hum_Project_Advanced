"""Interactive photo assistant using camera frames and the DeepSeek API."""

import argparse
import asyncio
import getpass
import codecs
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import traceback
import warnings

import numpy as np

from .api import session as direct_api
from .api.client import configured_api_key, read_project_settings
from .avatar.controller import AvatarController, AvatarState
from .avatar.frames import AvatarFrames, apply_teleport_effect
from .avatar.pose import AvatarPose, WHITE_DRAGON_CATALOG
from .console import TerminalOutput
from .paths import PROJECT_ROOT
from .persona import add_persona_options, load_persona_option
from .prompts import (
    CO_PHOTO_COMMAND_PROMPT, FIXED_CAMERA_CONTEXT, LIVE_ENTRY_CONTEXT,
    MOVING_CAMERA_CONTEXT, PHOTO_GUIDANCE_PROMPT, character_name_for,
)
from .protocol import Command, StreamingCommandParser
from .readiness import CoPhotoReadiness
from .speech_output import SpeechOutput
from .voice_input import recognize_microphone
from .vision.beside import side_placement
from .vision.camera import open_camera
from .vision.framing import DepartureWatch, Placement
from .vision.ground import camera_reference, projected_height
from .vision.head import measure_face
from .vision.layers import LiveMaskSource, SceneLayers
from .vision.occlusion import PersonOcclusion
from .vision.peek import peek_placement
from .vision.perspective import (
    PORTRAIT_CHEEK_RATIO, PORTRAIT_FACE_DROP, PersonSilhouettes, matched_standing_height_px,
    person_reference, standing_height_px,
)
from .vision.scene import SceneSegmenter
from .vision.seat import find_seat
from .vision.support import WallBrace, wall_brace


MODES = ("guidance", "co-photo")
DEFAULT_CAMERA_DEVICE = 701


class RealtimeServiceError(RuntimeError):
    pass


class FrontendEventLogger:
    """Thread-safe JSONL log for protocol, parser, control, and failure events."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def record(self, event: str, **fields) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._file.write(line + "\n")

    def record_exception(self, event: str, exc: BaseException) -> None:
        self.record(
            event,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(traceback.format_exception(exc)),
        )

    def close(self) -> None:
        with self._lock:
            self._file.close()


class ConsoleTextInput:
    """One nonblocking input source and pending queue for all model sessions.

    The sender acknowledges a line only after the session accepts it. Mode
    changes therefore cannot discard a line waiting to be sent. No blocking
    input()/executor thread is created for each reconnect.
    """

    def __init__(self, stream=None, *, terminal: TerminalOutput | None = None) -> None:
        self.stream = sys.stdin if stream is None else stream
        self.terminal = terminal if terminal is not None else TerminalOutput()
        self.pending: deque[str] = deque()
        self.eof = False
        self._line = ""
        self._posix_editor = False
        self._escape_prefix = ""
        self._high_surrogate = ""
        encoding = getattr(self.stream, "encoding", None) or "utf-8"
        self._decoder = codecs.getincrementaldecoder(encoding)(errors="replace")

    def put(self, text: str) -> None:
        if text.strip():
            self.pending.append(text.strip())

    def peek(self) -> str | None:
        return self.pending[0] if self.pending else None

    def acknowledge(self, text: str) -> None:
        if self.pending and self.pending[0] == text:
            self.pending.popleft()

    async def run_microphone(
        self, stop_event: threading.Event, *, api_key: str, device: str | int | None = None,
        speaking: threading.Event | None = None,
    ) -> None:
        def on_text(text: str) -> None:
            text = text.strip()
            if text and not stop_event.is_set():
                self.put(text)
                self.terminal.line(f"[mic] {text}")

        def on_status(message: str) -> None:
            self.terminal.line(f"[mic] {message}")

        try:
            await recognize_microphone(stop_event, on_text, on_status, api_key=api_key, device=device, speaking=speaking)
        except Exception as exc:
            on_status(f"麦克风识别中断，可继续打字。{exc}")

    def _accept_text(self, text: str, *, final: bool = False) -> None:
        self._line += text
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            self.put(line)
        if final:
            self.put(self._line)
            self._line = ""
            self.eof = True

    def _read_available(self) -> bytes | None:
        """None means no data yet; b'' means EOF, including a closed pipe."""
        descriptor = self.stream.fileno()
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
            if kernel.GetFileType(handle) == 3:  # FILE_TYPE_PIPE
                available = wintypes.DWORD()
                if not kernel.PeekNamedPipe(handle, None, 0, None,
                                            ctypes.byref(available), None):
                    error = ctypes.get_last_error()
                    if error in (109, 232):  # broken / disconnected pipe
                        return b""
                    raise OSError(error, "Cannot read console input pipe")
                if not available.value:
                    return None
                return os.read(descriptor, min(available.value, 4096))
        else:
            import select

            if not select.select([descriptor], [], [], 0)[0]:
                return None
        return os.read(descriptor, 4096)

    def _edit_key(self, char: str) -> None:
        if char == "\x03":
            raise KeyboardInterrupt
        if char in ("\x04", "\x1a") and not self._line:
            self.eof = True
        elif char in ("\r", "\n") or (char == "\x04" and self._posix_editor):
            submitted = self._line
            self._line = ""
            self.put(submitted)
            self.terminal.finish_draft(submitted)
        elif char in ("\b", "\x7f") and self._line:
            self._line = self._line[:-1]
            self.terminal.update_draft(self._line)
        elif char >= " " and char != "\x7f":
            self._line += char
            self.terminal.update_draft(self._line)

    def _edit_posix_text(self, text: str) -> None:
        for char in text:
            if char == "\x1b":
                self._escape_prefix = char
            elif self._escape_prefix == "\x1b":
                self._escape_prefix = char if char in "[O" else ""
            elif self._escape_prefix:
                if "@" <= char <= "~":
                    self._escape_prefix = ""
            elif char == "\x15" and self._line:  # Ctrl+U clears the current draft
                self._line = ""
                self.terminal.update_draft("")
            else:
                self._edit_key(char)
            if self.eof:
                break

    def poll(self) -> None:
        if self.eof:
            return
        if os.name == "nt" and self.stream.isatty():
            import msvcrt

            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):
                    msvcrt.getwch()  # consume the second half of an arrow/function key
                elif "\ud800" <= char <= "\udbff":
                    self._high_surrogate = char
                else:
                    if "\udc00" <= char <= "\udfff":
                        pair = (self._high_surrogate + char).encode("utf-16-le", "surrogatepass")
                        char = pair.decode("utf-16-le", "replace")
                    self._high_surrogate = ""
                    self._edit_key(char)
                if self.eof:
                    break
            return
        chunk = self._read_available()
        if chunk is not None:
            text = self._decoder.decode(chunk, final=not chunk)
            if self._posix_editor:
                self._edit_posix_text(text)
                if not chunk:
                    self._edit_key("\n")
                    self.eof = True
            else:
                self._accept_text(text, final=not chunk)

    async def run(self, stop_event: threading.Event) -> None:
        original_tty = None
        try:
            if os.name == "posix" and self.stream.isatty() and sys.stdout.isatty():
                import termios
                import tty

                descriptor = self.stream.fileno()
                original_tty = termios.tcgetattr(descriptor)
                tty.setcbreak(descriptor, termios.TCSANOW)
                self._posix_editor = True
            while not stop_event.is_set() and not self.eof:
                try:
                    self.poll()
                except (OSError, ValueError) as exc:
                    self.terminal.line(f"[input] console input unavailable: {exc}", file=sys.stderr)
                    self.eof = True
                await asyncio.sleep(0.05)
        finally:
            if original_tty is not None:
                termios.tcsetattr(descriptor, termios.TCSANOW, original_tty)
                self._posix_editor = False
            self.terminal.finish_draft()
        if self.eof and not stop_event.is_set():
            self.terminal.line("[input] EOF; camera monitoring continues")


class SpeechQueue:
    """Keep transcript fragments ordered and send speech clauses to the player."""

    # When an ASCII separator follows a digit, wait one character to distinguish
    # a real pause from 1,200 / 2.8 / 10:30, even if the network splits there.
    _pause = re.compile(
        r'(?:[，、。！？；：!?;…\n]|(?<!\d)[,:]|(?<=\d)[,:](?=\D)|\.(?=\s))+[”’」』]*'
    )

    def __init__(self, output: SpeechOutput | None = None) -> None:
        self._items: deque[str] = deque()
        self._lock = threading.Lock()
        self.output = output
        self._unfinished = ""

    def enqueue(self, text: str) -> None:
        if text:
            with self._lock:
                self._items.append(text)

    def pop_next(self) -> str | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def read_aloud(self, text: str | None = "", final: bool = False) -> None:
        if self.output is None:
            return
        if text is None:
            self._unfinished = ""
            return
        self._unfinished += text
        start = 0
        for boundary in self._pause.finditer(self._unfinished):
            clause = self._unfinished[start:boundary.end()].strip()
            if any(char.isalnum() for char in clause):
                self.output.say(clause)
            start = boundary.end()
        self._unfinished = self._unfinished[start:]
        if final:
            if any(char.isalnum() for char in self._unfinished):
                self.output.say(self._unfinished.strip())
            self._unfinished = ""


class ModeSelection:
    """Requested and confirmed prompt mode, independent of frontend actions."""

    def __init__(self, mode: str = "guidance") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        self._mode = mode
        self._revision = 0
        self._loaded_mode: str | None = None
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self.snapshot()[0]

    def snapshot(self) -> tuple[str, int]:
        with self._lock:
            return self._mode, self._revision

    @property
    def co_photo_active(self) -> bool | None:
        """None until the first local session is ready; ON/OFF records its prompt."""
        with self._lock:
            return None if self._loaded_mode is None else self._loaded_mode == "co-photo"

    def mark_loaded(self, mode: str, revision: int) -> bool:
        with self._lock:
            if (mode, revision) != (self._mode, self._revision):
                return False
            self._loaded_mode = mode
            return True

    def select(self, mode: str) -> bool:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        with self._lock:
            if mode == self._mode:
                return False
            self._mode = mode
            self._revision += 1
            return True


def alpha_composite_at(
    canvas: np.ndarray, avatar: np.ndarray, left: int, top: int,
    *foreground_masks: np.ndarray,
) -> bool:
    """Composite the avatar and report whether any of it survives clipping and occlusion."""
    import cv2

    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("canvas must be a BGR image")
    if avatar.ndim != 3 or avatar.shape[2] != 4:
        raise ValueError("avatar must be a BGRA image")

    canvas_h, canvas_w = canvas.shape[:2]
    avatar_h, avatar_w = avatar.shape[:2]
    dst_left = max(0, left)
    dst_top = max(0, top)
    dst_right = min(canvas_w, left + avatar_w)
    dst_bottom = min(canvas_h, top + avatar_h)
    if dst_left >= dst_right or dst_top >= dst_bottom:
        return False

    src_left = dst_left - left
    src_top = dst_top - top
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)
    source = avatar[src_top:src_bottom, src_left:src_right]
    foreground = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    alpha = source[:, :, 3].astype(np.float32) / 255.0
    return _blend_sprite(canvas, foreground, alpha, dst_left, dst_top, foreground_masks)


def _blend_sprite(canvas, foreground, alpha, left, top, foreground_masks) -> bool:
    """Blend a clipped projection with the current exposure's foreground masks."""
    import cv2

    height, width = alpha.shape
    rows, cols = slice(top, top + height), slice(left, left + width)
    if foreground_masks:
        regions = (mask[rows, cols] for mask in foreground_masks)
        covered = next(regions)
        for region in regions:
            covered = np.maximum(covered, region)
        alpha = alpha * (1.0 - covered)
    destination = canvas[rows, cols]
    cv2.blendLinear(destination, foreground, 1.0 - alpha, alpha, dst=destination)
    return bool(np.any(alpha >= 1 / 255))


class CameraSource:
    """Continuously retain only the newest non-stale webcam frame."""

    def __init__(
        self,
        device_index: int,
        stop_event: threading.Event,
        *,
        device_name: str | None = None,
        max_frame_age: float = 2.5,
        person_occlusion: PersonOcclusion | None = None,
    ) -> None:
        self.device_index = device_index
        self.device_name = device_name
        self.display_label = ""
        self.stop_event = stop_event
        self.max_frame_age = max_frame_age
        self.person_occlusion = person_occlusion
        self._capture = None
        self._thread = None
        self._frame: np.ndarray | None = None
        self._frame_time = float("-inf")
        self._lock = threading.Lock()

    def start(self) -> None:
        capture, frame, self.display_label = open_camera(self.device_index, self.device_name)
        self._publish_frame(frame)
        self._capture = capture
        self._thread = threading.Thread(
            target=self._capture_loop, name="camera-capture", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            self._publish_frame(frame)

    def _publish_frame(self, frame: np.ndarray) -> None:
        # Readers share a stable snapshot even if the camera reuses its buffer.
        snapshot = frame.copy()
        snapshot.setflags(write=False)
        with self._lock:
            if self.person_occlusion is not None:
                self.person_occlusion.submit(snapshot)
            self._frame = snapshot
            self._frame_time = time.monotonic()

    def latest_frame(self) -> np.ndarray | None:
        """Return the newest read-only camera snapshot while it is fresh."""
        with self._lock:
            if (
                self._frame is None
                or time.monotonic() - self._frame_time > self.max_frame_age
            ):
                return None
            return self._frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._capture = None


@dataclass
class PhotoCapture:
    camera: CameraSource | None
    directory: Path
    character_canvas: "CanvasSource | None" = None

    def save(self) -> Path:
        """Save the camera and any visible character at the instant of capture."""
        frame = self.camera.latest_frame() if self.camera is not None else None
        if frame is None or not frame.size:
            raise RuntimeError("当前没有可用的摄像头画面")

        import cv2

        try:
            if self.character_canvas is not None:
                frame = self.character_canvas.render_capture_frame(frame)
            ok, png = cv2.imencode(".png", frame)
        except cv2.error as exc:
            raise RuntimeError("照片渲染或 PNG 编码失败") from exc
        if not ok:
            raise RuntimeError("摄像头画面 PNG 编码失败")

        self.directory.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime("photo-%Y%m%d-%H%M%S-%f-")
        output = tempfile.NamedTemporaryFile(
            dir=self.directory, prefix=prefix, suffix=".tmp", delete=False,
        )
        pending = Path(output.name)
        try:
            with output:
                output.write(png)
            return pending.replace(pending.with_suffix(".png"))
        finally:
            pending.unlink(missing_ok=True)


@dataclass
class _PendingEntrance:
    initial: AvatarState
    placement: Placement | None = None
    pose: AvatarPose | None = None
    departure: DepartureWatch | None = None


@dataclass(frozen=True)
class _ProjectedSprite:
    source: np.ndarray
    geometry: tuple
    left: int
    top: int
    color: np.ndarray
    alpha: np.ndarray


class CanvasSource:
    """Compose the selected character sprite with the camera view."""

    WINDOW_TITLE = "Photo assistant - press Q to stop"

    def __init__(
        self,
        sprite_catalog: Path,
        width: int,
        height: int,
        jpeg_quality: int,
        stop_event: threading.Event,
        *,
        preview: bool,
        person_occlusion: PersonOcclusion | None = None,
        scene_layers: dict[str, np.ndarray | LiveMaskSource] | None = None,
        infer_obstacle: bool = False,
        project_scene: bool = False,
    ) -> None:
        self.sprite_catalog = Path(sprite_catalog)
        self.avatar_frames: AvatarFrames | None = None
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.stop_event = stop_event
        self.preview = preview
        if person_occlusion is not None:
            if scene_layers is not None:
                raise ValueError("supply person_occlusion or an ordered scene_layers mapping")
            scene_layers = {"person": person_occlusion}
        self.scene_layers = SceneLayers(scene_layers or {}, infer_obstacle=infer_obstacle)
        self.controller: AvatarController | None = None
        self._thread = None
        self.error: Exception | None = None
        self._preview_camera: CameraSource | None = None
        self._show_character = True
        self._placement_lock = threading.Lock()
        self._projected_sprite: _ProjectedSprite | None = None
        self._entrance: _PendingEntrance | None = None
        self._fixed: Placement | None = None
        self._seat = None
        self._brace: WallBrace | None = None
        self._departure: DepartureWatch | None = None
        self._pose_override: AvatarPose | None = None
        self._projection_anchor: Placement | None = None
        self._body_horizon: float | None = None
        self._project_scene = project_scene
        self._manual_layer = False
        self._revision = 0
        # Distinguish a newer user/view request from automatic entrance or pose changes.
        self._input_revision = 0

    def entrance_state(self, camera: CameraSource | None) -> AvatarState:
        pose = self.avatar_frames.catalog.entry_pose
        if camera is not None and self.scene_layers.names:
            return AvatarState(0.5, 0.95, 1.0, 0.0, layer_index=0, pose=pose)
        if camera is None:
            return AvatarState(0.5, 0.95, 1.0, 0.0, pose=pose)
        return replace(self.controller.state_at(), pose=pose)

    def _controller_for(self, state: AvatarState, *, follow_ground: bool = True,
                        arrival_at: float | None = None) -> AvatarController:
        return AvatarController(state, scene_layer_count=len(self.scene_layers.names),
                                pose_catalog=self.avatar_frames.catalog, follow_person_ground=follow_ground,
                                arrival_at=arrival_at)

    def configure_view(self, camera: CameraSource | None, *, show_character: bool,
                       entrance: AvatarState | None = None) -> None:
        """Begin the confirmed session's entrance before making its character visible."""
        if entrance is not None:
            self.avatar_frames.frame(entrance.pose)
        with self._placement_lock:
            self._revision += 1
            self._input_revision += 1
            self._preview_camera = camera
            self._show_character = show_character
            if entrance is not None:
                self._seat = None
                self._brace = None
                self._projection_anchor = self._body_horizon = None
                self._manual_layer = False
                self._fixed = self._departure = None
                self._pose_override = None
                wait_for_edge = camera is not None and bool(self.scene_layers.names)
                self.controller = self._controller_for(
                    entrance, arrival_at=None if wait_for_edge else time.monotonic()
                )
                self._entrance = _PendingEntrance(entrance) if wait_for_edge else None
            elif not show_character:
                self._seat = None
                self._brace = None
                self._projection_anchor = self._body_horizon = None
                self._manual_layer = False
                self._entrance = None
                self._fixed = self._departure = None
                self._pose_override = None
        if self.scene_layers.live is not None:
            self.scene_layers.live.set_enabled(camera is not None and show_character)

    def render_preview_frame(self) -> np.ndarray:
        camera = self._preview_camera
        frame = camera.latest_frame() if camera is not None else None
        return self.render_frame(camera_frame=frame, include_character=self._show_character, presenting=True)

    def render_capture_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.render_frame(camera_frame=frame, capture=True)

    def start(self) -> None:
        self.avatar_frames = AvatarFrames(self.sprite_catalog)
        initial = AvatarState(0.5, 0.95, 1.0, 0.0, pose=self.avatar_frames.catalog.default_pose)
        self.avatar_frames.frame(initial.pose)
        self.controller = self._controller_for(initial)
        if self.preview:
            self._thread = threading.Thread(target=self._preview_loop, name="canvas-preview", daemon=True)
            self._thread.start()

    def apply_expression(self, command: Command) -> AvatarState | None:
        """Change only the face on a visible pose, preserving its placement and entry behavior."""
        if command.kind != "tachie":
            raise ValueError("表情切换需要 tachie 指令")
        pose = AvatarPose.from_tags(command.params)
        with self._placement_lock:
            if not self._show_character or self._entrance is not None or self.controller is None:
                return None
            state = self.controller.state_at()
            if replace(state.pose, face=pose.face) != pose:
                return None
            if state.pose == pose:
                return state
            catalog = self.avatar_frames.catalog
            target = catalog.entry(pose)
            if self._seat is not None or self._brace is not None:
                current = catalog.entry(state.pose)
                contact = "seat_contact_px" if self._seat is not None else "wall_contact_px"
                if (current.anchor_px != target.anchor_px or current.standing_height_px != target.standing_height_px
                        or getattr(current, contact) != getattr(target, contact)):
                    return None
            self.avatar_frames.frame(pose)
            state = self.controller.apply_command(command)
            if self._pose_override is not None:
                self._pose_override = pose
            if self._brace is not None:
                self._brace = replace(self._brace, pose=pose)
            self._revision += 1
            return state

    def apply_command(self, command: Command) -> AvatarState | None:
        if self.controller is None:
            raise RuntimeError("canvas has not been started")
        if command.kind in ("peek", "sit", "brace"):
            return self._scene_pose_once(command)
        if command.kind == "stand":
            command = Command("tachie", vars(self.avatar_frames.catalog.standing_pose))
        if command.kind == "follow":
            if command.params:
                raise ValueError("跟到身边使用无参数的 <follow>")
            return self._follow_once()
        if command.kind == "tachie":
            # Decode before committing the switch, so a damaged PNG cannot replace a working pose.
            try:
                pose = AvatarPose.from_tags(command.params)
                self.avatar_frames.frame(pose)
            except OSError as exc:
                raise ValueError(f"无法读取立绘：{exc}") from exc
            if (self._project_scene and self._show_character
                    and self.avatar_frames.catalog.entry(pose).wall_contact_px is not None):
                with self._placement_lock:
                    if self._brace is not None and self._brace.pose == pose:
                        return self.controller.state_at()
                return self._scene_pose_once(Command("brace", {}), requested_pose=pose)
            if (pose.legs == "坐着" and self._project_scene and self._show_character
                    and getattr(self.avatar_frames.catalog.entry(pose), "seat_contact_px", None) is not None):
                # A seated artwork change should target the same supporting edge.
                target = {}
                with self._placement_lock:
                    state = self.controller.state_at()
                    if self._seat is not None:
                        if state.pose == pose:
                            return state
                        x, y = self._seat.support_xy
                        target = {"x": x, "y": y}
                return self._scene_pose_once(Command("sit", target), requested_pose=pose)
        with self._placement_lock:
            target = self.controller.apply_command(command)
            if target is None:
                return None
            self._revision += 1
            self._input_revision += 1
            if command.kind == "layer":
                self._manual_layer = True
            if self._brace is not None and (command.kind in ("tachie", "teleport")
                                            or (command.kind == "rotate" and command.params["value"] != 0)):
                self._brace = None
                if command.kind != "tachie":
                    self._pose_override = self.avatar_frames.catalog.standing_pose
                    target = self.controller.apply_command(Command("tachie", vars(self._pose_override)))
                if self._fixed is not None:
                    self._fixed = replace(self._fixed, source="指定构图")
            if command.kind == "tachie":
                self._pose_override = target.pose
                self._departure = None
                if self._seat is not None and target.pose.legs != "坐着":
                    x, y = self._seat.floor_xy
                    self._fixed = self._seat = None
                    target = self.controller.apply_command(Command("teleport", {"x": x, "y": y}))
            if command.kind == "teleport":
                if self._seat is not None:
                    self._seat = None
                    self._pose_override = self.avatar_frames.catalog.standing_pose
                    target = self.controller.apply_command(Command("tachie", vars(self._pose_override)))
                if self._manual_layer and self._project_scene:
                    self._manual_layer = False
                    target = self.controller.apply_command(Command("layer", {"index": 0}))
                self._entrance = self._fixed = self._departure = None
            return target

    def _scene_pose_once(self, command, requested_pose=None):
        """Solve the requested pose against one exposure, then commit pose and placement together."""
        with self._placement_lock:
            camera = self._preview_camera
            if camera is None or not self._show_character:
                return None
            self._input_revision += 1
            request = self._input_revision
            state = self.controller.state_at()
            anchor = self._projection_anchor or self._fixed
            current_peek = (self._fixed if self._fixed and self._fixed.source in ("自动侧探", "肩后探身")
                            and not self._manual_layer and self._seat is None and state.rotation == 0 else None)
            horizon = self._body_horizon
        frame = camera.latest_frame()
        if frame is None:
            return None
        frame, masks, person, scene = self.scene_layers.resolve(frame, 0, capture=True)
        if masks is None:
            return None
        catalog = self.avatar_frames.catalog
        seat = None
        brace = None
        if command.kind == "sit":
            if scene is None:
                return None
            pose = requested_pose or next((p for p, e in catalog.entries.items() if e.seat_contact_px), None)
            if pose is None:
                raise ValueError("当前立绘库没有带支撑点的坐姿")
            self.avatar_frames.frame(pose)
            seat = find_seat(frame, scene, catalog.entry(pose), anchor,
                             target_x=command.params.get("x"), target_y=command.params.get("y"),
                             distance=command.params.get("distance", command.params.get("table")))
            if seat is None:
                return None
            fixed = seat.placement
        elif command.kind == "brace":
            entries = ((p, e) for p, e in catalog.entries.items()
                       if e.wall_contact_px is not None and (requested_pose is None or p == requested_pose))
            brace = wall_brace(frame, scene, entries, anchor, target_x=command.params.get("x"))
            if brace is None:
                return None
            fixed, pose = brace.placement, brace.pose
            self.avatar_frames.frame(pose)
        else:
            pose = state.pose if state.pose.torso == "半身歪斜向右" and state.pose.legs == "站着" else catalog.entry_pose
            if horizon is None:
                horizon = scene.ground.horizon_y if scene is not None and scene.ground is not None else .4
            fixed = peek_placement(frame, person, masks, scene, self.avatar_frames.frame(pose), catalog.entry(pose),
                                   anchor, horizon, current_peek if pose == state.pose else None)
            if fixed is None:
                return None
        with self._placement_lock:
            if request != self._input_revision:
                return None
            if fixed is current_peek:
                return self.controller.state_at()
            self._revision += 1
            self._fixed, self._seat, self._pose_override = fixed, seat, pose
            self._brace = brace
            self._entrance = self._departure = None
            self._manual_layer = False
            if anchor is not None and self._projection_anchor is None:
                self._projection_anchor = anchor
            self.controller.apply_command(Command("layer", {"index": 0}))
            self.controller.apply_command(Command("rotate", {"mode": "to", "value": 0., "ms": 0}))
            self.controller.apply_command(Command("tachie", vars(pose)))
            return self.controller.apply_command(Command("teleport", {
                "x": fixed.x, "y": fixed.y, "scale": fixed.scale,
            }))

    def render_frame(
        self, now: float | None = None, *, camera_frame: np.ndarray | None = None,
        include_character: bool = True, capture: bool = False, presenting: bool = False,
    ) -> np.ndarray:
        frame, _ = self._compose_frame(now, camera_frame=camera_frame, include_character=include_character,
                                       capture=capture, presenting=presenting)
        return frame

    @staticmethod
    def _framing(reference):
        if reference is None:
            return ""
        if not reference.estimated:
            return "全身"
        visible = (1 - reference.foot_y + reference.height) / reference.height
        return "头肩近景" if visible < 0.46 else "下半身出框"

    @staticmethod
    def _standing_height(entry, face, reference, height, y):
        cheeks = entry.face_box_px
        matched = (matched_standing_height_px(face.width, cheeks[2] - cheeks[0], entry.standing_height_px)
                   if face is not None and cheeks is not None else None)
        return standing_height_px(reference, height, y, fallback_height=height * .72, matched_height_px=matched)

    def _beside(self, frame, masks, person, state, *, entering, pose_override=None, scene=None):
        height, width = frame.shape[:2]
        silhouettes = PersonSilhouettes.from_mask(person)
        face = measure_face(frame, person, silhouettes=silhouettes)
        reference = person_reference(person, face.center_x / width if face else state.x,
                                     face.center_y / height if face else state.y, silhouettes=silhouettes)
        catalog = self.avatar_frames.catalog
        framing = self._framing(reference)
        if entering and reference is None and face is None and scene is not None and scene.ground is not None:
            # Place the feet on the floor just beyond a detected table. Its visible
            # mask can end at the tabletop, far above the hidden ground contact.
            tables = [(obj, info) for obj, info in zip(scene.objects, scene.descriptions)
                      if obj.label == "dining table" and "behind_y" in info]
            if tables:
                _, table = max(tables, key=lambda item: np.count_nonzero(item[0].mask > 0.5))
                x, y = table["center_x"], table["behind_y"]
                ratio = scene.ground.height_ratio(*camera_reference(), x, y)
                return Placement(x, y, 1.0, ratio, "", "桌后站位"), catalog.standing_pose, None
        if entering:
            pose = catalog.standing_pose if framing == "全身" else catalog.entry_pose
        else:
            pose = pose_override or (catalog.standing_pose if framing == "全身" else state.pose)
        entry = catalog.entry(pose)
        if not entering and reference is None and face is None:
            return None
        opening = side_placement(person, self.avatar_frames.frame(pose), entry.anchor_px, entry.standing_height_px,
                                 foreground_masks=masks, face_box_px=entry.face_box_px, head=face,
                                 preferred_x=None if entering else state.x, subject_reference=reference,
                                 scene=scene, silhouettes=silhouettes)
        if opening is not None:
            x, y, scale = opening.x, opening.y, opening.scale
            if face is None or entry.face_box_px is None:
                reference = person_reference(person, x, y, silhouettes=silhouettes)
                framing = self._framing(reference)
            size = scale * self._standing_height(entry, face, reference, height, y)
        elif not entering and framing != "全身" and face is not None and entry.face_box_px:
            # A front layer or upright pose may offer no occluded flank. Put its
            # face beside the subject while preserving the selected layer and pose.
            left, top, right, bottom = entry.face_box_px
            size = matched_standing_height_px(face.width, right - left, entry.standing_height_px,
                                              cheek_ratio=PORTRAIT_CHEEK_RATIO)
            zoom = size / entry.standing_height_px
            sx = face.center_x
            offset = face.width * 1.25
            choices = [v for v in (sx - offset, sx + offset) if .5 * face.width <= v <= width - .5 * face.width]
            if not choices:
                return None
            target_x = min(choices, key=lambda v: abs(v - width * state.x))
            x = (target_x - ((left + right) / 2 - entry.anchor_px[0]) * zoom) / width
            y = (face.center_y + PORTRAIT_FACE_DROP * face.width
                 - ((top + bottom) / 2 - entry.anchor_px[1]) * zoom) / height
            if not (-2 <= x <= 3 and -2 <= y <= 3):
                return None
            scale = size / self._standing_height(entry, face, reference, height, y)
        else:
            return None
        source = ("并排站位" if framing == "全身" else "自动侧探") if entering else "按指令跟随"
        return Placement(x, y, scale, size / height, framing, source), pose, face

    def _follow_once(self):
        """Sample the current scene once. A missing subject leaves no pending request."""
        with self._placement_lock:
            camera = self._preview_camera
            state = self.controller.state_at()
            slot = 0 if self._project_scene else state.layer_index
            if camera is None or not self._show_character:
                return None
            self._input_revision += 1
            request = self._input_revision
            override = self._pose_override
            leaving_pose = self._seat is not None or self._brace is not None or (
                self._fixed is not None and self._fixed.source == "肩后探身")
            if leaving_pose:
                override = None
        frame = camera.latest_frame()
        if frame is None:
            return None
        frame, masks, person, scene = self.scene_layers.resolve(frame, slot, capture=True)
        if masks is None or person is None:
            return None
        if scene is not None and slot == 0:
            masks = (*masks, *(obj.mask for obj in scene.objects))
        height, width = frame.shape[:2]
        # Reuse this exact snapshot if an automatic entrance or straighten-up
        # completes during inference. Only a newer input can cancel the command.
        while True:
            beside_state = replace(state, pose=self.avatar_frames.catalog.entry_pose) if leaving_pose else state
            solved = self._beside(frame, masks, person, beside_state, entering=False, pose_override=override, scene=scene)
            with self._placement_lock:
                if request != self._input_revision:
                    return None
                current = self.controller.state_at()
                if (current.x, current.y, current.pose) != (state.x, state.y, state.pose):
                    state = current
                    continue
                if solved is None:
                    return None
                placement, pose, face = solved
                entry = self.avatar_frames.catalog.entry(pose)
                fixed = self._fixed or (self._entrance.placement if self._entrance is not None else None)
                if (not leaving_pose and fixed is not None and state.pose == pose
                        and fixed.already_beside(placement, entry, face, width, height)):
                    if self._project_scene and self._manual_layer:
                        self._manual_layer = False
                        self._revision += 1
                        return self.controller.apply_command(Command("layer", {"index": 0}))
                    return current
                self._revision += 1
                self._entrance = self._departure = None
                self._seat = None
                self._brace = None
                if leaving_pose:
                    self._pose_override = None
                if override is None and pose.torso == "半身歪斜向右" and placement.framing != "全身":
                    self._departure = DepartureWatch.at_entry(person, face, frame)
                self._fixed = placement
                self._projection_anchor = placement
                if scene is not None and scene.ground is not None:
                    self._body_horizon = scene.ground.horizon_y
                if self._project_scene:
                    self._manual_layer = False
                    self.controller.apply_command(Command("layer", {"index": 0}))
                if pose != state.pose:
                    self.controller.apply_command(Command("tachie", vars(pose)))
                return self.controller.apply_command(Command("teleport", {
                    "x": placement.x, "y": placement.y, "scale": placement.scale,
                }))

    def _compose_frame(self, now=None, **options) -> tuple[np.ndarray, dict | None]:
        image, placement, raw = self.compose_view(now, **options)
        return (image.copy() if image is raw else image), placement

    def compose_view(
        self, now: float | None = None, *, camera_frame: np.ndarray | None = None,
        include_character: bool = True, capture: bool = False, presenting: bool = False,
    ) -> tuple[np.ndarray, dict | None, np.ndarray | None]:
        """Return the view, its placement, and the matching camera exposure before compositing."""
        if camera_frame is None:
            canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        if not include_character:
            return (canvas if camera_frame is None else camera_frame), None, camera_frame
        if self.controller is None or self.avatar_frames is None:
            raise RuntimeError("canvas has not been started")
        # Preserve the shutter's original pose and layer while segmentation runs.
        with self._placement_lock:
            controller, entrance, fixed = self.controller, self._entrance, self._fixed
            revision = self._revision
            anchor, manual_layer = self._projection_anchor, self._manual_layer
            seat = self._seat
            brace = self._brace
            body_horizon = self._body_horizon
            override = self._pose_override
            composing = self._preview_camera is not None and self._show_character
            state, follow_ground, arrival = controller.render_state(now)
            needs_opening = entrance is not None and entrance.placement is None
        solution = watch = None
        masks, person = (), None
        scene = None
        background = None
        if camera_frame is not None:
            latest_frame = camera_frame
            camera_frame, masks, person, scene = self.scene_layers.resolve(
                camera_frame, state.layer_index, capture=capture, seat=None if manual_layer else seat)
            if presenting and latest_frame.shape != camera_frame.shape:
                return latest_frame.copy(), None, latest_frame
            # The fixed-camera worker updates the person separately from the room.
            # Display that person's own exposure so a moving hand cannot leave its mask behind.
            live_background = presenting and (scene is None or not scene.fixed_camera)
            background = latest_frame if live_background else camera_frame
            canvas = background.copy()
            if masks is None:
                return canvas, None, background
            if needs_opening:
                geometry_masks = masks
                if scene is not None and state.layer_index == 0:
                    geometry_masks = (*geometry_masks, *(obj.mask for obj in scene.objects))
                if entrance.initial.layer_index != state.layer_index:
                    # Solve behind the original flank, but draw using the explicitly
                    # requested foreground slot and this shutter's exact image.
                    _, geometry_masks, _ = self.scene_layers.foreground(camera_frame, entrance.initial.layer_index,
                                                                       capture=True)
                if geometry_masks is not None:
                    solution = self._beside(camera_frame, geometry_masks, person, entrance.initial, entering=True,
                                            scene=scene)
                    if solution is not None and solution[1] != self.avatar_frames.catalog.standing_pose:
                        watch = DepartureWatch.at_entry(person, solution[2], camera_frame)
        elif composing:
            return canvas, None, None
        height, width = canvas.shape[:2]
        catalog = self.avatar_frames.catalog
        with self._placement_lock:
            current = revision == self._revision and controller is self.controller
            if current:
                fixed = self._fixed
            if scene is not None and (body_horizon is None or needs_opening):
                if scene.ground is not None:
                    body_horizon = scene.ground.horizon_y
                elif body_horizon is None:
                    body_horizon = 0.4
                if current:
                    self._body_horizon = body_horizon
                    self._project_scene = True
            if entrance is not None:
                # The first matched render chooses the geometry for *all* consumers.
                # Only the visible preview starts the entrance animation.
                if entrance.placement is None:
                    if solution is None:
                        if scene is None:
                            return canvas, None, background
                        return canvas, {"avatar_visible": False, "scene_objects": scene.descriptions}, background
                    entrance.placement, entrance.pose, _ = solution
                    entrance.departure = watch
                fixed = entrance.placement
                state = replace(state, x=fixed.x, y=fixed.y, scale=fixed.scale, pose=override or entrance.pose)
                arrival = 0.0
                if current and self._entrance is entrance:
                    reveal = presenting or not self.preview
                    started_at = (time.monotonic() if now is None else now) if reveal else None
                    controller.update_side_placement(fixed.x, fixed.y, fixed.scale, arrival_at=started_at)
                    if controller.state_at(now).pose != state.pose:
                        controller.apply_command(Command("tachie", vars(state.pose)), now=now)
                    self._projection_anchor = fixed
                    if reveal:
                        self._fixed = fixed
                        self._departure = entrance.departure if override is None else None
                        self._entrance = None
                        self._revision += 1
            elif fixed is not None:
                # Occlusion still uses the current image. Only the one-time
                # departure gesture observes the person after entry.
                if (current and self._departure is not None and (presenting or not self.preview)
                        and self._departure.departed(camera_frame, person)):
                    state = controller.apply_command(Command("tachie", vars(catalog.standing_pose)), now=now)
                    self._fixed = fixed = replace(fixed, source="原地站立")
                    self._departure = None
                    self._revision += 1
            else:
                entry = catalog.entry(state.pose)
                silhouettes = PersonSilhouettes.from_mask(person)
                face = measure_face(camera_frame, person, silhouettes=silhouettes)
                reference = person_reference(person, state.x, state.y, silhouettes=silhouettes)
                y = reference.foot_y if follow_ground and reference else state.y
                # A portrait may provide a reliable body size from the face but
                # no visible feet. Project that measurement from a stable point.
                face_reference = (composing and not follow_ground and face is not None and entry.face_box_px
                                  and (reference is None or reference.estimated))
                if scene is not None or anchor is not None or face_reference:
                    if anchor is None:
                        ax, ay, ah = camera_reference()
                        if reference is not None:
                            ay = reference.foot_y
                        if reference is not None or face_reference:
                            if face is not None:
                                ax = face.center_x / width
                            ah = self._standing_height(entry, face, reference, height, ay) / height
                        anchor = Placement(ax, ay, 1.0, ah)
                        if current and composing:
                            self._projection_anchor = anchor
                    horizon = body_horizon if body_horizon is not None else 0.4
                    size = (height * projected_height(anchor.y, anchor.standing_height_ratio, y, horizon)
                            * state.scale / anchor.scale)
                else:
                    size = self._standing_height(entry, face, reference, height, y) * state.scale
                framing = self._framing(reference)
                fixed = Placement(state.x, y, state.scale, size / height,
                                  framing, "指定构图" if framing or scene is not None else "")
                if current and composing:
                    self._fixed = fixed
                    controller.update_side_placement(fixed.x, fixed.y, fixed.scale)
                elif follow_ground and reference is not None:
                    controller.observe_ground(y, time.monotonic())
        entry = catalog.entry(state.pose)
        size = fixed.standing_height_ratio * height
        if scene is not None and seat is not None and state.layer_index == 0 and not manual_layer:
            furniture, foreground_person = scene.projected_foreground(fixed.x, fixed.y, seat=seat)
            masks = (*(mask for mask in masks if mask is not person), furniture, foreground_person)
        elif scene is not None:
            person_cover = scene.foreground_person
            if state.layer_index == 0:
                cover = scene.object_cover
                if scene.ground is not None and not manual_layer:
                    depth_order_person = brace is not None or fixed.source in ("指定构图", "桌后站位")
                    cover, person_cover = scene.projected_foreground(
                        fixed.x, fixed.y, depth_order_person=depth_order_person)
                if scene.objects:
                    masks = (*masks, cover)
            masks = tuple(person_cover if mask is person else mask for mask in masks)
        visible = self._draw_sprite(canvas, self.avatar_frames.frame(state.pose), entry.anchor_px,
                                    fixed.x, fixed.y, size / entry.standing_height_px, state.rotation,
                                    masks, arrival=arrival if presenting else 1.0)
        placement = {"x": fixed.x, "y": fixed.y, "scale": fixed.scale}
        if seat is not None:
            placement["seat_contact_xy"] = seat.support_xy
            if seat.host_label is not None:
                placement["seat_host"] = seat.host_label
        if brace is not None:
            placement["wall_contact_xy"] = brace.contact_xy
        if scene is not None:
            placement.update(scene_objects=scene.descriptions, standing_height_ratio=round(fixed.standing_height_ratio, 4))
        if fixed.source:
            placement["placement_source"] = fixed.source
        if fixed.framing:
            placement["person_framing"] = fixed.framing
        if composing:
            with self._placement_lock:
                settled = (revision == self._revision and controller is self.controller and arrival >= 1
                           and controller.rotation_at_target(state.rotation))
            placement.update(current_tachie=state.pose.command(), avatar_visible=visible,
                             pose_settled=settled, rotation=state.rotation, layer_index=state.layer_index)
        return canvas, placement, background

    def _draw_sprite(self, canvas, source, anchor, x, y, scale, rotation, masks, *, arrival=1.0) -> bool:
        """Map the sprite's ground point to the camera and warp only its visible area."""
        import cv2

        height, width = canvas.shape[:2]
        foot_x, foot_y = round(x * width), round(y * height)
        geometry = (width, height, foot_x, foot_y, scale, rotation, anchor)
        cached = self._projected_sprite
        if arrival >= 1.0 and cached is not None and cached.source is source and cached.geometry == geometry:
            return _blend_sprite(canvas, cached.color, cached.alpha, cached.left, cached.top, masks)
        sprite_h, sprite_w = source.shape[:2]
        ax, ay = anchor
        matrix = cv2.getRotationMatrix2D((ax, ay), -rotation, scale)
        matrix[:, 2] += (foot_x - ax, foot_y - ay)
        corners = np.array([(0, 0), (sprite_w, 0), (0, sprite_h), (sprite_w, sprite_h)])
        projected = corners @ matrix[:, :2].T + matrix[:, 2]
        left = max(0, math.floor(projected[:, 0].min()))
        right = min(width, math.ceil(projected[:, 0].max()))
        top = max(0, math.floor(projected[:, 1].min()))
        bottom = min(height, math.ceil(projected[:, 1].max()))
        if right <= left or bottom <= top:
            return False
        matrix[:, 2] -= (left, top)
        sprite = cv2.warpAffine(source, matrix, (right - left, bottom - top))
        if arrival < 1.0:
            sprite = apply_teleport_effect(sprite, arrival)
            return alpha_composite_at(canvas, sprite, left, top, *masks)
        color = cv2.cvtColor(sprite, cv2.COLOR_BGRA2BGR)
        alpha = sprite[:, :, 3].astype(np.float32) / 255.0
        # Each reader keeps its own reference even if another view replaces the cache.
        self._projected_sprite = _ProjectedSprite(source, geometry, left, top, color, alpha)
        return _blend_sprite(canvas, color, alpha, left, top, masks)

    def latest_jpeg(
        self, camera_frame: np.ndarray | None = None, *, include_character: bool = True
    ) -> bytes:
        jpeg, _ = self.jpeg_with_placement(camera_frame, include_character=include_character)
        return jpeg

    def jpeg_with_placement(
        self, camera_frame: np.ndarray | None = None, *, include_character: bool = True,
    ) -> tuple[bytes, dict | None]:
        frame, placement = self.frame_with_placement(camera_frame, include_character=include_character)
        return self.encode_jpeg(frame), placement

    def frame_with_placement(
        self, camera_frame: np.ndarray | None = None, *, include_character: bool = True,
    ) -> tuple[np.ndarray, dict | None]:
        """Return the model image and matching placement. Treat the image as read-only."""
        image, placement, _ = self.compose_view(camera_frame=camera_frame, include_character=include_character)
        return image, placement

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("failed to JPEG-encode canvas")
        return encoded.tobytes()

    def _preview_loop(self) -> None:
        import cv2

        try:
            while not self.stop_event.is_set():
                cv2.imshow(self.WINDOW_TITLE, self.render_preview_frame())
                if cv2.waitKey(16) & 0xFF in (ord("q"), ord("Q")):
                    self.stop_event.set()
                    break
        except Exception as exc:
            self.error = exc
            self.stop_event.set()
        finally:
            cv2.destroyAllWindows()

    def close(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.avatar_frames is not None:
            self.avatar_frames.close()
            self.avatar_frames = None
        self._projected_sprite = None


def decode_output_event(event: dict) -> tuple[str, object]:
    """Convert an API session event into the client-facing streaming value."""
    event_type = event.get("type")
    if event_type == "error":
        error = event.get("error") or {}
        code = error.get("code", "unknown_error")
        message = error.get("message", "no message")
        raise RealtimeServiceError(f"{code}: {message}")
    if event_type == "session.closed":
        return "closed", event.get("reason", "unknown")
    if event_type == "response.done":
        return "done", event.get("text", "")
    if event_type == "response.interrupted":
        return "interrupted", event.get("reason", "new_input")
    if event_type != "response.output.delta":
        return "other", event
    kind = event.get("kind")
    if kind == "text":
        return "text", event.get("text", "")
    return "other", event


async def send_realtime_inputs(
    session, canvas: CanvasSource, camera: CameraSource | None, stop_event: threading.Event,
    *, event_log: FrontendEventLogger | None = None, include_character: bool = True,
    text_input: ConsoleTextInput | None = None, report_placement: bool = False,
    readiness: CoPhotoReadiness | None = None,
    terminal: TerminalOutput | None = None,
    observation_interval: float | None = None,
) -> None:
    """Give user input priority and send the current camera view on each idle observation tick."""
    if terminal is None:
        terminal = TerminalOutput()
    entry_status = None
    started_at = time.monotonic()
    entry_hint_after = started_at + 3
    next_observation_at = started_at + observation_interval if observation_interval is not None else None
    pending_frame: tuple[np.ndarray, dict | None] | None = None
    while not stop_event.is_set():
        frame = camera.latest_frame() if camera is not None else None
        text = text_input.peek() if text_input is not None else None
        image = placement = None
        if frame is not None or (camera is None and text is not None):
            image, placement, _ = canvas.compose_view(camera_frame=frame, include_character=include_character)
        if report_placement and camera is not None:
            placed = placement is not None and placement.get("avatar_visible", True)
            if placed:
                status, message = "placed", None
                entry_hint_after = time.monotonic() + 3
            elif frame is None:
                status, message = "waiting_for_camera", "正在等待摄像头画面。"
            elif placement and "x" in placement:
                status, message = "out_of_view", "当前合照画面中看不到角色。"
            else:
                status, message = "waiting_for_placement", "正在寻找合照位置。"
            if status != entry_status and (placed or time.monotonic() >= entry_hint_after):
                if message:
                    terminal.line(f"[合照] {message}")
                if event_log is not None:
                    event_log.record("canvas.entry", status=status, placement=placement)
                entry_status = status
        if readiness is not None:
            if text is not None:
                readiness.begin_turn()
            elif placement is not None:
                ready_event = readiness.observe(placement)
                if ready_event is not None:
                    placement["readiness_event"] = ready_event
                    terminal.line("[合照] 角色已就位。")
                    if event_log is not None:
                        event_log.record("canvas.ready", reason=ready_event)
            if placement is None:
                placement = {"camera_frame_available": frame is not None, "avatar_visible": False}
            if readiness.last_action_succeeded is not None:
                placement["last_action_succeeded"] = readiness.last_action_succeeded
        if image is not None:
            pending_frame = image, placement
        elif placement is not None and placement.get("camera_frame_available") is False:
            pending_frame = None

        jpeg = None
        if text is not None and pending_frame is not None:
            image, placement = pending_frame
            jpeg = canvas.encode_jpeg(image)
            pending_frame = None  # The session now caches the encoded image for this turn.
        if report_placement and placement is not None:
            session.submit_frame(jpeg, text, placement=placement)
        else:
            session.submit_frame(jpeg, text)
        if text is not None:
            text_input.acknowledge(text)
        if observation_interval is not None:
            now = time.monotonic()
            if text is not None or now >= next_observation_at:
                # Skip missed or busy ticks. A user submission starts a new interval.
                next_observation_at = now + observation_interval
                if text is None and frame is not None and image is not None and session.can_observe:
                    observation_jpeg = canvas.encode_jpeg(image)
                    submitted = session.submit_observation(observation_jpeg, placement=placement)
                    if submitted and event_log is not None:
                        event_log.record("api.observation", jpeg_bytes=len(observation_jpeg), character_placement=placement)
        if event_log is not None:
            frame_info = {"character_placement": placement} if report_placement else {}
            event_log.record("api.input", jpeg_bytes=len(jpeg) if jpeg else 0, text=text, **frame_info)
        delay = 1.0
        if next_observation_at is not None:
            delay = min(delay, max(0.0, next_observation_at - time.monotonic()))
        await asyncio.sleep(delay)


def dispatch_model_commands(
    commands: list[Command],
    canvas: CanvasSource,
    speech_queue: SpeechQueue,
    event_log: FrontendEventLogger | None = None,
    *,
    verbose: bool = False,
    mode_selection: ModeSelection | None = None,
    photo_capture: PhotoCapture | None = None,
    terminal: TerminalOutput | None = None,
    brace_missing: bool = False,
    readiness: CoPhotoReadiness | None = None,
    speech_already_streamed: bool = False,
) -> bool:
    """Carry a failed wall placement across chunks of this response so its shutter can wait."""
    standalone = terminal is None
    if standalone:
        terminal = TerminalOutput()

    def emit_speech():
        for text in iter(speech_queue.pop_next, None):
            terminal.speech(text)
            if not speech_already_streamed:
                speech_queue.read_aloud(text, final=True)
            if event_log is not None:
                event_log.record("speak", text=text)

    for command in commands:
        if command.kind == "speak":
            speech_queue.enqueue(str(command.params["text"]))
            continue
        emit_speech()
        if command.kind == "blank":
            if verbose:
                terminal.line("[blank]")
        elif command.kind == "mode":
            if mode_selection is not None:
                changed = mode_selection.select(str(command.params["mode"]))
                if event_log is not None:
                    event_log.record("session.mode_requested", mode=mode_selection.mode,
                                     changed=changed, co_photo_active=mode_selection.co_photo_active)
        elif command.kind == "capture":
            if brace_missing:
                terminal.line("[合照] 没找到可扶的位置，这张还没拍。")
                if event_log is not None:
                    event_log.record("command.skipped", command=asdict(command), reason="wall_contact_missing")
                continue
            try:
                if photo_capture is None:
                    raise RuntimeError("当前没有可用的摄像头画面")
                path = photo_capture.save()
            except (OSError, RuntimeError) as exc:
                terminal.line(f"[photo] 拍照失败：{exc}")
                if event_log is not None:
                    event_log.record("command.skipped", command=asdict(command), error=str(exc))
            else:
                terminal.line(f"[photo] {path}")
                if event_log is not None:
                    event_log.record("photo.saved", path=str(path))
        elif command.kind == "photo":
            # Old recordings remain parseable; mode tags now select prompts.
            if event_log is not None:
                event_log.record(
                    "command.skipped", command=asdict(command),
                    reason="legacy_photo_entry_use_mode_tag",
                )
        else:
            wants_brace = command.kind == "brace" or (
                command.kind == "tachie" and command.params.get("arm") in ("左手扶墙", "右手扶墙"))
            if wants_brace:
                brace_missing = True
            try:
                target = canvas.apply_command(command)
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                if readiness is not None:
                    readiness.action_result(command.kind, applied=False)
                if event_log is not None:
                    event_log.record(
                        "command.skipped", command=asdict(command), error=str(exc)
                    )
                continue
            if readiness is not None:
                readiness.action_result(command.kind, applied=target is not None)
            if target is not None:
                if command.kind in {"brace", "stand", "follow", "peek", "sit", "teleport", "tachie"}:
                    brace_missing = False
                terminal.line(f"[canvas] {command.kind} {json.dumps(command.params, ensure_ascii=False)}")
                if event_log is not None:
                    event_log.record(
                        "command.applied",
                        command=asdict(command),
                        target=asdict(target),
                    )
            elif wants_brace or command.kind in {"follow", "sit", "peek"}:
                terminal.line("[合照] 当前画面未找到符合要求的位置。")
                if event_log is not None:
                    event_log.record("command.skipped", command=asdict(command), reason="no_scene_placement")

    emit_speech()
    if standalone:
        terminal.end_speech()
    return brace_missing


async def receive_realtime_outputs(
    session,
    stop_event: threading.Event,
    canvas: CanvasSource,
    speech_queue: SpeechQueue,
    *,
    verbose: bool,
    event_log: FrontendEventLogger | None = None,
    mode_selection: ModeSelection | None = None,
    photo_capture: PhotoCapture | None = None,
    terminal: TerminalOutput | None = None,
    readiness: CoPhotoReadiness | None = None,
) -> tuple[str, list[Command]] | None:
    """Consume the turn, returning any actions that need a newly selected view."""
    if terminal is None:
        terminal = TerminalOutput()
    parser = StreamingCommandParser(
        allow_plain_speech=True,
        on_speech_delta=speech_queue.read_aloud if speech_queue.output is not None else None,
    )
    active_response_id = None
    active_source = None
    # The direct session drops superseded events. Also reject replays from other event sources.
    retired_responses = deque(maxlen=64)
    wait_for_turn_end = mode_selection is not None and getattr(session, "emits_response_done", False)
    requested_mode = None
    after_switch: list[Command] = []
    brace_missing = False

    def observation_action(command):
        if command.kind != "tachie" or mode_selection is None or not mode_selection.co_photo_active:
            reason = "observation_feedback_only"
        elif readiness is not None and readiness.pending_action:
            reason = "user_placement_pending"
        else:
            try:
                target = canvas.apply_expression(command)
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                reason = str(exc)
            else:
                if target is not None:
                    if event_log is not None:
                        event_log.record("command.applied", command=asdict(command), target=asdict(target),
                                         response_id=active_response_id, turn_source="observation")
                    return
                reason = "expression_requires_current_pose"
        if event_log is not None:
            event_log.record("command.skipped", command=asdict(command), reason=reason,
                             response_id=active_response_id, turn_source="observation")

    while not stop_event.is_set():
        event = await session.recv()
        if event_log is not None:
            event_log.record("api.receive", message=event)
        response_id = event.get("response_id")
        source = event.get("turn_source")
        if source is None:
            source = active_source if active_source is not None and response_id in (None, active_response_id) else "user"
        if response_id is not None and response_id in retired_responses:
            if event_log is not None:
                event_log.record("parser.stale_output_ignored", response_id=response_id, kind=event.get("type"))
            continue
        failed_observation = event.get("type") == "error" and source == "observation"
        if event.get("type") == "response.interrupted" or failed_observation:
            interrupted_id = response_id or active_response_id
            if interrupted_id is not None:
                retired_responses.append(interrupted_id)
            if interrupted_id == active_response_id:
                if readiness is not None and source != "observation":
                    readiness.cancel_turn()
                speech_queue.read_aloud(None)
                parser.flush()
                active_response_id = None
                active_source = None
                requested_mode = None
                after_switch.clear()
                terminal.end_speech()
            if event_log is not None:
                event_log.record("observation.failed" if failed_observation else "parser.interrupted",
                                 response_id=interrupted_id, turn_source=source)
            continue

        kind, value = decode_output_event(event)
        if kind == "text":
            if active_source is None or response_id != active_response_id:
                if active_source == "user" and source == "observation":
                    if response_id is not None:
                        retired_responses.append(response_id)
                    if event_log is not None:
                        event_log.record("parser.stale_output_ignored", response_id=response_id, kind=kind,
                                         turn_source=source)
                    continue
                if active_response_id is not None:
                    retired_responses.append(active_response_id)
                if readiness is not None and source != "observation":
                    readiness.begin_turn()
                terminal.end_speech()
                speech_queue.read_aloud(None)
                parser.flush()
                active_response_id = response_id
                active_source = source
                requested_mode = None
                after_switch.clear()
                brace_missing = False
            previous_plain_count = parser.plain_speech_count
            commands = parser.feed(response_id, str(value))
            if event_log is not None:
                if parser.plain_speech_count != previous_plain_count:
                    event_log.record("parser.plain_text_adapted", response_id=response_id, turn_source=active_source,
                                     command_count=parser.plain_speech_count - previous_plain_count)
                for command in commands:
                    event_log.record("parser.command", command=asdict(command),
                                     response_id=response_id, turn_source=active_source)
            immediate = []
            for command in commands:
                if active_source == "observation":
                    if command.kind in {"speak", "blank"}:
                        dispatch_model_commands([command], canvas, speech_queue, event_log, verbose=verbose,
                                                terminal=terminal, speech_already_streamed=True)
                    else:
                        observation_action(command)
                elif wait_for_turn_end and command.kind == "mode":
                    previous = requested_mode or mode_selection.mode
                    requested_mode = command.params["mode"]
                    if event_log is not None:
                        event_log.record("session.mode_requested", mode=requested_mode,
                                         changed=previous != requested_mode,
                                         co_photo_active=mode_selection.co_photo_active)
                elif (requested_mode is not None and (after_switch or requested_mode != mode_selection.mode)
                      and command.kind not in {"speak", "blank"}):
                    # Capture and character actions need the destination view.
                    after_switch.append(command)
                else:
                    immediate.append(command)
            if immediate:
                brace_missing = dispatch_model_commands(
                    immediate, canvas, speech_queue, event_log, verbose=verbose, mode_selection=mode_selection,
                    photo_capture=photo_capture, terminal=terminal, brace_missing=brace_missing, readiness=readiness,
                    speech_already_streamed=True)
        elif kind == "done":
            if response_id is not None and response_id != active_response_id:
                continue
            parser.flush(response_id)
            terminal.end_speech()
            if active_response_id is not None:
                retired_responses.append(active_response_id)
            active_response_id = None
            active_source = None
            brace_missing = False
            if requested_mode is not None:
                if mode_selection.select(requested_mode):
                    return requested_mode, after_switch
                # Multiple tags can resolve back to the already loaded mode.
                dispatch_model_commands(after_switch, canvas, speech_queue, event_log,
                                        verbose=verbose, photo_capture=photo_capture, terminal=terminal, readiness=readiness,
                                        speech_already_streamed=True)
                requested_mode = None
                after_switch.clear()
            if readiness is not None and source != "observation":
                readiness.finish_turn()
        elif kind == "closed":
            speech_queue.read_aloud(None)
            parser.flush()
            terminal.line(f"[session] closed: {value}")
            stop_event.set()
            return

        if verbose and event.get("metadata"):
            terminal.line(f"[api] {json.dumps(event['metadata'], ensure_ascii=False)}", file=sys.stderr)


def build_control_prompt(
    canvas: CanvasSource, args: argparse.Namespace, *, mode: str | None = None,
    entrance: AvatarState | None = None, scene_context: str | None = None,
) -> str:
    if canvas.controller is None or canvas.avatar_frames is None:
        raise RuntimeError("canvas has not been started")
    mode = "guidance" if mode is None else mode
    template = {"guidance": PHOTO_GUIDANCE_PROMPT, "co-photo": CO_PHOTO_COMMAND_PROMPT}[mode]
    input_context = "当前没有摄像头输入。" if args.no_camera else "画面来自真人摄像头。"
    common = dict(
        character_name=args.character_name,
        character_persona=args.character_persona,
        response_language="默认使用中文。用户用英文时用英文回复，用户指明语言时按指定语言回复。",
        recent_context="",
        extra_instructions=f"用户通过文字输入与你交流。{input_context}",
    )
    if mode == "guidance":
        return template.format(**common)

    if args.no_camera:
        entry_context = "无相机时显示起始立绘。"
    else:
        if scene_context is None:
            scene_context = MOVING_CAMERA_CONTEXT if args.moving_camera else FIXED_CAMERA_CONTEXT
        entry_context = f"{LIVE_ENTRY_CONTEXT}\n{scene_context}"
    state = entrance if entrance is not None else canvas.controller.render_state()[0]
    catalog = canvas.avatar_frames.catalog
    return template.format(
        **common,
        pose_catalog=catalog.describe(),
        tachie_fields=catalog.entry_pose.command_fields(),
        entry_context=entry_context,
        current_character_state=json.dumps(
            {**asdict(state), "scene_layers": canvas.scene_layers.names},
            ensure_ascii=False,
        ),
    )


async def wait_for_mode_change(selection, revision, stop_event) -> bool:
    while not stop_event.is_set():
        if selection.snapshot()[1] != revision:
            return True
        await asyncio.sleep(0.05)
    return False


async def run_realtime_sessions(
    api_key, canvas, camera,
    speech_queue, stop_event, args, *, event_log=None, mode_selection=None, text_input=None,
) -> None:
    """Reload task prompts from model mode tags; mode never gates actions."""
    selection = mode_selection if mode_selection is not None else ModeSelection()
    terminal = text_input.terminal if text_input is not None else TerminalOutput()
    queued_for_view = None
    handoff_messages = ()
    while not stop_event.is_set():
        mode, revision = selection.snapshot()
        include_character = bool(args.prompt) or mode == "co-photo"
        entrance = canvas.entrance_state(camera) if include_character else None
        system_prompt = args.prompt or build_control_prompt(canvas, args, mode=mode, entrance=entrance)
        session = await direct_api.open_session(
            system_prompt, api_key=api_key, handoff_messages=handoff_messages,
        )
        tasks = []
        switched = False
        try:
            if stop_event.is_set():
                return
            if not args.prompt and not selection.mark_loaded(mode, revision):
                # A newer request superseded the prompt during initialization.
                switched = True
                continue
            canvas.configure_view(camera, show_character=include_character, entrance=entrance)
            readiness = CoPhotoReadiness() if mode == "co-photo" and camera is not None and not args.prompt else None
            photo_capture = PhotoCapture(camera, PROJECT_ROOT / "photos",
                                        character_canvas=canvas if include_character else None)
            terminal.line(f"[session] created: {session.session_id}")
            if args.prompt:
                terminal.line("[mode] custom prompt; automatic mode switching disabled")
            else:
                terminal.line(f"[mode] loaded: {'ON' if selection.co_photo_active else 'OFF'} ({mode})")
            if event_log is not None:
                event_log.record("session.mode_ready", mode="custom" if args.prompt else mode, revision=revision,
                                 co_photo_active=selection.co_photo_active,
                                 include_character=include_character,
                                 custom_prompt=bool(args.prompt), session_id=session.session_id,
                                 system_prompt=system_prompt)
            if queued_for_view is not None:
                target_mode, commands = queued_for_view
                if mode == target_mode:
                    dispatch_model_commands(commands, canvas, speech_queue, event_log,
                                            verbose=args.verbose, photo_capture=photo_capture, terminal=terminal,
                                            readiness=readiness, speech_already_streamed=True)
                    if readiness is not None:
                        readiness.finish_turn()
                queued_for_view = None
            sender = asyncio.create_task(send_realtime_inputs(
                session, canvas, camera, stop_event, event_log=event_log,
                include_character=include_character,
                text_input=text_input, report_placement=mode == "co-photo" and not args.prompt,
                terminal=terminal, readiness=readiness,
                observation_interval=2.0 if camera is not None and not args.prompt else None,
            ), name="realtime-sender")
            receiver = asyncio.create_task(receive_realtime_outputs(
                session, stop_event, canvas, speech_queue, verbose=args.verbose,
                event_log=event_log, mode_selection=None if args.prompt else selection,
                photo_capture=photo_capture, terminal=terminal, readiness=readiness,
            ), name="realtime-receiver")
            tasks.extend([sender, receiver])
            switch = asyncio.create_task(wait_for_mode_change(selection, revision, stop_event))
            tasks.append(switch)
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if receiver in done:
                queued_for_view = receiver.result()
            switched = not stop_event.is_set() and (
                queued_for_view is not None or (switch in done and switch.result())
            )
            if switched and hasattr(session, "handoff_messages"):
                handoff_messages = session.handoff_messages()
        finally:
            # The camera and console stay open across sessions.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            terminal.end_speech()
            speech_queue.read_aloud(None)
            await session.close()
        if not switched:
            return
        if event_log is not None:
            event_log.record("session.prompt_refresh", mode=selection.mode)
        terminal.line("[session] switching prompt")


def prompt_hidden_key(prompt: str) -> str:
    if sys.stdin.isatty():
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                return getpass.getpass(prompt).strip()
            except (EOFError, getpass.GetPassWarning):
                pass
    return ""


def read_api_key() -> str:
    key = configured_api_key() or prompt_hidden_key("DeepSeek API Key (hidden): ")
    if not key:
        raise RuntimeError("请在 settings.json 中填写 api_key，设置 DEEPSEEK_API_KEY，或在终端提示时输入密钥。")
    return key


def read_assemblyai_api_key(settings: dict) -> str:
    key = settings.get("assemblyai_api_key", "")
    if not isinstance(key, str):
        raise ValueError("settings.json 中的 assemblyai_api_key 必须是字符串。")
    return (key.strip() or os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
            or prompt_hidden_key("AssemblyAI API Key（隐藏输入，直接回车跳过）："))


async def run_probe(
    args: argparse.Namespace, event_log: FrontendEventLogger | None = None
) -> None:
    api_key = read_api_key()
    settings = read_project_settings()
    microphone = args.microphone if args.microphone is not None else settings.get("microphone", True)
    if not isinstance(microphone, bool):
        raise ValueError("settings.json 中的 microphone 必须为 true 或 false。")
    tts = args.tts if args.tts is not None else settings.get("tts", True)
    if not isinstance(tts, bool):
        raise ValueError("settings.json 中的 tts 必须为 true 或 false。")
    assembly_key = read_assemblyai_api_key(settings) if microphone or tts else ""
    mic_device = args.microphone_device if args.microphone_device is not None else settings.get("microphone_device")
    if mic_device is not None and (isinstance(mic_device, bool) or not isinstance(mic_device, (str, int))):
        raise ValueError("settings.json 中的 microphone_device 需要填写输入设备编号或名称。")
    if isinstance(mic_device, str):
        mic_device = mic_device.strip() or None
        if mic_device and mic_device.isdecimal():
            mic_device = int(mic_device)
    stop_event = threading.Event()
    occlusion = None if args.no_camera else PersonOcclusion(lambda: SceneSegmenter(fixed_camera=not args.moving_camera))
    canvas = CanvasSource(
        args.sprite_catalog,
        args.canvas_width,
        args.canvas_height,
        args.jpeg_quality,
        stop_event,
        preview=not args.no_preview,
        scene_layers={"person": occlusion} if occlusion is not None else {},
        infer_obstacle=occlusion is not None,
        project_scene=occlusion is not None,
    )
    camera = None
    if not args.no_camera:
        name = None
        if sys.platform == "win32" and args.camera_device is None:
            name = settings.get("camera_name", "")
            if not isinstance(name, str):
                raise ValueError("settings.json 中的 camera_name 必须是摄像头名称字符串。")
            name = name.strip() or None
        index = DEFAULT_CAMERA_DEVICE if args.camera_device is None else args.camera_device
        camera = CameraSource(index, stop_event, device_name=name, person_occlusion=occlusion)
    canvas.configure_view(camera, show_character=False)
    text_input = ConsoleTextInput()
    loop = asyncio.get_running_loop()

    def tts_status(message: str) -> None:
        loop.call_soon_threadsafe(text_input.terminal.line, f"[tts] {message}")

    speech_output = SpeechOutput(on_status=tts_status, api_key=assembly_key) if tts else None
    speech_queue = SpeechQueue(speech_output)
    input_tasks = []
    try:
        if speech_output is not None:
            speech_output.start()
        if camera is not None:
            camera.start()
        canvas.start()
        if event_log is not None:
            event_log.record(
                "client.start", url=direct_api.ENDPOINT, backend="api",
                sprite_catalog=str(args.sprite_catalog),
                initial_state=asdict(canvas.controller.state_at()), canvas_width=args.canvas_width,
                canvas_height=args.canvas_height, camera_enabled=camera is not None,
                camera_device=camera.device_index if camera and not camera.device_name else None,
                camera_description=camera.display_label if camera else None,
                fixed_camera=not args.moving_camera if camera is not None else None,
            )

        print(f"[api] {direct_api.ENDPOINT}", flush=True)
        print(f"[camera] {camera.display_label if camera else 'disabled'}", flush=True)
        print("[controls] press Q in the preview or Ctrl+C to stop", flush=True)
        print("[input] type a goal, correction, or mode request and press Enter", flush=True)
        input_tasks.append(asyncio.create_task(text_input.run(stop_event)))
        if microphone and assembly_key:
            speaking = speech_output.speaking if speech_output is not None else None
            input_tasks.append(asyncio.create_task(text_input.run_microphone(
                stop_event, api_key=assembly_key, device=mic_device, speaking=speaking,
            )))
        elif microphone:
            text_input.terminal.line("[mic] 缺少 AssemblyAI Key，听写已关闭。重启时输入密钥，或设置 ASSEMBLYAI_API_KEY。")
        else:
            text_input.terminal.line("[mic] 麦克风已关闭。")
        await run_realtime_sessions(
            api_key, canvas, camera, speech_queue, stop_event, args,
            event_log=event_log, text_input=text_input,
        )
    finally:
        stop_event.set()
        for task in input_tasks:
            task.cancel()
        await asyncio.gather(*input_tasks, return_exceptions=True)
        if speech_output is not None:
            speech_output.close()
        canvas.close()
        if camera is not None:
            camera.close()
        if occlusion is not None:
            occlusion.close()
        if canvas.error is not None:
            raise canvas.error


def parse_args(argv=None, *, default_catalog: Path = WHITE_DRAGON_CATALOG) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive photo guidance using text, camera frames and the DeepSeek API."
    )
    parser.add_argument("--backend", choices=("api",), default="api", help=argparse.SUPPRESS)
    parser.add_argument("--sprite-catalog", type=Path, default=default_catalog,
                        help="JSON catalog of tagged PNG character sprites (default: %(default)s)")
    parser.add_argument("--canvas-width", type=int, default=1280)
    parser.add_argument("--canvas-height", type=int, default=720)
    parser.add_argument(
        "--jpeg-quality", type=int, default=80, help="JPEG quality from 1 to 100"
    )
    parser.add_argument(
        "--prompt", help="override the generated streaming-command system prompt"
    )
    parser.add_argument("--character-name", help="character name used in conversation")
    add_persona_options(parser)
    parser.add_argument(
        "--camera-device", type=int,
        default=os.environ.get("HUM_CAMERA_DEVICE") or None,
        help="OpenCV camera index; overrides camera_name in settings.json (also via HUM_CAMERA_DEVICE)",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="send a blank frame in guidance, or the character canvas in co-photo/custom",
    )
    parser.add_argument("--no-preview", action="store_true", help="hide canvas preview")
    parser.add_argument("--moving-camera", action="store_true",
                        help="update furniture and depth continuously for a handheld or moving camera")
    parser.add_argument(
        "--tts", action=argparse.BooleanOptionalAction, default=None,
        help="speak English with AssemblyAI and Mandarin with Edge Xiaoyi (default: on)",
    )
    parser.add_argument(
        "--microphone", action=argparse.BooleanOptionalAction, default=None,
        help="use AssemblyAI streaming microphone recognition alongside typed input (default: on)",
    )
    parser.add_argument("--microphone-device", metavar="ID_OR_NAME",
                        help="sounddevice input device index or name (default: system input device)")
    parser.add_argument("--verbose", action="store_true", help="print API response metadata and blank commands")
    parser.add_argument(
        "--log-file",
        type=Path,
        help="JSONL log path (default: experiments/runtime/frontend-<timestamp>.jsonl)",
    )
    args = parser.parse_args(argv)
    if not args.prompt:
        load_persona_option(parser, args)
    if not args.character_name:
        args.character_name = character_name_for(args.sprite_catalog)
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.canvas_width <= 0 or args.canvas_height <= 0:
        parser.error("canvas dimensions must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    log_path = args.log_file or PROJECT_ROOT / "experiments/runtime" / (
        f"frontend-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.jsonl"
    )
    event_log = FrontendEventLogger(log_path)
    print(f"[log] {log_path.resolve()}", flush=True)
    try:
        asyncio.run(run_probe(args, event_log))
    except KeyboardInterrupt:
        print("\n[client] interrupted", flush=True)
        event_log.record("client.interrupted")
        return 130
    except Exception as exc:
        print(f"\n[client] error: {exc}", file=sys.stderr, flush=True)
        event_log.record_exception("client.error", exc)
        return 1
    finally:
        event_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
