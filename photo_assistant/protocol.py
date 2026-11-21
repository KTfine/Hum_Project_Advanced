"""Forgiving incremental parser for compact streamed character commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

from .avatar.pose import AvatarPose

DEFAULT_TRANSITION_MS = 350
# The visible frame is 0..1. Extra travel accommodates cropped, enlarged poses.
MIN_POSITION = -2.0
MAX_POSITION = 3.0
CHARACTER_ACTIONS = frozenset({"follow", "peek", "sit", "brace", "stand", "teleport", "tachie", "layer", "rotate"})
_PREFIX_RE = re.compile(r"<(speak|teleport|tachie|rotate|layer|photo|mode|wait|sit|brace):"
                        r"|<(blank|(?:capture|follow|peek|sit|brace|stand)(?=>))")
_STARTERS = (
    "<speak:",
    "<teleport:",
    "<tachie:",
    "<rotate:",
    "<layer:",
    "<photo:",
    "<capture>",
    "<follow>",
    "<peek>",
    "<sit>",
    "<sit:",
    "<brace>",
    "<brace:",
    "<stand>",
    "<mode:",
    "<wait:",
    "<blank>",
)


@dataclass(frozen=True)
class Command:
    kind: str
    params: dict[str, Any]


def teleport_destination(params: Mapping[str, Any]) -> dict[str, float | str] | None:
    """Read a complete absolute destination from a wire tag or a local command."""
    if not isinstance(params, Mapping) or not {"x", "y"} <= params.keys() <= {"x", "y", "scale", "depth"}:
        return None
    destination: dict[str, float | str] = {}
    for key, raw in params.items():
        if key == "depth":
            if not isinstance(raw, str) or raw not in ("near", "far", "alongside"):
                return None
            destination[key] = raw
            continue
        value = _finite_number(raw)
        low, high = (0.1, 3.0) if key == "scale" else (MIN_POSITION, MAX_POSITION)
        if value is None or not low <= value <= high:
            return None
        destination[key] = value
    return destination


class StreamingCommandParser:
    """Extract valid commands, optionally adapting untagged speech fragments.

    Strict parsing is the default for callers that require model-authored tags.
    Plain speech never implies an avatar action.
    """

    def __init__(self, max_buffer: int = 4096, *, allow_plain_speech: bool = False) -> None:
        if max_buffer < 16:
            raise ValueError("max_buffer must be at least 16")
        self.max_buffer = max_buffer
        self.allow_plain_speech = allow_plain_speech
        self.plain_speech_count = 0
        self._response_id: str | None = None
        self._buffer = ""
        self._discarding_tag = False

    def feed(self, response_id: str | None, text: str) -> list[Command]:
        if response_id != self._response_id:
            self._buffer = ""
            self._discarding_tag = False
            self._response_id = response_id
        self._buffer += str(text)
        commands: list[Command] = []

        while self._buffer:
            if self._discarding_tag:
                # After an oversized tag, do not speak its remaining parameters.
                nested = _PREFIX_RE.search(self._buffer)
                closing = self._buffer.find(">")
                if nested is not None and (closing < 0 or nested.start() < closing):
                    self._buffer = self._buffer[nested.start():]
                    self._discarding_tag = False
                    continue
                if closing < 0:
                    self._buffer = self._partial_prefix_suffix(self._buffer)
                    break
                self._buffer = self._buffer[closing + 1:]
                self._discarding_tag = False
                continue
            match = _PREFIX_RE.search(self._buffer)
            if match is None:
                if self.allow_plain_speech:
                    opening = self._buffer.find("<")
                    if opening < 0:
                        commands.extend(self._plain_speech(self._buffer))
                        self._buffer = ""
                        break
                    if opening > 0:
                        commands.extend(self._plain_speech(self._buffer[:opening]))
                        self._buffer = self._buffer[opening:]
                    closing = self._buffer.find(">")
                    if closing >= 0:
                        # Unknown angle-bracket commands are never speech.
                        self._buffer = self._buffer[closing + 1:]
                        continue
                    if len(self._buffer) > self.max_buffer:
                        self._discarding_tag = True
                        self._buffer = self._partial_prefix_suffix(self._buffer)
                    break
                self._buffer = self._partial_prefix_suffix(self._buffer)
                break
            if match.start() > 0:
                if self.allow_plain_speech:
                    commands.extend(self._plain_speech(self._buffer[:match.start()]))
                self._buffer = self._buffer[match.start() :]
                match = _PREFIX_RE.match(self._buffer)
                assert match is not None

            kind = match.group(1) or match.group(2)
            body_start = match.end()
            nested = _PREFIX_RE.search(self._buffer, body_start)
            closing = self._buffer.find(">", body_start)
            if nested is not None and (closing < 0 or nested.start() < closing):
                prefix_text = self._buffer[body_start : nested.start()]
                # `<speak: <teleport:>` is still a recoverable outer speech tag.
                # With real speech before the nested prefix, prefer resyncing so
                # a missing speech delimiter cannot swallow a following action.
                if kind != "speak" or prefix_text.strip():
                    self._buffer = self._buffer[nested.start() :]
                    continue
            if closing < 0:
                if len(self._buffer) > self.max_buffer:
                    self._buffer = ""
                    self._discarding_tag = self.allow_plain_speech
                break

            closing_run_end = closing + 1
            while (
                closing_run_end < len(self._buffer)
                and self._buffer[closing_run_end] == ">"
            ):
                closing_run_end += 1
            if (
                kind == "speak"
                and closing_run_end == len(self._buffer)
                and closing_run_end == closing + 1
                and not self._buffer[body_start:closing].strip()
            ):
                # An empty-looking speech tag at a chunk boundary may be the
                # first half of `<speak: >>`; wait for one more text fragment.
                break
            # Repeated `>` is ambiguous only for speech: preserve preceding
            # brackets as spoken content there, but treat them as harmless
            # duplicate delimiters for structured commands.
            delimiter = closing_run_end - 1 if kind == "speak" else closing
            body = self._buffer[body_start:delimiter]
            self._buffer = self._buffer[closing_run_end:]

            # Some model replies mix the compact speech opener with an XML closer.
            if kind == "speak" and body.endswith(("</speak", "</speak:")):
                if self.allow_plain_speech:
                    commands.extend(self._plain_speech(body.rsplit("</speak", 1)[0].strip()))
                continue
            command = _parse_command(kind, body, parameterized=match.group(1) is not None)
            if command is not None:
                commands.append(command)

        return commands

    def flush(self, response_id: str | None = None) -> list[Command]:
        if response_id is None or response_id == self._response_id:
            self._buffer = ""
            self._discarding_tag = False
            if response_id is not None:
                self._response_id = response_id
        return []

    def _plain_speech(self, text: str) -> list[Command]:
        # A prefix before a valid command may contain an unknown/broken tag.
        # Strip such spans instead of exposing their contents as speech.
        text = re.sub(r"<[^>]*(?:>|$)", "", text)
        if not text.strip():
            return []
        self.plain_speech_count += 1
        return [Command("speak", {"text": text})]

    @staticmethod
    def _partial_prefix_suffix(text: str) -> str:
        limit = min(len(text), max(map(len, _STARTERS)) - 1)
        for length in range(limit, 0, -1):
            suffix = text[-length:]
            if any(starter.startswith(suffix) for starter in _STARTERS):
                return suffix
        return ""


def _parse_command(kind: str, body: str, *, parameterized: bool) -> Command | None:
    if not parameterized:
        return Command(kind, {}) if not body.strip() else None
    if kind == "speak":
        text = body.strip()
        return Command("speak", {"text": text}) if text else None
    if kind == "photo":
        return Command("photo", {"action": "enter"}) if body.strip() == "enter" else None
    if kind == "mode":
        mode = body.strip()
        return Command("mode", {"mode": mode}) if mode in {"guidance", "co-photo"} else None
    if kind == "wait":
        task = body.strip()
        return Command("wait", {"task": task}) if task in {"feedback", "capture"} else None
    parts = [part.strip() for part in body.split(",")]
    mode = None
    if kind in {"rotate", "layer"}:
        mode = parts.pop(0)
        if mode not in {"to", "by"}:
            return None
    values: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            return None
        key, value = (piece.strip() for piece in part.split("=", 1))
        if not key or not value or key in values:
            return None
        values[key] = value

    if kind == "tachie":
        try:
            AvatarPose.from_tags(values)
        except ValueError:
            return None
        return Command("tachie", values)

    if kind == "teleport":
        target = teleport_destination(values)
        return Command("teleport", target) if target is not None else None

    if kind == "brace" and values.keys() == {"x"}:
        x = _finite_number(values["x"])
        if x is not None and 0 <= x <= 1:
            return Command("brace", {"x": x})

    if kind == "sit":
        if values.keys() == {"table"} and values["table"] in {"near", "far"}:
            return Command("sit", values)
        if values.keys() == {"x", "y"}:
            x, y = _finite_number(values["x"]), _finite_number(values["y"])
            if x is not None and y is not None and 0 <= x <= 1 and 0 <= y <= 1:
                return Command("sit", {"x": x, "y": y})

    if kind == "layer":
        if mode != "to" or set(values) != {"index"} or not re.fullmatch(r"[0-9]+", values["index"]):
            return None
        try:
            index = int(values["index"])
        except ValueError:  # Python limits conversion of very long digit strings.
            return None
        return Command("layer", {"index": index})

    if kind == "rotate":
        ms = _finite_number(values.pop("ms", str(DEFAULT_TRANSITION_MS)))
        if ms is None or not ms.is_integer() or not 0 <= ms <= 3000:
            return None
        if set(values) != {"value"}:
            return None
        value = _finite_number(values["value"])
        if value is None or not -180.0 <= value <= 180.0:
            return None
        return Command("rotate", {"mode": mode, "ms": int(ms), "value": value})

    return None


def _finite_number(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None
