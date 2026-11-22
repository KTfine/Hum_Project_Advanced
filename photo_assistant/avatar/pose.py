"""Exact-match index of tagged, transparent character illustrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct

from photo_assistant.paths import PROJECT_ROOT

WHITE_DRAGON_CATALOG = PROJECT_ROOT / "models/avatars/white_dragon/catalog.json"

SPRITE_OPTIONS = {
    "face": ("微笑", "龇牙笑", "高冷表情", "害羞", "惊讶", "思考", "难过", "鼓脸小生气"),
    "hand": ("比耶", "无动作"),
    "arm": ("自然下垂", "伸直举高过头", "背在身后", "左手扶墙", "右手扶墙"),
    "torso": ("正着站", "半身歪斜向右"),
    "legs": ("站着", "坐着", "蹲下"),
}


@dataclass(frozen=True)
class AvatarPose:
    face: str = "微笑"
    hand: str = "无动作"
    arm: str = "自然下垂"
    torso: str = "正着站"
    legs: str = "站着"

    @classmethod
    def from_tags(cls, tags: Mapping[str, str]) -> AvatarPose:
        if not isinstance(tags, Mapping) or tags.keys() != SPRITE_OPTIONS.keys():
            raise ValueError("立绘组合须包含 face、hand、arm、torso、legs 五项")
        for key, options in SPRITE_OPTIONS.items():
            if tags[key] not in options:
                raise ValueError(f"未知的立绘标签 {key}={tags[key]}")
        return cls(**tags)

    def command_fields(self) -> str:
        return ",".join(f"{key}={value}" for key, value in asdict(self).items())

    def command(self) -> str:
        return f"<tachie:{self.command_fields()}>"


@dataclass(frozen=True)
class SpriteEntry:
    path: Path
    standing_height_px: float
    anchor_px: tuple[float, float]
    face_box_px: tuple[float, float, float, float] | None = None
    seat_contact_px: tuple[float, float] | None = None
    wall_contact_px: tuple[float, float] | None = None


def _patterns(rows: Sequence[tuple[str, ...]], options: tuple[tuple[str, ...], ...]) -> list[tuple[str, ...]]:
    """Group identical suffixes, preserving the exact set of available combinations."""
    if not options:
        return [()]
    groups = {}
    for value in options[0]:
        suffixes = tuple(sorted(row[1:] for row in rows if row[0] == value))
        if suffixes:
            groups.setdefault(suffixes, []).append(value)
    patterns = []
    for suffixes, values in groups.items():
        prefix = "*" if len(values) == len(options[0]) else "/".join(values)
        for suffix in _patterns(suffixes, options[1:]):
            patterns.append((prefix,) + suffix)
    return patterns


class PoseCatalog:
    def __init__(self, entries: dict[AvatarPose, SpriteEntry]):
        if not entries:
            raise ValueError("立绘数据库至少需要一张立绘")
        self.entries = entries
        self.default_pose = next(iter(entries))
        self.entry_pose = max(
            (pose for pose in entries if pose.torso == "半身歪斜向右"),
            key=lambda pose: (pose.legs == "站着", pose.hand == "无动作",
                              pose.arm == "自然下垂", pose.face == "微笑"),
            default=self.default_pose,
        )
        self.standing_pose = max(
            (pose for pose in entries if pose.torso == "正着站" and pose.legs == "站着"),
            key=lambda pose: (pose.hand == "无动作", pose.arm == "自然下垂", pose.face == "微笑"),
            default=self.entry_pose,
        )

    def entry(self, pose: AvatarPose) -> SpriteEntry:
        if pose not in self.entries:
            raise ValueError(f"库中没有此立绘组合 {pose.command()}")
        return self.entries[pose]

    def select(self, params: Mapping[str, str]) -> AvatarPose:
        pose = AvatarPose.from_tags(params)
        self.entry(pose)
        return pose

    def describe(self) -> str:
        if len(self.entries) <= 16:
            return "\n".join(pose.command() for pose in self.entries)

        options = {}
        for field, allowed in SPRITE_OPTIONS.items():
            used = {getattr(pose, field) for pose in self.entries}
            options[field] = tuple(value for value in allowed if value in used)
        lines = [f"{field} 可选 {'、'.join(values)}" for field, values in options.items()]
        lines.append("组合按 " + " | ".join(options) + " 排列（* 为该项任一可选值，/ 分隔备选值）。")
        rows = [tuple(asdict(pose).values()) for pose in self.entries]
        lines.extend(" | ".join(pattern) for pattern in _patterns(rows, tuple(options.values())))
        return "\n".join(lines)


def _number(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _point(value, field, filename, max_x, max_y) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2 or not all(_number(v) for v in value)
            or not 0 <= value[0] <= max_x or not 0 <= value[1] <= max_y):
        raise ValueError(f"{field} 须为图内的 [x, y]：{filename}")
    return float(value[0]), float(value[1])


def _entry(row: dict, directory: Path) -> SpriteEntry:
    filename = row.get("file")
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        raise ValueError("file 需要目录内的相对 PNG 路径")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() != ".png":
        raise ValueError(f"立绘文件必须是目录内的 PNG：{filename}")
    with path.open("rb") as stream:
        header = stream.read(29)
    if (len(header) < 29 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            or header[24:26] != bytes((8, 6))):
        raise ValueError(f"立绘须为 8 位 RGBA PNG：{filename}")
    width, height = struct.unpack(">II", header[16:24])
    size = row.get("standing_height_px")
    if not _number(size) or size <= 0:
        raise ValueError("standing_height_px 须为正数")
    anchor = _point(row.get("anchor_px"), "anchor_px", filename, width, height)
    face_box = row.get("face_box_px")
    if face_box is not None:
        if (not isinstance(face_box, list) or len(face_box) != 4 or not all(_number(v) for v in face_box)
                or not 0 <= face_box[0] < face_box[2] <= width
                or not 0 <= face_box[1] < face_box[3] <= height):
            raise ValueError(f"face_box_px 须为图内的 [左, 上, 右, 下]：{filename}")
        face_box = tuple(face_box)
    seat = row.get("seat_contact_px")
    if seat is not None:
        seat = _point(seat, "seat_contact_px", filename, width, height)
    palm = row.get("wall_contact_px")
    if palm is not None:
        palm = _point(palm, "wall_contact_px", filename, width - 1, height - 1)
    return SpriteEntry(path, float(size), anchor, face_box, seat, palm)


def load_pose_catalog(path: Path) -> PoseCatalog:
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = document.get("sprites") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError("立绘数据库须提供 sprites 数组")
    entries = {}
    for index, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict):
                raise ValueError("立绘记录须为对象")
            pose = AvatarPose.from_tags(row.get("tags"))
            if pose in entries:
                raise ValueError("数据库中存在重复的五项标签组合")
            entries[pose] = _entry(row, path.parent)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path.name} 第 {index} 条立绘记录：{exc}") from exc
    return PoseCatalog(entries)
