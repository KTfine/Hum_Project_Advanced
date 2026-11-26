from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import random
from urllib.request import Request, urlopen
import zipfile

from evaluation.data.penn_action import exclusive_download_lock
from evaluation.io import atomic_write_json


RAVDESS_RECORD_ID = 1188976
RAVDESS_API_URL = f"https://zenodo.org/api/records/{RAVDESS_RECORD_ID}"
RAVDESS_ARCHIVES = ("Video_Speech_Actor_01.zip", "Video_Speech_Actor_02.zip")
RAVDESS_LABELS = ("neutral", "happy", "sad", "angry", "fear", "disgust", "surprise")
RAVDESS_EMOTIONS = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry",
    6: "fear", 7: "disgust", 8: "surprise",
}


def parse_ravdess_filename(name: str | Path) -> dict:
    parts = Path(name).stem.split("-")
    if len(parts) != 7 or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid RAVDESS filename: {name}")
    modality, channel, emotion, intensity, statement, repetition, actor = map(int, parts)
    if modality != 2 or channel != 1:
        raise ValueError("RAVDESS sample is not video-only speech")
    if emotion not in RAVDESS_EMOTIONS:
        raise ValueError(f"unknown emotion code: {emotion}")
    if intensity not in (1, 2) or statement not in (1, 2) or repetition not in (1, 2):
        raise ValueError(f"invalid RAVDESS factors: {name}")
    return {
        "modality": "video_only", "vocal_channel": "speech",
        "emotion": RAVDESS_EMOTIONS[emotion],
        "intensity": "normal" if intensity == 1 else "strong",
        "statement": statement, "repetition": repetition, "actor": actor,
    }


def select_ravdess_samples(
    entries: list[dict], *, count_per_class: int = 8, seed: int = 3800,
) -> list[dict]:
    generator = random.Random(seed)
    selected = []
    for label in RAVDESS_LABELS:
        candidates = sorted(
            (item for item in entries if item.get("emotion") == label),
            key=lambda item: item["sample_id"],
        )
        if len(candidates) < count_per_class:
            raise ValueError(
                f"RAVDESS class {label} has {len(candidates)} samples; requires {count_per_class}"
            )
        selected.extend(generator.sample(candidates, count_per_class))
    return sorted(selected, key=lambda item: item["sample_id"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_ravdess_manifest(
    video_root: str | Path, *, count_per_class: int = 8, seed: int = 3800,
) -> Path:
    root = Path(video_root)
    entries = []
    for video_path in sorted(root.rglob("*.mp4")):
        try:
            parsed = parse_ravdess_filename(video_path.name)
        except ValueError as exc:
            if "video-only speech" in str(exc):
                continue
            raise
        entries.append({
            "sample_id": video_path.stem,
            "video_path": str(video_path.resolve()),
            "video_sha256": _sha256_file(video_path),
            **parsed,
        })
    samples = select_ravdess_samples(entries, count_per_class=count_per_class, seed=seed)
    return atomic_write_json(root / "manifest.json", {
        "schema_version": 1, "dataset": "RAVDESS video-only speech",
        "source": f"https://doi.org/10.5281/zenodo.{RAVDESS_RECORD_ID}",
        "record_id": RAVDESS_RECORD_ID, "seed": seed,
        "count_per_class": count_per_class, "count": len(samples),
        "label_order": list(RAVDESS_LABELS), "excluded_labels": ["calm"],
        "samples": samples,
    })


def fetch_ravdess_file_metadata() -> dict[str, dict]:
    request = Request(RAVDESS_API_URL, headers={"User-Agent": "HumProjectEvaluation/2.0"})
    with urlopen(request, timeout=120) as response:
        record = json.load(response)
    metadata = {}
    for item in record.get("files", []):
        if item.get("key") in RAVDESS_ARCHIVES:
            metadata[item["key"]] = {
                "size": int(item["size"]),
                "url": (
                    f"https://zenodo.org/records/{RAVDESS_RECORD_ID}/files/"
                    f"{item['key']}?download=1"
                ),
                "checksum": item.get("checksum"),
            }
    missing = set(RAVDESS_ARCHIVES) - set(metadata)
    if missing:
        raise RuntimeError(f"Zenodo record missing archives: {sorted(missing)}")
    return metadata


def download_ravdess_archive(path: str | Path, *, url: str, expected_bytes: int) -> Path:
    output = Path(path)
    with exclusive_download_lock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file() and output.stat().st_size == expected_bytes:
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".part")
        for _attempt in range(100):
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset == expected_bytes:
                break
            if offset > expected_bytes:
                raise RuntimeError(
                    f"oversized RAVDESS partial archive: {offset} > {expected_bytes}"
                )
            range_end = min(expected_bytes - 1, offset + 16 * 1024 * 1024 - 1)
            request = Request(url, headers={
                "User-Agent": "HumProjectEvaluation/2.0",
                "Range": f"bytes={offset}-{range_end}",
            })
            with urlopen(request, timeout=300) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if partial.stat().st_size <= offset:
                raise RuntimeError("RAVDESS download made no progress")
        if partial.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"incomplete RAVDESS archive: {partial.stat().st_size} of {expected_bytes} bytes"
            )
        partial.replace(output)
        return output


def extract_ravdess_archive(archive: str | Path, destination: str | Path) -> None:
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {info.filename}")
            if path.suffix.lower() == ".mp4":
                output = target / path.name
                if not output.is_file():
                    with bundle.open(info) as source, output.open("wb") as sink:
                        while block := source.read(8 * 1024 * 1024):
                            sink.write(block)


def prepare_ravdess(
    target_dir: str | Path, *, count_per_class: int = 8, seed: int = 3800,
) -> Path:
    target = Path(target_dir)
    videos = target / "videos"
    manifest = videos / "manifest.json"
    if manifest.is_file():
        return manifest
    metadata = fetch_ravdess_file_metadata()
    for name in RAVDESS_ARCHIVES:
        item = metadata[name]
        archive = download_ravdess_archive(
            target / name, url=item["url"], expected_bytes=item["size"],
        )
        extract_ravdess_archive(archive, videos)
    return build_ravdess_manifest(videos, count_per_class=count_per_class, seed=seed)
