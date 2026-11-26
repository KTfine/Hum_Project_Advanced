from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import random
import tarfile
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat

from evaluation.io import atomic_write_json
from evaluation.video import uniform_frame_indices


PENN_ACTION_URL = "https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz"
PENN_ACTION_ARCHIVE_BYTES = 3_235_203_923
PENN_ACTIONS = (
    "baseball_pitch", "baseball_swing", "bench_press", "bowl",
    "clean_and_jerk", "golf_swing", "jumping_jacks", "jump_rope",
    "pullup", "pushup", "situp", "squat", "strum_guitar",
    "tennis_forehand", "tennis_serve",
)


def _mat_string(value: object) -> str:
    array = np.asarray(value).squeeze()
    if array.ndim == 0:
        return str(array.item())
    if array.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in array.flat)
    if array.size == 1:
        return str(array.flat[0])
    raise ValueError(f"cannot decode MATLAB string with shape {array.shape}")


def load_penn_annotation(path: str | Path) -> dict:
    source = Path(path)
    data = loadmat(source)
    action = _mat_string(data["action"]).strip()
    if action not in PENN_ACTIONS:
        raise ValueError(f"unknown Penn Action label: {action}")
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    visibility = np.asarray(data["visibility"], dtype=bool)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 13:
        raise ValueError(f"invalid Penn joint arrays: x={x.shape}, y={y.shape}")
    if visibility.shape != x.shape:
        raise ValueError(f"visibility shape {visibility.shape} != joint shape {x.shape}")
    return {
        "action": action,
        "joints_xy": np.stack((x, y), axis=-1),
        "visibility": visibility,
        "split": "train" if int(np.asarray(data.get("train", [[0]])).squeeze()) else "test",
    }


def select_penn_samples(
    entries: list[dict], *, count_per_class: int = 4, seed: int = 3800
) -> list[dict]:
    selected = []
    generator = random.Random(seed)
    for action in PENN_ACTIONS:
        candidates = sorted(
            (entry for entry in entries if entry.get("action") == action),
            key=lambda item: item["sample_id"],
        )
        if len(candidates) < count_per_class:
            raise ValueError(
                f"Penn Action class {action} has {len(candidates)} samples; "
                f"requires {count_per_class}"
            )
        selected.extend(generator.sample(candidates, count_per_class))
    return sorted(selected, key=lambda item: item["sample_id"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_penn_manifest(
    dataset_root: str | Path, *, count_per_class: int = 4, seed: int = 3800
) -> Path:
    root = Path(dataset_root)
    entries = []
    for annotation_path in sorted((root / "labels").glob("*.mat")):
        sample_id = annotation_path.stem
        annotation = load_penn_annotation(annotation_path)
        frame_paths = sorted((root / "frames" / sample_id).glob("*.jpg"))
        if len(frame_paths) != annotation["joints_xy"].shape[0]:
            raise ValueError(
                f"Penn sample {sample_id} has {len(frame_paths)} frames but "
                f"{annotation['joints_xy'].shape[0]} annotation rows"
            )
        selected_indices = uniform_frame_indices(len(frame_paths), 8)
        entries.append({
            "sample_id": sample_id,
            "action": annotation["action"],
            "split": annotation["split"],
            "frame_paths": [str(path.resolve()) for path in frame_paths],
            "selected_frame_indices": selected_indices,
            "selected_frame_paths": [str(frame_paths[index].resolve()) for index in selected_indices],
            "joints_xy": annotation["joints_xy"].astype(float).tolist(),
            "visibility": annotation["visibility"].astype(bool).tolist(),
            "annotation_sha256": _sha256_file(annotation_path),
        })
    samples = select_penn_samples(entries, count_per_class=count_per_class, seed=seed)
    return atomic_write_json(root / "manifest.json", {
        "schema_version": 1,
        "dataset": "Penn Action",
        "source": PENN_ACTION_URL,
        "seed": seed,
        "count_per_class": count_per_class,
        "count": len(samples),
        "joint_order": [
            "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle",
        ],
        "label_order": list(PENN_ACTIONS),
        "samples": samples,
    })


def validate_archive_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return True


@contextmanager
def exclusive_download_lock(path: str | Path):
    import msvcrt

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"download already active for {lock_path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def download_penn_archive(path: str | Path) -> Path:
    output = Path(path)
    with exclusive_download_lock(output.with_suffix(output.suffix + ".lock")):
        if output.is_file() and output.stat().st_size == PENN_ACTION_ARCHIVE_BYTES:
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".part")
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {
            "User-Agent": "HumProjectEvaluation/2.0",
            "Range": f"bytes={offset}-",
        }
        request = Request(PENN_ACTION_URL, headers=headers)
        with urlopen(request, timeout=300) as response:
            append = offset > 0 and getattr(response, "status", None) == 206
            with partial.open("ab" if append else "wb") as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
        if partial.stat().st_size != PENN_ACTION_ARCHIVE_BYTES:
            raise RuntimeError(
                f"incomplete Penn Action archive: {partial.stat().st_size} of "
                f"{PENN_ACTION_ARCHIVE_BYTES} bytes"
            )
        partial.replace(output)
        return output


def extract_penn_archive(archive: str | Path, destination: str | Path) -> Path:
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            validate_archive_member(member.name)
        bundle.extractall(target)
    root = target / "Penn_Action"
    if not (root / "frames").is_dir() or not (root / "labels").is_dir():
        raise RuntimeError("Penn Action archive did not contain frames and labels")
    return root


def prepare_penn_action(
    target_dir: str | Path, *, count_per_class: int = 4, seed: int = 3800
) -> Path:
    target = Path(target_dir)
    root = target / "Penn_Action"
    manifest = root / "manifest.json"
    if manifest.is_file():
        return manifest
    archive = download_penn_archive(target / "Penn_Action.tar.gz")
    if not root.is_dir():
        root = extract_penn_archive(archive, target)
    return build_penn_manifest(root, count_per_class=count_per_class, seed=seed)
