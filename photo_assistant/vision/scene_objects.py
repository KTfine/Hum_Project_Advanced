"""COCO instance silhouettes and single-frame relative inverse depth using OpenCV DNN."""

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from photo_assistant.paths import PROJECT_ROOT


_MODEL_DIR = PROJECT_ROOT / "models"
_YOLO_REV = "0e2a3f195ae62a25bc038ee9665b8e4ac5b98793"
_YOLO_FILE = f"unileon-robotics/YOLO11-ONNX/resolve/{_YOLO_REV}/yolo11s-seg.onnx"


@dataclass(frozen=True)
class _Model:
    name: str
    size: int
    sha256: str
    urls: tuple[str, ...]

    def read(self) -> bytes:
        path = _MODEL_DIR / self.name
        if path.stat().st_size != self.size:
            raise ValueError(f"{self.name} 文件大小不符")
        weights = path.read_bytes()
        if hashlib.sha256(weights).hexdigest() != self.sha256:
            raise ValueError(f"{self.name} SHA-256 校验失败")
        return weights


_YOLO = _Model(
    "yolo11s-seg.onnx", 40_638_417,
    "a187920739d565aee2ff5ecc0c710eeb2387ffb105afc6ddbbd4652052e604ca",
    (f"https://huggingface.co/{_YOLO_FILE}", f"https://hf-mirror.com/{_YOLO_FILE}"),
)
_MIDAS = _Model(
    "midas-small-256.onnx", 66_764_249,
    "2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58",
    ("https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",),
)

# Contiguous COCO class indices used by this ONNX export. PPHumanSeg supplies person.
_LABELS = (
    "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|"
    "fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|"
    "bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|"
    "sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|"
    "bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|"
    "carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|"
    "laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|"
    "book|clock|vase|scissors|teddy bear|hair drier|toothbrush"
).split("|")


def install_models() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model in (_YOLO, _MIDAS):
        try:
            model.read()
            continue
        except (OSError, ValueError):
            pass
        for url in model.urls:
            try:
                with urlopen(url, timeout=30) as response:
                    data = response.read(model.size + 1)
                if len(data) != model.size or hashlib.sha256(data).hexdigest() != model.sha256:
                    raise ValueError(f"{model.name} 下载内容校验失败")
            except (OSError, ValueError) as exc:
                if url == model.urls[-1]:
                    raise RuntimeError(f"无法下载 {model.name}：{exc}") from exc
                continue
            with tempfile.NamedTemporaryFile(dir=_MODEL_DIR, delete=False) as output:
                pending = Path(output.name)
            try:
                pending.write_bytes(data)
                pending.replace(_MODEL_DIR / model.name)
            finally:
                pending.unlink(missing_ok=True)
            break


@dataclass(frozen=True)
class SceneObject:
    label: str
    confidence: float
    mask: np.ndarray


def _complete_tabletop(frame: np.ndarray, coverage: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Recover gaps in a smooth tabletop without painting its bounding box or leg openings."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
    seed = coverage > 0.85
    pixels = lab[seed]
    if len(pixels) < 64:
        return coverage
    # Learn the dominant finish from the detected surface, including wood and dark finishes.
    bins = pixels // 16
    codes = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    dominant = np.bincount(codes).argmax()
    cluster = pixels[codes == dominant]
    if len(cluster) < 0.14 * len(pixels):
        return coverage
    color = np.median(cluster, axis=0).astype(np.int16)
    difference = np.abs(lab - color)
    similar = (difference <= (16, 7, 7)).all(axis=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(similar.astype(np.uint8), connectivity=4)
    support = np.bincount(labels[seed], minlength=count)
    area = stats[:, cv2.CC_STAT_AREA]
    # A region that continues far beyond the detected furniture has little seed support.
    accepted = (area >= 40) & (support >= 0.24 * area)
    accepted[0] = False
    filled = np.maximum(coverage, accepted[labels])

    # The detector can bridge the gap between joined desks. Fit the dominant finish
    # to a physical plane in this frame's inverse-depth coordinates, then cut away
    # coherent regions that are well behind it. Original legs remain below the top.
    surface = (filled > 0.9) & (difference <= (4, 3, 3)).all(axis=2)
    ys = np.nonzero(surface)[0]
    if ys.size < 128 or np.ptp(ys) < 12:
        return filled
    slope, offset = np.polyfit(ys, depth[surface], 1)
    residual = depth - (offset + slope * np.arange(len(depth))[:, None])
    error = np.median(np.abs(residual[surface]))
    if slope <= 0 or error > 0.035:
        return filled
    edges = cv2.Canny(cv2.GaussianBlur(frame, (3, 3), 0), 15, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel)
    count, regions, stats, _ = cv2.connectedComponentsWithStats((edges == 0).astype(np.uint8), connectivity=4)
    area = stats[:, cv2.CC_STAT_AREA]
    behind = residual < -max(0.036, 3 * error)
    distant = np.bincount(regions[behind], minlength=count) > 0.4 * area
    same_finish = np.bincount(regions[surface], minlength=count) > 0.01 * area
    cut = distant & ~same_finish & (area > 32)
    cut[0] = False
    return filled * (1 - cv2.dilate(cut[regions].astype(np.uint8), kernel))


class SceneObjectEstimator:
    """One owner thread. Masks and normalized inverse depth use original BGR frame coordinates."""

    def __init__(self) -> None:
        self.person_boxes: tuple[tuple[int, int, int, int], ...] = ()
        self.person_patches = ()
        try:
            self._segmenter = cv2.dnn.readNetFromONNX(np.frombuffer(_YOLO.read(), dtype=np.uint8))
            self._depth = cv2.dnn.readNetFromONNX(np.frombuffer(_MIDAS.read(), dtype=np.uint8))
        except (OSError, ValueError, cv2.error) as exc:
            raise RuntimeError(
                "无法加载场景遮挡模型。请在项目目录运行 python -m photo_assistant.vision.scene_objects 后重启。"
            ) from exc

    def _objects(self, frame: np.ndarray, depth: np.ndarray) -> tuple[SceneObject, ...]:
        height, width = frame.shape[:2]
        scale = min(640 / width, 480 / height)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (640 - w) // 2, (480 - h) // 2
        padded = np.full((480, 640, 3), 114, np.uint8)
        padded[top:top + h, left:left + w] = cv2.resize(frame, (w, h))
        self._segmenter.setInput(cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True))
        detections, prototypes = self._segmenter.forward(("output0", "output1"))
        rows = detections[0].T
        class_scores = rows[:, 4:84]
        labels = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(rows)), labels]
        planes = prototypes[0].reshape(32, -1)

        def silhouette(coefficients, box):
            x0, y0, x1, y1 = box
            logits = (coefficients @ planes).reshape(prototypes.shape[2:])
            logits = cv2.resize(logits, (640, 480))[top:top + h, left:left + w]
            logits = cv2.resize(logits, (width, height))[y0:y1, x0:x1]
            return np.clip(0.5 + logits, 0.0, 1.0)

        # PPHumanSeg sometimes marks an entire chair as a person. Retain independent
        # same-frame person evidence for its caller without another model pass.
        people = (labels == 0) & (scores >= 0.20)
        person_boxes = rows[people, :4].copy()
        person_boxes[:, :2] -= person_boxes[:, 2:] / 2
        survivors = cv2.dnn.NMSBoxes(person_boxes.tolist(), scores[people].tolist(), 0.199, 0.5)
        patches = []
        person_rows = rows[people]
        for index in survivors:
            box = _unpad_box(person_boxes[index], left, top, w, h, width, height)
            if box is not None:
                patches.append((box, silhouette(person_rows[index, 84:], box)))
        self.person_patches = tuple(patches)
        self.person_boxes = tuple(box for box, _ in patches)
        # Joined tables have an ambiguous instance boundary. COCO's table score can be
        # lower even when the union gives a usable silhouette for occlusion.
        # Bags often sit on top of desks. Electronics, books and posters on shelving
        # are easily confused, so require stronger evidence for those categories.
        thresholds = np.select(
            [labels == 60, np.isin(labels, (24, 26, 28)), np.isin(labels, (56, 57, 58, 59))],
            [0.20, 0.25, 0.35], default=0.55,
        )
        keep = (labels != 0) & (scores >= thresholds)
        rows, labels, scores = rows[keep], labels[keep], scores[keep]
        if not len(rows):
            return ()
        # NMS accepts top-left and extent. Suppress duplicates within the same class.
        boxes = rows[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        kept_indices = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), labels.tolist(), 0.199, 0.5)
        result = []
        for idx in kept_indices:
            box = _unpad_box(boxes[idx], left, top, w, h, width, height)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            coverage = silhouette(rows[idx, 84:], box)
            if np.count_nonzero(coverage >= 0.5) < max(16, 0.0005 * width * height):
                continue
            if labels[idx] == 60:
                coverage = _complete_tabletop(frame[y0:y1, x0:x1], coverage, depth[y0:y1, x0:x1])
            mask = np.zeros((height, width), dtype=np.float32)
            mask[y0:y1, x0:x1] = coverage
            # Same pixels may be called both a backpack and a handbag. Distinct chairs
            # in front of the table have different silhouettes even when boxes overlap.
            label = _LABELS[labels[idx]]
            if any(_same_silhouette(mask, prior.mask, same_class=label == prior.label,
                                    table=label == "dining table") for prior in result):
                continue
            result.append(SceneObject(label, float(scores[idx]), mask))
            if len(result) == 12:
                break
        return tuple(result)

    def infer(self, frame: np.ndarray) -> tuple[tuple[SceneObject, ...], np.ndarray]:
        # The published v2.1 ONNX embeds channel normalization. It expects RGB in [0, 1].
        small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        self._depth.setInput(cv2.dnn.blobFromImage(small, 1 / 255.0, swapRB=True))
        depth = self._depth.forward()[0]
        low, high = np.percentile(depth, (2, 98))
        if np.isfinite([low, high]).all() and high > low:
            normalized = np.clip((depth - np.float32(low)) / np.float32(high - low), 0.0, 1.0)
            depth = cv2.resize(normalized, (frame.shape[1], frame.shape[0]))
        else:
            depth = np.zeros(frame.shape[:2], dtype=np.float32)
        return self._objects(frame, depth), depth


def _unpad_box(box, left, top, w, h, width, height):
    x, y, bw, bh = box
    x0 = max(0, int(np.floor((x - left) * width / w)))
    y0 = max(0, int(np.floor((y - top) * height / h)))
    x1 = min(width, int(np.ceil((x + bw - left) * width / w)))
    y1 = min(height, int(np.ceil((y + bh - top) * height / h)))
    if x0 < x1 and y0 < y1:
        return x0, y0, x1, y1
    return None


def _same_silhouette(left: np.ndarray, right: np.ndarray, *, same_class: bool, table: bool = False) -> bool:
    left = left[::4, ::4] >= 0.5
    right = right[::4, ::4] >= 0.5
    intersection = np.count_nonzero(left & right)
    candidate_area = np.count_nonzero(left)
    if same_class and intersection > 0.92 * candidate_area:
        # Less confident second proposal sometimes contains only the tabletop of
        # an already accepted desk. A separate desk has disjoint visible pixels.
        return True
    if table and same_class and intersection > .82 * candidate_area:
        # A completed tabletop sometimes spills past the stronger full-table
        # proposal. Look for the wide rim above that instance's narrow legs.
        _, top1, width1, height1 = cv2.boundingRect(left.astype(np.uint8))
        x, top2, width2, height2 = cv2.boundingRect(right.astype(np.uint8))
        bottom = top2 + height2
        legs = right[bottom - max(1, round(.12 * height2)):bottom, x:x + width2].any(axis=0)
        if (width1 >= .65 * width2 and height1 < .60 * height2 and top1 + height1 < bottom
                and 0 < legs.sum() <= .55 * width2):
            return True
    return intersection > 0.82 * np.count_nonzero(left | right)


if __name__ == "__main__":
    try:
        install_models()
        print(f"场景分割与相对深度模型已就绪：{_MODEL_DIR}")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
