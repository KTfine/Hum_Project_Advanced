"""Single-person foreground confidence from the OpenCV Zoo PPHumanSeg model."""

import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from photo_assistant.paths import PROJECT_ROOT


MODEL_NAME = "human_segmentation_pphumanseg_2023mar.onnx"
MODEL_PATH = PROJECT_ROOT / "models" / MODEL_NAME
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/4.10.0/"
    f"models/human_segmentation_pphumanseg/{MODEL_NAME}"
)
MODEL_SHA256 = "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24"
MODEL_BYTES = 6_163_938


def install_model() -> Path:
    """Install the official, pinned weights, leaving an existing file intact on failure."""
    if MODEL_PATH.is_file() and hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256:
        return MODEL_PATH
    with urlopen(MODEL_URL, timeout=60) as response:
        weights = response.read(MODEL_BYTES + 1)
    if len(weights) != MODEL_BYTES or hashlib.sha256(weights).hexdigest() != MODEL_SHA256:
        raise ValueError("下载的人物分割模型校验失败，请重试。")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as output:
        pending = Path(output.name)
    try:
        pending.write_bytes(weights)
        return pending.replace(MODEL_PATH)
    finally:
        pending.unlink(missing_ok=True)


class PersonSegmenter:
    """Use one instance from one inference thread. Input frames are uint8 BGR."""

    def __init__(self) -> None:
        try:
            weights = MODEL_PATH.read_bytes()
            if hashlib.sha256(weights).hexdigest() != MODEL_SHA256:
                raise ValueError("模型校验失败")
            self._net = cv2.dnn.readNetFromONNX(np.frombuffer(weights, dtype=np.uint8))
        except (OSError, ValueError, cv2.error) as exc:
            raise RuntimeError("无法加载人物遮挡模型。请在项目目录运行 python -m photo_assistant.vision.segmentation 后重启。") from exc

    def segment(self, frame: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("人物分割器已关闭")
        # OpenCV Zoo's model expects RGB channels normalized to [-1, 1].
        blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 127.5, (192, 192), (127.5, 127.5, 127.5), swapRB=True
        )
        self._net.setInput(blob)
        person_probability = self._net.forward()[0, 1]
        # An empty, low-texture view can produce scattered midrange scores.
        if person_probability.max() < 0.8:
            return np.zeros(frame.shape[:2], dtype=np.float32)
        # Keep the soft transition at the silhouette, with opaque interiors.
        confidence = np.clip((person_probability - 0.2) / 0.6, 0.0, 1.0)
        return cv2.resize(confidence, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        self._net = None


if __name__ == "__main__":
    try:
        print(f"人物遮挡模型已就绪：{install_model()}")
    except (OSError, ValueError) as error:
        raise SystemExit(f"无法安装人物遮挡模型：{error}") from error
