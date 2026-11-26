from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_CACHE_ROOT = Path("D:/AI_cache/hum_project_evaluation")


@dataclass(frozen=True)
class EvalPaths:
    cache_root: Path = DEFAULT_CACHE_ROOT

    def __post_init__(self) -> None:
        root = Path(self.cache_root)
        if root.drive.upper() != "D:":
            raise ValueError("Evaluation cache_root must be on the D: drive")
        object.__setattr__(self, "cache_root", root)

    @property
    def datasets(self) -> Path:
        return self.cache_root / "datasets"

    @property
    def models(self) -> Path:
        return self.cache_root / "models"

    @property
    def runs(self) -> Path:
        return self.cache_root / "runs"

    def ensure(self) -> "EvalPaths":
        for path in (self.cache_root, self.datasets, self.models, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        return self


def configure_external_caches(paths: EvalPaths) -> dict[str, str]:
    """Point common ML/package caches at D: and return the applied values."""
    values = {
        "HF_HOME": str(paths.cache_root / "huggingface"),
        "HF_HUB_CACHE": str(paths.cache_root / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(paths.cache_root / "huggingface" / "datasets"),
        "TORCH_HOME": str(paths.cache_root / "torch"),
        "PIP_CACHE_DIR": str(paths.cache_root / "pip"),
        "YOLO_CONFIG_DIR": str(paths.cache_root / "ultralytics"),
        "TMP": str(paths.cache_root / "tmp"),
        "TEMP": str(paths.cache_root / "tmp"),
    }
    for key, value in values.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)
    return values
