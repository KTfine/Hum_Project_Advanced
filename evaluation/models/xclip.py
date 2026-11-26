from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PIL import Image

from evaluation.data.penn_action import PENN_ACTIONS


XCLIP_VARIANTS = (
    ("large-patch14", "microsoft/xclip-large-patch14"),
    ("base-patch32", "microsoft/xclip-base-patch32"),
)


def xclip_model_name(variant: str) -> str:
    for name, _model_id in XCLIP_VARIANTS:
        if name == variant:
            return f"xclip-{name}"
    raise ValueError(f"unsupported X-CLIP variant: {variant}")


def penn_text_prompts() -> list[str]:
    phrases = {
        "bowl": "bowling", "pullup": "pull ups", "pushup": "push ups",
        "situp": "sit ups", "squat": "squats", "strum_guitar": "strumming a guitar",
    }
    return [
        f"a video of a person performing {phrases[label]}"
        if label in phrases
        else f"a video of a person performing a {label.replace('_', ' ')}"
        for label in PENN_ACTIONS
    ]


class XClipPennClassifier:
    def __init__(
        self, model_id: str, *, cache_dir: str | Path, device: str = "cuda",
    ):
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        self.torch = torch
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = XCLIPProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model = XCLIPModel.from_pretrained(
            model_id, cache_dir=str(cache_dir), dtype=dtype,
        ).to(device)
        self.model.eval()

    def predict(self, frame_paths: list[str | Path]) -> list[float]:
        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        inputs = self.processor(
            text=penn_text_prompts(), images=frames, return_tensors="pt", padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits_per_video[0]
            probabilities = self.torch.softmax(logits.float(), dim=-1)
        return probabilities.detach().cpu().numpy().astype(float).tolist()


def run_xclip_sample(
    sample: dict, classifier: XClipPennClassifier | None, *, model_name: str,
) -> dict:
    if classifier is None:
        return {
            "sample_id": sample["sample_id"], "model": model_name,
            "status": "blocked", "reason": "X-CLIP failed to initialize",
            "tasks": ["activity"],
        }
    started = time.perf_counter()
    try:
        probabilities = classifier.predict(sample["selected_frame_paths"])
        if len(probabilities) != len(PENN_ACTIONS) or not np.isclose(sum(probabilities), 1.0):
            raise ValueError("X-CLIP returned invalid Penn probabilities")
        return {
            "sample_id": sample["sample_id"], "model": model_name, "status": "ok",
            "prediction": {"activity_probabilities": probabilities},
            "elapsed_s": time.perf_counter() - started, "tasks": ["activity"],
        }
    except Exception as exc:
        return {
            "sample_id": sample["sample_id"], "model": model_name, "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started, "tasks": ["activity"],
        }
