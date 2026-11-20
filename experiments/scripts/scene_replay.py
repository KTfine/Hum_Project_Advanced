"""Replay a fixed camera exposure through the local co-photo renderer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.client import CanvasSource
from photo_assistant.protocol import Command
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneSegmenter


def save(path, frame):
    cv2.imencode(".png", frame)[1].tofile(str(path))


def replay(path, output, x, far_y, near_y, width_limit=640):
    raw = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"Cannot decode {path}")
    if raw.shape[1] > width_limit:
        raw = cv2.resize(raw, (width_limit, round(raw.shape[0] * width_limit / raw.shape[1])),
                         interpolation=cv2.INTER_AREA)
    raw.setflags(write=False)
    height, width = raw.shape[:2]
    output.mkdir(parents=True, exist_ok=True)
    worker = PersonOcclusion(SceneSegmenter, max_frame_age=120)
    canvas = CanvasSource(WHITE_DRAGON_CATALOG, width, height, 95, threading.Event(),
                          preview=False, scene_layers={"person": worker}, infer_obstacle=True)
    class StillCamera:
        def latest_frame(self):
            return raw

    try:
        camera = StillCamera()
        canvas.start()
        canvas.configure_view(camera, show_character=True, entrance=canvas.entrance_state(camera))
        started = time.monotonic()
        worker.submit(raw)
        scene = worker.mask_for_capture(raw)
        elapsed = time.monotonic() - started
        depth_image = cv2.applyColorMap((255 * scene.inverse_depth).astype(np.uint8), cv2.COLORMAP_INFERNO)
        save(output / "relative-depth.png", depth_image)
        annotated = raw.copy()
        palette = [(32, 180, 255), (240, 180, 55), (150, 60, 215), (60, 185, 100)]
        for i, (obj, info) in enumerate(zip(scene.objects, scene.descriptions)):
            color = palette[i % len(palette)]
            covered = obj.mask >= 0.5
            annotated[covered] = annotated[covered] * 0.55 + np.array(color) * 0.45
            xy = np.array(info["box"][:2]) * [width, height]
            cv2.putText(annotated, f"{i + 1} {obj.label}", tuple(xy.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        save(output / "instances.png", annotated)
        _, entry = canvas._compose_frame(camera_frame=raw, capture=True)
        metrics = dict(objects=dict(Counter(o.label for o in scene.objects)), cold_seconds=round(elapsed, 3),
                       floor_fitted=scene.ground is not None, entry=entry, moves={})
        samples = []
        for label, y in (("far", far_y), ("near", near_y), ("far-return", far_y)):
            canvas.apply_command(Command("teleport", {"x": x, "y": y, "depth": label.split("-")[0]}))
            shot = canvas.render_capture_frame(raw)
            _, pose = canvas.jpeg_with_placement(raw)
            if pose is None:
                raise RuntimeError("No matched preview")
            assert np.array_equal(canvas.render_frame(camera_frame=raw), shot)
            save(output / (label + ".png"), shot)
            # Replay the identical placement in front of all real layers.
            canvas.apply_command(Command("layer", {"index": 2}))
            unobscured = canvas.render_capture_frame(raw)
            occupied = np.any(unobscured != raw, axis=2)
            hidden = np.any(unobscured != shot, axis=2) & occupied
            metrics["moves"][label] = dict(foot=[pose["x"], pose["y"]],
                                           standing_px=round(pose["standing_height_ratio"] * height, 2),
                                           occluded_pixels=int(hidden.sum()), sprite_pixels=int(occupied.sum()))
            samples.append(shot)
        metrics["far_round_trip_equal"] = bool(np.array_equal(samples[0], samples[2]))
        save(output / "comparison.png", np.hstack(samples[:2]))
        (output / "measurements.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False))
    finally:
        canvas.close()
        worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--x", type=float, default=0.54)
    parser.add_argument("--far-y", type=float, default=0.64)
    parser.add_argument("--near-y", type=float, default=1.1)
    args = parser.parse_args()
    replay(args.image, args.output, args.x, args.far_y, args.near_y)
