"""Replay both hands against the same photographed pier using the live canvas and PNG shutter."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from experiments.scripts.co_photo_interaction import read_frame
from experiments.scripts.seat_placement import ReplayCamera
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.client import CanvasSource, PhotoCapture
from photo_assistant.paths import PROJECT_ROOT
from photo_assistant.protocol import StreamingCommandParser
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneSegmenter


def replay(frame, directory, *, mirrored):
    frame.setflags(write=False)
    camera = ReplayCamera(frame)
    worker = PersonOcclusion(SceneSegmenter, max_frame_age=120)
    height, width = frame.shape[:2]
    canvas = CanvasSource(WHITE_DRAGON_CATALOG, width, height, 90, threading.Event(), preview=False,
                          scene_layers={"person": worker}, infer_obstacle=True, project_scene=True)
    try:
        canvas.start()
        canvas.configure_view(camera, show_character=True, entrance=canvas.entrance_state(camera))
        worker.submit(frame)
        worker.mask_for_capture(frame)
        canvas.render_capture_frame(frame)

        # Independently marked outside corner of the red pier at this camera
        # position (x at ceiling and at floor, y at floor), in 640 x 480 pixels.
        top_x, base_x, base_y = 329, 330, 325
        if mirrored:
            top_x, base_x = width - 1 - top_x, width - 1 - base_x
        wire = f"<brace:x={base_x / width:.3f}>"
        command, = StreamingCommandParser().feed("wall", wire)
        state = canvas.apply_command(command)
        if state is None:
            raise AssertionError(f"No support found for {wire}")
        _, placement = canvas.jpeg_with_placement(frame)
        saved = PhotoCapture(camera, directory, canvas).save()
        photo = saved.replace(directory / ("mirror.png" if mirrored else "original.png"))
        pixels = cv2.imdecode(np.fromfile(photo, np.uint8), cv2.IMREAD_COLOR)
        changed = cv2.absdiff(frame, pixels).max(axis=2) > 10
        hand_x, hand_y = np.array(placement["wall_contact_xy"]) * [width, height]
        expected_x = top_x + (base_x - top_x) * hand_y / base_y
        hand_error = abs(hand_x - expected_x)
        x, y = round(hand_x), round(hand_y)
        touching_pixels = int(changed[y-4:y+5, x-4:x+5].sum())
        expected_arm = "右手扶墙" if mirrored else "左手扶墙"
        assert state.pose.arm == expected_arm, state.pose
        assert hand_error <= 4 and touching_pixels >= 6, (hand_error, touching_pixels)
        assert abs(placement["y"] * height - base_y) <= 12, placement["y"]
        assert (placement["x"] * width < hand_x) != mirrored

        return dict(photo=photo.name, command=wire, arm=state.pose.arm,
                    palm_px=[round(hand_x, 1), round(hand_y, 1)],
                    palm_distance_to_marked_edge_px=round(hand_error, 2),
                    changed_pixels_in_palm_patch=touching_pixels,
                    feet_px=[round(placement["x"] * width, 1), round(placement["y"] * height, 1)])
    finally:
        canvas.close()
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for both shutter images and measurements")
    args = parser.parse_args()
    folder = args.output or PROJECT_ROOT / "experiments/runtime" / datetime.now().strftime("wall-brace-%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=False)
    frame = read_frame(PROJECT_ROOT / "experiments/fixtures/library-pier.png")
    results = [replay(frame, folder, mirrored=False), replay(cv2.flip(frame, 1), folder, mirrored=True)]
    (folder / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(folder)


if __name__ == "__main__":
    main()
