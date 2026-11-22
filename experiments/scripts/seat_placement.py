"""Replay a single co-photo turn on a still scene with the production renderer and API."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import threading

import cv2
import numpy as np

from photo_assistant import client
from photo_assistant.api import session as direct
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.paths import PROJECT_ROOT
from photo_assistant.protocol import Command
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneSegmenter
from photo_assistant.vision.segmentation import PersonSegmenter


class ReplayCamera:
    def __init__(self, frame):
        self.frame = frame

    def latest_frame(self):
        return self.frame


class TurnLog:
    def __init__(self, stop):
        self.rows = []
        self.stop = stop

    def record(self, event, **fields):
        self.rows.append(dict(event=event, **fields))
        if event == "api.receive" and fields["message"]["type"] == "response.done":
            self.stop.set()


def save_frame(path, frame):
    ok, encoded = cv2.imencode(path.suffix, frame)
    if not ok:
        raise ValueError("Cannot encode frame")
    path.write_bytes(encoded.tobytes())


async def send_turn(system, jpeg, placement, text, canvas, stop, log):
    session = await direct.open_session(system)
    try:
        session.submit_frame(jpeg, text, placement=placement)
        await client.receive_realtime_outputs(session, stop, canvas, client.SpeechQueue(),
                                              verbose=False, event_log=log,
                                              mode_selection=client.ModeSelection("co-photo"))
    finally:
        await session.close()


def run(args):
    out = PROJECT_ROOT / "experiments/runtime/20261119-seat-test" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.image.stem
    )
    out.mkdir(parents=True)
    raw = args.image.read_bytes()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    resize = min(1, 960 / max(h, w))
    frame = cv2.resize(frame, (round(w * resize), round(h * resize)), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    camera = ReplayCamera(frame)
    settings = client.parse_args([], default_catalog=WHITE_DRAGON_CATALOG)
    stop = threading.Event()
    log = TurnLog(stop)
    worker = None
    if args.send:
        frame.setflags(write=False)
        worker = PersonOcclusion(SceneSegmenter, max_frame_age=120)
        person_layer = worker
    else:
        segmenter = PersonSegmenter()
        try:
            person_layer = segmenter.segment(frame)
        finally:
            segmenter.close()
    canvas = client.CanvasSource(settings.sprite_catalog, w, h, settings.jpeg_quality, stop,
                                 preview=False, scene_layers={"person": person_layer}, infer_obstacle=True,
                                 project_scene=args.send)
    canvas.start()
    record = dict(image=args.image.name, image_sha256=hashlib.sha256(raw).hexdigest(),
                  dimensions=[w, h], user=args.text, events=log.rows, send=args.send,
                  requested_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                  thinking=os.getenv("DEEPSEEK_THINKING", "low"))
    try:
        # Start each trial with the same visible standing character, already in co-photo.
        entrance = canvas.entrance_state(camera)
        scene_context = ("本次合照使用同一张静态照片，随图提供家具位置。" if args.send
                         else "本次合照使用同一张静态照片和固定的人体轮廓。")
        system = client.build_control_prompt(canvas, settings, mode="co-photo", entrance=entrance,
                                             scene_context=scene_context)
        canvas.configure_view(camera, show_character=True, entrance=entrance)
        if worker is not None:
            worker.submit(frame)
            worker.mask_for_capture(frame)
        canvas.apply_command(Command("tachie", asdict(canvas.avatar_frames.catalog.standing_pose)))
        canvas.apply_command(Command("teleport", {"x": args.x, "y": args.y}))
        jpeg, placement = canvas.jpeg_with_placement(frame)
        (out / "before.jpg").write_bytes(jpeg)
        (out / "system.txt").write_bytes(system.encode())
        record.update(before=placement, system_sha256=hashlib.sha256(system.encode()).hexdigest(),
                      catalog_sha256=hashlib.sha256(settings.sprite_catalog.read_bytes()).hexdigest())
        if args.send:
            asyncio.run(send_turn(system, jpeg, placement, args.text, canvas, stop, log))
        elif args.manual_seat:
            # Positive control: supply the support point and apparent height by hand.
            sx, sy, height_px = args.manual_seat
            catalog = canvas.avatar_frames.catalog
            pose, entry = next((p, e) for p, e in catalog.entries.items() if e.seat_contact_px is not None)
            ax, ay = entry.anchor_px
            contact_x, contact_y = entry.seat_contact_px
            art_scale = height_px / entry.standing_height_px
            commands = [
                Command("tachie", asdict(pose)),
                Command("teleport", {"x": (sx - (contact_x - ax) * art_scale) / w,
                                     "y": (sy - (contact_y - ay) * art_scale) / h,
                                     "scale": height_px / (canvas._fixed.standing_height_ratio * h)}),
            ]
            record["manual_seat"] = args.manual_seat
            client.dispatch_model_commands(commands, canvas, client.SpeechQueue(), log)
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        try:
            final = canvas.render_capture_frame(frame)
            save_frame(out / "after.png", final)
            state = canvas.controller.state_at()
            record["after"] = asdict(state)
            fixed = canvas._fixed
            if fixed is not None:
                record["rendered_placement"] = asdict(fixed)
            entry = canvas.avatar_frames.catalog.entry(state.pose)
            if entry.seat_contact_px is not None and fixed is not None:
                scale = fixed.standing_height_ratio * h / entry.standing_height_px
                ax, ay = entry.anchor_px
                transform = cv2.getRotationMatrix2D((ax, ay), -state.rotation, scale)
                transform[:, 2] += (round(fixed.x * w) - ax, round(fixed.y * h) - ay)
                record["contact_xy_px"] = (transform @ [*entry.seat_contact_px, 1]).round(1).tolist()
        finally:
            if worker is not None:
                worker.close()
            canvas.close()
            (out / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
            print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--text", default="坐下")
    parser.add_argument("--x", type=float, required=True, help="initial standing foot x, in frame widths")
    parser.add_argument("--y", type=float, default=.95)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="send one frame and turn to the configured API")
    mode.add_argument("--manual-seat", type=float, nargs=3, metavar=("X", "Y", "STANDING_HEIGHT"),
                      help="local positive control; all three values use resized-frame pixels")
    run(parser.parse_args())
