"""Evaluate a real multi-turn co-photo session with two exposures from a fixed camera."""
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

from experiments.scripts.seat_placement import ReplayCamera, TurnLog, save_frame
from photo_assistant import client
from photo_assistant.api import session as direct
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneSegmenter


def read_frame(path, crop=None):
    frame = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot read {path}")
    if crop is not None:
        x0, y0, x1, y1 = crop
        frame = frame[y0:y1, x0:x1]
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
    frame.setflags(write=False)
    return frame


def write_json(path, obj):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


async def replay(left, right, out, *, portrait_entry=False, standing_pair=False, table_choice=False):
    out.mkdir(parents=True, exist_ok=False)
    # Digital zoom of the original exposure keeps the subject and room constant.
    crop = (0, 175, 704, 703) if portrait_entry else None
    frames = {p.name: read_frame(p, crop) for p in ((left,) if portrait_entry else (left, right))}
    camera = ReplayCamera(frames[left.name])
    h, w = camera.frame.shape[:2]
    args = client.parse_args(["--no-preview"], default_catalog=WHITE_DRAGON_CATALOG)
    worker = PersonOcclusion(SceneSegmenter, max_frame_age=120)
    canvas = client.CanvasSource(args.sprite_catalog, w, h, args.jpeg_quality, threading.Event(), preview=False,
                                 scene_layers={"person": worker}, infer_obstacle=True, project_scene=True)
    modes = client.ModeSelection()
    session = None
    speech = client.SpeechQueue()
    record = dict(model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), thinking=os.getenv("DEEPSEEK_THINKING", "low"),
                  input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (left, right)},
                  dimensions=[w, h], crop_pixels=crop, prompts={}, turns=[])
    manifest = out / "manifest.json"

    async def open_view(handoff=(), pending=None):
        nonlocal session
        mode, revision = modes.snapshot()
        showing = mode == "co-photo"
        entrance = canvas.entrance_state(camera) if showing else None
        system = client.build_control_prompt(canvas, args, mode=mode, entrance=entrance)
        record["prompts"][mode] = hashlib.sha256(system.encode()).hexdigest()
        (out / (mode + "-system.txt")).write_text(system, encoding="utf-8")
        session = await direct.open_session(system, managed_observations=True, observe_on_start=showing,
                                            handoff_messages=handoff, pending_observation=pending)
        modes.mark_loaded(mode, revision)
        canvas.configure_view(camera, show_character=showing, entrance=entrance)
        if showing:
            worker.submit(camera.frame)
            worker.mask_for_capture(camera.frame)

    async def turn(stage, text, source):
        camera.frame = frames[source.name]
        showing = modes.mode == "co-photo"
        if showing:
            worker.submit(camera.frame)
            worker.mask_for_capture(camera.frame)
        jpeg, before = canvas.jpeg_with_placement(camera.frame, include_character=showing)
        (out / (stage + "-input.jpg")).write_bytes(jpeg)
        stop = threading.Event()
        log = TurnLog(stop)
        row = dict(stage=stage, user=text, source=source.name, mode=modes.mode, before=before,
                   pending_before=session.pending_observation, events=log.rows)
        record["turns"].append(row)
        write_json(manifest, record)
        previous_id = session._response_id
        session.submit_frame(jpeg, text, placement=before)
        if session._response_id != previous_id:
            shutter = client.PhotoCapture(camera, out / "photos", character_canvas=canvas if showing else None)
            try:
                transition = await asyncio.wait_for(client.receive_realtime_outputs(
                    session, stop, canvas, speech, verbose=False, event_log=log,
                    mode_selection=modes, photo_capture=shutter), timeout=105)
            finally:
                write_json(manifest, record)
            done = next((r["message"] for r in reversed(log.rows) if r["event"] == "api.receive"
                         and r["message"]["type"] == "response.done"), {})
            row["reply"] = done.get("text")
            row["metadata"] = done.get("metadata")
            if transition:
                history, pending = session.handoff_messages(), session.pending_observation
                await session.close()
                await open_view(history, pending)
                client.dispatch_model_commands(transition[1], canvas, speech, log, mode_selection=modes,
                                               photo_capture=client.PhotoCapture(camera, out / "photos", canvas))
        row["pending_after"] = session.pending_observation
        row["png"] = stage + ".png"
        _, row["after"] = canvas.jpeg_with_placement(camera.frame, include_character=modes.mode == "co-photo")
        row["state"] = asdict(canvas.controller.state_at())
        shot = canvas.render_capture_frame(camera.frame) if modes.mode == "co-photo" else camera.frame
        save_frame(out / row["png"], shot)
        write_json(manifest, record)
        print(stage, row.get("reply") or "(no response scheduled)", flush=True)

    try:
        canvas.start()
        await open_view()
        await turn("invite", "小白龙，一起来和我合照吧，先不拍。", left)
        if modes.mode != "co-photo":
            await turn("invite-retry", "请现在进入合照模式，到我旁边来。", left)
        if modes.mode != "co-photo":
            raise RuntimeError("The assistant failed to enter co-photo mode after two user requests")
        if portrait_entry:
            await turn("portrait-shutter", "保持刚入镜的位置，从我身后露着脸，帮我们拍一张。", left)
            return
        if standing_pair:
            await turn("wait-move", None, right)
            await turn("follow", "跟过来。", right)
            await turn("standing-shutter", "站在我旁边，一起看镜头，现在帮我们拍一张两个人都入镜的全身合照。", right)
            return
        if table_choice:
            await turn("far-table", "到远处那张白桌上坐着，腿垂在桌沿外。", left)
            await turn("near-table", "现在换到靠近镜头的那张桌子上坐，先别拍。", left)
            await turn("table-shutter", "保持现在的位置，拍一张。", left)
            return
        await turn("behind", "站到我身后一点，从我的肩膀旁边露出脸，让我能挡住你一部分。先别拍。", left)
        await turn("wait-move", None, right)
        await turn("follow", "我换到另一边了，跟过来，站到我身边。", right)
        await turn("seat", "你坐下来吧。", right)
        await turn("seat-table", "我指的是靠镜头的白桌，坐在它的前沿，腿自然垂下来。", right)
        await turn("destination", "现在站起来，去画面左边靠书柜的空地，面向镜头，给我右边留点空间。", right)
        await turn("pair", "最后还是过来和我并排站，咱俩都要入镜。先别拍。", right)
        await turn("shutter", "就这样，我们一起看镜头，拍一张合照吧。", right)
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        write_json(manifest, record)
        if session:
            await session.close()
        canvas.close()
        worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output_root", type=Path)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--portrait-entry", action="store_true", help="Replay a digital crop of the left exposure")
    profile.add_argument("--standing-pair", action="store_true", help="Follow and take a full-length photo in a fresh session")
    profile.add_argument("--table-choice", action="store_true", help="Sit on far and near tables, then take a seated photo")
    args = parser.parse_args()
    prefix = ("portrait" if args.portrait_entry else "standing" if args.standing_pair
              else "tables" if args.table_choice else "co-photo")
    folder = args.output_root / datetime.now().strftime(prefix + "-%Y%m%d-%H%M%S-%f")
    print(folder.resolve(), flush=True)
    asyncio.run(replay(args.left, args.right, folder, portrait_entry=args.portrait_entry,
                       standing_pair=args.standing_pair, table_choice=args.table_choice))
