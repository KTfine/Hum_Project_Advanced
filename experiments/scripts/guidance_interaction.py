"""Run the guidance prompt against a bookshelf-to-glass-wall pair of generated camera views."""

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from photo_assistant import client
from photo_assistant.api.session import open_session
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG


class Camera:
    def latest_frame(self):
        return self.frame


class TurnLog(client.FrontendEventLogger):
    def __init__(self, path):
        super().__init__(path)
        self.completed = asyncio.Queue()

    def record(self, event, **fields):
        super().record(event, **fields)
        if event == "api.receive" and fields["message"]["type"] in {"response.done", "error"}:
            self.completed.put_nowait(fields["message"])


async def run(folder):
    output = folder / ("guidance-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir()
    args = client.parse_args(["--no-preview"], default_catalog=WHITE_DRAGON_CATALOG)
    camera = Camera()
    stop = threading.Event()
    canvas = client.CanvasSource(args.sprite_catalog, 640, 480, args.jpeg_quality, stop, preview=False)
    canvas.start()
    canvas.configure_view(camera, show_character=False)
    system = client.build_control_prompt(canvas, args, mode="guidance")
    frames = {}
    for name in ("person-left", "person-right"):
        source = cv2.imdecode(np.fromfile(folder / (name + ".png"), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = source.shape[:2]
        frames[name] = cv2.resize(source, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
    logger = TurnLog(output / "guidance-events.jsonl")
    logger.record("experiment.setup", mode="guidance", system_prompt=system,
                  system_sha256=hashlib.sha256(system.encode()).hexdigest(), input_files=list(frames))
    session = await open_session(system)
    selection = client.ModeSelection()
    receiver = asyncio.create_task(client.receive_realtime_outputs(
        session, stop, canvas, client.SpeechQueue(), verbose=False, event_log=logger,
        mode_selection=selection, photo_capture=client.PhotoCapture(camera, output / "photos")))
    records = []

    async def step(label, image, instruction):
        camera.frame = frames[image]
        jpeg, placement = canvas.jpeg_with_placement(camera.frame, include_character=False)
        assert placement is None
        logger.record("experiment.input", turn=label, image=image, text=instruction)
        session.submit_frame(jpeg, instruction)
        event = await asyncio.wait_for(logger.completed.get(), timeout=100)
        record = dict(turn=label, image=image, instruction=instruction, event=event)
        record["photos_after"] = [p.name for p in sorted((output / "photos").glob("*.png"))]
        records.append(record)
        (output / "guidance-results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(label, event.get("text", ""), flush=True)
        if event["type"] == "error":
            raise RuntimeError(event["error"])
        if receiver.done():
            raise RuntimeError(f"Guidance exited unexpectedly: {receiver.result()!r}")

    try:
        await step("visual-advice", "person-left", "我想拍一张自然的全身照。你看我站在什么旁边？给我一个现在就能做的小调整，先别拍。")
        await step("preference", "person-left", "我还是想站在原地，不想靠着家具。只调整一下上半身，怎么做自然？")
        await step("arrived-at-glass", "person-right", "我换到玻璃隔断旁了。看现在的画面，我的站姿还需要调整吗？")
        await step("shutter", "person-right", "现在就帮我拍一张吧。")
        photos = list((output / "photos").glob("*.png"))
        exact = len(photos) == 1 and np.array_equal(cv2.imdecode(np.fromfile(photos[0], np.uint8), 1),
                                                   frames["person-right"])
        logger.record("experiment.check", exactly_one_correct_raw_photo=exact)
        print("one matching photo:", exact, "results:", output)
    finally:
        await session.close()
        await asyncio.gather(receiver, return_exceptions=True)
        canvas.close()
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    asyncio.run(run(parser.parse_args().folder))
