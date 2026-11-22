"""Replay 30 camera samples per simulated second, with analysis every eighth sample.

Run from the project root: python -m experiments.scripts.preview_cadence
"""
from dataclasses import replace
import json
import threading
import time

import cv2
import numpy as np

from photo_assistant.avatar.controller import TELEPORT_EFFECT_SECONDS
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.client import CanvasSource
from photo_assistant.protocol import Command
from photo_assistant.vision.ground import GroundProjection
from photo_assistant.vision.scene import SceneAnalysis
from photo_assistant.vision.scene_objects import SceneObject


class Replay:
    """Publish immutable, numbered exposures and independently advance the scene result."""

    def __init__(self, width, height):
        person = np.zeros((height, width), np.float32)
        cv2.ellipse(person, (width // 3, height // 5), (width // 36, height // 18), 0, 0, 360, 1, -1)
        person[height // 4:height * 9 // 10, width * 3 // 10:width * 11 // 30] = 1
        table = np.zeros_like(person)
        table[height // 2:height * 3 // 5, width * 2 // 3:width * 5 // 6] = 1
        depth = np.linspace(.1, 1, height, dtype=np.float32)[:, None] + np.zeros_like(person)
        self.scene = SceneAnalysis(person, (SceneObject("dining table", .9, table),), depth,
                                   GroundProjection(.9, .1, .4), table, [])
        self.background = np.full((height, width, 3), 110, np.uint8)
        self.background[person == 1] = (80, 90, 140)
        self.background[table == 1] = (50, 80, 100)
        self.advance(0)

    def advance(self, number):
        self.frame = self.background.copy()
        self.frame[:4, :4] = number
        self.frame.setflags(write=False)
        if number % 8 == 0:
            self.analyzed = self.frame, replace(self.scene)

    def set_enabled(self, enabled):
        pass

    def latest(self):
        return self.analyzed

    def latest_frame(self):
        return self.frame

    def mask_for_capture(self, frame):
        raise AssertionError("preview requested synchronous analysis")


def replay(width, height):
    source = Replay(width, height)
    canvas = CanvasSource(WHITE_DRAGON_CATALOG, width, height, 85, threading.Event(),
                          preview=False, person_occlusion=source, infer_obstacle=True, project_scene=True)
    canvas.start()
    try:
        for show_character in (False, True):
            source.advance(0)
            entrance = canvas.entrance_state(source) if show_character else None
            canvas.configure_view(source, show_character=show_character, entrance=entrance)
            if show_character:
                canvas.apply_command(Command("teleport", {"x": .58, "y": .88}))
            canvas.render_preview_frame()
            time.sleep(TELEPORT_EFFECT_SECONDS + .01)
            markers, elapsed = [], []
            for index in range(120):
                source.advance(index)
                start = time.perf_counter()
                output = canvas.render_preview_frame()
                elapsed.append(1000 * (time.perf_counter() - start))
                markers.append(int(output[0, 0, 0]))
            if show_character and np.array_equal(output[4:], source.frame[4:]):
                raise AssertionError("avatar did not appear")
            print(json.dumps(dict(size=[width, height], co_photo=show_character,
                                  input_frames=len(markers), unique_preview_frames=len(set(markers)),
                                  render_ms_p50=round(float(np.median(elapsed)), 2),
                                  render_ms_p95=round(float(np.percentile(elapsed, 95)), 2))))
    finally:
        canvas.close()


if __name__ == "__main__":
    for size in ((640, 480), (1280, 720)):
        replay(*size)
