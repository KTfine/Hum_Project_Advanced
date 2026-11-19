"""Exercise camera-ground calibration through PNG decoding, actions and compositing."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from photo_assistant.client import CanvasSource
from photo_assistant.protocol import StreamingCommandParser


class CameraProjectionTests(unittest.TestCase):
    def setUp(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = dict(face="微笑", hand="无动作", arm="自然下垂", torso="正着站", legs="站着")
        self.tags = [base, dict(base, arm="伸直举高过头"), dict(base, legs="坐着")]
        entries = []
        for idx, tags in enumerate(self.tags):
            image = np.zeros((180, 42, 4), np.uint8)
            # Same scale and non-centered foot anchor for each picture.
            crown = 85 if tags["legs"] == "坐着" else 30
            image[crown:170, 4:17] = 255
            if tags["arm"] == "伸直举高过头":
                image[14:80, 17:21] = (0, 255, 0, 255)
            filename = f"pose{idx}.png"
            (folder / filename).write_bytes(cv2.imencode(".png", image)[1].tobytes())
            entries.append(dict(file=filename, tags=tags, standing_height_px=140, anchor_px=[10, 170]))
        catalog = folder / "catalog.json"
        catalog.write_text(json.dumps({"sprites": entries}, ensure_ascii=False), encoding="utf-8")
        self.frame = np.zeros((200, 320, 3), np.uint8)
        self.mask = np.zeros((200, 320), np.float32)
        self.mask[30:180, 30:70] = 1
        self.source = Mock()
        self.source.latest.return_value = self.frame, self.mask
        self.source.mask_for_capture.return_value = self.mask
        self.canvas = CanvasSource(catalog, 320, 200, 95, threading.Event(),
                                   preview=False, scene_layers={"person": self.source})
        self.canvas.start()
        self.addCleanup(self.canvas.close)

    def send(self, text):
        command, = StreamingCommandParser().feed("r", text)
        self.canvas.apply_command(command)

    def select(self, tags):
        self.send("<tachie:" + ",".join(f"{k}={v}" for k, v in tags.items()) + ">")

    def preview(self):
        return self.canvas.render_frame(camera_frame=self.frame)

    @staticmethod
    def bounds(image, channel=0):
        ys, xs = np.where(image[:, :, channel] > 127)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    def test_teleport_sets_both_feet_and_leaves_automatic_ground_follow(self):
        # A 150 px tall subject on row 180 sets a 140 px tall standing sprite.
        self.assertEqual(self.bounds(self.preview()), (154, 40, 167, 180))
        self.send("<teleport:x=0.75,y=0.9>")
        self.assertEqual(self.bounds(self.preview()), (234, 40, 247, 180))
        self.mask[:] = 0
        self.mask[10:160, 30:70] = 1
        self.assertEqual(self.bounds(self.preview())[3], 180)
        self.assertEqual(self.canvas.controller.state_at().y, 0.9)

    def test_destination_depth_controls_perspective_independent_of_layer_slot(self):
        self.preview()
        self.send("<layer:to,index=1>")
        self.send("<teleport:x=0.5,y=0.75>")
        _, far_top, _, far_foot = self.bounds(self.preview())
        self.send("<teleport:x=0.5,y=0.95>")
        _, near_top, _, near_foot = self.bounds(self.preview())
        self.assertEqual((far_foot, near_foot), (150, 190))
        self.assertLess(far_foot - far_top, 140)
        self.assertGreater(near_foot - near_top, 140)

    def test_raised_hand_and_seated_head_use_common_scale_and_foot_anchor(self):
        standing = self.bounds(self.preview())
        self.select(self.tags[1])
        raised = self.preview()
        self.assertEqual(self.bounds(raised), standing)
        self.assertLess(self.bounds(raised, channel=1)[1], standing[1])
        self.select(self.tags[2])
        self.assertEqual(self.bounds(self.preview()), (154, 95, 167, 180))

    def test_shutter_uses_its_own_mask_and_pose_while_a_teleport_is_dispatched(self):
        self.preview()
        shutter_mask = np.zeros_like(self.mask)
        shutter_mask[10:160, 30:70] = 1

        def segment(frame):
            self.assertIs(frame, self.frame)
            # Dispatch while the shutter awaits exact-frame segmentation.
            self.send("<teleport:x=0.5,y=0.7>")
            self.select(self.tags[2])
            return shutter_mask

        self.source.mask_for_capture.side_effect = segment
        self.assertEqual(self.bounds(self.canvas.render_capture_frame(self.frame)), (154, 20, 167, 160))
        self.assertEqual(self.canvas.controller.state_at().y, 0.7)
        self.assertEqual(self.bounds(self.preview())[3], 140)

    def test_arrival_previews_at_destination_while_jpeg_and_shutter_are_clear(self):
        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=12.0):
            self.send("<teleport:x=0.75,y=0.9,scale=0.5>")
            settled = self.canvas.render_capture_frame(self.frame)
            self.assertEqual(self.bounds(settled), (237, 110, 244, 180))
            visible = np.any(settled != self.frame, axis=2)
            for now in (12.0, 12.17):
                transient = self.canvas.render_frame(now, camera_frame=self.frame, presenting=True)
                np.testing.assert_array_equal(np.any(transient != self.frame, axis=2), visible)
                self.assertFalse(np.array_equal(transient, settled))
            clear = self.canvas.render_frame(12.36, camera_frame=self.frame, presenting=True)
            np.testing.assert_array_equal(clear, settled)
            _, encoded = cv2.imencode(".jpg", settled, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.assertEqual(self.canvas.latest_jpeg(self.frame), encoded.tobytes())
