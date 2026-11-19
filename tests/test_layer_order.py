"""Verify the avatar's depth slot against a person and a fixed foreground table."""
import io
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from photo_assistant.protocol import StreamingCommandParser
import photo_assistant.client as client
from photo_assistant.avatar.pose import DEFAULT_SPRITE_CATALOG
from tests.avatar_fixture import rendered_avatar


class LayerOrderTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.avatar_color = (100, 140, 180)
        avatar = np.full((24, 24, 4), (*self.avatar_color, 255), np.uint8)
        self.enterContext(rendered_avatar(avatar))

        # The table is in front of the person, also where both masks overlap.
        self.person = np.zeros((64, 64), np.float32)
        self.person[:, :40] = 1
        table = np.zeros_like(self.person)
        table[36:, :] = 1
        self.frame = np.full((64, 64, 3), 10, np.uint8)
        self.frame[:, :40] = (30, 50, 70)
        self.frame[36:, :] = (20, 40, 60)
        self.worker = Mock(spec=client.PersonOcclusion)
        self.worker.latest.return_value = (self.frame, self.person)
        self.worker.mask_for_capture.return_value = self.person
        self.camera = Mock(spec=client.CameraSource)
        self.camera.latest_frame.return_value = self.frame
        self.canvas = client.CanvasSource(
            DEFAULT_SPRITE_CATALOG, 64, 64, 95, threading.Event(), preview=False,
            scene_layers={"person": self.worker, "table": table},
        )
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        self.canvas.configure_view(self.camera, show_character=True)
        self.shutter = client.PhotoCapture(self.camera, self.folder / "photos", self.canvas)
        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=0.0):
            self.dispatch("<teleport:x=0.625,y=0.75>")

    def dispatch(self, text):
        commands = StreamingCommandParser().feed("r", text)
        with redirect_stdout(io.StringIO()):
            client.dispatch_model_commands(commands, self.canvas, client.SpeechQueue(),
                                           photo_capture=self.shutter)

    def expected(self, slot):
        # The actual 24x24 avatar is at x=28..51, y=24..47.
        image = self.frame.copy()
        if slot == 0:  # Behind both real objects.
            image[24:36, 40:52] = self.avatar_color
        elif slot == 1:  # Ahead of the person, behind the table.
            image[24:36, 28:52] = self.avatar_color
        elif slot == 2:  # Ahead of both.
            image[24:48, 28:52] = self.avatar_color
        return image

    def test_all_slots_match_preview_api_and_photo_in_both_directions(self):
        initial = self.canvas.controller.state_at()
        self.assertEqual(initial.layer_index, 0)
        for requested in (None, 1, 2, 1, 0):
            with self.subTest(requested=requested):
                tag = "" if requested is None else f"<layer:to,index={requested}>"
                slot = 0 if requested is None else requested
                existing = set(self.shutter.directory.glob("*.png"))
                self.dispatch(tag + "<capture>")
                photo, = set(self.shutter.directory.glob("*.png")) - existing
                expected = self.expected(slot)
                decoded = cv2.imdecode(np.frombuffer(photo.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(decoded, expected)
                np.testing.assert_array_equal(self.canvas.render_preview_frame(), expected)
                _, jpeg = cv2.imencode(".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.assertEqual(self.canvas.latest_jpeg(self.frame), jpeg.tobytes())
                self.assertEqual(self.canvas.controller.state_at(), replace(initial, layer_index=slot))

    def test_invalid_slot_and_guidance_round_trip_preserve_placement(self):
        self.dispatch("<layer:to,index=1><layer:to,index=3>")
        self.assertEqual(self.canvas.controller.state_at().layer_index, 1)
        self.canvas.configure_view(self.camera, show_character=False)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)
        self.canvas.configure_view(self.camera, show_character=True)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.expected(1))

    def test_arrival_scan_stays_behind_opaque_foreground(self):
        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=12.0):
            self.dispatch("<teleport:x=0.625,y=0.75>")
        preview = self.canvas.render_frame(12.175, camera_frame=self.frame, presenting=True)
        settled = self.expected(0)
        occluded = np.all(settled == self.frame, axis=2)
        np.testing.assert_array_equal(preview[occluded], self.frame[occluded])
        self.assertTrue(np.any(preview[~occluded] != settled[~occluded]))

    def test_shutter_keeps_original_slot_during_mask_inference(self):
        def mask_at_shutter(frame):
            self.assertIs(frame, self.frame)
            self.dispatch("<layer:to,index=2>")
            return self.person

        self.worker.mask_for_capture.side_effect = mask_at_shutter
        np.testing.assert_array_equal(self.canvas.render_capture_frame(self.frame), self.expected(0))
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.expected(2))
