"""Fixed-camera compositions through the real scene and canvas paths, using numbered exposures."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.client import CameraSource, CanvasSource, PhotoCapture
from photo_assistant.protocol import Command
from photo_assistant.vision.ground import GroundProjection
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneSegmenter
from photo_assistant.vision.scene_objects import SceneObject
from tests.avatar_fixture import rendered_avatar


def exposure(number, person_x):
    frame = np.full((64, 64, 3), 20, np.uint8)
    frame[0, 0] = number
    frame[8:62, person_x:person_x + 8] = (220, 50, 40)
    return frame


def person_mask(frame):
    return (frame[:, :, 0] == 220).astype(np.float32)


class FixedCameraPreviewTests(unittest.TestCase):
    def setUp(self):
        self.first = exposure(1, 21)
        self.second = exposure(2, 34)
        # The background at the person's second location was far from the camera
        # during calibration. Its stored depth must never erase that nearer person.
        first_person = person_mask(self.first)
        cabinet = np.zeros_like(first_person)
        cabinet[4:16, :5] = 1
        self.estimator = Mock(person_boxes=[(21, 8, 29, 62)], person_patches=[])
        self.estimator.infer.return_value = ((SceneObject("cabinet", .9, cabinet),), first_person.copy())
        self.enterContext(patch("photo_assistant.vision.scene.PersonSegmenter",
                                return_value=Mock(segment=person_mask)))
        self.enterContext(patch("photo_assistant.vision.scene.SceneObjectEstimator", return_value=self.estimator))
        self.enterContext(patch("photo_assistant.vision.scene.fit_ground", return_value=GroundProjection(1, 0, .4)))
        self.segmenter = SceneSegmenter(fixed_camera=True)
        self.addCleanup(self.segmenter.close)
        self.worker = Mock(spec=PersonOcclusion)
        self.worker.mask_for_capture.side_effect = self.segmenter.segment
        self.worker.latest.return_value = (self.first, self.segmenter.segment(self.first))

        self.camera = Mock(spec=CameraSource)
        self.camera.latest_frame.return_value = self.second
        self.avatar_color = (100, 140, 180)
        self.enterContext(rendered_avatar(np.full((24, 24, 4), (*self.avatar_color, 255), np.uint8)))
        self.enterContext(patch("photo_assistant.client.projected_height", return_value=24/64))
        self.canvas = CanvasSource(WHITE_DRAGON_CATALOG, 64, 64, 90, threading.Event(),
                                   preview=False, person_occlusion=self.worker, project_scene=True)
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        self.canvas.configure_view(self.camera, show_character=True)
        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=0):
            self.canvas.apply_command(Command("teleport", {"x": .5, "y": .75}))

    def expected(self, frame):
        image = frame.copy()
        image[24:48, 20:44] = self.avatar_color
        person = person_mask(frame).astype(bool)
        image[person] = frame[person]
        return image

    def test_moving_person_has_matching_preview_mask_and_no_outline_at_old_position(self):
        # Camera is one exposure ahead of the worker. Both the hand and its RGB
        # underneath the avatar must come from the same exposure.
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.expected(self.first))
        self.worker.latest.return_value = self.second, self.segmenter.segment(self.second)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.expected(self.second))
        self.worker.mask_for_capture.assert_not_called()
        self.estimator.infer.assert_called_once()

    def test_photo_segments_the_requested_exposure_while_preview_still_shows_the_previous_one(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = PhotoCapture(self.camera, folder, self.canvas).save()
        photo = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(photo, self.expected(self.second))
        self.assertIs(self.worker.mask_for_capture.call_args.args[0], self.second)
        self.estimator.infer.assert_called_once()

    def test_preview_uses_new_raw_resolution_until_a_matching_analysis_is_ready(self):
        resized = cv2.resize(self.second, (32, 48))
        self.camera.latest_frame.return_value = resized
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), resized)
        self.worker.mask_for_capture.assert_not_called()
