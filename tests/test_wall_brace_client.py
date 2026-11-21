"""Exercise wall contact, shutter and departure on a fixed RGB/depth camera."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from photo_assistant.avatar.pose import AvatarPose, WHITE_DRAGON_CATALOG
from photo_assistant.client import CanvasSource, PhotoCapture
from photo_assistant.protocol import StreamingCommandParser
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneAnalysis


class WallBraceClientTests(unittest.TestCase):
    def setUp(self):
        self.camera = Mock()
        self.worker = Mock(spec=PersonOcclusion)
        self.canvas = CanvasSource(WHITE_DRAGON_CATALOG, 640, 480, 90, threading.Event(),
                                   preview=False, scene_layers={"person": self.worker}, project_scene=True)
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        self.canvas.configure_view(self.camera, show_character=True)
        self.parser = StreamingCommandParser()
        self.set_view(pillar=True)
        self.send("<stand><teleport:x=0.3,y=0.85>")
        self.canvas.render_capture_frame(self.frame)

    def set_view(self, *, pillar, person=False):
        frame = np.full((480, 640, 3), 180, np.uint8)
        depth = np.full((480, 640), .25, np.float32)
        mask = np.zeros_like(depth)
        if pillar:
            frame[:380, 325:535] = (64, 84, 175)
            depth[:380, 325:535] = .72
        if person:
            cv2.ellipse(mask, (135, 116), (19, 26), 0, 0, 360, 1, -1)
            cv2.rectangle(mask, (105, 137), (165, 290), 1, -1)
            cv2.rectangle(mask, (113, 280), (133, 418), 1, -1)
            cv2.rectangle(mask, (143, 280), (160, 418), 1, -1)
            frame[mask > 0] = (40, 65, 75)
        scene = SceneAnalysis(mask, (), depth, None, np.zeros_like(mask), [])
        self.frame = frame
        self.camera.latest_frame.return_value = frame
        self.worker.latest.return_value = frame, scene
        self.worker.mask_for_capture.return_value = scene

    def send(self, tags):
        commands = self.parser.feed("user", tags)
        self.assertTrue(commands)
        return [self.canvas.apply_command(c) for c in commands][-1]

    def snapshot(self):
        return self.canvas._compose_frame(camera_frame=self.frame, capture=True)

    def test_repeated_pose_and_saved_shutter_keep_the_same_palm_contact(self):
        state = self.send("<brace:x=0.51>")
        self.assertEqual(state.pose.arm, "左手扶墙")
        shot, placement = self.snapshot()
        self.assertAlmostEqual(placement["wall_contact_xy"][0] * 640, 325, delta=2)
        self.assertLess(placement["x"], 325 / 640)
        self.assertTrue(np.any(shot != self.frame))
        self.assertEqual(self.send(state.pose.command()), state)
        # The same camera view delivered in a new buffer must keep the pose settled.
        self.set_view(pillar=True)
        new_shot, new_placement = self.snapshot()
        self.assertEqual(new_placement, placement)
        np.testing.assert_array_equal(new_shot, shot)
        with tempfile.TemporaryDirectory() as tmp:
            saved = PhotoCapture(self.camera, Path(tmp), self.canvas).save()
            actual = cv2.imdecode(np.frombuffer(saved.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(actual, shot)
        self.send("<stand>")
        _, upright = self.snapshot()
        self.assertNotIn("wall_contact_xy", upright)
        self.assertEqual(self.canvas.controller.state_at().pose, self.canvas.avatar_frames.catalog.standing_pose)
        self.assertEqual((placement["x"], placement["y"]), (upright["x"], upright["y"]))

    def test_moving_elsewhere_releases_wall_contact_and_uses_standing_pose(self):
        for move in ("<teleport:x=0.35,y=0.84>", "<follow>"):
            with self.subTest(move=move):
                self.set_view(pillar=True, person=True)
                self.assertIsNotNone(self.send("<brace:x=0.83>"))
                _, braced = self.snapshot()
                self.assertIn("wall_contact_xy", braced)
                self.assertIsNotNone(self.send(move))
                _, moved = self.snapshot()
                self.assertNotIn("wall_contact_xy", moved)
                self.assertNotEqual(braced["x"], moved["x"])
                self.assertEqual(self.canvas.controller.state_at().pose, self.canvas.avatar_frames.catalog.standing_pose)

    def test_unsuitable_scene_preserves_previous_position_pose_and_capture(self):
        self.set_view(pillar=False)
        state = self.canvas.controller.state_at()
        before, place = self.snapshot()
        for request in ("<brace>", AvatarPose(arm="右手扶墙").command()):
            self.assertIsNone(self.send(request))
            after, current = self.snapshot()
            self.assertEqual(state, self.canvas.controller.state_at())
            self.assertEqual(place, current)
            np.testing.assert_array_equal(before, after)
