"""Exercise co-photo entry with real sprites over a known head-and-shoulders silhouette."""

import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from photo_assistant.avatar.pose import AvatarPose, DEFAULT_SPRITE_CATALOG
from photo_assistant.client import CameraSource, CanvasSource, PersonOcclusion
from photo_assistant.protocol import StreamingCommandParser


class ShoulderEntranceTests(unittest.TestCase):
    def setUp(self):
        self.camera = Mock(spec=CameraSource)
        self.worker = Mock(spec=PersonOcclusion)
        self.canvas = CanvasSource(DEFAULT_SPRITE_CATALOG, 640, 480, 90, threading.Event(),
                                   preview=False, person_occlusion=self.worker)
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        self.set_person(310)
        self.enter()

    def set_person(self, center):
        mask = np.zeros((480, 640), np.float32)
        cv2.ellipse(mask, (center, 73), (23, 29), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (center - 11, 92), (center + 11, 118), 1, -1)
        outline = np.int32([(-13, 108), (-53, 119), (-54, 233), (-37, 250), (-34, 422),
                            (-9, 422), (0, 270), (9, 422), (34, 422), (37, 250),
                            (54, 233), (53, 119), (13, 108)])
        cv2.fillPoly(mask, [outline + (center, 0)], 1)
        frame = np.full((480, 640, 3), (61, 67, 37), np.uint8)
        frame[mask == 1] = (77, 67, 109)
        self.frame, self.mask = frame, mask
        self.camera.latest_frame.return_value = frame
        self.worker.latest.return_value = (frame, mask)
        self.worker.mask_for_capture.return_value = mask

    def enter(self):
        self.canvas.configure_view(self.camera, show_character=True,
                                   entrance=self.canvas.entrance_state(self.camera))

    def test_api_frame_cannot_consume_the_first_visible_preview(self):
        expected = self.canvas.render_preview_frame()
        self.enter()
        # Exercise the preview-enabled policy without opening a desktop window.
        self.canvas.preview = True
        self.canvas.latest_jpeg(self.frame)
        self.send(AvatarPose(face="高冷表情").command())
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), expected)
        self.assertFalse(np.array_equal(self.canvas.render_preview_frame(), expected))

    def test_no_camera_shows_the_entry_pose_on_the_blank_canvas(self):
        self.canvas.configure_view(None, show_character=True, entrance=self.canvas.entrance_state(None))
        self.assertEqual(self.canvas.controller.state_at().pose.torso, "半身歪斜向右")
        self.assertTrue((self.canvas.render_preview_frame() != 255).any())

    def send(self, text):
        for command in StreamingCommandParser().feed("response", text):
            self.canvas.apply_command(command)

    def assert_peek_at(self, image, center):
        delta = np.any(image != self.frame, axis=2)
        # The real person's pixels are preserved. The visible sprite reaches
        # the band beside the shoulder instead of just appearing near the feet.
        self.assertFalse(np.any(delta & (self.mask == 1)))
        left = delta[78:160, center - 100:center - 24].sum()
        right = delta[78:160, center + 24:center + 100].sum()
        self.assertGreater(max(left, right), 180)
        self.assertEqual(self.canvas.controller.state_at().pose, AvatarPose(torso="半身歪斜向右"))
        self.assertEqual(self.canvas.controller.state_at().layer_index, 0)

    def test_wait_for_segmentation_and_reenter_at_new_shoulder_after_teleport(self):
        self.worker.latest.return_value = None
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)
        self.worker.latest.return_value = (self.frame, np.zeros_like(self.mask))
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)

        self.worker.latest.return_value = (self.frame, self.mask)
        peek = self.canvas.render_preview_frame()
        self.assert_peek_at(peek, 310)
        self.send("<layer:to,index=1>")
        unobscured = self.canvas.render_preview_frame()
        body_behind_person = np.any(unobscured != peek, axis=2) & (self.mask == 1)
        self.assertGreater(body_behind_person[160:].sum(), 500)

        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=0.0):
            self.send("<teleport:x=0.85,y=0.85>" + AvatarPose(face="高冷表情").command())
        moved = self.canvas.render_preview_frame()
        self.assertFalse(np.array_equal(peek, moved))
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), moved)
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.85)

        self.canvas.configure_view(self.camera, show_character=False)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)
        self.set_person(160)
        self.enter()
        self.assert_peek_at(self.canvas.render_preview_frame(), 160)
        self.assertLess(self.canvas.controller.state_at().x, 0.45)

    def test_explicit_teleport_supersedes_pending_entrance_and_keeps_pose(self):
        pose = AvatarPose(face="龇牙笑", legs="坐着")
        self.worker.latest.return_value = (self.frame, np.zeros_like(self.mask))
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)
        self.send(pose.command() + "<teleport:x=0.82,y=0.76,scale=0.9>")
        arrived = self.canvas.render_preview_frame()
        visible_x = np.where(np.any(arrived != self.frame, axis=2))[1]
        self.assertGreater(visible_x.min(), self.frame.shape[1] / 2)
        state = self.canvas.controller.state_at()
        self.assertEqual((state.x, state.y, state.scale, state.pose), (0.82, 0.76, 0.9, pose))

        # A mask becoming available later must not move the avatar back to the shoulder.
        self.worker.latest.return_value = (self.frame, self.mask)
        self.canvas.render_preview_frame()
        self.assertEqual(self.canvas.controller.state_at(), state)


if __name__ == "__main__":
    unittest.main()
