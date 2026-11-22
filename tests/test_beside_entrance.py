"""Exercise co-photo entry and preview timing with real sprites and fixed camera silhouettes."""

import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from photo_assistant.avatar.pose import AvatarPose, WHITE_DRAGON_CATALOG
from photo_assistant.client import CameraSource, CanvasSource, PersonOcclusion
from photo_assistant.protocol import StreamingCommandParser


class BesideEntranceTests(unittest.TestCase):
    def setUp(self):
        self.camera = Mock(spec=CameraSource)
        self.worker = Mock(spec=PersonOcclusion)
        self.canvas = CanvasSource(WHITE_DRAGON_CATALOG, 640, 480, 90, threading.Event(),
                                   preview=False, person_occlusion=self.worker)
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        self.set_person(310)
        self.enter()

    def set_person(self, center, *, closeup=False):
        mask = np.zeros((480, 640), np.float32)
        cv2.ellipse(mask, (center, 73), (23, 29), 0, 0, 360, 1, -1)
        cv2.rectangle(mask, (center - 11, 92), (center + 11, 118), 1, -1)
        outline = np.int32([(-13, 108), (-53, 119), (-54, 233), (-37, 250), (-34, 422),
                            (-9, 422), (0, 270), (9, 422), (34, 422), (37, 250),
                            (54, 233), (53, 119), (13, 108)])
        cv2.fillPoly(mask, [outline + (center, 0)], 1)
        if closeup:
            # Move the person closer to the fixed camera around the top of their head.
            zoom = 2.3
            transform = np.float32([[zoom, 0, center * (1 - zoom)], [0, zoom, 44 * (1 - zoom)]])
            mask = cv2.warpAffine(mask, transform, (640, 480), flags=cv2.INTER_NEAREST)
        frame = np.full((480, 640, 3), (61, 67, 37), np.uint8)
        frame[mask == 1] = (77, 67, 109)
        self.frame, self.mask = frame, mask
        self.camera.latest_frame.return_value = frame
        self.worker.latest.return_value = (frame, mask)
        self.worker.mask_for_capture.return_value = mask

    def enter(self):
        self.canvas.configure_view(self.camera, show_character=True,
                                   entrance=self.canvas.entrance_state(self.camera))

    def test_each_entry_shares_person_components_for_face_and_ground_measurements(self):
        for measurable_face in (True, False):
            with self.subTest(measurable_face=measurable_face):
                if not measurable_face:
                    # With no neck profile, the solver selects the person again
                    # at the final foot point. Both analyses sample this size 1:1.
                    self.mask = np.zeros((180, 240), np.float32)
                    self.mask[22:165, 112:140] = 1
                    self.frame = np.full((180, 240, 3), 75, np.uint8)
                    self.worker.mask_for_capture.return_value = self.mask
                    self.enter()
                with patch.object(cv2, "connectedComponentsWithStats", wraps=cv2.connectedComponentsWithStats) as count:
                    rendered = self.canvas.render_capture_frame(self.frame)
                self.assertEqual(count.call_count, 1)
                self.assertEqual(self.canvas._fixed.framing, "全身")
                self.assertFalse(np.array_equal(rendered, self.frame))

    def test_api_frame_cannot_consume_the_first_visible_preview(self):
        # Exercise the preview-enabled policy without opening a desktop window.
        self.canvas.preview = True
        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=10.0) as clock:
            self.canvas.latest_jpeg(self.frame)
            self.send(AvatarPose(face="高冷表情").command())
            sharp = self.canvas.render_capture_frame(self.frame)
            clock.return_value = 13.0
            first = self.canvas.render_preview_frame()
            self.assertFalse(np.array_equal(first, self.frame))
            self.assertFalse(np.array_equal(first, sharp))
            clock.return_value = 14.0
            np.testing.assert_array_equal(self.canvas.render_preview_frame(), sharp)
            self.assertEqual(self.canvas.controller.state_at().pose.face, "高冷表情")

    def test_no_camera_shows_the_entry_pose_on_the_blank_canvas(self):
        self.canvas.configure_view(None, show_character=True, entrance=self.canvas.entrance_state(None))
        self.assertEqual(self.canvas.controller.state_at().pose.torso, "半身歪斜向右")
        self.assertTrue((self.canvas.render_preview_frame() != 255).any())

    def test_ready_frame_requires_finished_arrival_and_rotation_and_visible_pixels(self):
        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=10.0) as clock:
            self.canvas.render_preview_frame()
            _, entering = self.canvas.jpeg_with_placement(self.frame)
            self.assertFalse(entering["pose_settled"])
            clock.return_value = 11.0
            self.send("<rotate:to,value=20,ms=1000>")
            _, moving = self.canvas.jpeg_with_placement(self.frame)
            self.assertFalse(moving["pose_settled"])
            clock.return_value = 12.0
            _, ready = self.canvas.jpeg_with_placement(self.frame)
            self.assertTrue(ready["pose_settled"] and ready["avatar_visible"])
            self.worker.latest.return_value = (self.frame, np.ones_like(self.mask))
            _, hidden = self.canvas.jpeg_with_placement(self.frame)
            self.assertFalse(hidden["avatar_visible"])
            self.send("<teleport:x=-2,y=-2>")
            clock.return_value = 13.0
            self.worker.latest.return_value = (self.frame, self.mask)
            _, outside = self.canvas.jpeg_with_placement(self.frame)
            self.assertFalse(outside["avatar_visible"])

    def send(self, text):
        for command in StreamingCommandParser().feed("response", text):
            self.canvas.apply_command(command)

    def test_coordinate_moves_project_from_the_same_reference_with_only_a_person_mask(self):
        self.canvas.render_capture_frame(self.frame)
        entry = self.canvas._fixed

        def moved_height(x, y, scale=""):
            self.send(f"<teleport:x={x},y={y}{scale}>")
            self.canvas.render_capture_frame(self.frame)
            return self.canvas._fixed.standing_height_ratio

        distant_height = moved_height(.2, .68)
        self.assertLess(distant_height, entry.standing_height_ratio)
        self.assertAlmostEqual(moved_height(.8, .68), distant_height)
        self.assertAlmostEqual(moved_height(.8, entry.y), entry.standing_height_ratio)
        self.assertAlmostEqual(moved_height(.8, .68, f",scale={entry.scale / 2}"), distant_height / 2)
        self.assertAlmostEqual(moved_height(.2, entry.y), entry.standing_height_ratio / 2)

    def test_wait_for_segmentation_and_reenter_beside_person_after_teleport(self):
        self.worker.latest.return_value = None
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)
        self.worker.latest.return_value = (self.frame, np.zeros_like(self.mask))
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)

        self.worker.latest.return_value = (self.frame, self.mask)
        self.canvas.render_preview_frame()
        _, full = self.canvas.jpeg_with_placement(self.frame)
        self.assertEqual(full["placement_source"], "并排站位")
        self.assertEqual(self.canvas.controller.state_at().pose, self.canvas.avatar_frames.catalog.standing_pose)
        self.assertAlmostEqual(full["y"] * 480, 422, delta=10)
        self.assertGreater(abs(full["x"] * 640 - 310), 50)

        with patch("photo_assistant.avatar.controller.time.monotonic", return_value=0.0):
            self.send("<teleport:x=0.85,y=0.85>" + AvatarPose(face="高冷表情").command())
        moved = self.canvas.render_preview_frame()
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), moved)
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.85)

        self.canvas.configure_view(self.camera, show_character=False)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.frame)
        self.set_person(160, closeup=True)
        self.enter()
        side = self.canvas.render_capture_frame(self.frame)
        _, portrait = self.canvas.jpeg_with_placement(self.frame)
        self.assertEqual(portrait["placement_source"], "自动侧探")
        self.assertGreater(portrait["y"], 1.)
        self.assertLess(portrait["x"], .5)
        self.assertEqual(self.canvas.controller.state_at().pose, self.canvas.avatar_frames.catalog.entry_pose)
        delta = np.any(side != self.frame, axis=2)
        self.assertFalse(np.any(delta & (self.mask == 1)))
        self.assertTrue(delta[60:265, :320].any())
        self.send("<layer:to,index=1>")
        in_front = self.canvas.render_capture_frame(self.frame)
        changed_on_person = np.any(in_front != side, axis=2) & (self.mask == 1)
        self.assertGreater(changed_on_person.sum(), 500)

    def test_explicit_teleport_supersedes_pending_entrance_and_keeps_pose(self):
        pose = AvatarPose(face="高冷表情", legs="坐着")
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
        first_height = self.canvas._fixed.standing_height_ratio
        self.send("<teleport:x=0.82,y=0.94>")
        self.canvas.render_preview_frame()
        self.assertGreater(self.canvas._fixed.standing_height_ratio, first_height)


if __name__ == "__main__":
    unittest.main()
