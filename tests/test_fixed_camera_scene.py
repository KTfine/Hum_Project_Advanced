"""Fixed-camera calibration, actor depth and segmentation worker lifetimes."""
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.ground import GroundProjection
from photo_assistant.vision.scene import SceneSegmenter
from photo_assistant.vision.scene_objects import SceneObject


def exposure(x, bottom=54, size=(60, 80)):
    image = np.zeros((*size, 3), np.uint8)
    image[bottom - 32:bottom, x:x + 10, 0] = 255
    return image


class FixedSceneTests(unittest.TestCase):
    def setUp(self):
        self.person = Mock()
        self.person.segment.side_effect = lambda frame: frame[:, :, 0].astype(np.float32) / 255
        self.estimator = Mock(person_boxes=(), person_patches=())
        self.estimator.infer.side_effect = self.infer
        self.enterContext(patch("photo_assistant.vision.scene.PersonSegmenter", return_value=self.person))
        self.enterContext(patch("photo_assistant.vision.scene.SceneObjectEstimator", return_value=self.estimator))

    def infer(self, frame):
        # A steady chair beside the initial subject. The detector incorrectly
        # extends its mask onto the subject; the depth network also sees their body.
        actor = (frame[:, :, 0] != 0).astype(np.float32)
        ys, xs = np.nonzero(actor)
        self.estimator.person_boxes = [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)]
        self.estimator.person_patches = [(self.estimator.person_boxes[0], np.ones((32, 10), np.float32))]
        chair = actor.copy()
        chair[25:51, 33:44] = 1
        return (SceneObject("chair", .92, chair),), np.where(actor > 0, .95, .30).astype(np.float32)

    def test_only_static_geometry_is_reused_as_the_person_crosses_the_image(self):
        model = SceneSegmenter(fixed_camera=True)
        first = model.segment(exposure(4))
        second = model.segment(exposure(55))
        self.assertTrue(first.fixed_camera and second.fixed_camera)
        self.assertIs(first.calibration, second.calibration)
        self.assertIs(first.inverse_depth, second.inverse_depth)
        self.assertIs(first.objects, second.objects)
        self.assertEqual(self.estimator.infer.call_count, 1)
        self.assertEqual(self.person.segment.call_count, 2)
        self.assertEqual(second.person[34, 8], 0)
        self.assertEqual(second.foreground_person[34, 59], 1)
        self.assertEqual(second.object_cover[34, 8], 0)
        self.assertEqual(second.object_cover[34, 39], 1)
        self.assertAlmostEqual(second.inverse_depth[34, 8], .30, places=3)
        self.assertFalse(second.calibration.frame.flags.writeable)
        self.assertEqual(second.calibration.person[34, 8], 1)
        # Far-wall depth at the new subject's pixels cannot cut holes in an actor
        # standing in front of the avatar. A farther person can still stand behind.
        _, nearer = second.projected_foreground(.5, .70, depth_order_person=True)
        self.assertEqual(nearer[34, 59], 1)
        third = model.segment(exposure(55, bottom=38))
        _, farther = third.projected_foreground(.5, .8, depth_order_person=True)
        self.assertEqual(farther[20, 59], 0)
        self.assertEqual(first.person[34, 8], 1)

    def test_recalibrates_for_new_shape_and_explicit_reset(self):
        model = SceneSegmenter(fixed_camera=True)
        first = model.segment(exposure(3))
        resized = model.segment(exposure(3, size=(72, 80)))
        self.assertIsNot(first.calibration, resized.calibration)
        model.reset_scene()
        latest = model.segment(exposure(31, size=(72, 80)))
        self.assertIsNot(latest.calibration, resized.calibration)
        self.assertEqual(self.estimator.infer.call_count, 3)

    def test_moving_camera_still_analyzes_every_exposure(self):
        model = SceneSegmenter()
        for x in (4, 41):
            scene = model.segment(exposure(x))
            self.assertFalse(scene.fixed_camera)
            self.assertAlmostEqual(scene.inverse_depth[34, x], .95)
        self.assertEqual(self.estimator.infer.call_count, 2)

    def test_rejected_chair_stays_quiet_but_a_person_at_the_same_pixels_is_kept(self):
        # Blue determines the mocked PPHumanSeg silhouette. Other channels carry appearance.
        room = exposure(20)
        mask = room[:, :, 0].astype(np.float32) / 255
        self.estimator.infer.side_effect = None
        self.estimator.infer.return_value = ((SceneObject("chair", .9, mask),), np.ones_like(mask))
        self.enterContext(patch("photo_assistant.vision.scene.fit_ground", return_value=GroundProjection(.6, .1, .4)))
        model = SceneSegmenter(fixed_camera=True)
        self.assertFalse(model.segment(room).person.any())
        noisy_room = room.copy()
        noisy_room[:, :, 1] = 4
        self.assertFalse(model.segment(noisy_room).foreground_person.any())

        actor = noisy_room.copy()
        actor[23:44, 20:30, 1] = 93
        np.testing.assert_array_equal(model.segment(actor).person, mask)
        # A limb connected to the old shape changes the silhouette even with identical clothes.
        joined = room.copy()
        joined[24:32, 30:45, 0] = 255
        np.testing.assert_array_equal(model.segment(joined).person, joined[:, :, 0] / 255)
        # A separate person leaves the rejected chair suppressed.
        separate = np.maximum(room, exposure(54))
        np.testing.assert_array_equal(model.segment(separate).person, exposure(54)[:, :, 0] / 255)
        self.assertFalse(model.segment(room).person.any())
        self.assertEqual(self.estimator.infer.call_count, 1)

        self.estimator.person_boxes = [(20, 22, 30, 54)]
        model.reset_scene()
        np.testing.assert_array_equal(model.segment(room).person, mask)
        np.testing.assert_array_equal(model.segment(room).person, mask)
        larger = exposure(20, size=(80, 100))
        large_mask = larger[:, :, 0].astype(np.float32) / 255
        self.estimator.infer.return_value = ((SceneObject("chair", .9, large_mask),), np.ones_like(large_mask))
        np.testing.assert_array_equal(model.segment(larger).person, large_mask)
        self.assertEqual(self.estimator.infer.call_count, 3)

    def test_switch_during_calibration_discards_obsolete_scene_before_shutter(self):
        started, resume = threading.Event(), threading.Event()
        frames = []

        def infer(frame):
            frames.append(frame)
            if len(frames) == 1:
                started.set()
                if not resume.wait(3):
                    raise RuntimeError("calibration test timed out")
            return self.infer(frame)

        self.estimator.infer.side_effect = infer
        worker = PersonOcclusion(lambda: SceneSegmenter(fixed_camera=True))
        self.addCleanup(worker.close)
        self.addCleanup(resume.set)
        worker.set_enabled(True)
        worker.submit(exposure(4))
        self.assertTrue(started.wait(3))
        worker.set_enabled(False)
        worker.set_enabled(True)
        resume.set()
        frame = exposure(55)
        scene = worker.mask_for_capture(frame)
        self.assertIs(frames[1], frame)
        self.assertEqual(scene.calibration.person[34, 59], 1)
        captured = worker.mask_for_capture(exposure(24))
        self.assertIs(captured.calibration, scene.calibration)
        self.assertEqual(captured.foreground_person[34, 28], 1)
        self.assertEqual(captured.foreground_person[34, 59], 0)
        self.assertEqual(len(frames), 2)
