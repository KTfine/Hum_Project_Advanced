import unittest

import numpy as np

from photo_assistant.vision.ground import GroundProjection, camera_reference, fit_ground


class GroundTests(unittest.TestCase):
    def test_fit_uses_floor_beyond_the_furniture_instead_of_depth_at_foot_pixel(self):
        h, w = 180, 240
        frame = np.zeros((h, w, 3), np.uint8)
        depth = np.broadcast_to(.15 + .65 * np.arange(h)[:, None] / (h - 1), (h, w)).copy()
        # Some furniture has a known mask, another nearer desk was missed.
        mask = np.zeros((h, w), np.uint8)
        mask[85:160, 30:150] = 1
        depth[mask != 0] = .95
        depth[115:140, 150:200] = .97
        ground = fit_ground(frame, depth, (mask,))
        self.assertIsNotNone(ground)
        self.assertAlmostEqual(ground.depth_at(.4, .7), .605, places=3)
        self.assertGreater(ground.depth_at(.4, 1.6), 1.0)
        self.assertGreater(depth[126, 96], ground.depth_at(.4, .7))

    def test_inadequate_floor_evidence_is_rejected(self):
        frame = np.zeros((80, 100, 3), np.uint8)
        depth = np.zeros(frame.shape[:2], np.float32)
        self.assertIsNone(fit_ground(frame, depth))
        depth[:] = np.linspace(0, .8, 80)[:, None]
        hidden = np.ones_like(depth)
        hidden[-9:] = 0
        self.assertIsNone(fit_ground(frame, depth, (hidden,)))

    def test_measured_body_reference_projects_without_compounding_or_metric_depth(self):
        ground = GroundProjection(1.4, -.55, .23)
        changed_depth_offset = GroundProjection(.75, .21, .23)
        for x, y, size in ((.6693, .9074, 284.24 / 480),  # full companion replay
                           (.4192, 1.0445, 397.11 / 480),  # half companion replay
                           (.3, 2.060, 783.1 / 480),  # existing portrait photo
                           camera_reference()):
            with self.subTest(anchor=(x, y, size)):
                self.assertEqual(ground.height_ratio(x, y, size, 1 - x, y), size)
                far = ground.height_ratio(x, y, size, x, .62)
                self.assertLess(far, size)
                self.assertAlmostEqual(far, changed_depth_offset.height_ratio(x, y, size, x, .62))
                self.assertGreater(ground.height_ratio(x, y, size, x, y + .15), size)
