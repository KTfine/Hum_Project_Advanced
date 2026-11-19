"""Ground-plane sizing and frame identity for masked camera previews and photos."""

import unittest
from unittest.mock import Mock

import numpy as np

from photo_assistant.vision.layers import SceneLayers
from photo_assistant.vision.perspective import person_reference, standing_height_px


class PerspectiveTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((200, 320), np.float32)
        self.mask[30:180, 70:110] = 1

    def test_same_ground_depth_is_half_a_head_shorter(self):
        ref = person_reference(self.mask, 0.5, 0.95)
        self.assertEqual(ref.height, 0.75)
        self.assertEqual(ref.foot_y, 0.9)
        self.assertEqual(standing_height_px(ref, 200, ref.foot_y, fallback_height=120), 140)

    def test_reference_projects_linearly_with_foot_depth_and_bounds_extremes(self):
        ref = person_reference(self.mask, 0.5, 0.95)
        height_at = lambda y: standing_height_px(ref, 200, y, fallback_height=120)
        self.assertLess(height_at(0.8), height_at(0.9))
        self.assertAlmostEqual(height_at(0.9), (height_at(0.8) + height_at(1.0)) / 2)
        self.assertGreater(height_at(-2), 0)
        self.assertLessEqual(height_at(3), 140 * 2.5)

    def test_nearest_complete_person_is_used_in_group(self):
        self.mask[60:140, 180:205] = 1
        self.mask[100, 160] = 1  # isolated noise
        self.assertEqual(person_reference(self.mask, 0.65, 0.9).height, 0.4)
        self.assertEqual(person_reference(self.mask, 0.15, 0.7).height, 0.75)

    def test_empty_cropped_and_wide_silhouettes_use_default_depth_curve(self):
        self.mask[:] = 0
        self.mask[0:60, 20:35] = 1
        self.mask[80:, 220:250] = 1
        self.mask[100:160, 40:140] = 1
        for mask in (None, np.zeros_like(self.mask), self.mask):
            self.assertIsNone(person_reference(mask, 0.5, 0.95))
        self.assertEqual(standing_height_px(None, 200, .95, fallback_height=110), 110)
        self.assertLess(standing_height_px(None, 200, .6, fallback_height=110), 110)

    def test_sampling_camera_resolution_preserves_foot_position_and_height(self):
        # Repeated pixels exactly represent the same ground plane at greater resolution.
        enlarged = self.mask.repeat(5, axis=0).repeat(5, axis=1)
        ref = person_reference(enlarged, 0.5, 0.95)
        self.assertAlmostEqual(ref.foot_y, 0.9)
        self.assertAlmostEqual(ref.height, 0.75)


class ReferenceLayerTests(unittest.TestCase):
    def setUp(self):
        self.source = Mock()
        self.layers = SceneLayers({"person": self.source})
        self.frame = np.zeros((32, 40, 3), np.uint8)
        self.mask = np.ones((32, 40), np.float32)

    def test_front_layer_uses_same_frame_person_reference_without_occluding(self):
        older = self.frame.copy()
        self.source.latest.return_value = older, self.mask
        frame, masks, reference = self.layers.foreground(self.frame, 1, capture=False)
        self.assertIs(frame, older)
        self.assertEqual(masks, ())
        self.assertIs(reference, self.mask)
        self.source.mask_for_capture.return_value = self.mask
        frame, masks, reference = self.layers.foreground(self.frame, 1, capture=True)
        self.assertIs(frame, self.frame)
        self.assertEqual(masks, ())
        self.assertIs(reference, self.mask)
        self.source.mask_for_capture.assert_called_once_with(self.frame)

    def test_preview_warming_only_hides_sprite_behind_live_layer(self):
        self.source.latest.return_value = None
        for slot, expected in [(0, None), (1, ())]:
            frame, masks, reference = self.layers.foreground(self.frame, slot, capture=False)
            self.assertIs(frame, self.frame)
            self.assertEqual(masks, expected)
            self.assertIsNone(reference)

    def test_table_mask_does_not_set_reference_height(self):
        layers = SceneLayers({"table": self.mask})
        _, masks, reference = layers.foreground(self.frame, 0, capture=False)
        self.assertEqual(len(masks), 1)
        self.assertIsNone(reference)
