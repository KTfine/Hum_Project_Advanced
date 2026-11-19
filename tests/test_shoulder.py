"""Project the shipped entry sprite against synthetic full and cropped camera silhouettes."""

import unittest

import cv2
import numpy as np

from photo_assistant.avatar.frames import AvatarFrames
from photo_assistant.avatar.pose import AvatarPose, DEFAULT_SPRITE_CATALOG
from photo_assistant.vision.perspective import person_reference, standing_height_px
from photo_assistant.vision.shoulder import shoulder_placement


def camera_person(center, height):
    mask = np.zeros((200, 320), np.float32)
    top = 12
    cv2.ellipse(mask, (center, top + round(0.07 * height)),
                (round(0.055 * height), round(0.07 * height)), 0, 0, 360, 1, -1)
    cv2.rectangle(mask, (center - round(0.12 * height), top + round(0.13 * height)),
                  (center + round(0.12 * height), top + round(0.57 * height)), 1, -1)
    cv2.rectangle(mask, (center - round(0.075 * height), top + round(0.5 * height)),
                  (center + round(0.075 * height), top + height - 1), 1, -1)
    return mask


class ShoulderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frames = AvatarFrames(DEFAULT_SPRITE_CATALOG)
        pose = AvatarPose(torso="半身歪斜向右")
        cls.sprite = frames.frame(pose)
        cls.entry = frames.catalog.entry(pose)
        frames.close()

    def place(self, mask, foreground_masks=()):
        return shoulder_placement(mask, self.sprite, self.entry.anchor_px, self.entry.standing_height_px,
                                  foreground_masks=foreground_masks)

    def inspect_placement(self, mask, placement):
        height, width = mask.shape
        ref = person_reference(mask, placement.x, placement.y)
        actual_height = placement.scale * standing_height_px(
            ref, height, placement.y, fallback_height=0.72 * height)
        px_scale = actual_height / self.entry.standing_height_px
        offset = np.array([round(placement.x * width), round(placement.y * height)]) - np.array(
            self.entry.anchor_px) * px_scale
        # Landmarks on the default tilted PNG: eye midpoint and chest/waist band.
        face_xy = np.rint(offset + np.array([187, 75]) * px_scale).astype(int)
        chest = self.sprite[:, :, 3].copy()
        chest[:145, :] = chest[300:, :] = 0
        projected = cv2.warpAffine(chest, np.float64([[px_scale, 0, offset[0]], [0, px_scale, offset[1]]]),
                                   (width, height)) > 128
        hidden = mask[projected].mean()
        return face_xy, hidden, actual_height

    def test_right_shoulder_face_is_visible_and_torso_remains_occluded(self):
        mask = camera_person(160, 160)
        placement = self.place(mask)
        face, hidden, height = self.inspect_placement(mask, placement)
        self.assertAlmostEqual(height, 160 * 14 / 15)
        self.assertAlmostEqual(placement.scale, 1)
        self.assertEqual(round(placement.y * 200), 172)
        self.assertTrue(178 < face[0] < 195 and 30 < face[1] < 52)
        self.assertEqual(mask[face[1], face[0]], 0)
        self.assertGreater(hidden, 0.55)

    def test_lower_body_offscreen_has_extrapolated_feet_and_face_at_shoulder(self):
        mask = camera_person(160, 290)
        placement = self.place(mask)
        face, hidden, height = self.inspect_placement(mask, placement)
        self.assertGreater(placement.y, 1)
        self.assertTrue(232 < height < 305)
        self.assertEqual(mask[face[1], face[0]], 0)
        self.assertTrue(193 < face[0] < 232 and 46 < face[1] < 85)
        self.assertGreater(hidden, 0.55)

    def test_right_edge_switches_to_the_visible_left_shoulder(self):
        mask = camera_person(317, 160)
        face, hidden, _ = self.inspect_placement(mask, self.place(mask))
        self.assertTrue(285 < face[0] < 312)
        self.assertGreater(hidden, 0.08)
        self.assertEqual(mask[face[1], face[0]], 0)

    def test_foreground_object_can_force_the_other_shoulder_or_block_both(self):
        mask = camera_person(160, 160)
        obstacle = np.zeros_like(mask)
        obstacle[:70, 164:] = 1
        placement = self.place(mask, (mask, obstacle))
        face, hidden, _ = self.inspect_placement(mask, placement)
        self.assertLess(face[0], 160)
        self.assertEqual(obstacle[face[1], face[0]], 0)
        self.assertGreater(hidden, 0.08)
        obstacle[:70, :] = 1
        self.assertIsNone(self.place(mask, (mask, obstacle)))

    def test_slightly_clipped_crown_with_clear_shoulders_and_torso_only_crop(self):
        full = camera_person(160, 160)
        mask = full[17:, :]  # five pixels clipped from the top of the head
        face, hidden, _ = self.inspect_placement(mask, self.place(mask))
        self.assertEqual(mask[face[1], face[0]], 0)
        self.assertTrue(13 < face[1] < 35)
        self.assertGreater(hidden, 0.55)
        self.assertIsNone(self.place(full[35:, :]))  # no head above the shoulder line

    def test_empty_noisy_or_unlocalized_mask_defers_placement(self):
        empty = np.zeros((200, 320), np.float32)
        noise = empty.copy()
        noise[::8, ::8] = 1
        for mask in (None, empty, noise, np.ones_like(empty)):
            self.assertIsNone(self.place(mask))
