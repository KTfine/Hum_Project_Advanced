"""Check where the companion stands in full-length and cropped camera views."""

import unittest

import cv2
import numpy as np

from photo_assistant.avatar.frames import AvatarFrames
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.vision.beside import side_placement
from photo_assistant.vision.head import measure_face
from photo_assistant.vision.perspective import matched_standing_height_px, person_reference, standing_height_px


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


class BesideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = AvatarFrames(WHITE_DRAGON_CATALOG)
        cls.addClassCleanup(cls.frames.close)

    def place(self, mask, *obstacles, portrait=False):
        pose = self.frames.catalog.entry_pose if portrait else self.frames.catalog.standing_pose
        entry = self.frames.catalog.entry(pose)
        masks = ((mask,) if mask is not None else ()) + obstacles
        return side_placement(mask, self.frames.frame(pose), entry.anchor_px, entry.standing_height_px,
                              foreground_masks=masks, face_box_px=entry.face_box_px,
                              head=measure_face(None, mask))

    def test_full_length_subject_has_neighboring_feet_on_the_same_ground(self):
        mask = camera_person(160, 160)
        p = self.place(mask)
        self.assertIsNotNone(p)
        x, y = round(p.x * 320), round(p.y * 200)
        self.assertAlmostEqual(y, 172, delta=5)
        self.assertTrue(20 < abs(x - 160) < 80)
        self.assertEqual(mask[y, x], 0)

    def test_portrait_keeps_feet_offscreen_with_a_visible_face_and_partial_body_overlap(self):
        mask = camera_person(160, 380)
        placement = self.place(mask, portrait=True)
        self.assertIsNotNone(placement)
        self.assertGreater(placement.y, 1)
        pose = self.frames.catalog.entry_pose
        entry = self.frames.catalog.entry(pose)
        left, top, right, bottom = entry.face_box_px
        person_face = measure_face(None, mask)
        matched = matched_standing_height_px(person_face.width, right - left, entry.standing_height_px)
        h, w = mask.shape
        ref = person_reference(mask, placement.x, placement.y)
        size = placement.scale * standing_height_px(ref, h, placement.y, fallback_height=.72 * h,
                                                    matched_height_px=matched)
        scale = size / entry.standing_height_px
        offset = np.rint([placement.x * w, placement.y * h]) - np.array(entry.anchor_px) * scale
        x, y = np.rint(offset + np.array([(left + right) / 2, (top + bottom) / 2]) * scale).astype(int)
        self.assertTrue(0 < x < w and 0 < y < h)
        self.assertEqual(mask[y, x], 0)
        self.assertAlmostEqual((right - left) * scale / person_face.width, 0.71, delta=.03)
        self.assertAlmostEqual((y - person_face.center_y) / person_face.width, 0.84, delta=.20)
        matrix = np.float32([[scale, 0, offset[0]], [0, scale, offset[1]]])
        ink = cv2.warpAffine(self.frames.frame(pose)[:, :, 3], matrix, (w, h)) > 128
        self.assertTrue(.20 < mask[ink].mean() < .80)

    def test_subject_at_either_edge_uses_free_space_inside_the_frame(self):
        for center in (3, 317):
            with self.subTest(center=center):
                p = self.place(camera_person(center, 160))
                self.assertIsNotNone(p)
                x = p.x * 320
                self.assertTrue(20 < x < 300)
                self.assertTrue(20 < abs(x - center) < 80)

    def test_foreground_obstacle_limits_the_available_side(self):
        mask = camera_person(160, 160)
        left = np.zeros_like(mask)
        left[:, :152] = 1
        right = np.zeros_like(mask)
        right[:, 168:] = 1
        self.assertGreater(self.place(mask, left).x, .5)
        self.assertLess(self.place(mask, right).x, .5)
        self.assertIsNone(self.place(mask, left, right))

    def test_cropped_crown_still_allows_side_entry_when_head_shape_is_measurable(self):
        full = camera_person(160, 380)
        clipped = full[17:]  # first five pixels of the head are outside the frame
        placement = self.place(clipped, portrait=True)
        self.assertIsNotNone(placement)
        self.assertGreater(placement.y, 1.)
        self.assertIsNone(self.place(full[60:], portrait=True))  # only a torso remains

    def test_no_person_can_use_the_edge_of_a_foreground_partition(self):
        panel = np.zeros((200, 320), np.float32)
        panel[12:, :100] = 1
        placement = self.place(None, panel, portrait=True)
        self.assertIsNotNone(placement)
        # The anchor remains behind the panel; the tilted face reaches its free edge.
        pose = self.frames.catalog.entry_pose
        entry = self.frames.catalog.entry(pose)
        left, _, right, _ = entry.face_box_px
        height = placement.scale * standing_height_px(None, 200, placement.y, fallback_height=144)
        foot_x = round(placement.x * 320)
        face_x = foot_x + ((left + right) / 2 - entry.anchor_px[0]) * height / entry.standing_height_px
        self.assertLess(foot_x, 100)
        self.assertTrue(100 < face_x < 160)

    def test_empty_noisy_or_unlocalized_mask_defers_placement(self):
        empty = np.zeros((200, 320), np.float32)
        noise = empty.copy()
        noise[::8, ::8] = 1
        for mask in (None, empty, noise, np.ones_like(empty)):
            self.assertIsNone(self.place(mask))
