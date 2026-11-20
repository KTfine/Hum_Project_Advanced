"""Verify seat support, projection and occlusion without requiring model weights."""
import unittest

import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG, load_pose_catalog
from photo_assistant.vision.framing import Placement
from photo_assistant.vision.ground import GroundProjection
from photo_assistant.vision.scene import SceneAnalysis
from photo_assistant.vision.scene_objects import SceneObject
from photo_assistant.vision.seat import table_seat


class TableSeatTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 320, 3), 40, np.uint8)
        depth = np.full((240, 320), .1, np.float32)
        person = np.zeros_like(depth)
        objects = []
        descriptions = []
        for name, box, value in (
            ("dining table", (90, 90, 160, 100), .38),
            ("dining table", (90, 135, 250, 157), .7),
            ("chair", (108, 152, 132, 193), .91),
        ):
            x0, y0, x1, y1 = box
            mask = np.zeros_like(depth)
            mask[y0:y1, x0:x1] = 1
            self.frame[y0:y1, x0:x1] = 218
            depth[mask > .5] = value
            if name == "dining table" and value == .7:
                # The front edge, not the bottom of this leg, must support the pose.
                mask[y1:215, 169:175] = 1
                y1 = 215
            objects.append(SceneObject(name, .85, mask))
            descriptions.append(dict(box=(x0 / 320, y0 / 240, x1 / 320, y1 / 240), center_x=(x0+x1)/640))
        person[80:120, 250:260] = 1
        depth[person > .5] = .92
        self.scene = SceneAnalysis(person, tuple(objects), depth, GroundProjection(1, -.15, .35),
                                   np.maximum.reduce([o.mask for o in objects]), descriptions)
        catalog = load_pose_catalog(WHITE_DRAGON_CATALOG)
        self.entry = next(e for p, e in catalog.entries.items() if p.legs == "坐着")
        self.anchor = Placement(.65, .85, 1.0, .6)

    def test_frontmost_table_supports_sprite_hands_and_dangling_feet(self):
        seat = table_seat(self.frame, self.scene, self.entry, self.anchor)
        self.assertIsNotNone(seat)
        p = seat.placement
        projected_contact = (np.array([p.x * 320, p.y * 240]) +
                             (np.array(self.entry.seat_contact_px) - self.entry.anchor_px)
                             * (p.standing_height_ratio * 240 / self.entry.standing_height_px))
        self.assertAlmostEqual(projected_contact[1], 156, delta=2.5)
        np.testing.assert_allclose(projected_contact, np.array(seat.support_xy) * (320, 240))
        self.assertLess(seat.support_xy[1], p.y)
        self.assertLess(p.y, seat.floor_xy[1])
        furniture, person = seat.foreground(self.scene)
        self.assertEqual(furniture[150, 170], 0)
        self.assertEqual(furniture[165, 115], 1)
        self.assertEqual(person[98, 253], 1)

    def test_box_center_targets_the_rear_table_and_front_table_can_occlude_it(self):
        seat = table_seat(self.frame, self.scene, self.entry, self.anchor, target_x=.39, target_y=.40)
        self.assertAlmostEqual(seat.support_xy[1] * 240, 99, delta=2)
        furniture, _ = seat.foreground(self.scene)
        self.assertEqual(furniture[96, 120], 0)
        self.assertEqual(furniture[148, 180], 1)

    def test_missing_target_yields_no_floating_pose(self):
        self.assertIsNone(table_seat(self.frame, self.scene, self.entry, self.anchor, target_x=0, target_y=0))
        self.assertIsNone(table_seat(self.frame, None, self.entry, self.anchor))

    def test_explicit_near_or_far_uses_depth_despite_wrong_numeric_center(self):
        near = table_seat(self.frame, self.scene, self.entry, self.anchor, table="near", target_x=.39, target_y=.40)
        far = table_seat(self.frame, self.scene, self.entry, self.anchor, table="far", target_x=.53, target_y=.73)
        self.assertAlmostEqual(near.support_xy[1] * 240, 156, delta=2.5)
        self.assertAlmostEqual(far.support_xy[1] * 240, 99, delta=2)
