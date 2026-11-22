"""Verify seat support, projection and occlusion without requiring model weights."""
from dataclasses import replace
import unittest

import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG, load_pose_catalog
from photo_assistant.vision.framing import Placement
from photo_assistant.vision.ground import GroundProjection
from photo_assistant.vision.scene import CameraCalibration, SceneAnalysis
from photo_assistant.vision.scene_objects import SceneObject
from photo_assistant.vision.seat import find_seat, table_seat


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

    def test_fixed_camera_reserves_space_for_people_who_walk_in_front_of_the_table(self):
        table = self.scene.objects[1]
        depth = np.where(table.mask > .5, .7, .1).astype(np.float32)
        room = replace(self.scene, person=np.zeros_like(self.scene.person), objects=(table,),
                       inverse_depth=depth, object_cover=table.mask, descriptions=[self.scene.descriptions[1]])
        calibration = CameraCalibration(self.frame, room)
        clear = replace(room, calibration=calibration)
        self.assertIsNotNone(table_seat(self.frame, clear, self.entry, self.anchor, table="near"))

        # People now stand between the table and the camera. Only the calibration
        # image had distant background above the rim and clear floor below it.
        people = np.zeros_like(room.person)
        for left in (94, 142, 190, 238):
            people[35:234, left:left + 44] = 1
        blocked = replace(clear, person=people)
        self.assertIsNone(table_seat(self.frame, blocked, self.entry, self.anchor, table="near"))

    def test_cached_occlusion_tracks_the_supporting_table(self):
        for which in ("near", "far", "near"):
            seat = table_seat(self.frame, self.scene, self.entry, self.anchor, table=which)
            masks = self.scene.projected_foreground(.4, .8, seat=seat)
            self.assertIs(masks, self.scene.projected_foreground(.4, .8, seat=seat))
            for actual, expected in zip(masks, seat.foreground(self.scene)):
                np.testing.assert_array_equal(actual, expected)

    def test_explicit_near_or_far_uses_depth_despite_wrong_numeric_center(self):
        near = table_seat(self.frame, self.scene, self.entry, self.anchor, table="near", target_x=.39, target_y=.40)
        far = table_seat(self.frame, self.scene, self.entry, self.anchor, table="far", target_x=.53, target_y=.73)
        self.assertAlmostEqual(near.support_xy[1] * 240, 156, delta=2.5)
        self.assertAlmostEqual(far.support_xy[1] * 240, 99, delta=2)

    def test_nearest_usable_chair_places_contact_on_the_seat_above_its_crossbar(self):
        mask = np.zeros_like(self.scene.person)
        mask[35:83, 104:192] = mask[94:109, 101:196] = 1
        mask[83:186, 105:110] = mask[83:186, 186:191] = mask[149:153, 105:191] = 1
        frame = self.frame.copy()
        frame[mask > .5] = 180
        chair = SceneObject("chair", .9, mask)
        scene = replace(self.scene, objects=(self.scene.objects[0], chair),
                        descriptions=[self.scene.descriptions[0], dict(box=(101/320, 35/240, 196/320, 186/240))],
                        inverse_depth=np.where(mask > .5, .82, self.scene.inverse_depth))
        seat = table_seat(frame, scene, self.entry, self.anchor)
        self.assertEqual(seat.host_label, "chair")
        self.assertEqual(seat.placement.source, "边缘坐姿")
        self.assertAlmostEqual(seat.support_xy[1] * 240, 108, delta=2)
        self.assertEqual(find_seat(frame, scene, self.entry, self.anchor, distance="near"), seat)
        self.assertEqual(table_seat(frame, scene, self.entry, self.anchor, table="near"), seat)

        # A swivel base is thick enough to survive the crossbar filter.
        mask[151:172, 105:191] = 1
        frame[151:172, 105:191] = 180
        seat = table_seat(frame, scene, self.entry, self.anchor, target_x=.46, target_y=.45)
        self.assertAlmostEqual(seat.support_xy[1] * 240, 108, delta=2)

    def test_distance_ranks_usable_supports_across_furniture_types(self):
        frame = np.zeros_like(self.frame)
        far_bench = np.zeros_like(self.scene.person)
        far_bench[94:108, 24:92] = 1
        far_bench[108:142, 27:31] = far_bench[108:142, 86:90] = 1
        near_table = np.zeros_like(far_bench)
        near_table[126:146, 184:290] = 1
        depth = np.where(near_table > .5, .68, np.where(far_bench > .5, .39, .1)).astype(np.float32)
        frame[(near_table + far_bench) > .5] = 180
        room = replace(self.scene, person=np.zeros_like(depth), inverse_depth=depth,
                       objects=(SceneObject("dining table", .9, near_table), SceneObject("bench", .9, far_bench)),
                       descriptions=[dict(box=(184/320, 126/240, 290/320, 146/240)),
                                     dict(box=(24/320, 94/240, 92/320, 142/240))],
                       object_cover=np.maximum(near_table, far_bench))
        fixed = CameraCalibration(frame, room)
        for distance, blocked, expected in (("far", None, "bench"), ("near", None, "dining table"),
                                           ("far", slice(12, 112), "dining table"),
                                           ("near", slice(170, 304), "bench")):
            with self.subTest(distance=distance, blocked=blocked):
                people = np.zeros_like(depth)
                if blocked is not None:
                    people[35:235, blocked] = 1
                scene = replace(room, person=people, calibration=fixed)
                seat = find_seat(frame, scene, self.entry, self.anchor, distance=distance)
                self.assertEqual(seat.host_label, expected)
                self.assertEqual(table_seat(frame, scene, self.entry, self.anchor, table=distance), seat)

    def test_fixed_camera_uses_cushion_seam_and_occludes_with_foreground_people(self):
        mask = np.zeros_like(self.scene.person)
        mask[75:205, 80:246] = 1
        frame = self.frame.copy()
        frame[75:151, 80:246] = 175
        frame[151:205, 80:246] = 90
        bench = SceneObject("couch", .95, mask)
        bag_mask = np.zeros_like(mask)
        bag_mask[132:178, 101:124] = 1
        bag = SceneObject("backpack", .9, bag_mask)
        depth = np.where(bag_mask > .5, .95, np.where(mask > .5, .7, .1)).astype(np.float32)
        room = replace(self.scene, objects=(bench, bag), descriptions=[dict(box=(.25,.3125,.77,.855)),
                       dict(box=(.316,.55,.388,.742))], inverse_depth=depth, object_cover=np.maximum(mask, bag_mask),
                       person=np.zeros_like(mask))
        person = np.zeros_like(mask)
        person[60:235, 202:232] = 1
        current = replace(room, calibration=CameraCalibration(frame, room), person=person)
        seat = table_seat(frame, current, self.entry, self.anchor, target_x=.5, target_y=.63)
        self.assertEqual(seat.host_label, "couch")
        self.assertAlmostEqual(seat.support_xy[1] * 240, 150, delta=2)
        furniture, people = current.projected_foreground(.5, .8, seat=seat)
        self.assertEqual(furniture[147, 145], 0)
        self.assertEqual(furniture[143, 115], 1)
        self.assertEqual(people[135, 219], 1)

    def test_pointed_unlabelled_step_requires_a_visible_raised_support(self):
        frame = np.zeros_like(self.frame)
        frame[96:112, 90:240] = 160  # Upper surface and its distinct front lip.
        frame[112:181, 90:240] = 85
        mask = np.zeros_like(self.scene.person)
        mask[96:181, 90:240] = 1
        empty = np.zeros_like(mask)
        scene = replace(self.scene, objects=(), descriptions=[], person=empty, object_cover=empty,
                        inverse_depth=np.where(mask > .5, .74, .1).astype(np.float32))
        def solve(room, y=112/240):
            return table_seat(frame, room, self.entry, self.anchor, target_x=.55, target_y=y)
        seat = solve(scene)
        self.assertIsNotNone(seat)
        self.assertIsNone(seat.host_label)
        self.assertEqual(seat.placement.source, "边缘坐姿")
        self.assertAlmostEqual(seat.support_xy[1] * 240, 111, delta=2)
        self.assertIsNone(solve(replace(scene, inverse_depth=empty)))  # Painted line on a flat wall.
        self.assertIsNone(solve(scene, 180/240))  # The base meets the floor here.
        screen = SceneObject("tv", .9, mask)
        self.assertIsNone(solve(replace(scene, objects=(screen,), descriptions=[dict(box=(.28,.4,.75,.76))],
                                         object_cover=mask)))
        frame[-2:] = 220
        self.assertIsNone(solve(scene, 228/239))
