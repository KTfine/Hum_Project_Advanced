"""Check scene-aware movement through the normal canvas command and shutter paths."""
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.client import CanvasSource
from photo_assistant.protocol import Command, StreamingCommandParser
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneAnalysis
from photo_assistant.vision.scene_objects import SceneObject


class Floor:
    horizon_y = 0.0

    def depth_at(self, x, y):
        return y

    def height_ratio(self, ax, ay, ratio, x, y):
        return ratio * y / ay


class SceneProjectionTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.full((240, 320, 3), 128, np.uint8)
        person = np.zeros(self.raw.shape[:2], np.float32)
        cover = np.zeros_like(person)
        cover[104:195, 100:220] = 1
        self.scene = SceneAnalysis(person, (SceneObject("dining table", .9, cover),),
                                   np.full_like(person, .75), Floor(), cover,
                                   [dict(label="dining table", box=(100/320, 104/240, 220/320, 195/240),
                                         center_x=.5, behind_y=.62, front_y=.91)])
        self.segmentation = Mock(spec=PersonOcclusion)
        self.segmentation.latest.return_value = self.raw, self.scene
        self.segmentation.mask_for_capture.return_value = self.scene
        self.canvas = CanvasSource(WHITE_DRAGON_CATALOG, 320, 240, 95, threading.Event(), preview=False,
                                   scene_layers={"person": self.segmentation}, infer_obstacle=True)
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        camera = Mock()
        camera.latest_frame.return_value = self.raw
        self.canvas.configure_view(camera, show_character=True, entrance=self.canvas.entrance_state(camera))

    def shot(self):
        return self.canvas._compose_frame(camera_frame=self.raw, capture=True)

    def move(self, y):
        command, = StreamingCommandParser().feed("r", f"<teleport:x=0.5,y={y}>")
        self.canvas.apply_command(command)
        return self.shot()

    def test_sit_then_stand_uses_the_table_and_keeps_the_initial_size_reference(self):
        self.shot()
        initial = self.canvas._projection_anchor
        target = self.canvas.apply_command(Command("sit", {"distance": "near"}))
        self.assertEqual(target.pose.legs, "坐着")
        seat = self.canvas._seat
        self.canvas.apply_command(Command("tachie", vars(target.pose)))
        self.assertIs(self.canvas._seat, seat)
        seated, placement = self.shot()
        self.assertEqual(placement["placement_source"], "桌沿坐姿")
        self.assertAlmostEqual(seat.support_xy[1] * 240, 194, delta=3)
        self.assertTrue(np.any(seated != self.raw))
        target = self.canvas.apply_command(Command("stand", {}))
        self.assertEqual(target.pose.legs, "站着")
        self.assertIsNone(self.canvas._seat)
        _, placement = self.shot()
        self.assertAlmostEqual(placement["y"], seat.floor_xy[1])
        self.assertIs(self.canvas._projection_anchor, initial)

    def test_bench_edge_is_used_for_the_shot_and_standing_back_up(self):
        mask = np.zeros(self.raw.shape[:2], np.float32)
        mask[122:140, 60:260] = 1
        mask[140:198, 76:83] = mask[140:198, 237:244] = 1
        self.raw[mask > .5] = 200
        bench = SceneObject("bench", .92, mask)
        table_mask = np.zeros_like(mask)
        table_mask[80:93, 24:98] = 1
        self.raw[table_mask > .5] = 195
        table = SceneObject("dining table", .9, table_mask)
        depth = np.where(mask > .5, .82, np.where(table_mask > .5, .42, .1)).astype(np.float32)
        scene = SceneAnalysis(self.scene.person, (bench, table), depth, Floor(), np.maximum(mask, table_mask), [
            dict(label="bench", box=(60/320, 122/240, 260/320, 198/240), center_x=.5),
            dict(label="dining table", box=(24/320, 80/240, 98/320, 93/240), center_x=61/320),
        ])
        self.segmentation.latest.return_value = self.raw, scene
        self.segmentation.mask_for_capture.return_value = scene

        parser = StreamingCommandParser()
        for tag, expected in (("<sit:distance=far>", "dining table"), ("<sit:distance=near>", "bench"),
                              ("<sit:table=near>", "bench")):
            with self.subTest(tag=tag):
                command, = parser.feed("r", tag)
                self.assertEqual(self.canvas.apply_command(command).pose.legs, "坐着")
                self.assertEqual(self.shot()[1]["seat_host"], expected)

        sit, = parser.feed("r", "<sit:x=0.50,y=0.58>")
        self.assertEqual(self.canvas.apply_command(sit).pose.legs, "坐着")
        shot, placement = self.shot()
        self.assertEqual(placement["seat_host"], "bench")
        self.assertAlmostEqual(placement["seat_contact_xy"][1] * 240, 139, delta=3)
        self.assertTrue(np.any(shot != self.raw))

        # An explicit point outside any support keeps the current seat.
        self.assertIsNone(self.canvas.apply_command(Command("sit", {"x": .03, "y": .05})))
        self.assertEqual(self.shot()[1]["seat_contact_xy"], placement["seat_contact_xy"])
        floor_xy = self.canvas._seat.floor_xy
        self.assertEqual(self.canvas.apply_command(Command("stand", {})).pose.legs, "站着")
        _, standing = self.shot()
        self.assertNotIn("seat_contact_xy", standing)
        self.assertAlmostEqual(standing["y"], floor_xy[1])

    def test_pointed_cabinet_front_remains_behind_the_dangling_legs(self):
        self.raw[:] = 0
        self.raw[96:112, 90:240] = 160
        self.raw[112:201, 90:240] = 85
        empty = np.zeros(self.raw.shape[:2], np.float32)
        depth = np.full_like(empty, .1)
        depth[96:201, 90:240] = .9
        scene = SceneAnalysis(empty, (), depth, Floor(), empty, [])
        self.segmentation.latest.return_value = self.raw, scene
        self.segmentation.mask_for_capture.return_value = scene
        # A broad cabinet front also looks like a foreground panel to the live renderer.
        _, masks, _, _ = self.canvas.scene_layers.resolve(self.raw, 0, capture=True)
        self.assertTrue(any(mask[145, 176] > .9 for mask in masks))
        target = self.canvas.apply_command(Command("sit", {"x": .55, "y": 112/240}))
        self.assertEqual(target.pose.legs, "坐着")
        seated, placement = self.shot()
        self.assertEqual(placement["placement_source"], "边缘坐姿")
        self.assertNotIn("seat_host", placement)
        legs = np.s_[118:180, 130:210]
        self.assertTrue(np.any(seated[legs] != self.raw[legs]))

    def test_two_sides_of_table_use_same_body_and_exact_exposure(self):
        far, placement = self.shot()
        self.assertEqual(placement["placement_source"], "桌后站位")
        self.assertEqual(placement["scene_objects"], self.scene.descriptions)
        covered = self.scene.object_cover > 0
        np.testing.assert_array_equal(far[covered], self.raw[covered])
        near, near_place = self.move(.93)
        self.assertTrue(np.any(near[covered] != self.raw[covered]))
        self.assertAlmostEqual(near_place["standing_height_ratio"] / placement["standing_height_ratio"],
                               .93 / .62, places=3)
        self.canvas.apply_command(Command("layer", {"index": 0}))
        forced, _ = self.shot()
        np.testing.assert_array_equal(forced[covered], self.raw[covered])
        automatic, _ = self.move(.93)
        np.testing.assert_array_equal(automatic, near)
        restored, restored_place = self.move(.62)
        np.testing.assert_array_equal(restored, far)
        self.assertEqual(restored_place["standing_height_ratio"], placement["standing_height_ratio"])
