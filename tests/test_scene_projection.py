"""Check scene-aware movement through the normal canvas command and shutter paths."""
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.client import CanvasSource
from photo_assistant.protocol import Command
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
        source = Mock(spec=PersonOcclusion)
        source.latest.return_value = self.raw, self.scene
        source.mask_for_capture.return_value = self.scene
        self.canvas = CanvasSource(WHITE_DRAGON_CATALOG, 320, 240, 95, threading.Event(), preview=False,
                                   scene_layers={"person": source}, infer_obstacle=True)
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        camera = Mock()
        camera.latest_frame.return_value = self.raw
        self.canvas.configure_view(camera, show_character=True, entrance=self.canvas.entrance_state(camera))

    def shot(self):
        return self.canvas._compose_frame(camera_frame=self.raw, capture=True)

    def move(self, y, depth):
        self.canvas.apply_command(Command("teleport", {"x": .5, "y": y, "depth": depth}))
        return self.shot()

    def test_sit_then_stand_uses_the_table_and_keeps_the_initial_size_reference(self):
        self.shot()
        initial = self.canvas._projection_anchor
        target = self.canvas.apply_command(Command("sit", {"table": "near"}))
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

    def test_two_sides_of_table_use_same_body_and_exact_exposure(self):
        far, placement = self.shot()
        self.assertEqual(placement["placement_source"], "桌后站位")
        self.assertEqual(placement["scene_objects"], self.scene.descriptions)
        covered = self.scene.object_cover > 0
        np.testing.assert_array_equal(far[covered], self.raw[covered])
        near, near_place = self.move(.93, "near")
        self.assertTrue(np.any(near[covered] != self.raw[covered]))
        self.assertAlmostEqual(near_place["standing_height_ratio"] / placement["standing_height_ratio"],
                               .93 / .62, places=3)
        self.canvas.apply_command(Command("layer", {"index": 0}))
        forced, _ = self.shot()
        np.testing.assert_array_equal(forced[covered], self.raw[covered])
        restored, restored_place = self.move(.62, "far")
        np.testing.assert_array_equal(restored, far)
        self.assertEqual(restored_place["standing_height_ratio"], placement["standing_height_ratio"])
