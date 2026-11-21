"""Standing beside a person uses the same foreground furniture as the shutter."""

import unittest

import numpy as np

from photo_assistant.vision.beside import side_placement
from photo_assistant.vision.ground import GroundProjection
from photo_assistant.vision.scene import SceneAnalysis
from photo_assistant.vision.scene_objects import SceneObject


class PairDepthTests(unittest.TestCase):
    def test_slot_uses_real_depth_and_keeps_manually_supplied_masks(self):
        person = np.zeros((200, 320), np.float32)
        person[25:47, 90:109] = 1
        person[45:117, 82:116] = 1
        person[115:180, 88:110] = 1
        across = np.ones_like(person)
        art = np.zeros((128, 54, 4), np.uint8)
        art[5:30, 14:37, 3] = 255
        art[30:126, 8:45, 3] = 255
        def place(depth, manual=False):
            obj = SceneObject("dining table", .9, across)
            scene = SceneAnalysis(person, (obj,), np.full_like(person, depth),
                                  GroundProjection(1, 0, .4), across, [])
            masks = (person, across, across.copy()) if manual else (person, across)
            return side_placement(person, art, (26, 126), 121, face_box_px=(17, 10, 34, 22),
                                  foreground_masks=masks, scene=scene)
        self.assertIsNotNone(place(.30))
        self.assertIsNone(place(.95))
        self.assertIsNone(place(.30, manual=True))

    def test_verified_feet_apply_to_duplicate_top_and_preserve_foreground_chair(self):
        person = np.zeros((150, 240), np.float32)
        table = np.zeros_like(person)
        table[60:74, 18:175] = 1
        table[72:119, 38:45] = table[72:119, 149:156] = 1
        top = np.zeros_like(table)
        top[62:72, 22:171] = 1
        chair = np.zeros_like(table)
        chair[96:, 178:220] = 1
        objects = (SceneObject("dining table", .9, table), SceneObject("dining table", .4, top),
                   SceneObject("chair", .85, chair))
        depth = np.full_like(table, .81)
        depth[table > 0] = .72
        depth[chair > 0] = .94
        scene = SceneAnalysis(person, objects, depth, GroundProjection(.80, .02, .4),
                              np.maximum(table, chair), [])
        self.assertGreater(scene.depth_visibility(.40, .84)[65, 100], .99)
        self.assertEqual(scene.furniture_cover(.40, .84)[65, 100], 0)
        self.assertEqual(scene.furniture_cover(.40, .70)[65, 100], 1)
        self.assertEqual(scene.furniture_cover(.40, .84)[120, 199], 1)
        # If the local floor is behind the detected support, a silhouette end
        # alone cannot establish that the camera sees the real legs.
        depth[120:125] = .4
        unresolved = SceneAnalysis(person, objects, depth, scene.ground, scene.object_cover, [])
        self.assertEqual(unresolved.furniture_cover(.40, .84)[65, 100], 1)
