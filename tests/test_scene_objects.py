import unittest
from unittest.mock import Mock

import numpy as np

from photo_assistant.vision.scene_objects import SceneObjectEstimator, _complete_tabletop, _same_silhouette


class SceneObjectsTests(unittest.TestCase):
    def test_instances_person_boxes_and_depth_use_original_coordinates(self):
        # A 640×320 image gets 80 pixels of padding above/below. Two chairs and
        # one person share the same forward pass, even on a wide camera frame.
        frame = np.zeros((320, 640, 3), np.uint8)
        rows = np.zeros((3, 116), np.float32)
        rows[:, :4] = [[90, 160, 40, 80], [560, 360, 60, 100], [320, 230, 50, 150]]
        rows[[0, 1], 4 + 56] = 0.85
        rows[2, 4] = 0.7
        rows[:, 84] = 2
        estimator = SceneObjectEstimator.__new__(SceneObjectEstimator)
        estimator._segmenter = Mock()
        estimator._depth = Mock()
        estimator._segmenter.forward.return_value = rows.T[None], np.ones((1, 32, 120, 160), np.float32)
        estimator._depth.forward.return_value = np.indices((256, 256), dtype=np.float32)[0][None]
        objects, depth = estimator.infer(frame)
        self.assertEqual([o.label for o in objects], ["chair", "chair"])
        self.assertEqual(estimator.person_boxes, ((295, 75, 345, 225),))
        self.assertEqual(estimator.person_patches[0][1].shape, (150, 50))
        self.assertEqual(estimator.person_patches[0][1][75, 25], 1)
        self.assertEqual(objects[0].mask.shape, (320, 640))
        self.assertEqual(objects[0].mask[80, 90], 1)
        self.assertEqual(objects[0].mask[160, 90], 0)
        self.assertEqual(objects[1].mask[280, 560], 1)
        self.assertEqual(depth.shape, frame.shape[:2])
        self.assertGreater(depth[280, 100], depth[80, 100])
        rows[:, 4:84] = 0
        objects, _ = estimator.infer(frame)
        self.assertEqual(objects, ())
        self.assertEqual(estimator.person_boxes, ())
        self.assertEqual(estimator.person_patches, ())

    def test_subset_of_the_same_table_does_not_count_as_another_desk(self):
        primary = np.zeros((80, 100), np.float32)
        primary[:25, 10:90] = 1
        primary[25:, 20:25] = primary[25:, 75:80] = 1
        top = primary.copy()
        top[25:] = 0
        self.assertTrue(_same_silhouette(top, primary, same_class=True))
        self.assertFalse(_same_silhouette(1 - primary, primary, same_class=True))

    def test_color_completion_fills_a_tabletop_gap_and_preserves_the_floor(self):
        frame = np.full((80, 100, 3), (20, 100, 35), np.uint8)
        frame[12:44, 5:94] = (110, 62, 39)
        partial = np.zeros(frame.shape[:2], np.float32)
        partial[12:44, 25:94] = 1
        depth = np.zeros(frame.shape[:2], np.float32)
        complete = _complete_tabletop(frame, partial, depth)
        self.assertEqual(complete[16, 10], 1)
        self.assertEqual(complete[60, 60], 0)

    def test_incomplete_table_mask_with_a_spill_and_distinct_overlapping_tables(self):
        table = np.zeros((120, 180), np.float32)
        table[20:52, 20:144] = 1
        table[52:112, 32:40] = table[52:112, 124:132] = 1
        fragment = np.zeros_like(table)
        fragment[16:48, 20:144] = 1
        # One eighth of the weaker tabletop extends into a gap in the main mask.
        self.assertTrue(_same_silhouette(fragment, table, same_class=True, table=True))
        separate = np.zeros_like(table)
        separate[28:60, 100:176] = 1
        separate[60:108, 104:112] = separate[60:108, 164:172] = 1
        self.assertFalse(_same_silhouette(separate, table, same_class=True, table=True))
