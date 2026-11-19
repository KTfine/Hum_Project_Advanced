import unittest

import numpy as np

from photo_assistant.avatar.frames import apply_teleport_effect


class TeleportEffectTests(unittest.TestCase):
    def test_scan_reveals_at_fixed_geometry_and_preserves_alpha_edges(self):
        sprite = np.zeros((101, 5, 4), dtype=np.uint8)
        sprite[:, :, :3] = (24, 28, 32)
        sprite[:, 1:4, 3] = (48, 255, 128)
        sprite.setflags(write=False)

        initial = apply_teleport_effect(sprite, 0.0)
        midway = apply_teleport_effect(sprite, 0.5)
        np.testing.assert_array_equal(initial[:, :, 3] > 0, sprite[:, :, 3] > 0)
        np.testing.assert_array_equal(midway[:, (0, 4)], sprite[:, (0, 4)])
        np.testing.assert_array_equal(midway[0], sprite[0])
        np.testing.assert_array_equal(midway[-1], initial[-1])
        self.assertLess(midway[-1, 2, 3], midway[50, 2, 3])
        self.assertLess(midway[50, 2, 3], midway[0, 2, 3])
        self.assertGreater(midway[50, 2, 0], sprite[50, 2, 0])
        self.assertTrue(np.all(midway[:, :, 3] <= sprite[:, :, 3]))
        np.testing.assert_array_equal(apply_teleport_effect(sprite, 1.0), sprite)
