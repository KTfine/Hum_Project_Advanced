"""Check a drawn palm against known vertical edges, without model downloads."""
import unittest

import cv2
import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG, load_pose_catalog
from photo_assistant.vision.framing import Placement
from photo_assistant.vision.scene import SceneAnalysis
from photo_assistant.vision.support import wall_brace


def pillar(left, right, foot=251):
    h, w = 320, 480
    image = np.full((h, w, 3), (180, 177, 168), np.uint8)
    image[181:] = (146, 149, 153)
    depth = np.full((h, w), .2, np.float32)
    depth[181:] += np.linspace(0, .7, h - 181)[:, None]
    cv2.rectangle(image, (left, 0), (right, foot), (82, 91, 160), -1)
    depth[:foot+1, left:right+1] = .73
    empty = np.zeros((h, w), np.float32)
    return image, SceneAnalysis(empty, (), depth, None, empty.copy(), [])


class WallGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_pose_catalog(WHITE_DRAGON_CATALOG)

    def test_projected_hand_meets_each_side_of_a_pillar(self):
        # Move the pillar and give the character a starting point in open space.
        for left, right, anchor_x, edge, sign in ((260, 410, .17, 260, 1), (52, 209, .81, 209, -1)):
            with self.subTest(edge=edge):
                image, scene = pillar(left, right)
                h, w = scene.person.shape
                result = wall_brace(image, scene, self.catalog.entries.items(),
                                    Placement(anchor_x, .85, 1., .56))
                self.assertIsNotNone(result)
                entry = self.catalog.entry(result.pose)
                location = result.placement
                zoom = location.standing_height_ratio * h / entry.standing_height_px
                offset = np.subtract(entry.wall_contact_px, entry.anchor_px)
                ink = np.rint([location.x*w, location.y*h]) + offset * zoom
                np.testing.assert_allclose(ink, np.multiply(result.contact_xy, (w, h)), atol=.01)
                self.assertLess(abs(ink[0] - edge), 3)
                self.assertGreater(sign*(edge - location.x*w), 12)
                self.assertLess(abs(location.y*h - 251), 8)
                self.assertEqual(int(np.sign(offset[0])), sign)

    def test_a_seam_on_a_flat_plane_is_not_a_support(self):
        image, scene = pillar(230, 350)
        scene.inverse_depth[:] = .55
        self.assertIsNone(wall_brace(image, scene, self.catalog.entries.items()))

    def test_no_floor_endpoint_or_no_depth_defers(self):
        image, scene = pillar(230, 350, foot=319)
        self.assertIsNone(wall_brace(image, scene, self.catalog.entries.items()))
        self.assertIsNone(wall_brace(image, None, self.catalog.entries.items()))

    def test_unrelated_target_and_a_person_in_front_of_support_are_rejected(self):
        image, scene = pillar(230, 350)
        self.assertIsNone(wall_brace(image, scene, self.catalog.entries.items(), target_x=.04))
        scene.person[:, 215:365] = 1
        self.assertIsNone(wall_brace(image, scene, self.catalog.entries.items()))

    def test_a_person_between_the_palm_and_the_requested_surface_blocks_contact(self):
        image, scene = pillar(190, 372)
        target = 190 / 480
        brace = wall_brace(image, scene, self.catalog.entries.items(), target_x=target)
        self.assertIsNotNone(brace)
        x, y = np.rint(np.multiply(brace.contact_xy, (480, 320))).astype(int)
        cv2.circle(scene.person, (x, y), 6, 1., -1)
        self.assertIsNone(wall_brace(image, scene, self.catalog.entries.items(), target_x=target))

    def test_near_border_contour_uses_only_the_visible_open_side(self):
        image, scene = pillar(1, 182)
        brace = wall_brace(image, scene, self.catalog.entries.items())
        self.assertIsNotNone(brace)
        self.assertLess(abs(brace.contact_xy[0] * 480 - 182), 3)
