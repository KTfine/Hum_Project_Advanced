"""Check the shutter geometry when a person asks the dragon to stand behind them."""

import threading
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG
from photo_assistant.client import CanvasSource
from photo_assistant.protocol import Command
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneAnalysis


def subject(center):
    """Upright 304 px subject with separately labelled shoes, on a 640 x 480 exposure."""
    person = np.zeros((480, 640), np.float32)
    shoes = np.zeros_like(person)
    cv2.ellipse(person, (center, 147), (16, 21), 0, 0, 360, 1, -1)
    cv2.rectangle(person, (center - 10, 158), (center + 10, 180), 1, -1)
    outline = [(-10, 174), (-40, 185), (-42, 278), (-29, 300), (-22, 307),
               (-25, 415), (-8, 415), (0, 316), (9, 415), (25, 415),
               (25, 309), (38, 291), (42, 265), (39, 185), (10, 174)]
    cv2.fillPoly(person, [np.int32(outline) + (center, 0)], 1)
    for offset in (-18, 18):
        cv2.ellipse(shoes, (center + offset, 420), (12, 10), 0, 0, 360, 1, -1)
    return np.maximum(person, shoes), shoes > .5


class PeekFeetTests(unittest.TestCase):
    def camera_view(self, person):
        raw = np.full((*person.shape, 3), (74, 99, 61), np.uint8)
        raw[person > .5] = (70, 66, 116)
        empty = np.zeros_like(person)
        scene = SceneAnalysis(person, (), empty, None, empty, [])
        worker = Mock(spec=PersonOcclusion)
        worker.latest.return_value = raw, scene
        worker.mask_for_capture.return_value = scene
        camera = Mock()
        camera.latest_frame.return_value = raw
        h, w = person.shape
        canvas = CanvasSource(WHITE_DRAGON_CATALOG, w, h, 95, threading.Event(), preview=False,
                              scene_layers={"person": worker}, project_scene=True)
        canvas.start()
        self.addCleanup(canvas.close)
        canvas.configure_view(camera, show_character=True, entrance=canvas.entrance_state(camera))
        return canvas, raw

    def test_full_body_peek_separates_shoes_and_keeps_face_and_body_overlap(self):
        for center in (120, 500):
            with self.subTest(center=center):
                person, shoes = subject(center)
                canvas, raw = self.camera_view(person)
                canvas.render_capture_frame(raw)
                self.assertIsNotNone(canvas.apply_command(Command("peek", {})))
                shot, place = canvas._compose_frame(camera_frame=raw, capture=True)
                state = canvas.controller.state_at()
                art = canvas.avatar_frames.frame(state.pose)
                entry = canvas.avatar_frames.catalog.entry(state.pose)
                self.assertEqual(entry.path.name, "peek-right.png")
                h, w = person.shape
                zoom = place["standing_height_ratio"] * h / entry.standing_height_px
                transform = np.array([[zoom, 0, round(place["x"] * w) - zoom * entry.anchor_px[0]],
                                      [0, zoom, round(place["y"] * h) - zoom * entry.anchor_px[1]]])

                def project(mask):
                    return cv2.warpAffine(mask.astype(np.uint8), transform, (w, h),
                                          flags=cv2.INTER_NEAREST) > 0

                opaque = art[:, :, 3] >= 220
                # Independent landmarks on the two shoe uppers and soles in peek-right.png.
                shoe_ink = np.zeros_like(opaque)
                shoe_ink[1410:1495, 342:425] = opaque[1410:1495, 342:425]
                shoe_ink[1410:1495, 500:592] = opaque[1410:1495, 500:592]
                dragon_shoes = project(shoe_ink)
                self.assertTrue(dragon_shoes.any())
                distance = cv2.distanceTransform((~shoes).astype(np.uint8), cv2.DIST_L2,
                                                 cv2.DIST_MASK_PRECISE)
                self.assertGreaterEqual(float(distance[dragon_shoes].min()), 4.0)
                sole_setback = np.nonzero(shoes)[0].max() - place["y"] * h
                self.assertGreater(sole_setback, 0)
                self.assertLessEqual(sole_setback, .11 * 304)
                changed = np.any(shot != raw, axis=2)
                left, top, right, bottom = map(int, entry.face_box_px)
                face = np.zeros_like(opaque)
                face[top:bottom, left:right] = opaque[top:bottom, left:right]
                self.assertGreater(changed[project(face)].mean(), .95)
                torso = np.zeros_like(opaque)
                hip = bottom + round(.34 * entry.standing_height_px)
                torso[bottom:hip] = opaque[bottom:hip]
                self.assertGreater(person[project(torso)].mean(), .50)
                body = project(opaque)
                hidden = body & (person > .5)
                self.assertGreater(hidden.sum(), .10 * body.sum())
                self.assertFalse(changed[hidden].any())
                self.assertIsNotNone(canvas.apply_command(Command("peek", {})))
                np.testing.assert_array_equal(canvas.render_capture_frame(raw), shot)

    def test_cropped_entry_and_peek_keep_shoes_out_of_frame(self):
        person, _ = subject(245)
        # Crop a waist-up view from the same silhouette so a foreground hem
        # cannot be confused with a person's shoes at the bottom of the image.
        cropped = cv2.resize(person[100:298, 140:404], (640, 480), interpolation=cv2.INTER_NEAREST)
        canvas, raw = self.camera_view(cropped)
        first, place = canvas._compose_frame(camera_frame=raw, capture=True)
        self.assertGreater(place["y"], 1)
        self.assertEqual(place["placement_source"], "自动侧探")
        self.assertIsNotNone(canvas.apply_command(Command("peek", {})))
        second, peek = canvas._compose_frame(camera_frame=raw, capture=True)
        self.assertGreater(peek["y"], 1)
        self.assertEqual(place["standing_height_ratio"], peek["standing_height_ratio"])
        np.testing.assert_array_equal(first, second)
