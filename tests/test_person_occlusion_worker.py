"""Exercise frame identity and mode changes with a controllable local segmenter."""
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from photo_assistant.vision.occlusion import PersonOcclusion


def camera_frame(value):
    return np.full((4, 5, 3), value, np.uint8)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.model = Mock(segment=self.segment)
        self.worker = PersonOcclusion(lambda: self.model)
        self.addCleanup(self.worker.close)
        self.addCleanup(self.release.set)

    def segment(self, frame):
        value = int(frame[0, 0, 0])
        self.calls.append(value)
        if value == 1:
            self.started.set()
            if not self.release.wait(3):
                raise RuntimeError("test inference timed out")
        if value == 9:
            raise ValueError("model error")
        return frame[:, :, 0].astype(np.float32) / 255.0

    def wait_for_preview(self, frame):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            pair = self.worker.latest()
            if pair is not None and pair[0] is frame:
                return pair[1]
            time.sleep(0.002)
        self.fail("preview did not publish expected frame")

    def test_drops_obsolete_preview_work_and_captures_exact_frame(self):
        self.worker.set_enabled(True)
        self.worker.submit(camera_frame(1))
        self.assertTrue(self.started.wait(3))
        self.worker.submit(camera_frame(2))
        newest = camera_frame(3)
        self.worker.submit(newest)
        self.release.set()
        mask = self.wait_for_preview(newest)
        self.worker.set_enabled(True)
        self.assertIs(self.worker.mask_for_capture(newest), mask)
        shutter_frame = camera_frame(4)
        shutter_mask = self.worker.mask_for_capture(shutter_frame)
        np.testing.assert_allclose(shutter_mask, 4 / 255)
        self.assertEqual(self.calls, [1, 3, 4])
        self.worker.close()
        self.model.close.assert_called_once()

    def test_switch_discards_in_flight_result_and_disabled_submissions(self):
        self.worker.submit(camera_frame(5))
        self.worker.set_enabled(True)
        self.worker.submit(camera_frame(1))
        self.assertTrue(self.started.wait(3))
        self.worker.set_enabled(False)
        self.worker.submit(camera_frame(6))
        self.worker.set_enabled(True)
        self.release.set()
        # An exact shutter request completes only after the obsolete inference.
        np.testing.assert_allclose(self.worker.mask_for_capture(camera_frame(7)), 7 / 255)
        self.assertIsNone(self.worker.latest())
        self.assertEqual(self.calls, [1, 7])

    def test_preview_expiry_uses_frame_arrival_time(self):
        self.worker.set_enabled(True)
        with patch("photo_assistant.vision.occlusion.time.monotonic", return_value=100.0) as clock:
            self.worker.submit(camera_frame(1))
            self.assertTrue(self.started.wait(3))
            clock.return_value = 104.0
            self.release.set()
            self.worker.mask_for_capture(camera_frame(7))
            self.assertIsNone(self.worker.latest())

    def test_error_reaches_both_consumers_and_mode_reentry_allows_retry(self):
        self.worker.set_enabled(True)
        with self.assertRaisesRegex(RuntimeError, "model error"):
            self.worker.mask_for_capture(camera_frame(9))
        with self.assertRaisesRegex(RuntimeError, "model error"):
            self.worker.latest()
        self.worker.set_enabled(False)
        self.worker.set_enabled(True)
        np.testing.assert_allclose(self.worker.mask_for_capture(camera_frame(8)), 8 / 255)
