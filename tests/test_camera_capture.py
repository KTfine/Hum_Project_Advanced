"""Exercise Windows camera selection with fake devices and capture handles."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from photo_assistant.vision.camera import open_camera


def device(name, index, backend, vid=123, pid=456):
    return SimpleNamespace(name=name, index=index, backend=backend, vid=vid, pid=pid)


class CameraCaptureTests(unittest.TestCase):
    def setUp(self):
        self.devices = {cv2.CAP_DSHOW: [], cv2.CAP_MSMF: []}
        self.enumerate = Mock(side_effect=self.devices.__getitem__)
        module = SimpleNamespace(enumerate_cameras=self.enumerate)
        self.enterContext(patch.dict(sys.modules, {"cv2_enumerate_cameras": module}))
        self.capture = self.enterContext(patch.object(cv2, "VideoCapture"))
        self.enterContext(patch("photo_assistant.vision.camera.time.sleep"))
        self.image = np.ones((2, 3, 3), dtype=np.uint8)

    def test_name_selects_external_camera_after_its_index_changes(self):
        self.devices[cv2.CAP_DSHOW] = [
            device("FHD Camera", 1, cv2.CAP_DSHOW),
            device("2K USB Camera", 3, cv2.CAP_DSHOW),
        ]
        handle = self.capture.return_value
        handle.isOpened.return_value = True
        handle.read.return_value = (True, self.image)

        capture, frame, label = open_camera(701, "2k usb CAMERA")

        self.capture.assert_called_once_with(3, cv2.CAP_DSHOW)
        self.assertIs(capture, handle)
        self.assertIs(frame, self.image)
        self.assertIn("2K USB Camera", label)
        handle.release.assert_not_called()

    def test_failed_first_frame_releases_dshow_then_uses_same_camera_on_msmf(self):
        self.devices[cv2.CAP_DSHOW] = [device("2K USB Camera", 1, cv2.CAP_DSHOW)]
        self.devices[cv2.CAP_MSMF] = [
            device("FHD Camera", 1, cv2.CAP_MSMF),
            device("2K USB Camera", 4, cv2.CAP_MSMF),
        ]
        failed, working = Mock(), Mock()
        failed.isOpened.return_value = working.isOpened.return_value = True
        failed.read.return_value = (False, None)
        working.read.return_value = (True, self.image)

        def open_handle(index, backend):
            if backend == cv2.CAP_MSMF:
                failed.release.assert_called_once_with()
                return working
            return failed

        self.capture.side_effect = open_handle
        capture, frame, label = open_camera(701, "2K USB Camera")

        self.assertEqual(self.capture.call_args_list, [call(1, cv2.CAP_DSHOW), call(4, cv2.CAP_MSMF)])
        self.assertIs(capture, working)
        self.assertIs(frame, self.image)
        self.assertIn("MSMF", label)
        working.release.assert_not_called()

    def test_missing_camera_does_not_open_built_in_or_similarly_named_device(self):
        self.devices[cv2.CAP_DSHOW] = [device("FHD Camera", 1, cv2.CAP_DSHOW)]
        self.devices[cv2.CAP_MSMF] = [device("2K USB Camera Virtual", 1, cv2.CAP_MSMF)]
        with self.assertRaisesRegex(RuntimeError, "2K USB Camera"):
            open_camera(701, "2K USB Camera")
        self.capture.assert_not_called()

    def test_rejects_conflicting_identity_without_opening_a_camera(self):
        self.devices[cv2.CAP_DSHOW] = [device("2K USB Camera", 1, cv2.CAP_DSHOW)]
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            with self.subTest(backend=backend):
                self.devices[backend].append(device("2K USB Camera", 2, backend, pid=789))
                with self.assertRaisesRegex(RuntimeError, "2K USB Camera"):
                    open_camera(701, "2K USB Camera")
                self.capture.assert_not_called()
                self.devices[backend].pop()

    def test_numeric_override_skips_device_enumeration(self):
        handle = self.capture.return_value
        handle.isOpened.return_value = True
        handle.read.return_value = (True, self.image)
        capture, _, _ = open_camera(0, None)
        self.assertIs(capture, handle)
        self.capture.assert_called_once_with(0)
        self.enumerate.assert_not_called()
