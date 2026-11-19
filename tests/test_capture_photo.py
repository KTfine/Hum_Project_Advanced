"""Exercise the model shutter with synthetic camera frames and API events."""
import asyncio
import io
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import cv2
import numpy as np

import photo_assistant.client as client


def delta(response_id, text):
    return {"type": "response.output.delta", "kind": "text", "response_id": response_id, "text": text}


class PhotoCaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory())) / "照片"
        self.camera = Mock(spec=client.CameraSource)
        self.capture = client.PhotoCapture(self.camera, self.directory)
        self.frame = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)

    def test_each_capture_persists_its_own_raw_frame_as_a_png(self):
        next_frame = np.flip(self.frame, axis=1).copy()
        self.camera.latest_frame.side_effect = [self.frame, next_frame]
        first = self.capture.save()
        second = self.capture.save()

        self.assertNotEqual(first, second)
        self.assertEqual(set(self.directory.iterdir()), {first, second})
        for path, frame in [(first, self.frame), (second, next_frame)]:
            self.assertEqual(path.suffix, ".png")
            png = path.read_bytes()
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, frame)

    def test_unavailable_camera_or_frame_creates_no_photo(self):
        self.camera.latest_frame.return_value = None
        canvas = Mock(spec=client.CanvasSource)
        for camera in (None, self.camera):
            with self.subTest(camera=camera):
                with self.assertRaises(RuntimeError):
                    client.PhotoCapture(camera, self.directory, character_canvas=canvas).save()
                self.assertEqual(list(self.directory.glob("*.png")), [])
        canvas.render_frame.assert_not_called()

    def receive(self, events):
        session = Mock()
        session.recv = AsyncMock(side_effect=events)
        log = Mock(spec=client.FrontendEventLogger)
        stop = threading.Event()
        with redirect_stdout(io.StringIO()):
            asyncio.run(client.receive_realtime_outputs(
                session, stop, Mock(spec=client.CanvasSource), client.SpeechQueue(),
                verbose=False, event_log=log, photo_capture=self.capture,
            ))
        self.assertTrue(stop.is_set())
        return log

    def test_shutter_fires_on_complete_live_tag_and_ignores_stale_and_done_text(self):
        self.camera.latest_frame.return_value = self.frame

        def events():
            yield delta("old", "<capt")
            yield {"type": "response.interrupted", "response_id": "old"}
            yield delta("old", "ure><capture>")
            yield delta("new", "<capture")
            self.camera.latest_frame.assert_not_called()
            yield delta("new", ">")
            self.camera.latest_frame.assert_called_once_with()
            yield {"type": "response.done", "response_id": "new", "text": "<capture>"}
            yield {"type": "session.closed"}

        log = self.receive(events())
        self.camera.latest_frame.assert_called_once_with()
        path, = self.directory.glob("*.png")
        log.record.assert_any_call("photo.saved", path=str(path))

    def test_stream_reports_missing_frame_and_can_take_the_next_photo(self):
        self.camera.latest_frame.side_effect = [None, self.frame]
        log = self.receive([
            delta("r1", "<capture>"),
            delta("r2", "<capture>"),
            {"type": "session.closed"},
        ])
        skipped, = [entry.kwargs for entry in log.record.call_args_list
                    if entry.args == ("command.skipped",)]
        self.assertEqual(skipped["command"]["kind"], "capture")
        self.assertTrue(skipped["error"])
        path, = self.directory.glob("*.png")
        log.record.assert_any_call("photo.saved", path=str(path))
