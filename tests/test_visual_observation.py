import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from photo_assistant import client


class VisualObservationTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_guidance_skips_missed_slots_while_busy_or_without_a_camera_frame(self):
        tick = 0
        stop = threading.Event()
        goal = "帮我调整一下构图。"
        session = Mock(can_observe=False)
        camera = Mock(latest_frame=lambda: None if 12 <= tick <= 15 else bytes([tick]))
        canvas = Mock()
        canvas.compose_view.side_effect = lambda *, camera_frame, include_character: (camera_frame, None, camera_frame)
        encoded = []

        def encode(image):
            encoded.append((tick, image))
            return b"image" + image

        canvas.encode_jpeg.side_effect = encode
        text_input = Mock(peek=lambda: goal if tick == 3 else None)

        async def advance(seconds):
            nonlocal tick
            self.assertEqual(seconds, 1.0)
            tick += 1
            session.can_observe = tick > 3 and not 6 <= tick <= 9
            if tick == 21:
                stop.set()

        with patch.object(client, "time", SimpleNamespace(monotonic=lambda: tick)), \
             patch.object(client, "asyncio", SimpleNamespace(sleep=advance)):
            await client.send_realtime_inputs(session, canvas, camera, stop, text_input=text_input,
                                              include_character=False, observation_interval=2.0)

        # The camera or session recovers on an off-second (10 and 16).
        # The next scheduled observations use those slots' latest images (11 and 17).
        self.assertEqual(encoded, [(t, bytes([t])) for t in (3, 5, 11, 17, 19)])
        self.assertEqual([call.args[0] for call in session.submit_observation.call_args_list],
                         [b"image" + bytes([t]) for t in (5, 11, 17, 19)])
        text_input.acknowledge.assert_called_once_with(goal)
