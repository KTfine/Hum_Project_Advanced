"""Replay a co-photo session from numbered camera exposures through streamed speech."""
import asyncio
import base64
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import cv2
import numpy as np

from photo_assistant import client
from photo_assistant.api.session import DirectSession


def exposure(second):
    # The person steps aside and back before speaking, then moves right and holds.
    left = 132 if 1 <= second < 4 or second >= 14 else 44
    frame = np.full((128, 192, 3), 48, np.uint8)
    mask = np.zeros(frame.shape[:2], np.float32)
    frame[26:62, left:left + 22] = (155, 170, 190)
    frame[62:116, left - 5:left + 27] = (130, 55, 35)
    mask[26:62, left:left + 22] = 1
    mask[62:116, left - 5:left + 27] = 1
    return frame, mask


class ObservationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_second_updates_send_matched_composites_even_when_nothing_moves(self):
        tick = 0
        stop = threading.Event()
        user_inputs = {8: "我想和你合照，帮我找一个合适的位置。", 19: "我保持这个位置。"}
        camera = Mock(latest_frame=lambda: exposure(tick)[0])
        # The segmentation worker finishes two seconds behind the live camera.
        worker = Mock(latest=lambda: exposure(tick - 2))
        canvas = client.CanvasSource(client.WHITE_DRAGON_CATALOG, 192, 128, 95, stop,
                                     preview=False, person_occlusion=worker)
        canvas.start()
        self.addCleanup(canvas.close)
        canvas.configure_view(camera, show_character=True)
        text_input = Mock(peek=lambda: user_inputs.get(tick))
        speech = {8: "请往右走一步。", 16: "现在的位置刚好。", 19: "好。"}
        received = []

        async def complete(messages, *, on_delta):
            received.append((tick, messages[-1]["content"]))
            reply = f"<speak:{speech[tick]}>" if tick in speech else "<blank>"
            on_delta(reply[:4])
            await asyncio.sleep(0)
            on_delta(reply[4:])
            return SimpleNamespace(text=reply, metadata={})

        provider = AsyncMock()
        provider.complete.side_effect = complete
        session = DirectSession("co-photo", provider)
        self.addAsyncCleanup(session.close)
        encoded = {}
        original_encode = canvas.encode_jpeg

        def encode(image):
            self.assertNotIn(tick, encoded, "the same exposure was encoded twice")
            encoded[tick] = original_encode(image)
            return encoded[tick]

        done = asyncio.Event()
        sources = []
        spoken = []

        def record(event, **fields):
            if event == "api.receive" and fields["message"]["type"] == "response.done":
                sources.append(fields["message"].get("turn_source", "user"))
                done.set()
            if event == "speak":
                spoken.append(fields["text"])

        async def advance(seconds):
            nonlocal tick
            self.assertEqual(seconds, 1.0)
            if tick in encoded:
                await done.wait()
                done.clear()
            tick += 1
            if tick == 22:
                await session.close()
                stop.set()

        log = Mock(record=record)
        terminal = Mock()
        with patch.object(canvas, "encode_jpeg", side_effect=encode), \
             patch.object(client, "time", SimpleNamespace(monotonic=lambda: tick)), \
             patch.object(client, "asyncio", SimpleNamespace(sleep=advance)):
            async with asyncio.timeout(5):
                await asyncio.gather(
                    client.send_realtime_inputs(session, canvas, camera, stop, text_input=text_input,
                                                report_placement=True, observation_interval=2.0,
                                                event_log=log, terminal=terminal),
                    client.receive_realtime_outputs(session, stop, canvas, client.SpeechQueue(),
                                                    verbose=False, event_log=log, terminal=terminal),
                )

        expected = [8, 10, 12, 14, 16, 18, 19, 21]
        self.assertEqual([second for second, _ in received], expected)
        self.assertEqual(list(encoded), expected)
        self.assertEqual(sources, ["user" if second in user_inputs else "observation" for second in expected])
        self.assertEqual([call.args[0] for call in text_input.acknowledge.call_args_list], list(user_inputs.values()))
        self.assertEqual(spoken, list(speech.values()))
        self.assertEqual(encoded[10], encoded[12])
        self.assertNotEqual(encoded[14], encoded[16])
        for second, blocks in received:
            if second in user_inputs:
                self.assertEqual(blocks[0]["text"], user_inputs[second])
            data = base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1])
            self.assertEqual(data, encoded[second])
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            raw, mask = exposure(second - 2)
            # At t=14 the live person has moved, but the composited exposure
            # still shows them on the left. The next scheduled update shows the step.
            person_x = 54 if second < 16 else 142
            np.testing.assert_allclose(image[45, person_x], raw[45, person_x], atol=6)
            # JPEG deviations well above compression noise outside the person are the avatar.
            changed = np.max(np.abs(image.astype(int) - raw), axis=2) > 32
            self.assertGreater(np.count_nonzero(changed & (mask == 0)), 100)
