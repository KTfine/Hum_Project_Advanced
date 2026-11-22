"""Exercise AssemblyAI streaming turns with deterministic audio and WebSocket events."""

import asyncio
import json
import os
import sys
import threading
import unittest
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aiohttp import WSMessage, WSMsgType

from photo_assistant import voice_input


class FakeSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.blocks = asyncio.Queue()
        self.controls = []
        self.closed = False
        self.event(type="Begin")

    def event(self, **payload):
        self.incoming.put_nowait(WSMessage(WSMsgType.TEXT, json.dumps(payload), ""))

    async def receive(self):
        return await self.incoming.get()

    async def send_bytes(self, pcm):
        self.blocks.put_nowait(pcm)

    async def send_json(self, data):
        self.controls.append(data)
        self.event(type="Termination")


class VoiceInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stop = threading.Event()
        self.speaking = threading.Event()
        self.closed = threading.Event()
        self.opened = asyncio.Event()
        self.delivered = []
        self.messages = []
        self.hardware = SimpleNamespace(query_devices=Mock(return_value={
            "name": "USB microphone", "default_samplerate": 48000,
        }), RawInputStream=Mock(side_effect=self.open_input))
        self.enterContext(patch.dict(sys.modules, {"sounddevice": self.hardware}))
        self.socket = FakeSocket()
        self.connect = self.enterContext(patch("aiohttp.ClientSession.ws_connect", side_effect=self.connect_ws))

    @asynccontextmanager
    async def connect_ws(self, endpoint, **kwargs):
        yield self.socket

    @contextmanager
    def open_input(self, **kwargs):
        self.capture = kwargs["callback"]
        self.opened.set()
        try:
            yield SimpleNamespace(active=True)
        finally:
            self.closed.set()

    async def start(self):
        self.worker = asyncio.create_task(voice_input.recognize_microphone(
            self.stop, self.delivered.append, self.messages.append, api_key="secret",
            device=3, speaking=self.speaking,
        ))
        self.addAsyncCleanup(self.finish)
        await asyncio.wait_for(self.opened.wait(), 2)

    async def finish(self):
        self.stop.set()
        if not self.worker.cancelled():
            await asyncio.wait_for(self.worker, 3)

    async def wait_for_text(self, text):
        async with asyncio.timeout(2):
            while text not in self.delivered:
                await asyncio.sleep(.005)

    def turn(self, order, text, final=True):
        self.socket.event(type="Turn", turn_order=order, transcript=text, end_of_turn=final)

    async def test_activity_and_unique_completed_turns_reach_client(self):
        await self.start()
        self.socket.event(type="SpeechStarted", timestamp=100)
        self.turn(0, "现在", False)
        self.turn(0, "现在拍照")
        self.turn(0, "现在拍照。")
        self.turn(1, "   ")
        self.turn(2, "Take a photo.")
        await self.wait_for_text("Take a photo.")
        await self.finish()
        self.assertEqual(self.delivered, ["现在拍照", "Take a photo."])
        self.assertTrue(any("检测到说话" in message for message in self.messages))
        self.assertTrue(any("USB microphone" in message for message in self.messages))
        self.assertTrue(self.closed.is_set())
        self.assertEqual(self.socket.controls, [{"type": "Terminate"}])
        options = self.connect.call_args.kwargs
        self.assertEqual(options["headers"], {"Authorization": "secret"})
        self.assertEqual(options["params"], {
            "sample_rate": 48000, "speech_model": "assemblyai-stt-3-7-realtime-agent", "mode": "balanced",
        })
        self.hardware.query_devices.assert_called_once_with(3, "input")
        options = self.hardware.RawInputStream.call_args.kwargs
        self.assertEqual((options["device"], options["blocksize"], options["channels"], options["dtype"]),
                         (3, 4800, 1, "int16"))

    async def test_playback_sends_silence_and_first_new_phrase_is_accepted(self):
        await self.start()
        pcm = bytes([17, 2]) * 4800
        self.speaking.set()
        self.capture(pcm, 4800, None, None)
        self.speaking.clear()
        self.capture(pcm, 4800, None, None)
        self.assertEqual(await asyncio.wait_for(self.socket.blocks.get(), 1), bytes(len(pcm)))
        self.assertEqual(await asyncio.wait_for(self.socket.blocks.get(), 1), pcm)
        self.turn(0, "再拍一张")
        await self.wait_for_text("再拍一张")
        self.assertEqual(self.delivered, ["再拍一张"])

    async def test_overflow_discards_damaged_utterance(self):
        await self.start()
        self.capture(b"", 4800, None, True)
        self.capture(bytes(9600), 4800, None, None)
        async with asyncio.timeout(1):
            while not any("丢帧" in status for status in self.messages):
                await asyncio.sleep(.005)
        self.turn(0, "不完整的指令")
        self.turn(1, "向左")
        await self.wait_for_text("向左")
        self.assertEqual(self.delivered, ["向左"])

    async def test_cancel_terminates_billable_session_and_closes_device(self):
        await self.start()
        self.worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.worker
        self.assertTrue(self.closed.is_set())
        self.assertEqual(self.socket.controls, [{"type": "Terminate"}])

    async def test_connection_error_and_missing_key_are_reported(self):
        self.socket.incoming = asyncio.Queue()
        self.socket.incoming.put_nowait(WSMessage(WSMsgType.CLOSE, 1008, "Invalid key secret"))
        await voice_input.recognize_microphone(self.stop, Mock(), self.messages.append, api_key="secret")
        self.assertTrue(any("1008" in s and "secret" not in s for s in self.messages))
        self.hardware.RawInputStream.assert_not_called()
        with patch.dict(os.environ, {"ASSEMBLYAI_API_KEY": ""}):
            await voice_input.recognize_microphone(self.stop, Mock(), self.messages.append)
        self.assertIn("未配置", self.messages[-1])
        self.connect.assert_called_once()

    async def test_device_failure_still_terminates_socket(self):
        self.hardware.RawInputStream.side_effect = OSError("device disconnected")
        await voice_input.recognize_microphone(self.stop, Mock(), self.messages.append, api_key="test")
        self.assertIn("device disconnected", self.messages[-1])
        self.assertEqual(self.socket.controls, [{"type": "Terminate"}])

    async def test_transient_close_reconnects_and_resets_turn_order(self):
        await self.start()
        self.turn(0, "第一句")
        await self.wait_for_text("第一句")
        original = self.socket
        self.socket = FakeSocket()
        original.incoming.put_nowait(WSMessage(WSMsgType.CLOSE, 3008, "session expired"))
        self.turn(0, "第二句")
        await self.wait_for_text("第二句")
        self.assertEqual(self.delivered, ["第一句", "第二句"])
        self.assertEqual(self.connect.call_count, 2)

    async def test_zero_input_diagnoses_selected_microphone_once(self):
        await self.start()
        for _ in range(52):
            self.capture(bytes(9600), 4800, None, None)
            await asyncio.wait_for(self.socket.blocks.get(), 1)
        self.assertEqual(sum("全零音频" in message for message in self.messages), 1)


if __name__ == "__main__":
    unittest.main()
