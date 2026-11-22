"""Exercise offline dictation with deterministic PCM input and no audio hardware."""

import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from photo_assistant import voice_input


class VoiceInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        model_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"HUM_VOSK_MODEL": model_dir}))
        self.recognizer = Mock()
        self.vosk = SimpleNamespace(
            Model=Mock(), KaldiRecognizer=Mock(return_value=self.recognizer), SetLogLevel=Mock(),
        )
        self.device = SimpleNamespace(
            query_devices=Mock(return_value={"default_samplerate": 48000.0}),
            RawInputStream=Mock(),
        )
        self.enterContext(patch.dict(sys.modules, {"sounddevice": self.device, "vosk": self.vosk}))
        self.stop = threading.Event()
        self.opened = threading.Event()
        self.closed = threading.Event()

    def feed_pcm(self, *blocks):
        @contextmanager
        def stream(**options):
            self.assertEqual(options["channels"], 1)
            self.assertEqual(options["dtype"], "int16")
            self.assertEqual(options["samplerate"], 48000)
            try:
                self.opened.set()
                for block in blocks:
                    options["callback"](block, len(block) // 2, None, None)
                yield SimpleNamespace(active=True)
            finally:
                self.closed.set()

        self.device.RawInputStream.side_effect = stream

    async def test_only_completed_nonempty_utterances_are_delivered_once(self):
        blocks = [bytes([sample, 0]) for sample in range(4)]
        self.feed_pcm(*blocks)
        self.recognizer.AcceptWaveform.side_effect = [False, True, True, True]
        self.recognizer.Result.side_effect = [
            json.dumps({"text": text}) for text in ("现在 拍 一 张", "", "向 左 移动")
        ]

        delivered = []
        event_loop_thread = threading.get_ident()

        def receive(text):
            self.assertEqual(threading.get_ident(), event_loop_thread)
            delivered.append(text)
            if len(delivered) == 2:
                self.stop.set()

        try:
            await asyncio.wait_for(voice_input.recognize_microphone(self.stop, receive, Mock()), 4)
        finally:
            self.stop.set()

        self.assertEqual(delivered, ["现在拍一张", "向左移动"])
        self.assertEqual(self.recognizer.AcceptWaveform.call_args_list, [call(b) for b in blocks])
        self.recognizer.FinalResult.assert_not_called()
        self.assertTrue(self.closed.is_set())

    async def test_input_overflow_skips_damaged_phrase_and_accepts_next_command(self):
        first, stale, overflow, tail, next_command = [bytes([i, 0]) for i in range(5)]
        self.feed_pcm(first, stale)
        last_result = ""

        def recognize(block):
            nonlocal last_result
            if block == first:
                # The device overruns while Vosk is still decoding this phrase.
                capture = self.device.RawInputStream.call_args.kwargs["callback"]
                capture(overflow, 1, None, SimpleNamespace(input_overflow=True))
                capture(tail, 1, None, None)
                capture(next_command, 1, None, None)
                return False
            last_result = "拍照" if block == tail else "向 左 移动"
            return True

        self.recognizer.AcceptWaveform.side_effect = recognize
        self.recognizer.Result.side_effect = lambda: json.dumps({"text": last_result})
        delivered = []

        def receive(text):
            delivered.append(text)
            if text == "向左移动":
                self.stop.set()

        try:
            await asyncio.wait_for(voice_input.recognize_microphone(self.stop, receive, Mock()), 4)
        finally:
            self.stop.set()
        self.assertEqual(delivered, ["向左移动"])
        self.assertEqual(self.recognizer.AcceptWaveform.call_args_list, [call(first), call(tail), call(next_command)])
        self.recognizer.Reset.assert_called_once()
        self.device.RawInputStream.assert_called_once()
        self.assertTrue(self.closed.is_set())

    async def test_cancelling_while_waiting_for_audio_closes_stream(self):
        self.feed_pcm()
        on_text = Mock()
        task = asyncio.create_task(voice_input.recognize_microphone(self.stop, on_text, Mock()))
        try:
            self.assertTrue(await asyncio.to_thread(self.opened.wait, 3))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        finally:
            self.stop.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(self.closed.is_set())
        on_text.assert_not_called()

    async def test_audio_device_error_reaches_status_without_stopping_client(self):
        self.device.RawInputStream.side_effect = OSError("device disconnected")
        messages = []
        on_text = Mock()
        await asyncio.wait_for(
            voice_input.recognize_microphone(self.stop, on_text, messages.append), 3,
        )
        self.assertTrue(any("device disconnected" in message for message in messages), messages)
        self.assertFalse(self.stop.is_set())
        on_text.assert_not_called()
