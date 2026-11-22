import asyncio
import contextlib
import io
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from photo_assistant import client


class MicrophoneInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_and_keyboard_share_queue_and_preserve_draft(self):
        stop = threading.Event()
        source = client.ConsoleTextInput()
        source.put("先向左一点")
        source._edit_key("好")

        async def recognize(stop_event, on_text, on_status):
            self.assertIs(stop_event, stop)
            on_status("就绪")
            on_text(" ")
            on_text("  现在拍一张  ")

        output = io.StringIO()
        with patch.object(client, "recognize_microphone", recognize), contextlib.redirect_stdout(output):
            await source.run_microphone(stop)
            source._edit_key("\r")

        self.assertEqual(list(source.pending), ["先向左一点", "现在拍一张", "好"])
        self.assertIn("[mic] 现在拍一张", output.getvalue())
        self.assertIn("[mic] 就绪", output.getvalue())

    async def test_engine_failure_leaves_keyboard_available(self):
        source = client.ConsoleTextInput()
        output = io.StringIO()
        with patch.object(client, "recognize_microphone", side_effect=OSError("设备已断开")), \
             contextlib.redirect_stdout(output):
            await source.run_microphone(threading.Event())
            source._accept_text("现在拍照\n")
        self.assertIn("可继续打字", output.getvalue())
        self.assertEqual(source.peek(), "现在拍照")

    async def test_launcher_selects_microphone_from_settings_and_cli_and_cleans_up(self):
        for settings, switches, enabled in (
            ({}, [], False),
            ({"microphone": True}, [], True),
            ({"microphone": False}, ["--microphone"], True),
            ({"microphone": True}, ["--no-microphone"], False),
        ):
            with self.subTest(settings=settings, switches=switches):
                stop_seen = []
                started = asyncio.Event()

                async def recognize(stop_event, on_text, on_status):
                    try:
                        on_text("现在拍一张")
                        started.set()
                        await asyncio.Future()
                    finally:
                        stop_seen.append(stop_event.is_set())

                async def sessions(*args, text_input, **kwargs):
                    await asyncio.sleep(0)
                    if enabled:
                        await asyncio.wait_for(started.wait(), 1)
                        self.assertEqual(text_input.peek(), "现在拍一张")

                args = client.parse_args(["--no-camera", "--no-preview", *switches])
                engine = AsyncMock(side_effect=recognize)
                canvas = Mock(error=None)
                with patch.object(client, "read_api_key", return_value="test"), \
                     patch.object(client, "read_project_settings", return_value=settings), \
                     patch.object(client, "CanvasSource", return_value=canvas), \
                     patch.object(client, "recognize_microphone", engine), \
                     patch.object(client.ConsoleTextInput, "run", new_callable=AsyncMock), \
                     patch.object(client, "run_realtime_sessions", sessions), \
                     contextlib.redirect_stdout(io.StringIO()):
                    await client.run_probe(args)
                self.assertEqual(engine.call_count, int(enabled))
                self.assertEqual(stop_seen, [True] if enabled else [])
