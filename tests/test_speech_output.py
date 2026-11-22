import asyncio
from array import array
from contextlib import nullcontext
import json
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from photo_assistant import speech_output


class SpeechOutputTests(unittest.TestCase):
    def setUp(self):
        self.report = Mock()
        self.player = speech_output.SpeechOutput(self.report, api_key="unit-test")
        self.stream = Mock()
        self.device = Mock()
        self.device.RawOutputStream.return_value = nullcontext(self.stream)
        self.edge = Mock()
        self.decoder = Mock()
        self.enterContext(patch.dict(sys.modules, {
            "edge_tts": self.edge, "miniaudio": self.decoder, "sounddevice": self.device,
        }))
        self.addCleanup(self.player.close)

    def test_voice_uses_clause_language_and_previous_voice_for_a_countdown(self):
        chinese, english = "zh-CN-XiaoyiNeural", "aai-audrey"
        for text, previous, expected in (
            ("肩膀放松，", english, chinese), ("Try turning left.", chinese, english),
            ("3、2、1！", english, english), ("3、2、1！", chinese, chinese),
            ("试试这个 OK 的角度。", english, chinese), ("Look at 小白龙 and smile!", chinese, chinese),
        ):
            with self.subTest(text=text):
                self.assertEqual(speech_output._voice_for(text, previous), expected)

    def test_edge_audio_packets_are_combined_and_decoded_in_memory(self):
        async def stream():
            yield {"type": "audio", "data": b"first"}
            yield {"type": "WordBoundary", "text": "Smile"}
            yield {"type": "audio", "data": b"second"}
        self.edge.Communicate.return_value.stream = stream
        pcm = array("h", [11, -12])
        self.decoder.decode.return_value = SimpleNamespace(samples=pcm)
        result = asyncio.run(speech_output._synthesize("微笑，", speech_output.CHINESE_VOICE))
        self.assertEqual(result, pcm.tobytes())
        self.assertEqual(self.edge.Communicate.call_args.args, ("微笑，", "zh-CN-XiaoyiNeural"))
        self.decoder.decode.assert_called_once_with(b"firstsecond",
            output_format=self.decoder.SampleFormat.SIGNED16, nchannels=1, sample_rate=24000)

    def test_assembly_stream_starts_before_http_body_completes(self):
        playing, finished = threading.Event(), threading.Event()
        played = []

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"a"
                yield b"bcd"
                if not await asyncio.to_thread(playing.wait, 2):
                    raise AssertionError("first packet was buffered until the end of the response")
                yield b"ef"

        def handler(request):
            self.assertEqual(str(request.url), "https://tts.us.assemblyai.com/v1/speech")
            self.assertEqual(request.headers["Authorization"], "unit-test")
            self.assertEqual(json.loads(request.content), {"text": "Smile,", "voice_id": "aai-audrey",
                "language": "english", "encoding": "pcm_s16le", "sample_rate": 24000,
                "container": "raw", "stream": True})
            return httpx.Response(200, stream=Body())

        def write(pcm):
            played.append(pcm)
            (playing if len(played) == 1 else finished).set()

        self.stream.write.side_effect = write
        connection = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(speech_output.httpx, "AsyncClient", return_value=connection):
            self.player.start()
            self.player.say("Smile,")
            self.assertTrue(finished.wait(4))
            self.player.close()
        self.assertEqual(played, [b"abcd", b"ef"])
        self.device.RawOutputStream.assert_called_once()
        self.edge.Communicate.assert_not_called()

    def test_overlaps_synthesis_with_playback_and_preserves_order(self):
        writing, prefetched, finished = (threading.Event() for _ in range(3))
        chinese, english = "看镜头，", "Now smile."
        played = []

        async def assembly_audio(client, text, api_key):
            self.assertEqual((text, api_key), (english, "unit-test"))
            if not await asyncio.to_thread(writing.wait, 2):
                raise AssertionError("first clause did not begin playback")
            prefetched.set()
            yield text.encode("utf-8")

        def write(block):
            played.append((block, self.player.speaking.is_set()))
            if len(played) == 1:
                writing.set()
                prefetched.wait(2)
            else:
                finished.set()

        self.stream.write.side_effect = write
        with patch.object(speech_output, "_assembly_audio", new=assembly_audio), \
             patch.object(speech_output, "_synthesize", new=AsyncMock(return_value=chinese.encode("utf-8"))) as synth:
            self.player.start()
            self.player.say(chinese)
            self.player.say(english)
            self.assertTrue(finished.wait(4))
            self.player.close()
            synth.assert_awaited_once_with(chinese, "zh-CN-XiaoyiNeural")
        self.assertTrue(prefetched.is_set())
        self.assertEqual(played, [(chinese.encode("utf-8"), True), (english.encode("utf-8"), True)])
        self.assertFalse(self.player.speaking.is_set())
        self.device.RawOutputStream.assert_called_with(samplerate=24000, channels=1, dtype="int16")

    def test_missing_key_keeps_chinese_but_does_not_send_english(self):
        finished = threading.Event()
        self.stream.write.side_effect = lambda _: finished.set()
        player = speech_output.SpeechOutput(self.report)
        self.addCleanup(player.close)
        with patch.object(speech_output, "_synthesize", new=AsyncMock(return_value=b"next")) as edge, \
             patch.object(speech_output, "_assembly_audio") as assembly:
            player.start()
            player.say("Smile.")
            player.say("看镜头。")
            self.assertTrue(finished.wait(3))
            player.close()
            assembly.assert_not_called()
            edge.assert_awaited_once_with("看镜头。", speech_output.CHINESE_VOICE)

    def test_network_failure_skips_only_failed_clause(self):
        finished = threading.Event()
        self.stream.write.side_effect = lambda _: finished.set()
        with patch.object(speech_output, "_synthesize", new=AsyncMock(
                side_effect=[TimeoutError(), b"next"])) as synth:
            self.player.start()
            self.player.say("第一段，")
            self.player.say("第二段。")
            self.assertTrue(finished.wait(3))
            self.player.close()
            self.assertEqual(synth.await_count, 2)
        self.stream.write.assert_called_once_with(b"next")
        self.assertTrue(any("TimeoutError" in call.args[0] for call in self.report.call_args_list))

    def test_output_failure_clears_microphone_gate_and_keeps_subtitles_available(self):
        reported = threading.Event()
        self.report.side_effect = lambda message: reported.set() if "播放失败" in message else None
        self.device.RawOutputStream.side_effect = OSError("no output device")
        with patch.object(speech_output, "_synthesize", new=AsyncMock(return_value=b"pcm")):
            self.player.start()
            self.player.say("看镜头。")
            self.assertTrue(reported.wait(3))
            self.assertFalse(self.player.speaking.is_set())

    def test_close_cancels_network_and_discards_queued_clauses(self):
        entered, cancelled = threading.Event(), threading.Event()

        async def synthesize(text, voice):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with patch.object(speech_output, "_synthesize", side_effect=synthesize) as synth:
            self.player.start()
            self.player.say("第一句。")
            self.player.say("下一句。")
            self.assertTrue(entered.wait(3))
            self.player.close()
            self.assertTrue(cancelled.wait(1))
            self.assertFalse(self.player._thread.is_alive())
            self.assertEqual(synth.await_count, 1)
            self.device.RawOutputStream.assert_not_called()
