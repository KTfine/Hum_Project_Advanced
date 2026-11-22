import asyncio
import contextlib
import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from photo_assistant import client


class SpokenReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_preserves_live_transcript_and_reads_sentences_in_order(self):
        output = Mock()
        terminal = Mock()
        delta = {"type": "response.output.delta", "kind": "text", "response_id": "1"}
        events = iter([
            {**delta, "text": "先往"},
            {**delta, "text": "左。<capture><unknown:secret><speak:看镜头>"},
            {"type": "response.done", "response_id": "1"},
            {**delta, "text": "Keep your", "response_id": "2"},
            {**delta, "text": " chin up", "response_id": "2"},
            {"type": "response.interrupted", "response_id": "2"},
            {**delta, "text": "stale", "response_id": "2"},
            {**delta, "text": "好！<mode:co-photo>", "response_id": "3"},
            {"type": "response.done", "response_id": "3"},
        ])
        async def recv():
            if terminal.speech.call_count == 1:
                terminal.speech.assert_called_once_with("先往")
                output.say.assert_not_called()
            return next(events)

        result = await client.receive_realtime_outputs(
            SimpleNamespace(recv=recv, emits_response_done=True), threading.Event(),
            Mock(), client.SpeechQueue(output), verbose=False,
            terminal=terminal, mode_selection=client.ModeSelection(),
        )
        self.assertEqual(result, ("co-photo", []))
        self.assertEqual(output.say.call_args_list, [
            call("先往左。"), call("看镜头"), call("好！"),
        ])
        self.assertTrue(any("[photo]" in args[0][0] for args in terminal.line.call_args_list))

    async def test_commas_play_before_speak_closes_and_actions_keep_their_order(self):
        output, canvas, terminal = Mock(), Mock(), Mock()
        delta = {"type": "response.output.delta", "kind": "text", "response_id": "r"}
        async def events():
            yield {**delta, "text": "<sp"}
            yield {**delta, "text": "eak:放松肩膀，"}
            self.assertEqual(output.say.call_args_list, [call("放松肩膀，")])
            terminal.speech.assert_not_called()
            yield {**delta, "text": "Look this way,"}
            self.assertEqual(output.say.call_args_list[-1], call("Look this way,"))
            yield {**delta, "text": "很好><follow><speak:Smile; hold still!>"}
            canvas.apply_command.assert_called_once()
            self.assertEqual(terminal.speech.call_args_list, [
                call("放松肩膀，Look this way,很好"), call("Smile; hold still!"),
            ])
            yield {"type": "response.done", "response_id": "r"}
            yield {"type": "session.closed"}

        stream = events()
        await client.receive_realtime_outputs(
            SimpleNamespace(recv=stream.__anext__), threading.Event(), canvas,
            client.SpeechQueue(output), verbose=False, terminal=terminal,
        )
        self.assertEqual(output.say.call_args_list, [
            call("放松肩膀，"), call("Look this way,"), call("很好"), call("Smile;"), call("hold still!"),
        ])

    async def test_incomplete_speech_tail_is_discarded_on_end_and_interruption(self):
        output = Mock()
        def text(rid, body):
            return {"type": "response.output.delta", "kind": "text", "response_id": rid, "text": body}
        events = [
            text("1", "<speak:Talk, half a wo"), {"type": "response.done", "response_id": "1"},
            text("2", "<speak:再试一次；未完"), {"type": "response.interrupted", "response_id": "2"},
            text("2", "旧片段>"), text("3", "Eyes"), text("3", " "), text("3", "here: smile"),
            {"type": "response.done", "response_id": "3"}, {"type": "session.closed"},
        ]
        await client.receive_realtime_outputs(
            SimpleNamespace(recv=AsyncMock(side_effect=events)), threading.Event(), Mock(),
            client.SpeechQueue(output), verbose=False, terminal=Mock(),
        )
        self.assertEqual(output.say.call_args_list, [call("Talk,"), call("再试一次；"), call("Eyes here:"), call("smile")])

    def test_punctuation_inside_numbers_survives_every_chunk_boundary(self):
        for text, clauses in [
            ("Set ISO 1,200, then f/2.8. Ready?", ["Set ISO 1,200,", "then f/2.8.", "Ready?"]),
            ("Start at 10:30; smile!", ["Start at 10:30;", "smile!"]),
            ("Use edge.tts in the U.S.?", ["Use edge.tts in the U.S.?"]),
            ("转身、抬头：很好。", ["转身、", "抬头：", "很好。"]),
        ]:
            with self.subTest(text=text):
                output = Mock()
                queue = client.SpeechQueue(output)
                for char in text:
                    queue.read_aloud(char)
                queue.read_aloud(final=True)
                self.assertEqual([args.args[0] for args in output.say.call_args_list], clauses)

    async def test_idle_ready_event_updates_status_without_encoding_an_image(self):
        stop = threading.Event()
        session = Mock()
        session.submit_frame.side_effect = lambda *args, **kwargs: stop.set()
        readiness = Mock(last_action_succeeded=True)
        readiness.observe.return_value = "action"
        terminal = Mock()
        canvas = Mock()
        canvas.compose_view.return_value = object(), {"pose_settled": True}, object()
        with patch.object(client.asyncio, "sleep", new_callable=AsyncMock):
            await client.send_realtime_inputs(
                session, canvas, Mock(), stop, terminal=terminal, readiness=readiness, report_placement=True,
            )
        terminal.line.assert_called_once_with("[合照] 角色已就位。")
        terminal.speech.assert_not_called()
        session.submit_frame.assert_called_once_with(
            None, None, placement={"pose_settled": True, "readiness_event": "action",
                                      "last_action_succeeded": True},
        )
        canvas.encode_jpeg.assert_not_called()

    async def test_launcher_config_and_microphone_gate_use_one_player(self):
        for settings, switches, enabled in [
            ({}, [], True),
            ({"tts": False}, [], False),
            ({"tts": False}, ["--tts"], True),
            ({"tts": True}, ["--no-tts"], False),
        ]:
            with self.subTest(settings=settings, switches=switches):
                args = client.parse_args(["--no-camera", "--no-preview", "--microphone", *switches])
                def make_player(on_status, *, api_key):
                    self.assertEqual(api_key, "test-assembly")
                    on_status("就绪")
                    return player
                player = Mock(speaking=threading.Event())
                async def sessions(*args, **_):
                    await asyncio.sleep(0)
                    self.assertIs(args[3].output, player if enabled else None)
                transcript = io.StringIO()
                with patch.object(client, "read_api_key", return_value="test"), \
                     patch.object(client, "read_assemblyai_api_key", return_value="test-assembly"), \
                     patch.object(client, "read_project_settings", return_value=settings), \
                     patch.object(client, "CanvasSource", return_value=Mock(error=None)), \
                     patch.object(client, "SpeechOutput", side_effect=make_player) as make, \
                     patch.object(client.ConsoleTextInput, "run", new_callable=AsyncMock), \
                     patch.object(client, "recognize_microphone", new_callable=AsyncMock) as mic, \
                     patch.object(client, "run_realtime_sessions", sessions), \
                     contextlib.redirect_stdout(transcript):
                    await client.run_probe(args)
                self.assertEqual(make.call_count, int(enabled))
                self.assertEqual(player.start.call_count, int(enabled))
                self.assertEqual(player.close.call_count, int(enabled))
                self.assertEqual("[tts] 就绪" in transcript.getvalue(), enabled)
                mic.assert_awaited_once()
                self.assertEqual(mic.await_args.kwargs["api_key"], "test-assembly")
                self.assertIs(mic.await_args.kwargs.get("speaking"), player.speaking if enabled else None)
