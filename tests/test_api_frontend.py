"""API frontend checks without opening a camera, microphone, or remote API."""
import asyncio
import contextlib
import io
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG, load_pose_catalog

import photo_assistant.client as client


class Log:
    def __init__(self):
        self.rows = []

    def record(self, event, **fields):
        self.rows.append(dict(event=event, **fields))

    def record_message(self, event, message):
        self.record(event, message=message)


class DirectApiStartupTests(unittest.TestCase):
    def setUp(self):
        temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.settings = Path(temp_dir) / "settings.json"
        self.enterContext(patch("photo_assistant.api.client.SETTINGS_PATH", self.settings))

    def test_only_direct_api_is_available(self):
        self.assertEqual(client.parse_args([]).backend, "api")
        self.assertEqual(client.parse_args(["--backend", "api"]).backend, "api")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            client.parse_args(["--backend", "minicpm"])

    def test_key_is_prompted_once_if_missing_and_never_saved(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
             patch.object(client.sys.stdin, "isatty", return_value=True), \
             patch.object(client.getpass, "getpass", return_value="one-run-key") as prompt:
            self.assertEqual(client.read_api_key(), "one-run-key")
            prompt.assert_called_once()
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "")

    def test_noninteractive_start_without_key_reports_setup_step(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
             patch.object(client.sys.stdin, "isatty", return_value=False), \
             self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            client.read_api_key()

    def test_assemblyai_key_uses_its_own_setting_env_or_hidden_prompt(self):
        with patch.dict(os.environ, {"ASSEMBLYAI_API_KEY": " env ", "DEEPSEEK_API_KEY": "different"}), \
             patch.object(client.sys.stdin, "isatty", return_value=True), \
             patch.object(client.getpass, "getpass", return_value=" prompt ") as prompt:
            self.assertEqual(client.read_assemblyai_api_key({"assemblyai_api_key": " file "}), "file")
            self.assertEqual(client.read_assemblyai_api_key({}), "env")
            prompt.assert_not_called()
            os.environ["ASSEMBLYAI_API_KEY"] = ""
            self.assertEqual(client.read_assemblyai_api_key({"api_key": "different"}), "prompt")
            prompt.assert_called_once()
            self.assertEqual(os.environ["ASSEMBLYAI_API_KEY"], "")
            self.assertFalse(self.settings.exists())
            with patch.object(client.sys.stdin, "isatty", return_value=False):
                self.assertEqual(client.read_assemblyai_api_key({}), "")

    def test_run_from_another_folder_reads_project_key_without_prompt(self):
        self.settings.write_text('{"api_key":"file-key"}', encoding="utf-8-sig")
        other = self.settings.parent / "other"
        other.mkdir()
        (other / "settings.json").write_text('{"api_key":"wrong-folder"}', encoding="utf-8")
        with contextlib.chdir(other), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}), \
             patch.object(client.getpass, "getpass") as prompt:
            self.assertEqual(client.read_api_key(), "file-key")
            prompt.assert_not_called()

    def test_missing_or_blank_settings_fall_back_to_environment(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
            self.assertEqual(client.read_api_key(), "env-key")
            self.settings.write_text('{"api_key":"  "}', encoding="utf-8")
            self.assertEqual(client.read_api_key(), "env-key")

    def test_bad_settings_report_the_problem_without_echoing_the_key(self):
        for contents in ('{"api_key":"private-fixture"', '{"api_key":["private-fixture"]}'):
            with self.subTest(contents=contents):
                self.settings.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError) as error:
                    client.read_api_key()
                self.assertIn("settings.json", str(error.exception))
                self.assertNotIn("private-fixture", str(error.exception))

    def test_prompts_describe_camera_and_text_in_both_modes(self):
        canvas = SimpleNamespace(
            avatar_frames=SimpleNamespace(catalog=load_pose_catalog(WHITE_DRAGON_CATALOG)),
            scene_layers=client.SceneLayers({}),
            controller=SimpleNamespace(render_state=lambda: (client.AvatarState(.5, .95, 1, 0), True, 1.0)),
        )
        for mode in client.MODES:
            prompt = client.build_control_prompt(canvas, client.parse_args([]), mode=mode)
            self.assertIn("文字输入", prompt)
            self.assertIn("用户用英文时用英文回复", prompt)
            self.assertNotIn("麦克风", prompt)
            if mode == "co-photo":
                self.assertIn(canvas.avatar_frames.catalog.describe(), prompt)
            else:
                self.assertNotIn("<tachie:", prompt)

        for options, description in (
            ([], client.FIXED_CAMERA_CONTEXT),
            (["--moving-camera"], client.MOVING_CAMERA_CONTEXT),
            (["--no-camera"], None),
            (["--no-camera", "--moving-camera"], None),
        ):
            with self.subTest(options=options):
                args = client.parse_args(options)
                co_photo = client.build_control_prompt(canvas, args, mode="co-photo")
                guidance = client.build_control_prompt(canvas, args, mode="guidance")
                for context in (client.FIXED_CAMERA_CONTEXT, client.MOVING_CAMERA_CONTEXT):
                    self.assertEqual(context in co_photo, context == description)
                    self.assertNotIn(context, guidance)
                self.assertEqual(client.LIVE_ENTRY_CONTEXT in co_photo, description is not None)
                for prompt in (guidance, co_photo):
                    self.assertEqual("画面来自真人摄像头" in prompt, description is not None)
                    self.assertEqual("当前没有摄像头输入" in prompt, description is None)

    def test_persona_file_reaches_both_modes_from_another_working_directory(self):
        from photo_assistant.persona import DEFAULT_PERSONA_FILE

        canvas = SimpleNamespace(
            avatar_frames=SimpleNamespace(catalog=load_pose_catalog(WHITE_DRAGON_CATALOG)),
            scene_layers=client.SceneLayers({}),
            controller=SimpleNamespace(render_state=lambda: (client.AvatarState(.5, .95, 1, 0), True, 1.0)),
        )
        with contextlib.chdir(self.settings.parent):
            custom = Path("另一份 角色.txt")
            custom.write_text("温和、好奇。\n喜欢用短句。\n", encoding="utf-8-sig")
            self.assertEqual(client.parse_args([]).character_persona,
                             DEFAULT_PERSONA_FILE.read_text(encoding="utf-8-sig").strip())
            args = client.parse_args(["--character-persona-file", str(custom)])
            custom.unlink()
            for mode in client.MODES:
                self.assertIn("温和、好奇。\n喜欢用短句。", client.build_control_prompt(canvas, args, mode=mode))
            self.assertEqual(client.parse_args(["--character-persona", "直率"]).character_persona, "直率")


class ConsoleInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_pipe_poll_does_not_block_and_preserves_split_utf8_and_eof(self):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "r", encoding="utf-8")
        source = client.ConsoleTextInput(stream)
        stop = threading.Event()
        try:
            source.poll()  # no data available: must return without waiting
            self.assertIsNone(source.peek())
            encoded = "向左移动\n我想站着".encode("utf-8")
            os.write(write_fd, encoded[:2])
            source.poll()
            self.assertIsNone(source.peek())
            os.write(write_fd, encoded[2:])
            source.poll()
            self.assertEqual(source.peek(), "向左移动")
            os.close(write_fd)
            write_fd = None
            with contextlib.redirect_stdout(io.StringIO()):
                await asyncio.wait_for(source.run(stop), .5)
            self.assertTrue(source.eof)
            self.assertFalse(stop.is_set())
            self.assertEqual(list(source.pending), ["向左移动", "我想站着"])
        finally:
            stream.close()
            if write_fd is not None:
                os.close(write_fd)

    async def test_cancelled_reader_does_not_create_blocking_input_threads(self):
        source = client.ConsoleTextInput()
        before = set(threading.enumerate())
        with patch.object(source, "poll"):
            task = asyncio.create_task(source.run(threading.Event()))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(set(threading.enumerate()), before)


class ApiInputSenderTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_failure_retains_text_until_next_session_accepts_it(self):
        source = client.ConsoleTextInput()
        source.put("和我合照")
        source.put("我想站着")
        canvas = MagicMock()
        canvas.compose_view.return_value = object(), None, None
        canvas.encode_jpeg.return_value = b"camera-jpeg"
        old = MagicMock()
        old.submit_frame.side_effect = ValueError("closed")
        with self.assertRaisesRegex(ValueError, "closed"):
            await client.send_realtime_inputs(old, canvas, None, threading.Event(), text_input=source)
        self.assertEqual(source.peek(), "和我合照")

        stop = threading.Event()
        new = MagicMock()
        new.submit_frame.side_effect = lambda *_: stop.set()
        log = Log()
        await client.send_realtime_inputs(new, canvas, None, stop, text_input=source, event_log=log)
        new.submit_frame.assert_called_once_with(b"camera-jpeg", "和我合照")
        self.assertEqual(log.rows, [{"event": "api.input", "jpeg_bytes": 11, "text": "和我合照"}])
        self.assertEqual(source.peek(), "我想站着")


class InterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_interruption_discards_partial_tag_and_all_late_old_output(self):
        def text(response, value):
            return dict(type="response.output.delta", kind="text", response_id=response, text=value)

        incoming = iter([
            text("old", "<teleport:x="),
            dict(type="response.interrupted", response_id="old", reason="new_input"),
            text("new", "<speak:现在站好><teleport:x="),
            text("old", "0.1,y=0.8><mode:co-photo><speak:过时指令>"),
            dict(type="response.done", response_id="old"),
            text("new", "0.7,y=0.9><rotate:to,value=20,ms=0>"),
            dict(type="response.done", response_id="new"),
            dict(type="session.closed"),
        ])

        async def receive():
            return next(incoming)

        canvas = MagicMock()
        canvas.apply_command.return_value = None
        selection = client.ModeSelection()
        log = Log()
        with contextlib.redirect_stdout(io.StringIO()):
            await client.receive_realtime_outputs(
                SimpleNamespace(recv=receive), threading.Event(), canvas, client.SpeechQueue(),
                verbose=False, event_log=log, mode_selection=selection,
            )
        applied = [call.args[0] for call in canvas.apply_command.call_args_list]
        self.assertEqual([command.kind for command in applied], ["teleport", "rotate"])
        self.assertEqual(applied[0].params["x"], .7)
        self.assertEqual(selection.mode, "guidance")
        self.assertEqual([r["text"] for r in log.rows if r["event"] == "speak"], ["现在站好"])
        self.assertEqual(sum(r["event"] == "parser.stale_output_ignored" for r in log.rows), 2)


if __name__ == "__main__":
    unittest.main()
