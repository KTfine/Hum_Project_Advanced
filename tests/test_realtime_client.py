import asyncio
import contextlib
import io
import os
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import photo_assistant.client as client
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG, AvatarPose, load_pose_catalog
from photo_assistant.protocol import Command, StreamingCommandParser
from tests.avatar_fixture import rendered_avatar


class CameraSelectionTests(unittest.TestCase):
    def test_camera_is_enabled_by_default_and_can_be_disabled(self):
        with patch.dict(os.environ):
            os.environ.pop("HUM_CAMERA_DEVICE", None)
            defaults = client.parse_args([])
            disabled = client.parse_args(["--no-camera"])

        self.assertFalse(defaults.no_camera)
        self.assertFalse(defaults.moving_camera)
        self.assertTrue(client.parse_args(["--moving-camera"]).moving_camera)
        self.assertIsNone(defaults.camera_device)  # permit camera_name lookup at startup
        self.assertEqual(client.DEFAULT_CAMERA_DEVICE, 701)
        self.assertTrue(disabled.no_camera)

    def test_camera_argument_overrides_configured_default(self):
        for index in (0, 2):
            with self.subTest(index=index), patch.dict(os.environ, {"HUM_CAMERA_DEVICE": str(index)}):
                self.assertEqual(client.parse_args([]).camera_device, index)
                self.assertEqual(client.parse_args(["--camera-device", "1"]).camera_device, 1)


class OutputEventTests(unittest.TestCase):
    def test_streaming_text_delta_is_preserved(self):
        decoded = client.decode_output_event(
            {"type": "response.output.delta", "kind": "text", "text": "有人挥手"}
        )

        self.assertEqual(decoded, ("text", "有人挥手"))

    def test_api_error_becomes_an_exception_with_code_and_message(self):
        with self.assertRaisesRegex(client.RealtimeServiceError, "api_error.*Provider failed"):
            client.decode_output_event(
                {
                    "type": "error",
                    "error": {"code": "api_error", "message": "Provider failed"},
                }
            )


class AvatarControllerTests(unittest.TestCase):
    def test_teleport_commits_both_axes_and_size_before_first_effect_frame(self):
        controller = client.AvatarController(client.AvatarState(0.2, 0.8, 1, -12))
        command, = StreamingCommandParser().feed("r", "<teleport:x=0.9,y=0.5,scale=1.6>")
        target = controller.apply_command(command, now=10)
        self.assertEqual(target, client.AvatarState(0.9, 0.5, 1.6, -12))
        state, follows_ground, progress = controller.render_state(10)
        self.assertEqual(state, target)
        self.assertFalse(follows_ground)
        self.assertEqual(progress, 0)
        controller.observe_ground(0.95, 10.1)
        self.assertEqual(controller.state_at(10.15), target)
        self.assertAlmostEqual(controller.render_state(10.175)[2], 0.5)
        self.assertEqual(controller.render_state(11)[2], 1)

    def test_second_teleport_restarts_effect_and_preserves_current_size_and_pose(self):
        pose = AvatarPose(face="高冷表情")
        controller = client.AvatarController(client.AvatarState(0.3, 0.8, 1, -12, layer_index=1, pose=pose))
        controller.apply_command(Command("teleport", dict(x=0, y=0, scale=1.4)), now=1)
        target = controller.apply_command(Command("teleport", dict(x=1, y=1)), now=1.1)
        self.assertEqual(target, client.AvatarState(1, 1, 1.4, -12, layer_index=1, pose=pose))
        self.assertEqual(controller.render_state(1.1), (target, False, 0))

    def test_teleport_during_rotation_preserves_rotation_timing(self):
        controller = client.AvatarController(client.AvatarState(0.3, 0.8, 1, 0))
        controller.apply_command(Command("rotate", dict(mode="to", value=90, ms=1000)), now=1)
        controller.apply_command(Command("teleport", dict(x=0.9, y=0.3)), now=1.5)
        self.assertEqual(controller.state_at(1.5), client.AvatarState(0.9, 0.3, 1, 45))
        controller.apply_command(Command("rotate", dict(mode="by", value=30, ms=500)), now=1.5)
        self.assertEqual(controller.state_at(2).rotation, 120)

    def test_expression_preserves_all_geometry(self):
        controller = client.AvatarController(
            client.AvatarState(0.3, 0.8, 1.4, -12.0),
            pose_catalog=load_pose_catalog(WHITE_DRAGON_CATALOG),
        )
        pose = AvatarPose(face="高冷表情")
        controller.apply_command(Command("tachie", vars(pose)), now=1.0)
        self.assertEqual(controller.state_at(1.0), client.AvatarState(0.3, 0.8, 1.4, -12.0, pose=pose))

    def test_blank_command_does_not_change_state(self):
        initial = client.AvatarState(0.3, 0.8, 1.4, -12.0)
        controller = client.AvatarController(initial)

        result = controller.apply_command(Command("blank", {}), now=1.0)

        self.assertIsNone(result)
        self.assertEqual(controller.state_at(1.0), initial)


class SpeechQueueTests(unittest.TestCase):
    def test_speech_is_returned_in_fifo_order(self):
        queue = client.SpeechQueue()
        queue.enqueue("第一句")
        queue.enqueue("第二句")

        self.assertEqual(queue.pop_next(), "第一句")
        self.assertEqual(queue.pop_next(), "第二句")
        self.assertIsNone(queue.pop_next())


class CommandDispatchTests(unittest.TestCase):
    def test_complete_streamed_command_is_dispatched_before_session_close(self):

        class RecordingCanvas:
            def __init__(self):
                self.commands = []

            def apply_command(self, command):
                self.commands.append(command)
                return client.AvatarState(0.2, 0.95, 1.0, 0.0)

        canvas = RecordingCanvas()
        stop_event = threading.Event()
        session = FakeSession(
            [
                {
                    "type": "response.output.delta",
                    "kind": "text",
                    "text": "<teleport:x=0.2,y=0.95>",
                    "response_id": "r1",
                },
                {"type": "session.closed", "reason": "test"},
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(
                client.receive_realtime_outputs(
                    session,
                    stop_event,
                    canvas,
                    client.SpeechQueue(),
                    verbose=False,
                )
            )

        self.assertEqual(
            canvas.commands,
            [Command("teleport", {"x": 0.2, "y": 0.95})],
        )

    def test_malformed_stream_is_silent_and_later_command_still_dispatches(self):

        class RecordingCanvas:
            def __init__(self):
                self.commands = []

            def apply_command(self, command):
                self.commands.append(command)
                return None

        canvas = RecordingCanvas()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            client.dispatch_model_commands(
                client.StreamingCommandParser().feed(
                    "r1", "<teleport:x=>garbage<tachie:face=微笑,hand=无动作,arm=自然下垂,torso=正着站,legs=站着>"
                ),
                canvas,
                client.SpeechQueue(),
                verbose=False,
            )

        self.assertEqual([command.kind for command in canvas.commands], ["tachie"])
        self.assertEqual(stderr.getvalue(), "")

class CanvasCompositionTests(unittest.TestCase):
    def test_rgba_avatar_is_alpha_composited_and_clipped_on_white_canvas(self):
        composite = client.alpha_composite_at
        canvas = np.full((3, 3, 3), 255, dtype=np.uint8)
        avatar = np.array(
            [
                [[0, 0, 0, 255], [0, 0, 0, 128]],
                [[0, 0, 0, 0], [0, 0, 0, 255]],
            ],
            dtype=np.uint8,
        )

        composite(canvas, avatar, left=2, top=1)

        self.assertEqual(canvas[1, 2].tolist(), [0, 0, 0])
        self.assertEqual(canvas[2, 2].tolist(), [255, 255, 255])


class FakeSession:
    def __init__(self, incoming):
        self.incoming = iter(incoming)

    async def recv(self):
        return next(self.incoming)


class RealtimeInputSenderTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_composition_is_encoded_only_when_requested(self):
        import cv2
        stop_event = threading.Event()
        session = Mock()
        camera = Mock()
        first = np.full((180, 320, 3), 190, dtype=np.uint8)
        last = np.zeros_like(first)
        camera.latest_frame.side_effect = [first, last, None]
        inputs = Mock()
        inputs.peek.side_effect = [None, None, "看一下"]
        inputs.acknowledge.side_effect = lambda _: stop_event.set()
        pixels = np.full((50, 100, 4), (20, 80, 180, 255), dtype=np.uint8)
        self.enterContext(rendered_avatar(pixels))
        canvas = client.CanvasSource(WHITE_DRAGON_CATALOG, 320, 180, 90, stop_event, preview=False)
        self.addCleanup(canvas.close)
        canvas.start()
        with (patch.object(cv2, "imencode", wraps=cv2.imencode) as encode,
              patch.object(client.asyncio, "sleep", new_callable=AsyncMock)):
            await client.send_realtime_inputs(session, canvas, camera, stop_event, text_input=inputs)
        encode.assert_called_once()
        idle1, idle2, submitted = session.submit_frame.call_args_list
        self.assertEqual(idle1.args, (None, None))
        self.assertEqual(idle2.args, (None, None))
        jpeg, text = submitted.args
        self.assertEqual(text, "看一下")
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(frame.shape, (180, 320, 3))
        np.testing.assert_allclose(frame[145, 160], (20, 80, 180), atol=4)
        self.assertLess(frame[:90].max(), 10)

    async def test_missing_co_photo_camera_discards_pending_image_before_question(self):
        stop_event = threading.Event()
        session, canvas, camera, inputs = Mock(), Mock(), Mock(), Mock()
        camera.latest_frame.side_effect = [object(), None, None]
        canvas.compose_view.return_value = object(), {"avatar_visible": True}, object()
        inputs.peek.side_effect = [None, None, "还看得到吗"]
        inputs.acknowledge.side_effect = lambda _: stop_event.set()
        with patch.object(client.asyncio, "sleep", new_callable=AsyncMock):
            await client.send_realtime_inputs(session, canvas, camera, stop_event, text_input=inputs,
                                              report_placement=True, readiness=client.CoPhotoReadiness())
        canvas.encode_jpeg.assert_not_called()
        self.assertEqual(session.submit_frame.call_args.args, (None, "还看得到吗"))
        self.assertIs(session.submit_frame.call_args.kwargs["placement"]["camera_frame_available"], False)


if __name__ == "__main__":
    unittest.main()
