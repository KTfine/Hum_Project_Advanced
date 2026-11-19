"""Mode metadata must never gate the sprite frontend."""
import contextlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from photo_assistant.protocol import StreamingCommandParser
import photo_assistant.client as client
from photo_assistant.avatar.pose import AvatarPose, DEFAULT_SPRITE_CATALOG
from tests.avatar_fixture import rendered_avatar


class Log:
    def __init__(self):
        self.rows = []
    def record(self, event, **fields):
        self.rows.append(dict(event=event, **fields))
    def record_message(self, event, message):
        self.record(event, message=message)


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.rendered = np.full((10, 10, 4), (100, 200, 250, 128), dtype=np.uint8)
        self.enterContext(rendered_avatar(self.rendered))
        self.stop = threading.Event()
        self.canvas = client.CanvasSource(DEFAULT_SPRITE_CATALOG, 200, 100, 95,
                                          self.stop, preview=False)
        self.canvas.start()
        self.selection = client.ModeSelection()
        self.log = Log()

    def tearDown(self):
        self.canvas.close()
        self.directory.cleanup()

    def dispatch(self, text):
        commands = StreamingCommandParser(allow_plain_speech=True).feed('r', text)
        with contextlib.redirect_stdout(io.StringIO()):
            client.dispatch_model_commands(commands, self.canvas, client.SpeechQueue(), self.log,
                                           mode_selection=self.selection)

    def test_all_actions_execute_in_both_modes(self):
        for mode in client.MODES:
            self.selection.select(mode)
            self.selection.mark_loaded(*self.selection.snapshot())
            self.dispatch('<teleport:x=0.2,y=0.95,scale=0.5>'
                          '<rotate:to,value=10,ms=0><tachie:face=微笑,hand=无动作,arm=自然下垂,torso=正着站,legs=站着><speak:保持。>')
            self.assertEqual(self.canvas.controller.state_at(),
                             client.AvatarState(0.2, 0.95, 0.5, 10.0, pose=AvatarPose(face='微笑')))
        self.assertEqual(sum(r['event'] == 'command.applied' for r in self.log.rows), 6)
        self.assertFalse(any(r['event'] == 'command.blocked' for r in self.log.rows))

    def test_requested_mode_is_not_reported_as_loaded(self):
        self.assertIsNone(self.selection.co_photo_active)
        self.selection.mark_loaded(*self.selection.snapshot())
        self.dispatch('<mode:co-photo><teleport:x=0.4,y=0.95>')
        self.assertFalse(self.selection.co_photo_active)
        self.assertEqual(self.selection.mode, 'co-photo')
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.4)
        self.selection.mark_loaded(*self.selection.snapshot())
        self.assertTrue(self.selection.co_photo_active)
        self.dispatch('<mode:guidance>')
        self.assertTrue(self.selection.co_photo_active)
        self.selection.mark_loaded(*self.selection.snapshot())
        self.assertFalse(self.selection.co_photo_active)

    def test_duplicate_requests_are_idempotent_and_stale_confirmation_is_rejected(self):
        self.dispatch('<mode:co-photo>')
        first = self.selection.snapshot()
        self.dispatch('<mode:co-photo>')
        self.assertEqual(self.selection.snapshot(), first)
        self.dispatch('<mode:guidance>')
        self.assertFalse(self.selection.mark_loaded(*first))
        self.assertIsNone(self.selection.co_photo_active)

    def test_plain_words_legacy_tags_and_invalid_commands_do_not_switch(self):
        self.dispatch('和我合照。恢复指导模式。<photo:enter><mode:invalid><mode:co-photo,x=1>'
                      '<teleport:x=nan,y=1><teleport:x=1,y=1,scale=9><tachie:face=missing><rotate:to,value=999>')
        self.assertEqual(self.selection.snapshot(), ('guidance', 0))
        self.assertEqual(self.canvas.controller.state_at(),
                         client.AvatarState(0.5, 0.95, 1.0, 0.0))

    def test_split_mode_and_action_tags_execute_before_turn_end(self):
        parser = StreamingCommandParser()
        self.assertEqual(parser.feed('r', '<mo'), [])
        self.assertEqual(parser.feed('r', 'de:co-photo'), [])
        commands = parser.feed('r', '><teleport:x=0.4,y=0.95><speak:好。>')
        self.assertEqual([c.kind for c in commands], ['mode', 'teleport', 'speak'])
        with contextlib.redirect_stdout(io.StringIO()):
            client.dispatch_model_commands(commands, self.canvas, client.SpeechQueue(), self.log,
                                           mode_selection=self.selection)
        self.assertEqual(self.selection.mode, 'co-photo')
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.4)

    def test_co_photo_preview_and_jpeg_blend_avatar_into_the_current_camera_frame(self):
        camera = np.full((160, 320, 3), (10, 20, 30), dtype=np.uint8)
        source = type('Camera', (), {'latest_frame': lambda _: camera})()
        self.canvas.configure_view(source, show_character=True)
        # Rendered avatar is 10 x 10 pixels; make a transparent hole in its translucent body.
        self.rendered[4:6, 4:6, 3] = 0
        for value in [10, 40]:
            with self.subTest(camera_blue=value):
                camera[:, :, 0] = value
                original = camera.copy()
                expected = original.copy()
                blended = np.rint((np.array([100, 200, 250]) * 128
                                   + np.array([value, 20, 30]) * 127) / 255)
                # Bottom-center anchor (0.5, 0.95) on the 320 x 160 camera.
                expected[142:152, 155:165] = blended
                expected[146:148, 159:161] = original[146:148, 159:161]
                np.testing.assert_array_equal(self.canvas.render_preview_frame(), expected)
                ok, encoded = cv2.imencode('.jpg', expected, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.assertTrue(ok)
                self.assertEqual(self.canvas.latest_jpeg(camera), encoded.tobytes())
                np.testing.assert_array_equal(camera, original)

    def test_person_occludes_clipped_avatar_in_preview_model_image_and_shutter_frame(self):
        worker = Mock(spec=client.PersonOcclusion)
        self.canvas = client.CanvasSource(DEFAULT_SPRITE_CATALOG, 200, 100, 95,
                                         self.stop, preview=False, scene_layers={'person': worker})
        self.canvas.start()
        source = client.CameraSource(0, self.stop, person_occlusion=worker)
        self.canvas.configure_view(source, show_character=True)
        self.rendered[:] = (200, 200, 200, 255)
        with patch('photo_assistant.avatar.controller.time.monotonic', return_value=0):
            self.dispatch('<teleport:x=0,y=0.5>')
        preview_raw = np.full((100, 100, 3), 20, dtype=np.uint8)
        capture_raw = np.full_like(preview_raw, 60)
        source._publish_frame(capture_raw)
        queued = worker.submit.call_args.args[0]
        self.assertIs(source.latest_frame(), queued)
        self.assertFalse(queued.flags.writeable)
        worker.latest.return_value = None
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), capture_raw)

        # The worker is one frame behind. The preview must use its matched
        # image and mask, while the shutter segments the newly captured frame.
        mask = np.zeros((100, 100), np.float32)
        mask[:, :2] = 1
        mask[:, 2] = 0.5
        worker.latest.return_value = (preview_raw, mask)
        expected = preview_raw.copy()
        expected[40:50, 2] = 110
        expected[40:50, 3:5] = 200
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), expected)
        _, jpeg = cv2.imencode('.jpg', expected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertEqual(self.canvas.latest_jpeg(queued), jpeg.tobytes())

        def mask_at_shutter(frame):
            self.assertIs(frame, queued)
            self.dispatch('<teleport:x=0.8,y=0.5>')
            return 1.0 - mask

        worker.mask_for_capture.side_effect = mask_at_shutter
        path = client.PhotoCapture(source, Path(self.directory.name), self.canvas).save()
        expected_photo = capture_raw.copy()
        expected_photo[40:50, :2] = 200
        expected_photo[40:50, 2] = 130
        photo = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(photo, expected_photo)

        self.canvas.configure_view(source, show_character=False)
        worker.set_enabled.assert_called_with(False)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), capture_raw)

    def test_offscreen_teleports_crop_only_pixels_outside_the_frame(self):
        camera = np.zeros((100, 100, 3), dtype=np.uint8)
        avatar = self.rendered
        avatar[:, :, 3] = 255
        # Distinguish the head from the feet and the left side from the right.
        avatar[:, :, 0] = np.arange(10)[:, None] + 10
        avatar[:, :, 1] = np.arange(10) + 20
        cases = [
            (-0.02, 0.5, np.s_[40:50, :3], np.s_[:, 7:]),
            (1.02, 0.5, np.s_[40:50, 97:], np.s_[:, :3]),
            (0.5, 0.05, np.s_[:5, 45:55], np.s_[5:, :]),
            (0.5, 1.05, np.s_[95:, 45:55], np.s_[:5, :]),
        ]
        for x, y, visible_area, avatar_area in cases:
            with self.subTest(x=x, y=y):
                self.dispatch(f'<teleport:x={x},y={y}>')
                expected = camera.copy()
                expected[visible_area] = avatar[avatar_area][:, :, :3]
                np.testing.assert_array_equal(self.canvas.render_frame(camera_frame=camera), expected)

    def test_teleports_can_leave_and_return_from_outside_the_frame(self):
        self.dispatch('<teleport:x=-0.25,y=1.25>')
        state = self.canvas.controller.state_at()
        self.assertEqual((state.x, state.y), (-0.25, 1.25))
        self.dispatch('<teleport:x=-2,y=3>')
        state = self.canvas.controller.state_at()
        self.assertEqual((state.x, state.y), (-2, 3))
        camera = np.zeros((100, 100, 3), dtype=np.uint8)
        np.testing.assert_array_equal(self.canvas.render_frame(camera_frame=camera), camera)
        self.dispatch('<teleport:x=0.5,y=0.8>')
        self.assertTrue(self.canvas.render_frame(camera_frame=camera).any())

    def test_co_photo_without_camera_uses_configured_canvas_size_and_still_draws_avatar(self):
        self.canvas.configure_view(None, show_character=True)
        preview = self.canvas.render_preview_frame()
        self.assertEqual(preview.shape, (100, 200, 3))
        np.testing.assert_array_equal(preview[0, 0], [255, 255, 255])
        self.assertFalse(np.all(preview[90, 100] == 255))
        ok, encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertTrue(ok)
        self.assertEqual(self.canvas.latest_jpeg(None), encoded.tobytes())

    def test_teleport_preview_effect_and_clean_capture_share_the_destination(self):
        camera = np.zeros((100, 200, 3), dtype=np.uint8)
        source = Mock()
        source.latest_frame.return_value = camera
        self.canvas.configure_view(source, show_character=True)
        self.rendered[:] = (70, 140, 210, 255)
        original = self.canvas.render_preview_frame()
        with patch('photo_assistant.avatar.controller.time.monotonic', return_value=10):
            self.dispatch('<teleport:x=0.85,y=0.8,scale=2>')
        with patch('photo_assistant.avatar.controller.time.monotonic', return_value=10.175):
            preview = self.canvas.render_preview_frame()
            photo = self.canvas.render_capture_frame(camera)
            jpeg = self.canvas.latest_jpeg(camera)
        _, expected_jpeg = cv2.imencode('.jpg', photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertEqual(jpeg, expected_jpeg.tobytes())
        destination = np.any(photo != camera, axis=2)
        np.testing.assert_array_equal(np.any(preview != camera, axis=2), destination)
        self.assertFalse(preview[np.any(original != camera, axis=2)].any())
        self.assertFalse(np.array_equal(preview, photo))
        self.assertEqual(photo[75, 170].tolist(), [70, 140, 210])
        self.assertGreater(destination.sum(), np.any(original != camera, axis=2).sum())
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), photo)

    def test_guidance_preview_and_model_frame_contain_only_the_camera(self):
        camera = np.full((100, 200, 3), (10, 20, 30), dtype=np.uint8)
        source = type('Camera', (), {'latest_frame': lambda _: camera})()
        self.canvas.configure_view(source, show_character=False)
        preview = self.canvas.render_preview_frame()
        np.testing.assert_array_equal(preview, camera)
        self.assertIsNot(preview, camera)
        jpeg = self.canvas.latest_jpeg(camera, include_character=False)
        ok, expected = cv2.imencode('.jpg', camera, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self.assertTrue(ok)
        self.assertEqual(jpeg, expected.tobytes())
        # A valid model action remains executable while its character is hidden.
        self.dispatch('<teleport:x=0.2,y=0.95><tachie:face=微笑,hand=无动作,arm=自然下垂,torso=正着站,legs=站着>')
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), camera)
        self.assertEqual(self.canvas.controller.state_at().pose.face, '微笑')
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.2)
        self.canvas.configure_view(source, show_character=True)
        self.assertFalse(np.array_equal(self.canvas.render_preview_frame(), camera))
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.2)

    def test_guidance_without_camera_does_not_fall_back_to_a_character(self):
        self.canvas.configure_view(None, show_character=False)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(),
                                      np.full((100, 200, 3), 255, dtype=np.uint8))
        jpeg = self.canvas.latest_jpeg(None, include_character=False)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.all(decoded == 255))

    def test_removed_diagnostic_and_manual_mode_cli_are_rejected(self):
        for args in [['--action-only-test'], ['--mode', 'co-photo']]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    client.parse_args(args)


if __name__ == '__main__':
    unittest.main()
