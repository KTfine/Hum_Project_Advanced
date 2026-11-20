"""Exercise model-selected prompt reloads, with actual parsing and dispatch."""
import asyncio
import contextlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import photo_assistant.client as client
from photo_assistant.avatar.frames import AvatarFrames
from photo_assistant.avatar.pose import DEFAULT_SPRITE_CATALOG
from photo_assistant.vision.perspective import standing_height_px
from tests.avatar_fixture import rendered_avatar


class Log:
    def __init__(self):
        self.rows = []
    def record(self, event, **fields):
        self.rows.append(dict(event=event, **fields))
    def record_message(self, event, message):
        self.record(event, message=message)


class Session:
    def __init__(self, events):
        self.incoming = asyncio.Queue()
        for event in events:
            self.incoming.put_nowait(event)
        self.sent = []
        self.placements = []
        self.closed = False
        self.session_id = "local-test"
    async def recv(self):
        return await self.incoming.get()
    def submit_frame(self, jpeg, text, *, placement=None):
        self.sent.append(jpeg)
        self.placements.append(placement)
    async def close(self):
        self.closed = True


def text(value):
    return {'type': 'response.output.delta', 'kind': 'text', 'text': value, 'response_id': 'r'}


class PromptHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.rendered = np.full((10, 10, 4), 255, dtype=np.uint8)
        self.enterContext(rendered_avatar(self.rendered))
        self.stop = threading.Event()
        self.canvas = client.CanvasSource(DEFAULT_SPRITE_CATALOG, 200, 100, 80,
                                         self.stop, preview=False)
        self.camera = type('Camera', (), {'latest_frame': lambda _: np.zeros((100, 200, 3), dtype=np.uint8)})()
        self.canvas.start()
        self.args = client.parse_args(['--no-preview'])
        self.log = Log()
        self.prompts = []
        self.selection = client.ModeSelection()

    async def asyncTearDown(self):
        self.canvas.close()
        self.folder.cleanup()

    async def run_sessions(self, opener):
        with patch.object(client.direct_api, 'open_session', side_effect=opener), contextlib.redirect_stdout(io.StringIO()):
            await asyncio.wait_for(client.run_realtime_sessions(
                'test-key', self.canvas, self.camera, client.SpeechQueue(), self.stop,
                self.args, event_log=self.log, mode_selection=self.selection,
            ), timeout=2)

    async def test_both_mode_transitions_preserve_geometry_and_confirm_after_open(self):
        sockets = [
            Session([text('<teleport:x=0.4,y=0.8><mode:co-photo>')]),
            Session([text('<mode:co-photo><teleport:x=0.3,y=0.8><mode:guidance>')]),
            Session([text('<teleport:x=0.3,y=0.8,scale=0.8><speak:继续。>'), {'type':'session.closed'}]),
        ]
        async def opener(*args, **kwargs):
            index = len(self.prompts)
            self.prompts.append(args[0])
            if index:
                self.assertTrue(sockets[index-1].closed)
                self.assertEqual(self.selection.co_photo_active, index == 2)
                self.assertFalse(self.stop.is_set())
            return sockets[index]
        await self.run_sessions(opener)
        self.assertEqual(len(self.prompts), 3)
        for prompt, task in zip(self.prompts, ['真人拍照指导', '合照互动', '真人拍照指导']):
            self.assertIn(f'当前任务：{task}', prompt)
        self.assertIn('"x": 0.4', self.prompts[1])
        self.assertNotIn('"x":', self.prompts[0])
        self.assertNotIn('"x":', self.prompts[2])
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.3)
        self.assertAlmostEqual(self.canvas.controller.state_at().y, 0.8)
        self.assertEqual(self.canvas.controller.state_at().scale, 0.8)
        self.assertEqual([r['co_photo_active'] for r in self.log.rows if r['event']=='session.mode_ready'],
                         [False, True, False])
        self.assertEqual([r['text'] for r in self.log.rows if r['event']=='speak'], ['继续。'])
        self.assertTrue(all(s.closed for s in sockets))
        self.assertEqual([r['include_character'] for r in self.log.rows if r['event']=='session.mode_ready'],
                         [False, True, False])
        for index, socket in enumerate(sockets):
            self.assertTrue(socket.sent)
            data = socket.sent[0]
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if index == 1:
                self.assertGreater(float(frame.mean()), 1)
                self.assertEqual(socket.placements[0], {"x": 0.4, "y": 0.8, "scale": 1.0})
            else:
                self.assertTrue(np.all(frame == 0))
                self.assertIsNone(socket.placements[0])
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.camera.latest_frame())

    async def test_shutter_saves_visible_composition_across_mode_switches(self):
        # Distinct upper and lower colors expose which half made it into the photo.
        self.rendered[:5, :, :3] = (30, 80, 200)
        sessions = [
            Session([text('<capture><mode:co-photo>')]),
            Session([text('<teleport:x=0.4,y=1.05><capture><mode:guidance>')]),
            Session([text('<capture>'), {'type': 'session.closed'}]),
        ]
        with patch.object(client, 'PROJECT_ROOT', Path(self.folder.name)):
            await self.run_sessions(AsyncMock(side_effect=sessions))

        paths = [Path(row['path']) for row in self.log.rows if row['event'] == 'photo.saved']
        self.assertEqual(len(paths), 3)
        self.assertEqual(set(paths), set((Path(self.folder.name) / 'photos').glob('*.png')))
        raw = self.camera.latest_frame()
        half_portrait = raw.copy()
        half_portrait[95:100, 75:85] = (30, 80, 200)
        for path, expected in zip(paths, (raw, half_portrait, raw)):
            frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(frame, expected)

    async def test_mode_tag_alone_places_real_sprite_beside_full_body_before_first_capture(self):
        frame = self.camera.latest_frame()
        mask = np.zeros(frame.shape[:2], np.float32)
        cv2.circle(mask, (47, 17), 7, 1, -1)
        cv2.rectangle(mask, (37, 23), (57, 93), 1, -1)
        canvas = client.CanvasSource(DEFAULT_SPRITE_CATALOG, 200, 100, 80, self.stop,
                                     preview=False, scene_layers={"person": mask})
        with patch.object(client, "AvatarFrames", AvatarFrames):
            canvas.start()
        self.addCleanup(canvas.close)
        sessions = [Session([text('<mode:co-photo>')]),
                    Session([text('<capture>'), {'type': 'session.closed'}])]
        with (patch.object(self, 'canvas', canvas), patch.object(client, 'standing_height_px', standing_height_px),
              patch.object(client, 'PROJECT_ROOT', Path(self.folder.name))):
            await self.run_sessions(AsyncMock(side_effect=sessions))
        state = canvas.controller.state_at()
        placement = sessions[1].placements[0]
        self.assertEqual((placement["x"], placement["y"], placement["scale"]), (state.x, state.y, state.scale))
        self.assertEqual((placement["person_framing"], placement["placement_source"]), ("全身", "并排站位"))
        self.assertEqual(state.pose, canvas.avatar_frames.catalog.standing_pose)
        self.assertEqual(placement["current_tachie"], state.pose.command())
        person_y, person_x = np.nonzero(mask)
        foot_x, foot_y = state.x * frame.shape[1], state.y * frame.shape[0]
        self.assertFalse(person_x.min() <= foot_x <= person_x.max())
        self.assertLess(abs(foot_y - person_y.max()), 4)
        path, = (Path(self.folder.name) / 'photos').glob('*.png')
        photo = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        visible_y, visible_x = np.nonzero(np.any(photo != frame, axis=2))
        self.assertGreater(len(visible_x), 0)
        self.assertLess(abs(visible_y.max() - foot_y), 4)
        self.assertTrue(visible_x.min() <= foot_x <= visible_x.max())
        np.testing.assert_array_equal(photo[mask == 1], frame[mask == 1])

    async def test_same_mode_and_plain_invitation_do_not_reconnect(self):
        socket = Session([text('和我合照。<mode:guidance><speak:你好。>'), {'type':'session.closed'}])
        async def opener(*args, **kwargs):
            self.prompts.append(args[0])
            return socket
        await self.run_sessions(opener)
        self.assertEqual(len(self.prompts), 1)
        self.assertFalse(self.selection.co_photo_active)

    async def test_failed_reload_keeps_last_confirmed_mode_and_closes_old_socket(self):
        socket = Session([text('<mode:co-photo>')])
        async def opener(*args, **kwargs):
            self.prompts.append(args[0])
            if len(self.prompts) == 2:
                raise RuntimeError('reload failed')
            return socket
        with self.assertRaisesRegex(RuntimeError, 'reload failed'):
            await self.run_sessions(opener)
        self.assertTrue(socket.closed)
        self.assertFalse(self.selection.co_photo_active)
        self.assertEqual(self.selection.mode, 'co-photo')
        self.assertEqual(sum(r['event']=='session.mode_ready' for r in self.log.rows), 1)
        np.testing.assert_array_equal(self.canvas.render_preview_frame(), self.camera.latest_frame())

    async def test_custom_override_has_no_builtin_mode_claim_or_reload(self):
        self.args.prompt = 'custom instructions'
        socket = Session([text('<mode:co-photo><teleport:x=0.2,y=0.95>'), {'type':'session.closed'}])
        async def opener(*args, **kwargs):
            self.prompts.append(args[0])
            return socket
        await self.run_sessions(opener)
        self.assertEqual(self.prompts, ['custom instructions'])
        self.assertIsNone(self.selection.co_photo_active)
        self.assertAlmostEqual(self.canvas.controller.state_at().x, 0.2)
        self.assertEqual([r['mode'] for r in self.log.rows if r['event']=='session.mode_ready'], ['custom'])

    async def test_user_stop_closes_session_without_mode_reload(self):
        socket = Session([])
        async def opener(*args, **kwargs):
            self.prompts.append(args[0])
            asyncio.get_running_loop().call_later(0.01, self.stop.set)
            return socket
        await self.run_sessions(opener)
        self.assertEqual(len(self.prompts), 1)
        self.assertTrue(socket.closed)

    async def test_device_cleanup_on_session_failure(self):
        camera = MagicMock()
        with patch.object(client, 'CanvasSource', return_value=self.canvas), \
             patch.object(client, 'CameraSource', return_value=camera), \
             patch.object(client, 'read_api_key', return_value='test-key'), \
             patch.object(client, 'run_realtime_sessions', new=AsyncMock(side_effect=RuntimeError('failed'))), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                await client.run_probe(self.args, self.log)
        camera.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
