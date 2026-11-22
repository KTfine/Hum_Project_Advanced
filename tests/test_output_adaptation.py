"""Regression coverage for mixed native speech and streamed tag commands."""
import asyncio
import contextlib
from dataclasses import replace
import io
import threading
import unittest
from unittest.mock import Mock

from photo_assistant.protocol import StreamingCommandParser
import photo_assistant.client as client


class OutputParserTests(unittest.TestCase):
    def parser(self, **kwargs):
        return StreamingCommandParser(allow_plain_speech=True, **kwargs)

    def test_strict_parser_still_ignores_untagged_text(self):
        self.assertEqual(StreamingCommandParser().feed('r', '请向你自己的左边挪。'), [])

    def test_native_speech_is_available_on_each_fragment(self):
        parser = self.parser()
        first = parser.feed('r', '请向你自己的')
        self.assertEqual(first[0].params['text'], '请向你自己的')
        second = parser.feed('r', '左边挪一点。')
        self.assertEqual(second[0].params['text'], '左边挪一点。')
        self.assertEqual(parser.plain_speech_count, 2)

    def test_mixed_text_and_split_tags_keep_order_without_speaking_parameters(self):
        parser = self.parser()
        commands = parser.feed('r', '先保持。<tele')
        self.assertEqual([c.params['text'] for c in commands], ['先保持。'])
        commands += parser.feed('r', 'port:x=0.2,y=0.9><speak:好的。>再调整。')
        self.assertEqual([c.kind for c in commands], ['speak', 'teleport', 'speak', 'speak'])
        self.assertEqual(commands[1].params['x'], 0.2)
        self.assertEqual([c.params['text'] for c in commands if c.kind == 'speak'],
                         ['先保持。', '好的。', '再调整。'])

    def test_unknown_and_invalid_commands_are_not_spoken_or_executed(self):
        commands = self.parser().feed('r', '<unknown:secret><teleport:x=nan,y=1>继续。')
        self.assertEqual([(c.kind, c.params) for c in commands], [('speak', {'text': '继续。'})])

    def test_unknown_tag_can_span_fragments_without_leaking_body(self):
        parser = self.parser()
        self.assertEqual(parser.feed('r', '<unk'), [])
        self.assertEqual(parser.feed('r', 'nown:secret'), [])
        commands = parser.feed('r', '>完成。')
        self.assertEqual([c.params['text'] for c in commands], ['完成。'])

    def test_broken_unknown_prefix_does_not_swallow_later_valid_tag(self):
        commands = self.parser().feed('r', '提示。<unknown:secret<speak:继续。>')
        self.assertEqual([c.params['text'] for c in commands], ['提示。', '继续。'])

    def test_oversized_tag_discards_its_tail_until_recovery(self):
        parser = self.parser(max_buffer=32)
        self.assertEqual(parser.feed('r', '<teleport:' + 'x' * 40), [])
        self.assertEqual(parser.feed('r', 'x=0.9,y=0.3'), [])
        commands = parser.feed('r', '><speak:继续。>')
        self.assertEqual([(c.kind, c.params) for c in commands], [('speak', {'text': '继续。'})])

    def test_new_response_and_flush_discard_unfinished_tags(self):
        parser = self.parser()
        self.assertEqual(parser.feed('r1', '<teleport:x=0.'), [])
        self.assertEqual(parser.feed('r2', '你好。')[0].params['text'], '你好。')
        self.assertEqual(parser.feed('r2', '<unknown:secret'), [])
        parser.flush('r2')
        self.assertEqual(parser.feed('r2', '继续。')[0].params['text'], '继续。')

    def test_mixed_sprite_and_geometry_commands_can_share_one_line(self):
        text = ('<speak:好。><teleport:x=0.35,y=0.95,scale=0.8>'
                '<tachie:face=微笑,hand=比耶,arm=自然下垂,torso=正着站,legs=站着>'
                '<rotate:to,value=10,ms=300><rotate:by,value=-5,ms=300><blank>')
        parser = self.parser()
        commands = parser.feed('r', text)
        self.assertEqual([c.kind for c in commands],
                         ['speak', 'teleport', 'tachie', 'rotate', 'rotate', 'blank'])
        self.assertEqual(parser.plain_speech_count, 0)

    def test_explicit_speech_is_not_duplicated_as_plain_text(self):
        parser = self.parser()
        commands = parser.feed('r', ' \n<speak:保持。>\n')
        self.assertEqual([c.params['text'] for c in commands], ['保持。'])
        self.assertEqual(parser.plain_speech_count, 0)


class EventLog:
    def __init__(self):
        self.rows = []
    def record(self, event, **fields):
        self.rows.append({'event': event, **fields})
    def record_message(self, event, message):
        self.record(event, message=message)


class RecordingCanvas:
    def __init__(self):
        self.commands = []
    def apply_command(self, command):
        self.commands.append(command)
        return client.AvatarState(0.2, 0.95, 1.0, 0.0)


class Socket:
    def __init__(self, events, before_second=None):
        self.events = iter(events)
        self.count = 0
        self.before_second = before_second
    async def recv(self):
        if self.count == 1 and self.before_second:
            self.before_second()
        self.count += 1
        return next(self.events)


class ReceiverAdaptationTests(unittest.TestCase):
    def test_missing_wall_contact_skips_later_shutter_chunk_only_in_same_response(self):
        log = EventLog()
        canvas = Mock(spec=RecordingCanvas)
        canvas.apply_command.return_value = None
        shutter = Mock(spec=client.PhotoCapture)
        shutter.save.return_value = "next-photo.png"
        delta = {'type': 'response.output.delta', 'kind': 'text'}
        socket = Socket([
            {**delta, 'response_id': 'wall', 'text': '<brace>'},
            {**delta, 'response_id': 'wall', 'text': '<capture>'},
            {'type': 'response.done', 'response_id': 'wall'},
            {**delta, 'response_id': 'next', 'text': '<capture>'},
            {'type': 'session.closed', 'reason': 'test'},
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(client.receive_realtime_outputs(socket, threading.Event(), canvas,
                        client.SpeechQueue(), verbose=False, event_log=log, photo_capture=shutter))

        canvas.apply_command.assert_called_once_with(client.Command('brace', {}))
        shutter.save.assert_called_once_with()
        next_turn = next(i for i, row in enumerate(log.rows) if row['event'] == 'api.receive'
                         and row['message'].get('response_id') == 'next')
        first = log.rows[:next_turn]
        self.assertEqual([(r['command']['kind'], r['reason']) for r in first if r['event'] == 'command.skipped'],
                         [('brace', 'no_scene_placement'), ('capture', 'wall_contact_missing')])
        self.assertFalse(any(r['event'] == 'photo.saved' for r in first))
        self.assertEqual([r['path'] for r in log.rows[next_turn:] if r['event'] == 'photo.saved'], ['next-photo.png'])

    def test_spoken_reply_executes_expression_and_shutter_in_tag_order(self):
        log = EventLog()
        canvas = RecordingCanvas()
        output = Mock()
        shutter = Mock(spec=client.PhotoCapture)
        shutter.save.return_value = 'portrait.png'
        socket = Socket([
            {'type': 'response.output.delta', 'kind': 'text', 'response_id': 'r',
             'text': '<peek><tachie:face=龇牙笑,hand=无动作,arm=自然下垂,torso=半身歪斜向右,legs=站着>'
                     '<speak:看见了！><capture>'},
            {'type': 'session.closed', 'reason': 'test'},
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(client.receive_realtime_outputs(socket, threading.Event(), canvas,
                        client.SpeechQueue(output), verbose=False, event_log=log, photo_capture=shutter))
        self.assertEqual([command.kind for command in canvas.commands], ['peek', 'tachie'])
        self.assertEqual(canvas.commands[1].params['face'], '龇牙笑')
        output.say.assert_called_once_with('看见了！')
        shutter.save.assert_called_once_with()
        self.assertEqual([r['event'] for r in log.rows if r['event'] in {'command.applied', 'speak', 'photo.saved'}],
                         ['command.applied', 'command.applied', 'speak', 'photo.saved'])
        self.assertEqual([r['text'] for r in log.rows if r['event'] == 'speak'], ['看见了！'])

    def test_native_text_dispatches_before_next_fragment_or_turn_end(self):
        log = EventLog()
        canvas = RecordingCanvas()

        def check_incremental_dispatch():
            self.assertEqual([r['text'] for r in log.rows if r['event'] == 'speak'], ['请向你自己的'])

        socket = Socket([
            {'type': 'response.output.delta', 'kind': 'text', 'text': '请向你自己的', 'response_id': 'r'},
            {'type': 'response.output.delta', 'kind': 'text', 'text': '左边挪。<speak:保持。>', 'response_id': 'r'},
            {'type': 'session.closed', 'reason': 'test'},
        ], before_second=check_incremental_dispatch)
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(client.receive_realtime_outputs(socket, threading.Event(), canvas,
                        client.SpeechQueue(), verbose=False, event_log=log))
        self.assertEqual([r['text'] for r in log.rows if r['event'] == 'speak'],
                         ['请向你自己的', '左边挪。', '保持。'])
        self.assertEqual(canvas.commands, [])

    def test_actions_and_speech_dispatch_without_a_permission_filter(self):
        log = EventLog()
        canvas = RecordingCanvas()
        socket = Socket([
            {'type': 'response.output.delta', 'kind': 'text',
             'text': '<teleport:x=0.2,y=0.95><tachie:face=微笑,hand=无动作,arm=自然下垂,torso=正着站,legs=站着>'
                     '<speak:保持。>', 'response_id': 'r'},
            {'type': 'session.closed', 'reason': 'test'},
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(client.receive_realtime_outputs(socket, threading.Event(), canvas,
                        client.SpeechQueue(), verbose=False, event_log=log))
        self.assertEqual([c.kind for c in canvas.commands], ['teleport', 'tachie'])
        self.assertEqual(len([r for r in log.rows if r['event'] == 'command.blocked']), 0)
        self.assertEqual([r['text'] for r in log.rows if r['event'] == 'speak'], ['保持。'])


class CameraObservationReceiverTests(unittest.TestCase):
    def setUp(self):
        self.canvas = client.CanvasSource(client.WHITE_DRAGON_CATALOG, 160, 120, 85,
                                          threading.Event(), preview=False)
        self.canvas.start()
        self.addCleanup(self.canvas.close)
        self.pose = self.canvas.controller.state_at().pose
        self.grin = replace(self.pose, face='龇牙笑')
        self.log = EventLog()
        self.voice = Mock()
        self.shutter = Mock()
        self.shutter.save.return_value = 'shot.png'
        self.readiness = client.CoPhotoReadiness()
        self.modes = client.ModeSelection('co-photo')
        self.modes.mark_loaded(*self.modes.snapshot())

    def delta(self, text, rid='camera', source='observation'):
        return {'type': 'response.output.delta', 'response_id': rid, 'turn_source': source,
                'kind': 'text', 'text': text}

    def receive(self, *events):
        socket = Socket([*events, {'type': 'session.closed'}])
        socket.emits_response_done = True
        return asyncio.run(client.receive_realtime_outputs(
            socket, threading.Event(), self.canvas, client.SpeechQueue(self.voice), verbose=False,
            event_log=self.log, mode_selection=self.modes, terminal=Mock(), readiness=self.readiness,
            photo_capture=self.shutter))

    def test_observation_can_speak_and_change_face_without_changing_body_or_position(self):
        original = self.canvas.controller.state_at()
        departure = self.canvas._departure = object()
        body_change = replace(self.grin, torso='正着站').command()
        self.receive(self.delta('<speak:光线柔和了。><cap'), self.delta('ture><mode:guidance><follow>'),
                     self.delta(self.grin.command() + body_change + '<blank>'),
                     {'type': 'response.done', 'response_id': 'camera', 'turn_source': 'observation'})
        self.voice.say.assert_called_once_with('光线柔和了。')
        self.shutter.save.assert_not_called()
        self.assertEqual(self.modes.mode, 'co-photo')
        self.assertEqual(self.canvas.controller.state_at(), replace(original, pose=self.grin))
        self.assertIs(self.canvas._departure, departure)
        self.assertIsNone(self.canvas._pose_override)
        self.assertIsNone(self.readiness.last_action_succeeded)
        self.assertEqual([r['command']['kind'] for r in self.log.rows if r['event'] == 'command.skipped'],
                         ['capture', 'mode', 'follow', 'tachie'])

    def test_guidance_and_pending_user_placement_keep_current_expression(self):
        self.modes.select('guidance')
        self.modes.mark_loaded(*self.modes.snapshot())
        self.receive(self.delta(self.grin.command()))
        self.assertEqual(self.canvas.controller.state_at().pose, self.pose)

        self.modes.select('co-photo')
        self.modes.mark_loaded(*self.modes.snapshot())
        settled = dict(x=.5, y=.9, scale=1., current_tachie=self.pose.command(),
                       avatar_visible=True, pose_settled=True)
        self.assertEqual(self.readiness.observe(settled), 'entry')
        self.readiness.action_result('teleport', applied=True)
        self.readiness.finish_turn()
        self.receive(self.delta(self.grin.command() + '<speak:现在光线很好。>'),
                     {'type': 'response.done', 'response_id': 'camera'})
        self.assertEqual(self.canvas.controller.state_at().pose, self.pose)
        self.assertTrue(self.readiness.pending_action)
        self.assertTrue(self.readiness.last_action_succeeded)
        self.assertEqual(self.readiness.observe({**settled, 'x': .3}), 'action')

    def test_face_variants_keep_the_same_seat_or_wall_contact(self):
        entries = self.canvas.avatar_frames.catalog.entries
        placement = self.canvas._fixed = client.Placement(.3, .8, 1., .6)
        for field in ('seat_contact_px', 'wall_contact_px'):
            with self.subTest(field=field):
                self.canvas.controller.apply_command(client.Command('tachie', vars(self.pose)))
                self.canvas._pose_override = self.pose
                seat = self.canvas._seat = object() if field == 'seat_contact_px' else None
                self.canvas._brace = None if seat else client.WallBrace(placement, self.pose, (.3, .4), (.3, .1, .3, .8))
                entries[self.pose] = replace(entries[self.pose], **{field: (50, 80)})
                entries[self.grin] = replace(entries[self.grin], **{field: (50, 81)})
                command = client.Command('tachie', vars(self.grin))
                self.assertIsNone(self.canvas.apply_expression(command))
                self.assertEqual(self.canvas.controller.state_at().pose, self.pose)
                entries[self.grin] = replace(entries[self.grin], **{field: (50, 80)})
                self.assertEqual(self.canvas.apply_expression(command).pose, self.grin)
                self.assertIs(self.canvas._fixed, placement)
                self.assertIs(self.canvas._seat, seat)
                self.assertEqual(self.canvas._pose_override, self.grin)
                if not seat:
                    self.assertEqual(self.canvas._brace.contact_xy, (.3, .4))
                    self.assertEqual(self.canvas._brace.pose, self.grin)

    def test_interrupted_and_failed_observations_discard_stale_speech_and_accept_user_shutter(self):
        self.receive(self.delta('<speak:有人路过，未说完'),
                     {'type': 'response.interrupted', 'response_id': 'camera', 'turn_source': 'observation'},
                     self.delta('的旧话。><capture>'),
                     self.delta('<speak:半句话', rid='failure'),
                     {'type': 'error', 'response_id': 'failure', 'turn_source': 'observation',
                      'error': {'code': 'provider_error'}},
                     self.delta('过期。>', rid='failure'),
                     self.delta('<speak:可以拍。><capture>', rid='person', source='user'),
                     {'type': 'response.done', 'response_id': 'person'},
                     self.delta('延迟到达。', rid='person', source='user'))
        self.shutter.save.assert_called_once_with()
        self.assertEqual([entry.args[0] for entry in self.voice.say.call_args_list], ['有人路过，', '可以拍。'])
        self.assertEqual(sum(row['event'] == 'observation.failed' for row in self.log.rows), 1)

    def test_user_speech_wins_when_an_observation_arrives_out_of_order(self):
        self.receive(self.delta('<speak:过期的开头'),
                     self.delta('<speak:保持', rid='person', source='user'),
                     self.delta('不应播报。>'),
                     self.delta('<speak:晚到的观察。>', rid='late'),
                     self.delta('这个姿势。>', rid='person', source='user'),
                     {'type': 'response.done', 'response_id': 'person'},
                     self.delta('过期后续。', rid='late'))
        self.voice.say.assert_called_once_with('保持这个姿势。')
        self.assertEqual([row['text'] for row in self.log.rows if row['event'] == 'speak'], ['保持这个姿势。'])


if __name__ == '__main__':
    unittest.main()
