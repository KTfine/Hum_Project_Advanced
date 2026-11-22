import unittest

from photo_assistant.avatar.pose import AvatarPose
from photo_assistant.readiness import CoPhotoReadiness


class CoPhotoReadinessTests(unittest.TestCase):
    def setUp(self):
        self.ready = CoPhotoReadiness()
        self.standing = dict(x=0.6, y=0.9, scale=1.0, current_tachie=AvatarPose().command(),
                             avatar_visible=True, pose_settled=True)
        self.seated = {**self.standing, "current_tachie": AvatarPose(legs="坐着").command()}

    def test_entry_waits_for_real_visible_settled_frame_then_stays_quiet(self):
        for placement in (None, {"avatar_visible": False}, {**self.standing, "avatar_visible": False},
                          {**self.standing, "pose_settled": False}):
            self.assertIsNone(self.ready.observe(placement))
        self.assertTrue(self.ready.observe(self.standing))
        for placement in (self.standing, None, {"avatar_visible": False}, self.standing, self.seated):
            self.assertIsNone(self.ready.observe(placement))

    def test_multi_command_entry_waits_for_whole_turn_and_reports_once(self):
        self.ready.action_result("teleport", applied=True)
        self.assertIsNone(self.ready.observe(self.standing))
        self.ready.begin_turn()  # The caller can see a new-input event and its first streamed chunk.
        self.ready.action_result("tachie", applied=True)
        self.assertIsNone(self.ready.observe(self.seated))
        self.ready.finish_turn()
        self.assertIsNone(self.ready.observe({**self.seated, "pose_settled": False}))
        self.assertTrue(self.ready.observe(self.seated))
        self.assertIsNone(self.ready.observe(self.seated))

    def test_pose_change_and_return_to_earlier_pose_each_announce_once(self):
        self.assertTrue(self.ready.observe(self.standing))
        for final in (self.seated, self.standing):
            self.ready.action_result("tachie", applied=True)
            self.ready.finish_turn()
            self.assertTrue(self.ready.observe(final))
            self.assertIsNone(self.ready.observe(final))

    def test_noop_and_camera_jitter_are_silent_but_real_move_announces(self):
        self.ready.observe(self.standing)
        self.ready.action_result("follow", applied=True)
        self.ready.finish_turn()
        self.assertIsNone(self.ready.observe({**self.standing, "x": 0.6002}))
        self.ready.action_result("teleport", applied=True)
        self.ready.finish_turn()
        self.assertTrue(self.ready.observe({**self.standing, "x": 0.8}))

    def test_failed_or_interrupted_compound_action_never_becomes_an_entry_cue(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                ready = CoPhotoReadiness()
                ready.action_result("teleport", applied=True)
                if interrupted:
                    ready.cancel_turn()
                else:
                    ready.action_result("sit", applied=False)
                    ready.action_result("rotate", applied=True)
                ready.finish_turn()
                for placement in (None, self.standing, self.seated):
                    self.assertIsNone(ready.observe(placement))
                ready.begin_turn()
                ready.finish_turn()
                self.assertIsNone(ready.observe(self.seated))
                ready.action_result("stand", applied=True)
                ready.finish_turn()
                self.assertTrue(ready.observe(self.standing))

    def test_new_input_supersedes_pending_action(self):
        for entered in (False, True):
            with self.subTest(entered=entered):
                ready = CoPhotoReadiness()
                if entered:
                    ready.observe(self.standing)
                ready.action_result("sit", applied=True)
                ready.finish_turn()
                ready.begin_turn()
                ready.finish_turn()
                self.assertIsNone(ready.observe(self.seated))

    def test_message_uses_rendered_pose_and_placement(self):
        for source, pose, expected in (
            ("并排站位", AvatarPose(hand="比耶"), "我在你旁边，剪刀手摆好了。"),
            ("桌沿坐姿", AvatarPose(legs="坐着"), "我在桌沿坐好了。"),
            ("桌沿坐姿", AvatarPose(), "我到位了，姿势也摆好了。"),
            ("桌后站位", AvatarPose(), "我在桌子后面，姿势摆好了。"),
            ("指定构图", AvatarPose(hand="比耶"), "我到位了，剪刀手摆好了。"),
        ):
            with self.subTest(source=source, pose=pose):
                placement = dict(self.standing, placement_source=source, current_tachie=pose.command())
                self.assertEqual(CoPhotoReadiness().observe(placement), expected)

    def test_last_action_result_survives_questions_but_not_incomplete_actions(self):
        self.assertIsNone(self.ready.last_action_succeeded)
        self.ready.action_result("stand", applied=True)
        self.assertIsNone(self.ready.last_action_succeeded)
        self.ready.finish_turn()
        self.assertIs(self.ready.last_action_succeeded, True)
        self.ready.begin_turn()
        self.ready.finish_turn()
        self.assertIs(self.ready.last_action_succeeded, True)
        self.ready.action_result("sit", applied=False)
        self.ready.action_result("rotate", applied=True)
        self.ready.finish_turn()
        self.assertIs(self.ready.last_action_succeeded, False)
        self.ready.begin_turn()
        self.ready.cancel_turn()
        self.assertIs(self.ready.last_action_succeeded, False)
        self.ready.action_result("stand", applied=True)
        self.ready.cancel_turn()
        self.ready.finish_turn()
        self.assertIsNone(self.ready.last_action_succeeded)
