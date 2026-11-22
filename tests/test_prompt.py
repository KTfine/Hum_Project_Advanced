import json
import re
import unittest
from unittest.mock import patch

from photo_assistant import prompts
from photo_assistant.avatar.pose import WHITE_DRAGON_CATALOG, PoseCatalog, load_pose_catalog
from photo_assistant.protocol import StreamingCommandParser


class StreamingCommandPromptTests(unittest.TestCase):
    def render_prompt(self, template=None, catalog=None):
        catalog = catalog or load_pose_catalog(WHITE_DRAGON_CATALOG)
        return (template or prompts.PHOTO_GUIDANCE_PROMPT).format(
            character_name="小航",
            character_persona="活泼但不打扰用户",
            response_language="中文",
            pose_catalog=catalog.describe(),
            tachie_fields=catalog.entry_pose.command_fields(),
            entry_context=prompts.LIVE_ENTRY_CONTEXT,
            current_character_state="x=0.5,y=0.95,scale=1.0,rotation=0,face=微笑",
            recent_context="用户希望拍一张严肃的证件照。",
            extra_instructions="保持简短。",
        )

    def test_guidance_prompt_contains_context_and_guidance_commands(self):
        rendered = self.render_prompt()

        self.assertIn("当前任务：真人拍照指导", rendered)
        self.assertEqual(set(re.findall(r'<([a-z]+)(?=[:>])', rendered)), {'speak', 'blank', 'mode', 'capture'})
        self.assertNotIn('<tachie:', rendered)
        self.assertNotIn('入场', rendered)
        self.assertNotIn('初始角色状态', rendered)
        self.assertNotIn('x=0.5', rendered)

    def test_co_photo_prompt_describes_available_actions_and_poses(self):
        catalog = load_pose_catalog(WHITE_DRAGON_CATALOG)
        rendered = self.render_prompt(prompts.CO_PHOTO_COMMAND_PROMPT, catalog=catalog)
        self.assertIn("当前任务：合照互动", rendered)
        self.assertIn("表情", rendered)
        self.assertIn(prompts.LIVE_ENTRY_CONTEXT, rendered)
        self.assertIn(catalog.describe(), rendered)
        self.assertIn('<sit:distance=near>', rendered)
        self.assertIn('<sit:distance=far>', rendered)
        self.assertNotIn('<sit:table=', rendered)
        self.assertNotIn('depth=', rendered)
        self.assertIn('x=0.5', rendered)
        advertised = set()
        advertised_poses = set()
        for example in re.findall(r'<[^<>\n]+>', rendered):
            with self.subTest(example=example):
                commands = StreamingCommandParser().feed("prompt", example)
                self.assertEqual(len(commands), 1)
                command = commands[0]
                advertised.add(command.kind)
                if command.kind == "tachie":
                    advertised_poses.add(catalog.select(command.params))
        self.assertEqual(advertised_poses, set(catalog.entries))
        self.assertEqual(advertised, {
            'speak', 'teleport', 'rotate', 'tachie', 'layer', 'blank', 'mode', 'capture', 'follow',
            'peek', 'sit', 'brace', 'stand',
        })

    def test_live_scene_geometry_is_explained_only_in_co_photo(self):
        live = self.render_prompt(prompts.CO_PHOTO_COMMAND_PROMPT)
        guidance = self.render_prompt()
        for field in ("scene_objects", "current_tachie", "avatar_visible"):
            self.assertIn(field, live)
            self.assertNotIn(field, guidance)

    def test_all_pose_examples_are_compatible_with_a_sparse_catalog(self):
        from experiments.scripts.hum_api.session import build_prompt

        bundled = load_pose_catalog(WHITE_DRAGON_CATALOG)
        pose = next(p for p in bundled.entries if p.torso == "正着站" and p.face == "高冷表情")
        sparse = PoseCatalog({pose: bundled.entry(pose)})
        live = self.render_prompt(prompts.CO_PHOTO_COMMAND_PROMPT, catalog=sparse)
        with patch("experiments.scripts.hum_api.session.load_pose_catalog", return_value=sparse):
            experiment = build_prompt("co-photo")
        self.assertNotIn(prompts.LIVE_ENTRY_CONTEXT, experiment)
        for rendered in (live, experiment):
            examples = re.findall(r'<tachie:[^>]+>', rendered)
            self.assertTrue(examples)
            for example in examples:
                command, = StreamingCommandParser().feed("example", example)
                self.assertEqual(sparse.select(command.params), pose)

    def test_experiment_starts_in_the_bundled_peeking_pose(self):
        from experiments.scripts.hum_api.session import build_prompt

        rendered = build_prompt("co-photo")
        state_line = next(line for line in rendered.splitlines() if line.startswith("初始角色状态："))
        state = json.loads(state_line.removeprefix("初始角色状态："))
        self.assertEqual(state["pose"]["torso"], "半身歪斜向右")

    def test_both_modes_include_identity_context_and_switch_commands(self):
        for template in (prompts.PHOTO_GUIDANCE_PROMPT, prompts.CO_PHOTO_COMMAND_PROMPT):
            rendered = self.render_prompt(template)
            for value in ("小航", "活泼但不打扰用户", "中文", "保持简短。",
                          "用户希望拍一张严肃的证件照。", "<mode:co-photo>", "<mode:guidance>"):
                self.assertIn(value, rendered)
            self.assertIn("完整闭合", rendered)
            self.assertIn("单行连续输出", rendered)
            self.assertNotIn("{character_name}", rendered)


if __name__ == "__main__":
    unittest.main()
