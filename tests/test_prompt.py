import json
import re
import unittest
from unittest.mock import patch

from photo_assistant import prompts
from photo_assistant.avatar.pose import DEFAULT_SPRITE_CATALOG, SPRITE_OPTIONS, PoseCatalog, load_pose_catalog
from photo_assistant.protocol import StreamingCommandParser


class StreamingCommandPromptTests(unittest.TestCase):
    def render_prompt(self, template=None, catalog=None):
        catalog = catalog or load_pose_catalog(DEFAULT_SPRITE_CATALOG)
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

        self.assertIn("小航", rendered)
        self.assertIn("用户希望拍一张严肃的证件照", rendered)
        self.assertIn("当前任务：真人拍照指导", rendered)
        self.assertEqual(set(re.findall(r'<([a-z]+)(?=[:>])', rendered)), {'speak', 'blank', 'mode', 'capture'})
        self.assertNotIn('<tachie:', rendered)
        self.assertNotIn('入场', rendered)
        self.assertNotIn('初始角色状态', rendered)
        self.assertNotIn('x=0.5', rendered)

    def test_co_photo_prompt_describes_teleport_and_pose_selection(self):
        rendered = self.render_prompt(prompts.CO_PHOTO_COMMAND_PROMPT)
        self.assertIn("当前任务：合照互动", rendered)
        self.assertIn("表情", rendered)
        context, actions = rendered.split("动作写法", 1)
        self.assertNotIn("<tachie:", context)
        self.assertIn(prompts.LIVE_ENTRY_CONTEXT, context)
        self.assertIn("自行决定", actions)
        teleport_examples = re.findall(r'<teleport:[^>]+>', rendered)
        self.assertTrue(teleport_examples)
        for example in teleport_examples:
            command, = StreamingCommandParser().feed("prompt", example)
            self.assertEqual(command.kind, "teleport")
        sprite_examples = re.findall(r'<tachie:[^>]+>', actions)
        self.assertTrue(sprite_examples)
        for example in sprite_examples:
            commands = StreamingCommandParser().feed("prompt", example)
            self.assertEqual(len(commands), 1, example)
            self.assertEqual(commands[0].params.keys(), SPRITE_OPTIONS.keys())
        for values in SPRITE_OPTIONS.values():
            for value in values:
                self.assertIn(value, rendered)
        self.assertIn('x=0.5', rendered)
        self.assertEqual(set(re.findall(r'<([a-z]+)(?=[:>])', rendered)), {'speak', 'teleport', 'rotate', 'tachie', 'layer', 'blank', 'mode', 'capture'})

    def test_action_line_uses_a_complete_command_from_the_loaded_catalog(self):
        from experiments.scripts.hum_api.session import build_prompt

        bundled = load_pose_catalog(DEFAULT_SPRITE_CATALOG)
        pose = next(p for p in bundled.entries if p.torso == "正着站" and p.face == "高冷表情")
        sparse = PoseCatalog({pose: bundled.entry(pose)})
        live = self.render_prompt(prompts.CO_PHOTO_COMMAND_PROMPT, catalog=sparse)
        with patch("experiments.scripts.hum_api.session.load_pose_catalog", return_value=sparse):
            experiment = build_prompt("co-photo")
        self.assertNotIn(prompts.LIVE_ENTRY_CONTEXT, experiment)
        for rendered in (live, experiment):
            line = next(line for line in rendered.splitlines() if line.startswith("- 切换立绘"))
            command, = StreamingCommandParser().feed("example", line)
            self.assertEqual(sparse.select(command.params), pose)

    def test_experiment_starts_in_the_bundled_peeking_pose(self):
        from experiments.scripts.hum_api.session import build_prompt

        rendered = build_prompt("co-photo")
        state_line = next(line for line in rendered.splitlines() if line.startswith("初始角色状态："))
        state = json.loads(state_line.removeprefix("初始角色状态："))
        self.assertEqual(state["pose"]["torso"], "半身歪斜向右")

    def test_photography_guidance_is_derived_from_user_instruction(self):
        rendered = self.render_prompt()

        self.assertIn("根据用户当前的拍摄目标", rendered)
        self.assertIn("画面来自真人摄像头", rendered)
        self.assertNotIn("右上角白底小窗", rendered)
        self.assertNotIn("下巴", rendered)
        self.assertNotIn("侧脸", rendered)
        self.assertNotIn("微笑更好", rendered)

    def test_perspective_and_half_body_example_in_the_built_in_prompts(self):
        guidance = self.render_prompt()
        co_photo = self.render_prompt(prompts.CO_PHOTO_COMMAND_PROMPT)
        for rendered in (guidance, co_photo):
            self.assertIn("近大远小", rendered)
        self.assertIn("快门会保存实景", co_photo)
        example = next(line for line in co_photo.splitlines() if line.startswith("- 拍半身合照"))
        self.assertIn("矮约半个头", co_photo)
        teleport, = StreamingCommandParser().feed("example", example)
        self.assertEqual(teleport.kind, "teleport")
        self.assertGreater(teleport.params["y"], 1)

    def test_requires_tag_only_streaming_output_and_no_json(self):
        rendered = self.render_prompt()

        self.assertIn("不输出标签外文字", rendered)
        self.assertIn("完整闭合后即可执行", rendered)
        self.assertIn("单行连续输出", rendered)
        self.assertNotIn("JSON Lines", rendered)
        self.assertNotIn('{"type"', rendered)
        self.assertNotIn("{character_name}", rendered)


if __name__ == "__main__":
    unittest.main()
