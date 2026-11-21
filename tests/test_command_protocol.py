import unittest

SPRITE_PARAMS = {
    "face": "微笑", "hand": "无动作", "arm": "自然下垂", "torso": "正着站", "legs": "站着",
}
SPRITE_TAG = "<tachie:face=微笑,hand=无动作,arm=自然下垂,torso=正着站,legs=站着>"

class StreamingCommandParserTests(unittest.TestCase):
    def make_parser(self, **kwargs):
        from photo_assistant.protocol import StreamingCommandParser

        return StreamingCommandParser(**kwargs)

    def test_split_command_emits_only_after_final_delimiter(self):
        parser = self.make_parser()

        tag = "<teleport:x=-2,y=3,scale=0.1>"
        for split in range(1, len(tag)):
            with self.subTest(split=split):
                self.assertEqual(parser.feed("r1", tag[:split]), [])
                commands = parser.feed("r1", tag[split:] + "<capture>")
                self.assertEqual([(c.kind, c.params) for c in commands], [
                    ("teleport", {"x": -2, "y": 3, "scale": 0.1}), ("capture", {}),
                ])

    def test_multiple_commands_are_emitted_in_stream_order(self):
        parser = self.make_parser()

        commands = parser.feed(
            "r1", "noise<speak:向左一点。>" + SPRITE_TAG + "<blank>trailing"
        )

        self.assertEqual([command.kind for command in commands], ["speak", "tachie", "blank"])
        self.assertEqual(commands[0].params, {"text": "向左一点。"})
        self.assertEqual(commands[1].params, SPRITE_PARAMS)

    def test_composition_tags_survive_streaming_and_precede_the_shutter(self):
        stream = "<peek><sit><sit:table=near><sit:table=far><sit:y=0.42,x=0.6><brace><brace:x=.25><stand><follow><capture>"
        expected = [("peek", {}), ("sit", {}), ("sit", {"table": "near"}), ("sit", {"table": "far"}),
                    ("sit", {"x": 0.6, "y": 0.42}), ("brace", {}), ("brace", {"x": 0.25}),
                    ("stand", {}), ("follow", {}), ("capture", {})]
        for plain in (False, True):
            parser = self.make_parser(allow_plain_speech=plain)
            commands = []
            for char in stream:
                emitted = parser.feed("r", char)
                if char != ">":
                    self.assertEqual(emitted, [])
                commands.extend(emitted)
            self.assertEqual([(c.kind, c.params) for c in commands], expected)

    def test_malformed_scene_requests_do_not_move_the_avatar(self):
        malformed = ("<sit:>", "<sit:x=0.4>", "<sit:x=.4,y=-.01>", "<sit:x=nan,y=.2>",
                     "<sit:x=0.1,x=0.2,y=0.2>", "<sit:x=.1,y=.2,depth=far>",
                     "<sit:table=first>", "<sit:x=.5,y=.6,table=near>", "<stand:>",
                     "<peek:side=left>", "<brace:>", "<brace:x=nan>", "<brace:x=1.02>",
                     "<brace:side=left>", "<brace:x=.2,y=.4>", "<sitting>", "<standing>")
        for tag in malformed:
            with self.subTest(tag=tag):
                commands = self.make_parser(allow_plain_speech=True).feed("r", tag + "<capture>")
                self.assertEqual([c.kind for c in commands], ["capture"])

    def test_split_sprite_tag_switches_once_before_capture(self):
        for split in range(1, len(SPRITE_TAG)):
            with self.subTest(split=split):
                parser = self.make_parser(allow_plain_speech=True)
                self.assertEqual(parser.feed("r", SPRITE_TAG[:split]), [])
                commands = parser.feed("r", SPRITE_TAG[split:] + "<capture>")
                self.assertEqual([(c.kind, c.params) for c in commands],
                                 [("tachie", SPRITE_PARAMS), ("capture", {})])

    def test_invalid_sprite_tags_leave_following_tags_intact(self):
        for invalid in (
            SPRITE_TAG.replace("微笑", "大笑"),
            SPRITE_TAG.replace(",legs=站着", ""),
            SPRITE_TAG.replace(">", ",arm=背在身后>"),
            SPRITE_TAG.replace(">", ",path=foo.png>"),
            SPRITE_TAG.replace(",legs=", "，legs="),
            "<pose:face=joy>",
        ):
            with self.subTest(invalid=invalid):
                commands = self.make_parser(allow_plain_speech=True).feed("r", invalid + "<blank>")
                self.assertEqual([c.kind for c in commands], ["blank"])

    def test_capture_emits_once_after_closing_at_every_chunk_split(self):
        tag = "<capture>"
        for split in range(1, len(tag)):
            with self.subTest(split=split):
                parser = self.make_parser()
                self.assertEqual(parser.feed("r1", tag[:split]), [])
                commands = parser.feed("r1", tag[split:] + ">")
                self.assertEqual([(c.kind, c.params) for c in commands], [("capture", {})])
                self.assertEqual(parser.flush("r1"), [])

    def test_capture_and_speech_are_emitted_in_order(self):
        parser = self.make_parser(allow_plain_speech=True)
        commands = parser.feed("r1", "<speak:看镜头。><capture><speak:好了。>")
        self.assertEqual([c.kind for c in commands], ["speak", "capture", "speak"])

    def test_mixed_speech_closer_is_adapted_without_reading_markup(self):
        for plain in (False, True):
            for closer in ("</speak>", "</speak:>"):
                parser = self.make_parser(allow_plain_speech=plain)
                commands = []
                for char in f"<speak:到了。{closer}<wait:feedback>":
                    commands.extend(parser.feed("r", char))
                expected = [("speak", {"text": "到了。"})] if plain else []
                expected.append(("wait", {"task": "feedback"}))
                self.assertEqual([(cmd.kind, cmd.params) for cmd in commands], expected)
                self.assertEqual(parser.plain_speech_count, int(plain))

    def test_layer_slot_emits_on_close_and_in_order_with_shutter(self):
        tag = "<layer:to,index=2>"
        for split in range(1, len(tag)):
            with self.subTest(split=split):
                parser = self.make_parser()
                self.assertEqual(parser.feed("r1", tag[:split]), [])
                commands = parser.feed("r1", tag[split:] + "<capture><layer:to,index=0>")
                self.assertEqual([(c.kind, c.params) for c in commands], [
                    ("layer", {"index": 2}), ("capture", {}), ("layer", {"index": 0}),
                ])

    def test_invalid_layer_slots_do_not_change_the_next_command(self):
        parser = self.make_parser(allow_plain_speech=True)
        commands = parser.feed(
            "r1", "<layer:by,index=1><layer:to><layer:to,index=1,index=0>"
            "<layer:to,index=1,ms=250><layer:to,target=person><layer:to,index=-1>"
            "<layer:to,index=+1><layer:to,index=1.0><layer:to,index=1e0><blank>",
        )
        self.assertEqual([c.kind for c in commands], ["blank"])

    def test_only_the_exact_capture_tag_can_take_a_photo(self):
        for plain_speech in (False, True):
            with self.subTest(plain_speech=plain_speech):
                parser = self.make_parser(allow_plain_speech=plain_speech)
                commands = parser.feed(
                    "r1", "<capture:><Capture><capture ><captured><photo:capture><capture"
                )
                self.assertEqual(commands, [])
                self.assertEqual(parser.flush("r1"), [])

    def test_last_bracket_in_a_run_closes_speak_and_earlier_brackets_are_content(self):
        parser = self.make_parser()

        commands = parser.feed("r1", "<speak: >>")

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].params, {"text": ">"})

    def test_doubled_speak_closer_recovers_when_brackets_cross_chunks(self):
        parser = self.make_parser()

        self.assertEqual(parser.feed("r1", "<speak: >"), [])
        commands = parser.feed("r1", ">")

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].params, {"text": ">"})

    def test_extra_closing_brackets_do_not_invalidate_structured_commands(self):
        parser = self.make_parser()

        commands = parser.feed(
            "r1", "<teleport:x=0.2,y=0.95,scale=0.8>>" + SPRITE_TAG + ">"
        )

        self.assertEqual([command.kind for command in commands], ["teleport", "tachie"])

    def test_new_known_prefix_resynchronizes_after_unclosed_command(self):
        parser = self.make_parser()

        commands = parser.feed("r1", "<speak:坏掉了 " + SPRITE_TAG)

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].kind, "tachie")
        self.assertEqual(commands[0].params, SPRITE_PARAMS)

    def test_speak_can_recover_literal_command_like_text_after_empty_prefix(self):
        parser = self.make_parser()

        commands = parser.feed("r1", "<speak: <teleport:>")

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].kind, "speak")
        self.assertEqual(commands[0].params, {"text": "<teleport:"})

    def test_invalid_and_unknown_commands_are_silently_skipped(self):
        parser = self.make_parser()

        commands = parser.feed(
            "r1",
            "<unknown:x><move:to,x=0.2><scale:to,value=1.2>"
            "<teleport:x=nan,y=0><teleport:x=0,y=inf><teleport:x=0,y=0,scale=nan>"
            "<teleport:x=0.2,y=0.95>",
        )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].params, {"x": 0.2, "y": 0.95})

    def test_teleport_requires_absolute_destination_and_accepts_only_one_optional_scale(self):
        for fields in ("", "x=0.5", "y=0.5", "scale=1", "to,x=0,y=0", "dx=0,dy=0",
                       "x=0,y=0,ms=0", "x=0,y=0,x=1", "x=0,y=0,scale=1,scale=2"):
            with self.subTest(fields=fields):
                cmds = self.make_parser(allow_plain_speech=True).feed(
                    "r", f"<teleport:{fields}><teleport:y=-2,scale=3,x=3>"
                )
                self.assertEqual([(c.kind, c.params) for c in cmds],
                                 [("teleport", {"x": 3, "y": -2, "scale": 3})])

    def test_out_of_range_values_are_skipped_instead_of_clamped(self):
        parser = self.make_parser()

        commands = parser.feed(
            "r1", "<teleport:x=-2.01,y=0><teleport:x=0,y=3.01>"
            "<teleport:x=1,y=1,scale=3.01><teleport:x=0,y=0,scale=0.09>"
            "<rotate:to,value=181><blank>"
        )

        self.assertEqual([command.kind for command in commands], ["blank"])

    def test_overlong_unclosed_candidate_is_discarded_and_parser_recovers(self):
        parser = self.make_parser(max_buffer=32)

        self.assertEqual(parser.feed("r1", "<speak:" + "x" * 40), [])
        commands = parser.feed("r1", "garbage<speak:好了>")

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].params, {"text": "好了"})

    def test_response_change_discards_unfinished_previous_response(self):
        parser = self.make_parser()

        self.assertEqual(parser.feed("r1", "<speak:old"), [])
        commands = parser.feed("r2", "<speak:new>")

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].params, {"text": "new"})

    def test_flush_silently_discards_incomplete_input(self):
        parser = self.make_parser()
        parser.feed("r1", "<teleport:x=0.4")

        self.assertEqual(parser.flush("r1"), [])


if __name__ == "__main__":
    unittest.main()
