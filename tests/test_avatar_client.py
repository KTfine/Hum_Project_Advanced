"""Exercise the bundled avatar through the streaming, preview, layer and shutter paths."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from photo_assistant.avatar.pose import AvatarPose, WHITE_DRAGON_CATALOG, load_pose_catalog
from photo_assistant.protocol import StreamingCommandParser
from experiments.scripts.hum_api import build_prompt
from photo_assistant.client import CanvasSource, PhotoCapture, build_control_prompt, parse_args


class AvatarClientTests(unittest.TestCase):
    def test_real_catalog_switches_only_complete_available_sprites_through_capture(self):
        args = parse_args([])
        self.assertEqual(args.sprite_catalog, WHITE_DRAGON_CATALOG)
        self.assertEqual(args.character_name, "小白龙")
        frame = np.full((480, 640, 3), 206, np.uint8)
        camera = SimpleNamespace(latest_frame=lambda: frame)
        canvas = CanvasSource(args.sprite_catalog, 640, 480, 90, threading.Event(), preview=False,
                              scene_layers={"person": np.ones(frame.shape[:2], np.float32)})
        self.addCleanup(canvas.close)
        canvas.start()
        canvas.configure_view(camera, show_character=True)
        np.testing.assert_array_equal(canvas.render_preview_frame(), frame)

        parser = StreamingCommandParser()
        self.assertEqual(parser.feed("one", "<tac"), [])
        commands = parser.feed(
            "one", "hie:face=惊讶,hand=无动作,arm=自然下垂,torso=正着站,legs=站着><layer:to,index=1>"
        )
        self.assertEqual([command.kind for command in commands], ["tachie", "layer"])
        for command in commands:
            canvas.apply_command(command)
        standing = canvas.render_preview_frame()
        self.assertFalse(np.array_equal(standing, frame))
        sit, = parser.feed("one", "<tachie:face=高冷表情,hand=无动作,arm=自然下垂,torso=正着站,legs=坐着>")
        state = canvas.apply_command(sit)
        self.assertEqual(state.pose, AvatarPose(face="高冷表情", legs="坐着"))
        self.assertEqual(parser.feed("one", "<tachie:face=微笑>"), [])
        self.assertEqual(canvas.controller.state_at(), state)
        preview = canvas.render_preview_frame()
        self.assertFalse(np.array_equal(standing, preview))
        with tempfile.TemporaryDirectory() as out:
            photo = PhotoCapture(camera, Path(out), canvas).save()
            decoded = cv2.imdecode(np.frombuffer(photo.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded, preview)

        for rendered in (build_control_prompt(canvas, args, mode="co-photo"), build_prompt("co-photo")):
            self.assertIn("小白龙", rendered)
            self.assertIn(AvatarPose(face="惊讶").command(), rendered)
            self.assertIn("<tachie:", rendered)
        self.assertNotIn("<tachie:", build_control_prompt(canvas, args, mode="guidance"))

    def test_corrupt_replacement_image_keeps_previous_sprite_visible(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        catalog = load_pose_catalog(WHITE_DRAGON_CATALOG)
        art = catalog.entry(catalog.default_pose)
        rows = []
        for name, pose in [('good', AvatarPose()), ('broken', AvatarPose(hand='比耶'))]:
            (folder / f'{name}.png').write_bytes(art.path.read_bytes())
            rows.append({'file': f'{name}.png', 'tags': vars(pose), 'anchor_px': art.anchor_px,
                         'standing_height_px': art.standing_height_px})
        manifest = folder / 'catalog.json'
        manifest.write_text(json.dumps({'sprites': rows}), encoding='utf-8')
        canvas = CanvasSource(manifest, 320, 240, 80, threading.Event(), preview=False)
        canvas.start()
        self.addCleanup(canvas.close)
        before = canvas.render_preview_frame()
        (folder / 'broken.png').write_bytes(b'')
        change, = StreamingCommandParser().feed('r', AvatarPose(hand='比耶').command())
        with self.assertRaises(ValueError):
            canvas.apply_command(change)
        self.assertEqual(canvas.controller.state_at().pose, AvatarPose())
        np.testing.assert_array_equal(canvas.render_preview_frame(), before)


if __name__ == "__main__":
    unittest.main()
