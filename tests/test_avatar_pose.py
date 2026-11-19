from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from photo_assistant.avatar.frames import AvatarFrames
from photo_assistant.avatar.pose import (
    AvatarPose, DEFAULT_SPRITE_CATALOG, PoseCatalog, SPRITE_OPTIONS, SpriteEntry, load_pose_catalog,
)


class AvatarPoseTests(unittest.TestCase):
    def test_every_advertised_combination_decodes_with_its_ground_point(self):
        frames = AvatarFrames(DEFAULT_SPRITE_CATALOG)
        self.addCleanup(frames.close)
        catalog = frames.catalog
        self.assertEqual(catalog.default_pose, AvatarPose())
        self.assertEqual(catalog.entry_pose, AvatarPose(torso="半身歪斜向右"))
        for pose in catalog.entries:
            self.assertEqual(catalog.select(asdict(pose)), pose)
            pixels = frames.frame(pose)
            self.assertIs(pixels, frames.frame(pose))
            self.assertEqual(pixels.shape[2], 4)
            self.assertGreater(pixels[:, :, 3].max(), 0)
            self.assertGreater(catalog.entry(pose).standing_height_px, 0)
        base = catalog.default_pose
        sitting = replace(base, legs="坐着")
        self.assertEqual(catalog.entry(sitting).standing_height_px, catalog.entry(base).standing_height_px)
        self.assertLess(frames.frame(sitting).shape[0], frames.frame(base).shape[0])

    def test_fields_are_mandatory_and_values_are_closed(self):
        catalog = load_pose_catalog(DEFAULT_SPRITE_CATALOG)
        for invalid in ({}, {"face": "微笑"}, {**asdict(AvatarPose()), "mood": "开心"},
                        {**asdict(AvatarPose()), "hand": "挥手"}):
            with self.assertRaises(ValueError):
                catalog.select(invalid)
        self.assertIn(AvatarPose().command(), catalog.describe())

    def test_sparse_entry_prefers_available_leaning_pose_in_priority_order(self):
        entry = SpriteEntry(Path("sample.png"), 400, (80, 400))
        ideal = AvatarPose(torso="半身歪斜向右")
        poses = [AvatarPose(), replace(ideal, legs="坐着"), replace(ideal, hand="比耶"),
                 replace(ideal, arm="背在身后"), replace(ideal, face="高冷表情"), ideal]
        entries = {}
        for pose in poses:
            entries[pose] = entry
            self.assertEqual(PoseCatalog(entries).entry_pose, pose)
        # Equivalent leaning poses keep manifest order.
        other_face = replace(ideal, face="龇牙笑")
        self.assertEqual(PoseCatalog({poses[-2]: entry, other_face: entry}).entry_pose, poses[-2])

    def test_compact_menu_expands_to_exactly_the_available_combinations(self):
        entry = SpriteEntry(Path("sample.png"), 400, (80, 400))
        poses = [AvatarPose(*values) for values in itertools.product(*SPRITE_OPTIONS.values())]
        entries = {pose: entry for pose in poses if pose.legs == "蹲下" or pose.hand == "比耶"}
        menu = PoseCatalog(entries).describe()
        advertised = set()
        for line in menu.splitlines():
            terms = line.split(" | ")
            if len(terms) == 5 and terms[0] == "*":
                choices = [options if term == "*" else term.split("/")
                           for term, options in zip(terms, SPRITE_OPTIONS.values())]
                advertised.update(AvatarPose(*values) for values in itertools.product(*choices))
        self.assertEqual(advertised, set(entries))

    def test_custom_sparse_catalog_defaults_to_first_entry_and_checks_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "catalog.json"
            pose = AvatarPose(face="高冷表情", legs="坐着")
            row = {"file": "样例.png", "tags": asdict(pose), "standing_height_px": 500, "anchor_px": [2, 10]}
            def write(*rows):
                path.write_text(json.dumps({"sprites": rows}, ensure_ascii=False), encoding="utf8")
            write(row)
            with self.assertRaisesRegex(ValueError, "样例.png"):
                load_pose_catalog(path)
            sample = load_pose_catalog(DEFAULT_SPRITE_CATALOG).entry(AvatarPose()).path
            (folder / row["file"]).write_bytes(sample.read_bytes())
            catalog = load_pose_catalog(path)
            self.assertEqual(catalog.default_pose, pose)
            self.assertEqual(catalog.entry_pose, pose)
            self.assertIn(pose.command(), catalog.describe())
            self.assertNotIn(AvatarPose().command(), catalog.describe())
            with self.assertRaisesRegex(ValueError, "没有此立绘组合"):
                catalog.select(asdict(AvatarPose()))
            frames = AvatarFrames(path)
            self.addCleanup(frames.close)
            (folder / row["file"]).write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "无法读取立绘"):
                frames.frame(pose)
            (folder / row["file"]).write_bytes(sample.read_bytes())
            self.assertEqual(frames.frame(pose).shape[2], 4)
            for rows in ((row, row), ({**row, "standing_height_px": 0},), ({**row, "anchor_px": [1, -3]},),
                         ({**row, "file": "../outside.png"},), ()):
                write(*rows)
                with self.assertRaises(ValueError):
                    load_pose_catalog(path)


if __name__ == "__main__":
    unittest.main()
