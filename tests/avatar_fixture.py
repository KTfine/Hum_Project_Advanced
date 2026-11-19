"""Provide deterministic images and projection size to pixel-level compositing tests."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from photo_assistant.avatar.frames import AvatarFrames
from photo_assistant.avatar.pose import AvatarPose, PoseCatalog, SpriteEntry


@contextmanager
def rendered_avatar(pixels):
    h, w = pixels.shape[:2]
    frames = Mock(spec=AvatarFrames)
    frames.catalog = PoseCatalog({AvatarPose(): SpriteEntry(Path("fixture.png"), h, (w / 2, h))})
    frames.frame.return_value = pixels
    with (patch("photo_assistant.client.AvatarFrames", return_value=frames),
          patch("photo_assistant.client.standing_height_px", return_value=h)):
        yield frames
