"""Place a calibrated palm at a floor-standing solid edge from the current exposure."""
from dataclasses import dataclass

import cv2
import numpy as np

from ..avatar.pose import AvatarPose
from .framing import Placement
from .ground import camera_reference, projected_height


@dataclass(frozen=True)
class WallBrace:
    placement: Placement
    pose: AvatarPose
    contact_xy: tuple[float, float]
    edge_xyxy: tuple[float, float, float, float]


def _upright_edges(lab):
    """Extend fragmented detected segments to the visible end of their contour."""
    h, w = lab.shape[:2]
    lines = cv2.createLineSegmentDetector().detect(lab[:, :, 0])[0]
    if lines is None:
        return
    rows = np.arange(h)
    seen = set()
    # A nearby gap can occur when the wall and background have similar color.
    bridge = np.ones((max(3, round(.025 * h)), 1), np.uint8)
    # Differences across each pixel rather than surface brightness: a pale
    # wall can have a dark edge, or the reverse.
    diff = np.linalg.norm(lab[:, 4:].astype(np.float32) - lab[:, :-4], axis=2)
    contrast = cv2.dilate(np.pad(diff, ((0, 0), (2, 2))), np.ones((1, 5), np.uint8))
    for x1, y1, x2, y2 in lines[:, 0]:
        dy = float(y2 - y1)
        if abs(dy) < .18 * h or abs(x2 - x1) > .10 * abs(dy):
            continue
        slope = float(x2 - x1) / dy
        xs = x1 + (rows - y1) * slope
        xi = np.rint(xs).astype(int)
        valid = (xi >= 2) & (xi < w - 2)
        strong = valid & (contrast[rows, xi.clip(0, w - 1)] >= 12)
        mid = int((y1 + y2) / 2)
        for sign in (1, -1):
            # Shadows and tile seams can continue the wall line onto the floor.
            # Trace only rows where the material just inside the line persists.
            inset = max(2, round(.005 * w))
            strip = lab[rows, (xi + sign * inset).clip(0, w - 1)].astype(float)
            seed = strip[int(min(y1, y2)):int(max(y1, y2)) + 1]
            template = np.median(seed, axis=0)
            distances = np.linalg.norm(seed - template, axis=1)
            tolerance = np.clip(np.percentile(distances, 85) + 10, 19, 33)
            material = np.linalg.norm(strip - template, axis=1) <= tolerance
            joined = cv2.morphologyEx((strong & material)[:, None].astype(np.uint8), cv2.MORPH_CLOSE, bridge)[:, 0]
            if not joined[mid]:
                continue
            gaps = np.flatnonzero(joined == 0)
            top = int(gaps[gaps < mid].max(initial=-1)) + 1
            bottom = int(gaps[gaps > mid].min(initial=h)) - 1
            if not (.53 * h < bottom < .97 * h and bottom - top >= .31 * h):
                continue
            key = (round(xs[mid] / 6), round(bottom / 8), round(slope * 12), sign)
            if key in seen:
                continue
            seen.add(key)
            yield (float(xs[top] / w), top / h, float(xs[bottom] / w), bottom / h), sign


def _solid_side(lab, depth, person, edge, sign, hand_y):
    """Require a broad near surface at hand height and a distinct floor at its base."""
    h, w = depth.shape
    x1, y1, x2, y2 = edge
    rows = np.linspace(max(y1 + .01, hand_y - .09), min(y2 - .05, hand_y + .12), 17)
    ys = np.rint(rows * h).astype(int).clip(0, h - 1)
    xs = w * (x1 + (x2 - x1) * ((rows - y1) / (y2 - y1)))
    offsets = np.maximum(2, w * np.array([.009, .018, .031]))
    inside = np.rint(xs[:, None] + sign * offsets).astype(int)
    outside = np.rint(xs[:, None] - sign * offsets).astype(int)
    if min(inside.min(), outside.min()) < 0 or max(inside.max(), outside.max()) >= w:
        return None
    if np.mean(person[ys[:, None], inside] >= .35) > .08:
        return None
    hand_x = round(w * (x1 + (x2 - x1) * (hand_y - y1) / (y2 - y1)))
    hand_row = round(hand_y * h)
    radius = max(2, round(.009 * h))
    palm = person[max(0, hand_row-radius):min(h, hand_row+radius+1),
                  max(0, hand_x-radius):min(w, hand_x+radius+1)]
    if not palm.size or np.any(palm >= .5):
        return None
    near, far = depth[ys[:, None], inside], depth[ys[:, None], outside]
    delta = near - far
    if not np.isfinite(delta).all() or np.percentile(delta, 25) < .045:
        return None
    surface = lab[ys[:, None], inside].reshape(-1, 3).astype(float)
    typical = np.median(surface, axis=0)
    if np.percentile(np.linalg.norm(surface - typical, axis=1), 80) > 35:
        return None
    # Inspect a narrow strip just inside the contour. Broad samples can
    # cross the seam of a receding side wall before they reach the floor.
    foot = int(round(y2 * h))
    base_x = int(round(x2 * w + sign * max(2, .005 * w)))
    if not 0 <= base_x < w:
        return None
    gap = max(3, round(.026 * h))
    base = lab[foot + gap:min(h, foot + 2 * gap), base_x]
    if not base.size or np.linalg.norm(np.median(base, axis=0) - typical) < 18:
        return None
    return min(1., float(np.median(delta)) * 2.)


def wall_brace(frame, scene, entries, anchor=None, *, target_x=None):
    """Find a visible wall edge and a pose whose palm reaches it from open floor.

    All returned geometry is normalized to the original image. A requested
    target_x selects nearby edges only. Missing depth or an ambiguous flat
    color boundary cannot establish a physical support.
    """
    if scene is None or scene.inverse_depth is None or frame is None or not frame.size:
        return None
    choices = [(p, e) for p, e in entries if e.wall_contact_px is not None and e.face_box_px is not None]
    if not choices or min(frame.shape[:2]) < 32:
        return None
    height, width = frame.shape[:2]
    factor = min(1., 640 / max(height, width))
    w, h = max(32, round(width * factor)), max(32, round(height * factor))
    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    depth = cv2.resize(scene.inverse_depth, (w, h))
    person = cv2.resize(scene.foreground_person, (w, h))
    obstacles = cv2.resize(scene.object_cover, (w, h))
    base_x, base_y, base_height = camera_reference()
    if anchor is not None:
        base_x, base_y, base_height = anchor.x, anchor.y, anchor.standing_height_ratio
    horizon = scene.ground.horizon_y if scene.ground is not None else .4
    best, best_score = None, -np.inf
    for edge, surface_side in _upright_edges(lab):
        x1, y1, x2, y2 = edge
        foot_y = y2 + .008
        size = projected_height(base_y, base_height, foot_y, horizon)
        for pose, entry in choices:
            zoom = size * height / entry.standing_height_px
            ax, ay = entry.anchor_px
            dx, dy = entry.wall_contact_px[0] - ax, entry.wall_contact_px[1] - ay
            hand_y = round(foot_y * height) / height + dy * zoom / height
            if dx * surface_side <= 0 or not (y1 + .01 <= hand_y <= y2 - .025):
                continue
            sign = surface_side
            edge_x = x1 + (x2-x1) * (hand_y-y1) / (y2-y1)
            if target_x is not None and abs(edge_x - target_x) > .20:
                continue
            quality = _solid_side(lab, depth, person, edge, sign, hand_y)
            if quality is None:
                continue
            # The drawing is anchored at a rounded shoe pixel by CanvasSource.
            # Put the opaque outer palm one pixel inside the supporting surface.
            foot_x = round(edge_x * width + sign - dx * zoom) / width
            left, top, right, bottom = entry.face_box_px
            fx = foot_x + (np.array([left, right]) - ax) * zoom / width
            fy = foot_y + (np.array([top, bottom]) - ay) * zoom / height
            if not (.01 < fx[0] < fx[1] < .99 and .01 < fy[0] < fy[1] < .99 and .03 < foot_x < .97):
                continue
            face = person[round(fy[0]*h):max(round(fy[0]*h)+1, round(fy[1]*h)),
                          round(fx[0]*w):max(round(fx[0]*w)+1, round(fx[1]*w))]
            sole = np.s_[int(foot_y*h)-2:int(foot_y*h)+2,
                         max(0, round(foot_x*w-.06*size*h)):min(w, round(foot_x*w+.06*size*h)+1)]
            if (np.mean(face > .35) > .08 or np.any(person[sole] > .5)
                    or np.mean(obstacles[sole] > .5) > .25):
                continue
            score = quality + (y2-y1) - .8 * abs(foot_x - base_x)
            if target_x is not None:
                score -= 5 * abs(edge_x - target_x)
            if score > best_score:
                best_score = score
                place = Placement(foot_x, foot_y, anchor.scale if anchor else 1., size,
                                  "全身", "扶墙站位")
                best = WallBrace(place, pose, (foot_x + dx*zoom/width, hand_y), edge)
    return best
