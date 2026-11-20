"""Place a calibrated seated sprite on the front edge of a segmented table."""
from dataclasses import dataclass
from math import cos, radians, sin

import cv2
import numpy as np

from .framing import Placement
from .ground import camera_reference, projected_height


def _overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1e-8)


@dataclass(frozen=True)
class TableSeat:
    placement: Placement
    support_xy: tuple[float, float]
    floor_xy: tuple[float, float]
    host_box: tuple[float, float, float, float]
    # Ground point below the rim. The sprite's dangling shoes are above it.
    ground_y: float

    def foreground(self, scene):
        """Exclude the supporting table; preserve nearer independent silhouettes."""
        h, w = scene.person.shape
        x, y = self.support_xy
        row, col = min(h - 1, max(0, round(y * h) - 2)), min(w - 1, max(0, round(x * w)))
        if scene.ground is not None:
            rim_depth = scene.ground.depth_at(x, self.ground_y)
        else:
            sample = scene.inverse_depth[max(0, row - 2):row + 1, max(0, col - 2):col + 3]
            rim_depth = float(np.median(sample))
        threshold = np.clip((scene.inverse_depth - rim_depth - 0.012) / .02 + .5, 0, 1)
        cover = np.zeros_like(scene.person)
        # Match just one host in the current exposure. A second table may cross
        # the support pixel while still legitimately occluding the seated body.
        tables = [(i, _overlap(info["box"], self.host_box)) for i, (obj, info)
                  in enumerate(zip(scene.objects, scene.descriptions)) if obj.label == "dining table"]
        host, agreement = max(tables, key=lambda item: item[1], default=(None, 0.))
        for i, obj in enumerate(scene.objects):
            if i == host and agreement > .48:
                continue
            np.maximum(cover, obj.mask, out=cover)
        return cover * threshold, scene.foreground_person * threshold


def _edge_sites(frame, mask):
    """Sample the camera-facing boundary of the broad tabletop, removing narrow legs."""
    binary = (mask >= .5).astype(np.uint8)
    left, top, width, height = cv2.boundingRect(binary)
    if width < max(16, frame.shape[1] * .07) or height < 5:
        return
    roi = binary[top:top + height, left:left + width]
    broad = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((1, max(3, round(.20 * width))), np.uint8))
    gray = cv2.cvtColor(frame[top:top + height, left:left + width], cv2.COLOR_BGR2GRAY)
    vertical_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    half = max(3, round(.037 * width))
    for fraction in (.5, .42, .58, .34, .66, .26, .74):
        x = round(fraction * (width - 1))
        patch = broad[:, x - half:x + half + 1]
        ys = np.where(patch.mean(axis=1) >= .65)[0]
        if len(ys) < 3:
            continue
        y = int(ys[-1])
        # A soft model silhouette can include the lip thickness. Snap the support
        # to the image edge, near the frontmost part of that broad surface.
        radius = max(2, round(.009 * frame.shape[0]))
        low, high = max(0, y - radius), min(height, y + radius + 1)
        contrast = np.median(vertical_gradient[low:high, x - half:x + half + 1], axis=1)
        if contrast.max() >= 25:
            y = low + int(np.argmax(contrast))
        yield left + x, top + y


def table_seat(frame, scene, entry, anchor: Placement | None, *, table=None, target_x=None, target_y=None,
               rotation=0.0) -> TableSeat | None:
    """Return a supported pose on the chosen table, or None when no usable rim exists."""
    if scene is None or entry.seat_contact_px is None or table not in (None, "near", "far"):
        return None
    tables = [(obj, info) for obj, info in zip(scene.objects, scene.descriptions) if obj.label == "dining table"]
    if table is None and target_x is not None and target_y is not None:
        def target_distance(item):
            left, top, right, bottom = item[1]["box"]
            return np.hypot(target_x - (left + right) / 2, target_y - (top + bottom) / 2)
        tables = sorted(tables, key=target_distance)[:1]
        if tables and target_distance(tables[0]) > .25:
            return None
    else:
        def distance_order(item):
            obj, info = item
            depth = scene.inverse_depth[obj.mask > .5]
            valid = depth[np.isfinite(depth)]
            inverse_distance = float(np.median(valid)) if valid.size else 0.0
            return inverse_distance, info["box"][3], depth.size
        tables.sort(key=distance_order, reverse=table != "far")
        if table is not None:
            tables = tables[:1]
    h, w = frame.shape[:2]
    _, base_y, base_height = camera_reference()
    if anchor is not None:
        base_y, base_height = anchor.y, anchor.standing_height_ratio
    blockers = scene.person.copy()
    for other in scene.objects:
        if other.label != "dining table":
            np.maximum(blockers, other.mask, out=blockers)
    dx = entry.seat_contact_px[0] - entry.anchor_px[0]
    dy = entry.seat_contact_px[1] - entry.anchor_px[1]
    angle = radians(rotation)
    dx, dy = cos(angle) * dx - sin(angle) * dy, sin(angle) * dx + cos(angle) * dy
    for obj, info in tables:
        best, best_score = None, float("inf")
        for x, y in _edge_sites(frame, obj.mask):
            # Read the table just inside its lip, avoiding the floor seen through the gap below.
            sample_h, sample_w = max(3, round(.012 * h)), max(3, round(.012 * w))
            patch = np.s_[max(0, y - sample_h):max(1, y), max(0, x - sample_w):min(w, x + sample_w + 1)]
            sample = scene.inverse_depth[patch][obj.mask[patch] > .5]
            if len(sample) < 3:
                continue
            rim_depth = float(np.median(sample))
            ground_y = info["box"][3]
            horizon = .4
            if scene.ground is not None:
                horizon = scene.ground.horizon_y
                intercept = scene.ground.depth_at(x / w, 0)
                slope = scene.ground.depth_at(x / w, 1) - intercept
                ground_y = (rim_depth - intercept) / slope
            ground_y = float(np.clip(max(ground_y, y / h), 0, 2))
            size = projected_height(base_y, base_height, ground_y, horizon)
            zoom = size * h / entry.standing_height_px
            foot_x, foot_y = (x - dx * zoom) / w, (y - dy * zoom) / h
            # Reserve space above the lip for the face, and below it for the legs.
            half_width = max(3, round(.135 * size * h))
            occupied = np.s_[max(0, round(y - .49 * size * h)):min(h, round(y + .30 * size * h) + 1),
                              max(0, x - half_width):min(w, x + half_width + 1)]
            nearer = scene.inverse_depth[occupied] >= rim_depth - .01
            obstruction = float(np.mean(nearer & (blockers[occupied] > .5)))
            if obstruction > .5:
                continue
            score = obstruction + .28 * abs(x / w - info["center_x"]) + max(0, foot_y - 1)
            if score < best_score:
                best_score = score
                place = Placement(foot_x, foot_y, anchor.scale if anchor else 1.0, size,
                                  anchor.framing if anchor else "", "桌沿坐姿")
                best = TableSeat(place, (x / w, y / h), (x / w, min(2, ground_y + .04)), tuple(info["box"]), ground_y)
        if best is not None:
            return best
    return None
