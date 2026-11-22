"""Place a calibrated seated sprite on a visible furniture edge."""
from dataclasses import dataclass
from math import cos, radians, sin

import cv2
import numpy as np

from .framing import Placement
from .ground import camera_reference, projected_height

_SEAT_LABELS = {"dining table", "bed", "bench", "chair", "couch"}
_LEDGE_DEPTH_TOLERANCE = .045


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
    host_label: str | None = "dining table"
    rim_depth: float | None = None

    def occluding_panel(self, panel, scene):
        """Keep nearer panel pixels while exposing the face of an unlabelled support."""
        if self.host_label is not None or self.rim_depth is None:
            return panel
        h, w = panel.shape
        left, top, right, bottom = self.host_box
        region = np.s_[round(top * h):round(bottom * h), round(left * w):round(right * w)]
        visible = panel.copy()
        visible[region] *= scene.inverse_depth[region] > self.rim_depth + _LEDGE_DEPTH_TOLERANCE
        return visible

    def foreground(self, scene):
        """Exclude the supporting instance; preserve nearer independent silhouettes."""
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
        matches = [(i, _overlap(info["box"], self.host_box)) for i, (obj, info)
                   in enumerate(zip(scene.objects, scene.descriptions)) if obj.label == self.host_label]
        host, agreement = max(matches, key=lambda item: item[1], default=(None, 0.))
        for i, obj in enumerate(scene.objects):
            if i == host and agreement > .48:
                continue
            np.maximum(cover, obj.mask, out=cover)
        return cover * threshold, scene.foreground_person * threshold


def _edge_sites(frame, mask, label, target_x=None, target_y=None):
    """Sample the broad rim above narrow legs, or the front seam of a solid seat."""
    binary = (mask >= .5).astype(np.uint8)
    left, top, width, height = cv2.boundingRect(binary)
    if width < max(16, frame.shape[1] * .07) or height < 5:
        return
    roi = binary[top:top + height, left:left + width]
    broad = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((1, max(3, round(.20 * width))), np.uint8))
    gray = cv2.cvtColor(frame[top:top + height, left:left + width], cv2.COLOR_BGR2GRAY)
    vertical_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    half = max(3, round(.037 * width))
    fractions = [.5, .42, .58, .34, .66, .26, .74]
    if target_x is not None:
        fractions.insert(0, np.clip((target_x * frame.shape[1] - left) / (width - 1), .15, .85))
    for fraction in fractions:
        x = int(np.clip(round(fraction * (width - 1)), half, width - half - 1))
        patch = broad[:, x - half:x + half + 1]
        ys = np.where(patch.mean(axis=1) >= .65)[0]
        if len(ys) < 3:
            continue
        if label in ("bench", "chair"):
            # A thin horizontal crossbar below the seat can survive leg removal.
            runs = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
            substantial = [run for run in runs if len(run) >= max(4, round(.05 * height))]
            if not substantial:
                continue
            if label == "chair":
                # The wheel base of an office chair can be wider and thicker
                # than the seat. Prefer the broad band at the indicated height.
                seat_level = .50 * height
                if target_y is not None and .22 <= (target_y * frame.shape[0] - top) / height <= .70:
                    seat_level = target_y * frame.shape[0] - top
                ys = min(substantial, key=lambda run: abs(run[-1] - seat_level))
            else:
                ys = np.concatenate(substantial)
        y = int(ys[-1])
        use_inner_seam = (label in ("couch", "bed") or (label == "chair" and y >= .70 * height)
                          or (label == "bench" and y >= .88 * height))
        if use_inner_seam:
            # A sofa front or bed base continues down to the floor. Look around
            # seat height for the front cushion seam of the supporting surface.
            low, high = {"couch": (.35, .74), "chair": (.22, .70)}.get(label, (.18, .82))
            rows = ys[(ys >= low * height) & (ys <= high * height)]
            if not len(rows):
                continue
            band = vertical_gradient[rows, x - half:x + half + 1]
            strength = np.median(band, axis=1)
            usable = strength >= 35
            if not usable.any():
                continue
            preferred = (.58 if label != "bed" else .45) * height
            if target_y is not None:
                pointed_y = target_y * frame.shape[0] - top
                if low * height <= pointed_y <= high * height:
                    preferred = pointed_y
            penalty = abs(rows - preferred) / height + .065 * (1 - np.minimum(strength / 120., 1))
            y = int(rows[usable][np.argmin(penalty[usable])])
        # A soft model silhouette can include the lip thickness. Snap the support
        # to the image edge, near the frontmost part of that broad surface.
        radius = max(2, round(.009 * frame.shape[0]))
        low, high = max(0, y - radius), min(height, y + radius + 1)
        contrast = np.median(vertical_gradient[low:high, x - half:x + half + 1], axis=1)
        if contrast.max() >= 25:
            y = low + int(np.argmax(contrast))
        yield left + x, top + y


def _pointed_ledge(frame, scene, x, y):
    """Find a broad, raised, depth-bounded surface at an unlabelled image point."""
    h, w = frame.shape[:2]
    col, row = round(x * (w - 1)), round(y * (h - 1))
    half_width, radius = max(6, round(.055 * w)), max(3, round(.045 * h))
    if not (half_width <= col < w - half_width and radius <= row < h - radius):
        return None
    depth = scene.calibration.background.inverse_depth if scene.fixed_camera else scene.inverse_depth
    room = scene.calibration.frame if scene.fixed_camera else frame
    gray = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
    gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    strip = slice(col - half_width, col + half_width + 1)
    strength = np.median(gradient[row - radius:row + radius + 1, strip], axis=1)
    rows = np.flatnonzero(strength >= max(35., .5 * strength.max())) + row - radius
    for edge in sorted(rows, key=lambda edge: abs(edge - row)):
        # The top of a cabinet shows its front face immediately below the rim.
        # A shallow shelf shows the supporting surface just above it instead.
        for seed_row in (edge + 2, edge - 2):
            if not 0 <= seed_row < h:
                continue
            if (np.mean(scene.object_cover[seed_row, strip] > .5) > .2
                    or np.mean(scene.foreground_person[seed_row, strip] > .5) > .2):
                continue
            values = depth[seed_row, strip]
            disparity = float(np.median(values))
            if not np.isfinite(disparity) or np.ptp(values) > .07:
                continue
            matching = (np.abs(depth - disparity) < _LEDGE_DEPTH_TOLERANCE).astype(np.uint8)
            if not matching[seed_row, col]:
                continue
            _, _, _, box = cv2.floodFill(matching, None, (col, seed_row), 2)
            left, top, width, height = box
            if (width < 2 * half_width or height < max(4, .025 * h) or top == 0 or top + height >= h
                    or (left == 0 and left + width == w) or width < .85 * height):
                continue
            # A broad middle seam on a vertical panel cannot support a pose.
            # Limit the contact to the top/lip, with a farther background behind it.
            if not max(2, .009 * h) <= edge - top <= max(.30 * height, .09 * h):
                continue
            above = depth[max(0, top - max(3, round(.02 * h))):top, strip]
            if not above.size or not disparity - float(np.median(above)) >= .065:
                continue
            mask = matching == 2
            if np.mean(mask[seed_row, strip]) < .85:
                continue
            return mask, (left / w, top / h, (left + width) / w, (top + height) / h), (col, int(edge))
    return None


def find_seat(frame, scene, entry, anchor: Placement | None, *, distance=None, table=None,
              target_x=None, target_y=None, rotation=0.0) -> TableSeat | None:
    """Choose a usable seat by camera distance or the requested object/edge point."""
    if distance is not None and table is not None and distance != table:
        return None
    distance = table if distance is None else distance
    if scene is None or entry.seat_contact_px is None or distance not in (None, "near", "far"):
        return None
    h, w = frame.shape[:2]
    room = scene.calibration.frame if scene.fixed_camera else frame
    furniture = [(obj, info) for obj, info in zip(scene.objects, scene.descriptions) if obj.label in _SEAT_LABELS]
    pointed = distance is None and target_x is not None and target_y is not None
    if pointed:
        def distance_to_center(item):
            left, top, right, bottom = item[1]["box"]
            return np.hypot(target_x - (left + right) / 2, target_y - (top + bottom) / 2)
        def contains_target(item):
            obj, info = item
            left, top, right, bottom = info["box"]
            if not (left <= target_x <= right and top <= target_y <= bottom):
                return False
            # The reported box center can lie in a gap between the seat and its
            # back. Away from that center, require actual material at the point.
            if (abs(target_x - (left + right) / 2) < .15 * (right - left)
                    and abs(target_y - (top + bottom) / 2) < .15 * (bottom - top)):
                return True
            col, row = round(target_x * (w - 1)), round(target_y * (h - 1))
            return (obj.mask[max(0, row - 3):row + 4, max(0, col - 3):col + 4] > .5).any()
        furniture = [item for item in furniture if contains_target(item)]
        furniture.sort(key=distance_to_center)
        furniture = furniture[:1]
    else:
        def distance_order(item):
            obj, info = item
            depth = scene.inverse_depth[obj.mask > .5]
            valid = depth[np.isfinite(depth)]
            inverse_distance = float(np.median(valid)) if valid.size else 0.0
            return inverse_distance, info["box"][3], depth.size
        furniture.sort(key=distance_order, reverse=distance != "far")
    point = (target_x, target_y) if pointed else (None, None)
    supports = [(obj.label, obj.mask, info["box"], _edge_sites(room, obj.mask, obj.label, *point))
                for obj, info in furniture]
    if pointed and not supports:
        ledge = _pointed_ledge(frame, scene, target_x, target_y)
        if ledge is not None:
            mask, box, site = ledge
            supports.append((None, mask, box, (site,)))
    _, base_y, base_height = camera_reference()
    if anchor is not None:
        base_y, base_height = anchor.y, anchor.standing_height_ratio
    dx = entry.seat_contact_px[0] - entry.anchor_px[0]
    dy = entry.seat_contact_px[1] - entry.anchor_px[1]
    angle = radians(rotation)
    dx, dy = cos(angle) * dx - sin(angle) * dy, sin(angle) * dx + cos(angle) * dy
    for label, mask, box, sites in supports:
        blockers = np.zeros_like(scene.person) if scene.fixed_camera else scene.person.copy()
        for other in scene.objects:
            if other.mask is not mask:
                np.maximum(blockers, other.mask, out=blockers)
        best, best_score = None, float("inf")
        for x, y in sites:
            # Sample the support, excluding floor and background on either side of its lip.
            sample_h, sample_w = max(3, round(.012 * h)), max(3, round(.012 * w))
            patch = np.s_[max(0, y - sample_h):max(1, y), max(0, x - sample_w):min(w, x + sample_w + 1)]
            sample = scene.inverse_depth[patch][mask[patch] > .5]
            if label is None and len(sample) < 3:
                patch = np.s_[y:min(h, y + sample_h + 1), max(0, x - sample_w):min(w, x + sample_w + 1)]
                sample = scene.inverse_depth[patch][mask[patch] > .5]
            sample = sample[np.isfinite(sample)]
            if len(sample) < 3:
                continue
            rim_depth = float(np.median(sample))
            ground_y = box[3]
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
            if label is None and (ground_y - y / h < .08 * size or foot_y > ground_y + .03):
                continue
            # Reserve space above the lip for the face, and below it for the legs.
            half_width = max(3, round(.135 * size * h))
            occupied = np.s_[max(0, round(y - .49 * size * h)):min(h, round(y + .30 * size * h) + 1),
                              max(0, x - half_width):min(w, x + half_width + 1)]
            nearer = scene.inverse_depth[occupied] >= rim_depth - .01
            blocked = nearer & (blockers[occupied] > .5)
            if scene.fixed_camera:
                blocked |= scene.person_in_front(ground_y)[occupied] > .5
            obstruction = float(np.mean(blocked))
            if obstruction > .5:
                continue
            preferred_x = target_x if pointed else (box[0] + box[2]) / 2
            score = obstruction + .28 * abs(x / w - preferred_x) + max(0, foot_y - 1)
            if pointed:
                score += .4 * np.hypot(target_x - x / w, target_y - y / h)
            if score < best_score:
                best_score = score
                place = Placement(foot_x, foot_y, anchor.scale if anchor else 1.0, size,
                                  anchor.framing if anchor else "", "桌沿坐姿" if label == "dining table" else "边缘坐姿")
                best = TableSeat(place, (x / w, y / h), (x / w, min(2, ground_y + .04)), tuple(box),
                                 ground_y, label, rim_depth)
        if best is not None:
            return best
    return None


table_seat = find_seat
