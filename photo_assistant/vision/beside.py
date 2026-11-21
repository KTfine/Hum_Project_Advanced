"""Find the companion's spot beside the subject on entry or a follow request."""

from dataclasses import dataclass

import cv2
import numpy as np

from .head import FaceReference
from .perspective import (
    PORTRAIT_CHEEK_RATIO, PORTRAIT_FACE_DROP, PersonReference,
    matched_standing_height_px, person_reference, silhouette_reference,
    standing_height_px as project_height,
)


@dataclass(frozen=True)
class SidePlacement:
    x: float
    y: float
    scale: float


def _stand_beside(person_mask, foreground_masks, sprite, anchor, standing_height, reference,
                  face_box, head, preferred_x, scene):
    """Place both soles on the same ground plane and choose the nearest clear horizontal slot."""
    height, width = person_mask.shape
    step = max(1, (max(height, width) + 255) // 256)
    person = (person_mask[::step, ::step] >= 0.5).astype(np.uint8)
    rows, cols = person.shape
    y = reference.foot_y - 0.02 * reference.height
    foot = y * height / step
    matched = None
    if head is not None and head.width > 0 and face_box is not None and face_box[2] > face_box[0]:
        face_width = face_box[2] - face_box[0]
        matched = matched_standing_height_px(head.width, face_width, standing_height)
    base = project_height(reference, height, y, fallback_height=0.72 * height, matched_height_px=matched)
    size = base
    if matched is not None:
        # The drawing and the real person have different head-to-body proportions.
        # Approach the desired face height without making the illustrated face oversized.
        face_to_foot = anchor[1] - (face_box[1] + face_box[3]) / 2
        if face_to_foot > 0:
            ideal = (y * height - head.center_y - 0.5 * head.width) * standing_height / face_to_foot
            size = float(np.clip(ideal, matched * 0.97, matched * 1.18))
    zoom = size / (standing_height * step)

    # The template spans the scene height, so correlation scans only along x.
    sample = max(1, max(sprite.shape[:2]) // 384)
    ink = (sprite[::sample, ::sample, 3] >= 64).astype(np.uint8)
    left, top, w, h = cv2.boundingRect(ink)
    if not w or not h:
        return None
    art_width = max(1, int(np.ceil(w * sample * zoom)))
    if art_width > cols:
        return None
    top_y = foot + (top * sample - anchor[1]) * zoom
    if top_y < -0.03 * size / step or top_y + h * sample * zoom > rows + 1:
        return None
    matrix = np.float32([[sample * zoom, 0, 0], [0, sample * zoom, top_y]])
    silhouette = cv2.warpAffine(ink[top:top+h, left:left+w], matrix, (art_width, rows),
                                flags=cv2.INTER_NEAREST)
    if not silhouette.any():
        return None
    clearance = max(1, round(0.012 * reference.height * height / step))
    padded = cv2.dilate(person, np.ones((2 * clearance + 1,) * 2, np.uint8))
    foot_column = (anchor[0] - left * sample) * zoom
    face_column = ((face_box[0] + face_box[2]) / 2 - left * sample) * zoom if face_box else foot_column
    columns = np.arange(art_width)
    # Use the trunk and the feet to set spacing. Wings and trailing hair are
    # often wider than the two shoulders combined; a little contact is natural.
    core_radius = 0.13 * size / step
    core = silhouette * ((columns >= min(foot_column, face_column) - core_radius)
                         & (columns <= max(foot_column, face_column) + core_radius))
    core_overlap = cv2.matchTemplate(padded, core, cv2.TM_CCORR)[0]
    overlap = cv2.matchTemplate(person, silhouette, cv2.TM_CCORR)[0]
    clear = (core_overlap < 0.5) & (overlap <= 0.045 * np.count_nonzero(silhouette))
    head_and_shoes = person.copy()
    top = (reference.foot_y - reference.height) * height / step
    head_bottom = int(np.ceil(top + 0.18 * reference.height * height / step))
    shoes_top = int((reference.foot_y - 0.08 * reference.height) * height / step)
    head_and_shoes[max(0, head_bottom):max(0, shoes_top)] = 0
    clear &= cv2.matchTemplate(head_and_shoes, silhouette, cv2.TM_CCORR)[0] < 0.5
    furniture_masks = {id(obj.mask) for obj in scene.objects} if scene is not None else set()
    furniture = np.zeros_like(person)
    for mask in foreground_masks:
        if mask is person_mask:
            continue
        obstacle = (mask[::step, ::step] >= 0.5).astype(np.uint8)
        if id(mask) in furniture_masks:
            furniture |= obstacle
        else:
            clear &= cv2.matchTemplate(obstacle, silhouette, cv2.TM_CCORR)[0] < 0.5
    covered = np.zeros_like(overlap)
    if furniture.any():
        # GroundProjection assumes an unrolled camera. All candidates at this y
        # share the same floor depth, including a slot in front of the table.
        if scene.ground is not None:
            furniture &= scene.furniture_cover(.5, y)[::step, ::step] >= .5
        covered = cv2.matchTemplate(furniture, silhouette, cv2.TM_CCORR)[0] / silhouette.sum()
        clear &= covered < .18
        row = np.arange(rows)[:, None]
        if face_box is not None:
            fx0, fy0, fx1, fy1 = face_box
            face_ink = silhouette * ((columns >= (fx0 - left * sample) * zoom)
                                     & (columns <= (fx1 - left * sample) * zoom)
                                     & (row >= foot + (fy0 - anchor[1]) * zoom)
                                     & (row <= foot + (fy1 - anchor[1]) * zoom))
            clear &= cv2.matchTemplate(furniture, face_ink, cv2.TM_CCORR)[0] <= .04 * face_ink.sum() + .1
        shoes = core * (row >= foot - .05 * size / step)
        clear &= cv2.matchTemplate(furniture, shoes, cv2.TM_CCORR)[0] <= .08 * shoes.sum() + .1
    slots = np.flatnonzero(clear)
    if not slots.size:
        return None
    anchors = slots + foot_column
    if head is not None:
        person_x = head.center_x / step
    else:
        _, _, stats, centers = cv2.connectedComponentsWithStats(person)
        if len(stats) < 2:
            return None
        # The normalized foot and standing height identify this particular person.
        error = (np.abs(stats[1:, 1] + stats[1:, 3] - reference.foot_y * height / step)
                 + np.abs(stats[1:, 3] - reference.height * height / step))
        person_x = centers[1 + error.argmin(), 0]
    distance = np.abs(anchors - person_x)
    # A clear slot across the room no longer reads as standing beside this person.
    feasible = distance <= 0.65 * reference.height * height / step
    if not feasible.any():
        return None
    distance = distance + covered[slots] * reference.height * height / step
    if preferred_x is not None:
        side = preferred_x * width / step - person_x
        if abs(side) > 0.025 * cols:
            distance = distance + cols * ((anchors - person_x) * side < 0)
    best = np.argmin(np.where(feasible, distance, np.inf))
    x = float(anchors[best] * step / width)
    if matched is None:
        # The renderer chooses its unmeasured ground reference at the character's x.
        render_ref = person_reference(person_mask, x, y)
        base = project_height(render_ref, height, y, fallback_height=0.72 * height)
    scale = size / base
    return SidePlacement(x, y, scale) if 0.1 <= scale <= 3 else None


def _sample_art(sprite, anchor, standing_height, face_box, torso_rows=12):
    """Sample opaque face/body points and horizontal cross sections of the upper body."""
    step = max(1, max(sprite.shape[:2]) // 140)
    ys, xs = np.nonzero(sprite[::step, ::step, 3] >= 128)
    if not len(xs):
        return None
    xs, ys = xs * step, ys * step
    if face_box is None:
        face_top = ys.min() + 0.055 * standing_height
        chin = ys.min() + 0.16 * standing_height
        head = (ys >= face_top) & (ys <= chin)
        if not head.any():
            return None
        left, right = np.percentile(xs[head], [18, 82])
    else:
        left, face_top, right, chin = face_box
    face = (ys >= face_top) & (ys <= chin) & (xs >= left) & (xs <= right)
    body = ys > chin
    if not face.any() or not body.any():
        return None

    def offsets(x, y):
        return (np.asarray(x) - anchor[0]) / standing_height, (np.asarray(y) - anchor[1]) / standing_height

    torso_x, torso_y = [], []
    for row in np.linspace(chin + 0.035 * standing_height, chin + 0.34 * standing_height, torso_rows):
        pixels = xs[np.abs(ys - row) <= step / 2]
        if len(pixels):
            torso_x.append(np.percentile(pixels, np.linspace(8, 92, 11)))
            torso_y.append(np.full(11, row))
    if len(torso_x) < 6:
        return None
    sample = max(1, int(body.sum()) // 300)
    head = ys <= chin
    head_sample = max(1, int(head.sum()) // 180)
    return (offsets(xs[face], ys[face]), offsets(torso_x, torso_y),
            offsets(xs[body][::sample], ys[body][::sample]),
            offsets(xs[head][::head_sample], ys[head][::head_sample]))


def _at_positions(region, anchors, foot, size, cover, target):
    """Return camera inclusion, actual visibility and occlusion by this specific target."""
    dx, dy = region
    xs = np.rint(anchors.reshape((-1,) + (1,) * dx.ndim) + dx * size).astype(int)
    ys = np.rint(foot + dy * size).astype(int)
    h, w = cover.shape
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xi, yi = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    return inside, inside & ~cover[yi, xi], inside & target[yi, xi]


def _fit(art, target, cover, size, foot, flank_top, cropped=False, portrait=False, *,
         preferred_x=None, subject_x=None, prefer_clear_face=False):
    face, torso, body, head = art
    w = cover.shape[1]
    # Search by projected face center, allowing a leaning sprite's foot off camera.
    anchors = np.arange(0, w, max(1.0, size / 120)) - np.median(face[0]) * size
    inside, visible, _ = _at_positions(face, anchors, foot, size, cover, target)
    clear_face = (inside.mean(axis=1) >= 0.98) & (visible.mean(axis=1) >= 0.84)
    anchors = anchors[clear_face]
    if not len(anchors):
        return None
    face_visibility = visible.mean(axis=1)[clear_face]
    head_inside, _, _ = _at_positions(head, anchors, foot, size, cover, target)
    head_fit = head_inside.mean(axis=1)

    inside, visible, hidden = _at_positions(torso, anchors, foot, size, cover, target)
    row_y = foot + torso[1][:, 0] * size
    if cropped:
        # A portrait only shows the head and some upper torso. Count actual
        # cross sections above the frame edge; cropping alone is never occlusion.
        in_view = (row_y >= 0) & (row_y < cover.shape[0])
        if in_view.sum() < (2 if portrait else 3):
            return None
        inside, visible, hidden = inside[:, in_view], visible[:, in_view], hidden[:, in_view]
        row_y = row_y[in_view]
    # Each row measures across the torso, so the lower half disappearing behind
    # a horizontal ledge cannot masquerade as half the body peeking out sideways.
    row_shown = visible.mean(axis=2)
    row_hidden = hidden.mean(axis=2)
    at_flank = row_y >= flank_top
    split_rows = ((row_shown >= 0.12) & (row_hidden >= 0.12) & at_flank).mean(axis=1)
    shown = row_shown.mean(axis=1)
    lower = row_shown[:, row_shown.shape[1] // 2:].mean(axis=1)
    # The face must fit even in close-ups, but horns and the outside sleeve
    # may legitimately cross an image edge. Require real torso occlusion.
    min_head, min_torso = (0.72, 0.60) if portrait else (0.90, 0.90)
    feasible = ((head_fit >= min_head) & (inside.mean(axis=(1, 2)) >= min_torso) & (split_rows >= 0.42)
                & (shown >= 0.31) & (shown <= 0.70) & (lower >= 0.19)
                & (row_hidden.mean(axis=1) >= 0.25))
    if not feasible.any():
        return None

    inside, visible, hidden = _at_positions(body, anchors, foot, size, cover, target)
    count = np.maximum(inside.sum(axis=1), 1)
    body_shown = visible.sum(axis=1) / count
    feasible &= (body_shown >= 0.24) & (body_shown <= 0.76) & (hidden.sum(axis=1) / count >= 0.20)
    if not feasible.any():
        return None
    score = (1 - abs(shown - 0.5) - 0.60 * abs(body_shown - 0.5)
             + 0.15 * split_rows + 0.15 * face_visibility + 0.5 * head_fit)
    if prefer_clear_face:
        score -= .5 * (face_visibility < .98)
    if preferred_x is not None:
        # Rank only physically feasible poses. A request to stay on one side
        # follows that flank, even if the person's silhouette changes shape.
        score -= 0.5 * np.minimum(1.0, np.abs(anchors - preferred_x) / w)
        if abs(preferred_x - subject_x) > 0.025 * w:
            face_centers = anchors + np.median(face[0]) * size
            wrong_side = (face_centers - subject_x) * (preferred_x - subject_x) < 0
            score -= 1.1 * wrong_side
    best = np.argmax(np.where(feasible, score, -np.inf))
    return float(score[best]), float(anchors[best])


def _flank_top(target, top, height):
    """Locate the torso below the widening from head to shoulders, including bust crops."""
    widths = target[top:top + height].sum(axis=1)
    wide = np.flatnonzero(widths >= 0.75 * np.percentile(widths, 90))
    return top + max(0.23 * height, wide[0] + 0.05 * height)


def _compositions(reference, top, height, frame_height, torso_center):
    """Put the feet upstage, then project the standing height at that ground point."""
    if reference is not None:
        # Distance in the image is relative to human height, so the shoes
        # separate at every camera resolution and on both near and far people.
        for setback in (0.10, 0.13, 0.17):
            y = reference.foot_y - setback * reference.height
            size = project_height(reference, frame_height, y, fallback_height=0.72 * frame_height)
            yield size, y * frame_height
        return
    # An isolated obstacle has no known real-world height or ground depth.
    standing = min(frame_height * 1.02, height * 1.1)
    for depth in (1.0, 0.83, 0.67):
        size = standing * depth
        for fraction in (0.52, 0.43, 0.62, 0.34, 0.72):
            yield size, top + height * fraction - torso_center * size


def _portrait_compositions(head, size, face_y_offset, step):
    """Keep the companion's calibrated face size while trying nearby shoulder heights."""
    for face_drop in (PORTRAIT_FACE_DROP, 0.65, 1.0, 0.42):
        face_y = (head.center_y + face_drop * head.width) / step
        yield size, face_y - face_y_offset * size


def side_placement(
    person_mask: np.ndarray | None, sprite_bgra: np.ndarray, anchor_px: tuple[float, float],
    standing_height_px: float, *, foreground_masks: tuple[np.ndarray, ...] = (),
    face_box_px: tuple[float, float, float, float] | None = None,
    head: FaceReference | None = None,
    preferred_x: float | None = None,
    subject_reference: PersonReference | None = None,
    scene=None,
) -> SidePlacement | None:
    """Use adjacent floor space for full-body photos, and side occlusion for portraits."""
    if person_mask is not None and person_mask.ndim == 2 and person_mask.size:
        if subject_reference is None:
            height, width = person_mask.shape
            subject_x = head.center_x / width if head is not None else 0.5
            subject_y = head.center_y / height if head is not None else 0.95
            subject_reference = person_reference(person_mask, subject_x, subject_y)
        if subject_reference is not None and not subject_reference.estimated:
            return _stand_beside(person_mask, foreground_masks, sprite_bgra, anchor_px, standing_height_px,
                                 subject_reference, face_box_px, head, preferred_x, scene)
    else:
        subject_reference = None
    if not foreground_masks:
        return None
    shape = foreground_masks[0].shape
    if len(shape) != 2 or not min(shape):
        return None
    step = max(1, (max(shape) + 255) // 256)
    masks = [(m is person_mask, m[::step, ::step] >= 0.5) for m in foreground_masks]
    cover = np.logical_or.reduce([mask for _, mask in masks])
    if not cover.any() or cover.all():
        return None
    matched_height = None
    if head is not None and head.width > 0 and face_box_px is not None and face_box_px[2] > face_box_px[0]:
        face_width = face_box_px[2] - face_box_px[0]
        matched_height = matched_standing_height_px(head.width, face_width, standing_height_px)
        portrait_height = matched_standing_height_px(head.width, face_width, standing_height_px,
                                                    cheek_ratio=PORTRAIT_CHEEK_RATIO)
        face_y_offset = ((face_box_px[1] + face_box_px[3]) / 2 - anchor_px[1]) / standing_height_px
    art = _sample_art(sprite_bgra, anchor_px, standing_height_px, face_box_px,
                      torso_rows=32 if matched_height is not None else 12)
    if art is None:
        return None
    torso_center = np.mean(art[1][1])
    rows, cols = cover.shape
    frame_height = shape[0] / step
    portrait = matched_height is not None
    if portrait:
        face_col = int(np.clip(round(head.center_x / step), 0, cols - 1))
        face_row = int(np.clip(round(head.center_y / step), 0, rows - 1))
    people = []
    for person, mask in sorted(masks, key=lambda item: not item[0]):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for label in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
            left, top, width, height, area = map(int, stats[label])
            if height < max(8, rows * 0.16) or width < 3 or area < 0.003 * rows * cols:
                continue
            target = labels == label
            flank_top = _flank_top(target, top, height) if person else top
            center_x = (left + width / 2) * step / shape[1]
            if portrait:
                if person and not target[face_row, face_col]:
                    continue
                reference = subject_reference
                if person:
                    flank_top = min(flank_top, (head.center_y + 0.85 * head.width) / step)
            elif person:
                reference = silhouette_reference(labels, label, stats[label], frame_height)
                if reference is None:
                    continue
                if any(r.estimated and area < 0.22 * a for _, r, a in people):
                    continue
                people.append((center_x, reference, area))
            else:
                # An obstacle can still occlude the character when the person's
                # own flank is blocked. Retain that person's ground depth.
                reference = person_reference(person_mask, center_x, (top + height) / frame_height)
                if reference is None and people:
                    reference = min(people, key=lambda item: abs(item[0] - center_x))[1]
            best_score, placement = -np.inf, None
            if portrait:
                options = _portrait_compositions(head, portrait_height / step, face_y_offset, step)
                if reference is not None and scene is not None:
                    # When the subject's shoes are far out of frame, a face-matched
                    # sprite can inadvertently show both feet halfway up a wall.
                    # Keep the calibrated body size and crop the shoe band instead.
                    shoe_band = .10 * portrait_height / step
                    if reference.foot_y * frame_height >= frame_height + shoe_band:
                        options = dict.fromkeys((size_at_y, max(foot, frame_height + shoe_band))
                                                for size_at_y, foot in options)
            else:
                options = _compositions(reference, top, height, frame_height, torso_center)
            desired_x = None if preferred_x is None else preferred_x * shape[1] / step
            subject_x = head.center_x / step if portrait else left + width / 2
            for order, (size, foot) in enumerate(options):
                cropped = portrait or (reference is not None and reference.estimated and foot >= rows)
                if reference is not None and reference.estimated and reference.foot_y > 1.25 and not cropped:
                    continue
                result = _fit(art, target, cover, size, foot, flank_top, cropped, portrait,
                              preferred_x=desired_x, subject_x=subject_x,
                              prefer_clear_face=portrait and scene is not None)
                if result is None:
                    continue
                score, ax = result
                score -= 0.018 * order
                if portrait:
                    face_drop = ((foot + face_y_offset * size) * step - head.center_y) / head.width
                    score -= 0.16 * abs(face_drop - PORTRAIT_FACE_DROP)
                if score <= best_score:
                    continue
                x, y = float(ax * step / shape[1]), float(foot * step / shape[0])
                ref = subject_reference if portrait else person_reference(person_mask, x, y)
                if ref is not None and not portrait and y > ref.foot_y - 0.085 * ref.height:
                    continue
                base = project_height(ref, shape[0], y, fallback_height=0.72 * shape[0],
                                      matched_height_px=matched_height)
                scale = float(size * step / base)
                if -2 <= x <= 3 and -2 <= y <= 3 and 0.1 <= scale <= 3:
                    best_score, placement = score, SidePlacement(x, y, scale)
            if placement is not None:
                return placement
    return None
