"""Place a visible sprite face beside a person while part of its body stays behind them."""

import cv2
import numpy as np

from .framing import Placement
from .ground import projected_height
from .head import measure_face
from .perspective import PORTRAIT_CHEEK_RATIO, PersonSilhouettes, matched_standing_height_px


# Fraction of projected opaque pixels below the face that the person should
# hide for a full-length shoulder peek. The torso carries most of the weight.
PEEK_BODY_OVERLAP = .65


def peek_placement(frame, person, masks, scene, sprite, entry, anchor, horizon, current=None):
    if person is None or entry.face_box_px is None:
        return None
    h, w = person.shape
    silhouettes = PersonSilhouettes.from_mask(person)
    if silhouettes is None:
        return None
    head = measure_face(frame, person, silhouettes=silhouettes)
    ref = silhouettes.reference_at(head.center_x / w if head else .5, head.center_y / h if head else .9)
    if ref is None:
        return None
    top = round((ref.foot_y - ref.height) * h)
    face_y = head.center_y if head else top + ref.height * h * .085
    head_rows, head_cols = np.nonzero(person[max(0, int(face_y) - 3):max(1, int(face_y) + 4)] > .5)
    if not head_cols.size:
        return None
    cx = head.center_x if head else float(np.median(head_cols))
    cheek = head.width if head else max(5, np.percentile(head_cols, 85) - np.percentile(head_cols, 15))
    l, t, r, b = entry.face_box_px
    ax, ay = entry.anchor_px
    if anchor is not None and anchor.framing:
        body = anchor.standing_height_ratio * h
    else:
        body = (matched_standing_height_px(cheek, r-l, entry.standing_height_px, cheek_ratio=PORTRAIT_CHEEK_RATIO)
                if ref.estimated else .93 * ref.height * h)
    if ref.estimated:
        # Preserve the calibrated portrait size. Keep cropped feet below the
        # frame where possible while bringing the face level with the shoulder.
        foot = (face_y + .9 * cheek - ((t+b)/2 - ay) * body / entry.standing_height_px) / h
        options = [(max(foot, 1.055), body), (max(foot, 1.015), body), (foot, body)]
    else:
        options = []
        for setback in (.045, .065, .08, .092, .10, .11):
            y = ref.foot_y - setback * ref.height
            size = projected_height(anchor.y if anchor and anchor.framing else ref.foot_y,
                                    body, y, horizon)
            options.append((y, size))

    # Sample opaque sprite pixels in art coordinates once, then check actual
    # projected positions against the foreground, including the face itself.
    step = max(1, max(sprite.shape[:2]) // 150)
    sy, sx = np.nonzero(sprite[::step, ::step, 3] >= 155)
    sx, sy = sx * step, sy * step
    face = (sx >= l) & (sx < r) & (sy >= t) & (sy < b)
    if not face.any():
        return None
    below_face = sy > b
    torso = below_face & (sy <= b + .34 * entry.standing_height_px)
    person_cover = scene.foreground_person if scene is not None else person
    shoe_zone = None
    if not ref.estimated:
        # A tilted illustration can put either shoe well away from the foot
        # anchor. Protect the photographed shoes including their soft edges.
        body_px = ref.height * h
        sole_y = ref.foot_y * h
        shoe_top = max(0, int(sole_y - .08 * body_px))
        shoe_bottom = min(h, int(np.ceil(sole_y + .025 * body_px)))
        shoes = np.zeros((h, w), np.uint8)
        shoes[shoe_top:shoe_bottom] = person_cover[shoe_top:shoe_bottom] >= .25
        margin = max(2, round(.020 * body_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        shoe_zone = cv2.dilate(shoes, kernel)
        # Sample the sprite's opaque ankle-to-sole band before occlusion:
        # hiding a shoe behind the person's shoe still reads as stepping on it.
        sprite_shoes = sy >= ay - .11 * entry.standing_height_px
    base = person_cover.copy()
    for mask in masks:
        np.maximum(base, mask, out=base)
    best, chosen = -np.inf, None
    candidates = [(y, size, None, i) for i, (y, size) in enumerate(options)]
    if current is not None and (ref.estimated or options[-1][0] - .5/h <= current.y <= options[0][0] + .5/h):
        nearest = min(range(len(options)), key=lambda i: abs(options[i][0] - current.y))
        candidates.insert(0, (current.y, current.standing_height_ratio * h, current.x * w, nearest))
    for y, size, settled_x, i in candidates:
        zoom = size / entry.standing_height_px
        cover = base
        if scene is not None and scene.objects:
            cover = np.maximum(base, scene.furniture_cover(cx/w, y))
        yy = np.rint(y*h + (sy-ay)*zoom).astype(int)
        valid_y = (yy >= 0) & (yy < h)
        yy = np.clip(yy, 0, h-1)
        offsets = np.linspace(cx-2.8*cheek, cx+2.8*cheek, 81) - ((l+r)/2 - ax)*zoom
        for foot_x in offsets if settled_x is None else (settled_x,):
            xx = np.rint(foot_x + (sx-ax)*zoom).astype(int)
            inside = valid_y & (xx >= 0) & (xx < w)
            xx = np.clip(xx, 0, w-1)
            if shoe_zone is not None:
                feet = inside & sprite_shoes
                if not feet.any() or np.any(shoe_zone[yy[feet], xx[feet]]):
                    continue
            shown = inside & (cover[yy, xx] < .15)
            exposed = shown[face].mean()
            blocked = inside & (person_cover[yy, xx] > .75)
            overlap = blocked.sum() / max(1, inside.sum())
            if exposed < .97 or overlap < .12 or shown.mean() < .15:
                continue
            if ref.estimated:
                overlap_error = abs(overlap - .34)
            else:
                body_hidden = blocked[below_face].sum() / max(1, inside[below_face].sum())
                torso_hidden = blocked[torso].sum() / max(1, inside[torso].sum())
                overlap_error = (.7 * abs(torso_hidden - PEEK_BODY_OVERLAP)
                                 + .3 * abs(body_hidden - PEEK_BODY_OVERLAP))
            if settled_x is not None and ref.estimated:
                return current
            side_crop = (valid_y & ~inside).sum() / max(1, valid_y.sum())
            score = exposed - overlap_error - (.04 if ref.estimated else .018)*i - 2.0*side_crop
            if score > best:
                best = score
                chosen = current if settled_x is not None else Placement(
                    float(foot_x/w), float(y), anchor.scale if anchor else 1.,
                    float(size/h), "下半身出框" if ref.estimated else "全身", "肩后探身")
    return chosen
