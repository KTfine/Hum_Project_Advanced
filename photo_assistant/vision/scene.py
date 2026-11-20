"""Keep person, furniture and depth measurements tied to a single camera exposure."""

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from .ground import GroundProjection, fit_ground
from .scene_objects import SceneObject, SceneObjectEstimator
from .segmentation import PersonSegmenter


def _confirmed_people(person, boxes):
    """Keep soft silhouettes connected to a person box, including limbs outside it."""
    h, w = person.shape
    _, components = cv2.connectedComponents((person > .05).astype(np.uint8))
    keep = np.zeros(int(components.max()) + 1, bool)
    for x0, y0, x1, y1 in boxes:
        if y1 - y0 < .09 * h or (x1 - x0) > 1.6 * (y1 - y0):
            continue
        core = person[y0:y1, x0:x1] >= .5
        if np.count_nonzero(core) >= .1 * core.size:
            keep[components[y0:y1, x0:x1][core]] = True
    keep[0] = False
    return person * keep[components]


@dataclass(frozen=True)
class SceneAnalysis:
    person: np.ndarray
    objects: tuple[SceneObject, ...]
    inverse_depth: np.ndarray
    ground: GroundProjection | None
    object_cover: np.ndarray
    descriptions: list[dict]
    person_cover: np.ndarray | None = None

    @property
    def foreground_person(self):
        return self.person_cover if self.person_cover is not None else self.person

    def depth_visibility(self, x, y):
        # Both terms are in this exposure's relative inverse-depth coordinates.
        delta = self.inverse_depth - self.ground.depth_at(x, y)
        return np.clip(delta / 0.016 + 0.5, 0.0, 1.0)

    def _ground_contact(self, obj, ink):
        """Verify thin supports against the visible floor immediately beyond them."""
        if obj.label not in ("chair", "dining table") or self.ground is None:
            return None
        h = ink.shape[0]
        left, top, width, height = cv2.boundingRect(ink.astype(np.uint8))
        bottom = top + height
        if height < .10 * h or bottom >= .98 * h or width == 0:
            return None
        feet = ink[bottom - max(1, round(.12 * height)):bottom, left:left + width].any(axis=0)
        if not 0 < feet.sum() <= .55 * width:
            return None
        foot_cols = np.flatnonzero(ink[bottom - max(1, round(.02 * height)):bottom].any(axis=0))
        end = min(h, bottom + max(3, round(.012 * h)))
        floor = np.ix_(np.arange(bottom + 1, end), foot_cols)
        clear = (self.object_cover[floor] < .2) & (self.foreground_person[floor] < .2)
        samples = self.inverse_depth[floor][clear]
        if samples.size >= 6 and np.percentile(self.inverse_depth[ink], 90) <= np.median(samples):
            return (bottom + .010 * h) / h
        return None

    @cached_property
    def furniture_fronts(self):
        """Ground ordinate in front of each supported instance, when measurable."""
        masks = [obj.mask >= .5 for obj in self.objects]
        fronts = [self._ground_contact(obj, mask) for obj, mask in zip(self.objects, masks)]
        grounded = [i for i, front in enumerate(fronts) if front is not None]
        # The detector sometimes reports the same table twice, once with feet
        # and once as only a tabletop. A weaker overlapping instance cannot
        # put that same surface back in front of the character.
        for i, (obj, ink) in enumerate(zip(self.objects, masks)):
            if fronts[i] is not None:
                continue
            area = np.count_nonzero(ink)
            if area == 0:
                continue
            for j in grounded:
                owner = self.objects[j]
                if owner.label != obj.label or owner.confidence <= obj.confidence:
                    continue
                if np.count_nonzero(ink & masks[j]) >= .82 * area:
                    fronts[i] = fronts[j]
                    break
        return tuple(fronts)

    def furniture_cover(self, x, y, closer=None):
        """Return the actual furniture silhouettes that can cover this ground point."""
        if self.ground is None:
            return self.object_cover
        if closer is None:
            closer = self.depth_visibility(x, y)
        retained = [obj.mask for obj, front in zip(self.objects, self.furniture_fronts)
                    if front is None or y < front]
        if len(retained) == len(self.objects):
            return self.object_cover * closer
        if not retained:
            return np.zeros_like(self.object_cover)
        return np.maximum.reduce(retained) * closer


class SceneSegmenter:
    def __init__(self):
        self.person = PersonSegmenter()
        self.estimator = SceneObjectEstimator()

    def segment(self, frame):
        person = self.person.segment(frame)
        objects, depth = self.estimator.infer(frame)
        h, w = person.shape
        cover = np.zeros_like(person)
        descriptions = []
        for obj in objects:
            np.maximum(cover, obj.mask, out=cover)
            x, y, width, height = cv2.boundingRect((obj.mask >= 0.5).astype(np.uint8))
            box = [round(v, 3) for v in (x / w, y / h, (x + width) / w, (y + height) / h)]
            descriptions.append(dict(label=obj.label, box=box, confidence=round(obj.confidence, 3),
                                     center_x=round((x + width / 2) / w, 3), foot_y=box[3]))
        ground = fit_ground(frame, depth, (person, cover))
        # Empty rooms can produce a human-shaped PPHumanSeg response on chair
        # backs or clothing. In a wide furniture shot, check for corroborating
        # person pixels before using that response as a body-size reference.
        if ground is not None and objects and person.max() >= 0.5:
            confirmed = _confirmed_people(person, self.estimator.person_boxes)
            if not np.array_equal(confirmed, person):
                person = confirmed
                ground = fit_ground(frame, depth, (person, cover))
        if ground is not None:
            for obj, info in zip(objects, descriptions):
                x = info["center_x"]
                baseline = ground.depth_at(x, 0)
                slope = ground.depth_at(x, 1) - baseline
                if slope <= 0:
                    continue
                far, near = np.percentile(depth[obj.mask >= 0.5], (10, 90))
                # The visible bottom of a table is often its rim. Project the
                # two sides onto the inferred floor to guide a requested move.
                info["behind_y"] = round(float(np.clip((far - 0.035 - baseline) / slope, 0, 2)), 3)
                info["front_y"] = round(float(np.clip((near + 0.035 - baseline) / slope, 0, 2)), 3)
        # PPHumanSeg measures the subject for the existing body calibration.
        # A second silhouette can fill opaque trousers or sleeves that it misses.
        person_cover = None
        for (x0, y0, x1, y1), patch in self.estimator.person_patches:
            shape = patch >= .5
            support = person[y0:y1, x0:x1] >= .5
            if np.count_nonzero(shape & support) < max(16, .35 * np.count_nonzero(shape)):
                continue
            if person_cover is None:
                person_cover = person.copy()
            region = person_cover[y0:y1, x0:x1]
            np.maximum(region, patch, out=region)
        return SceneAnalysis(person, objects, depth, ground, cover, descriptions, person_cover)

    def close(self):
        self.person.close()
        self.estimator = None
