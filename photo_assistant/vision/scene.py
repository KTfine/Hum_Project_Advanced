"""Analyze a full exposure or track people against a calibrated fixed camera."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import cv2
import numpy as np

from .ground import GroundProjection, fit_ground
from .perspective import silhouette_reference
from .scene_objects import SceneObject, SceneObjectEstimator
from .seat import TableSeat
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
class _RejectedRegion:
    rows: slice
    cols: slice
    silhouette: np.ndarray
    area: int


def _rejected_regions(raw, confirmed):
    """Remember whole PPHumanSeg components rejected by same-exposure person detection."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(((raw > 0) & (confirmed == 0)).astype(np.uint8))
    regions = []
    for label, (x, y, w, h, area) in enumerate(stats[1:], start=1):
        rows, cols = slice(y, y + h), slice(x, x + w)
        silhouette = labels[rows, cols] == label
        if np.any(raw[rows, cols][silhouette] >= .5):
            regions.append(_RejectedRegion(rows, cols, silhouette, int(area)))
    return tuple(regions)


@dataclass(frozen=True)
class CameraCalibration:
    frame: np.ndarray
    background: SceneAnalysis
    rejected_regions: tuple[_RejectedRegion, ...] = ()

    @property
    def person(self):
        return self.background.foreground_person

    def current_person(self, frame, person):
        """Suppress a chair-like silhouette only while its full shape and appearance still match."""
        if not self.rejected_regions:
            return person
        count, labels, stats, _ = cv2.connectedComponentsWithStats((person > 0).astype(np.uint8))
        keep = np.ones(count, bool)
        for region in self.rejected_regions:
            inside = labels[region.rows, region.cols][region.silhouette]
            overlap = np.bincount(inside, minlength=count)
            overlap[0] = 0
            label = int(overlap.argmax())
            inter = overlap[label]
            area = stats[label, cv2.CC_STAT_AREA]
            if label == 0 or inter < .88 * (area + region.area - inter):
                continue
            x, y, w, h = stats[label, :4]
            roi = np.s_[y:y + h, x:x + w]
            delta = cv2.absdiff(frame[roi], self.frame[roi])
            changed = (delta.max(axis=2) > 20) & (labels[roi] == label)
            if np.count_nonzero(changed) <= .015 * area:
                keep[label] = False
        return person if keep.all() else person * keep[labels]


@dataclass(frozen=True)
class SceneAnalysis:
    person: np.ndarray
    objects: tuple[SceneObject, ...]
    inverse_depth: np.ndarray
    ground: GroundProjection | None
    object_cover: np.ndarray
    descriptions: list[dict]
    person_cover: np.ndarray | None = None
    calibration: CameraCalibration | None = field(default=None, repr=False, compare=False)
    _projection_cache: tuple[TableSeat | tuple[float, float, bool], tuple[np.ndarray, np.ndarray]] | None = field(
        default=None, init=False, repr=False, compare=False)

    @property
    def foreground_person(self):
        return self.person_cover if self.person_cover is not None else self.person

    @property
    def fixed_camera(self):
        return self.calibration is not None

    @cached_property
    def _person_feet(self):
        """Estimate each complete silhouette's depth from its feet in the current image."""
        person = self.foreground_person
        _, labels, stats, _ = cv2.connectedComponentsWithStats((person > 0).astype(np.uint8))
        feet = np.full(len(stats), np.inf)
        for i in range(1, len(stats)):
            ref = silhouette_reference(labels, i, stats[i], person.shape[0])
            # Detached limbs and cropped subjects cannot establish a farther floor position.
            if ref is not None and not ref.estimated:
                feet[i] = ref.foot_y
        return labels, feet

    def person_in_front(self, y):
        labels, feet = self._person_feet
        in_front = feet + .02 >= y
        if in_front.all():
            return self.foreground_person
        return self.foreground_person * in_front[labels]

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

    def projected_foreground(self, x, y, *, depth_order_person: bool = False, seat: TableSeat | None = None):
        """Reuse the occluders for a stationary avatar until the next analyzed exposure."""
        key = seat if seat is not None else (x, y, depth_order_person)
        cached = self._projection_cache
        if cached is not None and key == cached[0]:
            return cached[1]
        if self.calibration is not None:
            furniture, _ = self.calibration.background.projected_foreground(x, y, seat=seat)
            person = self.foreground_person
            if seat is not None or depth_order_person:
                person = self.person_in_front(seat.ground_y if seat is not None else y)
            masks = furniture, person
        elif seat is not None:
            masks = seat.foreground(self)
        elif self.ground is not None:
            closer = self.depth_visibility(x, y)
            furniture = self.furniture_cover(x, y, closer)
            person = self.foreground_person
            masks = furniture, person * closer if depth_order_person else person
        else:
            masks = self.object_cover, self.foreground_person
        # Publish the key with the masks; preview and shutter can request different points.
        object.__setattr__(self, "_projection_cache", (key, masks))
        return masks


def _exclude_subject(objects, depth, subject):
    """Remove a calibration subject from the static masks and approximate the depth behind them."""
    if not np.any(subject > .1):
        return objects, depth
    radius = max(1, round(max(depth.shape) / 256))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    hole = cv2.dilate((subject > .1).astype(np.uint8), kernel)
    visible = hole == 0
    remaining = []
    for obj in objects:
        mask = obj.mask * visible
        if np.any(mask >= .5):
            remaining.append(replace(obj, mask=mask))
    # MiDaS ran at 256px. Fill on that grid, keeping measured background pixels intact.
    h, w = depth.shape
    ratio = min(1., 256 / max(h, w))
    size = max(1, round(w * ratio)), max(1, round(h * ratio))
    small_depth = cv2.resize(depth, size, interpolation=cv2.INTER_AREA)
    hidden = cv2.resize(hole.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
    filled = (np.zeros_like(small_depth) if hidden.all() else
              cv2.inpaint(small_depth, hidden.astype(np.uint8), 3, cv2.INPAINT_NS))
    return tuple(remaining), np.where(visible, depth, cv2.resize(filled, (w, h)))


class SceneSegmenter:
    def __init__(self, *, fixed_camera=False):
        self.fixed_camera = fixed_camera
        self.calibration: CameraCalibration | None = None
        self.person = PersonSegmenter()
        self.estimator = SceneObjectEstimator()

    def reset_scene(self):
        self.calibration = None

    def segment(self, frame):
        person = self.person.segment(frame)
        calibration = self.calibration
        if calibration is not None and frame.shape == calibration.frame.shape:
            person = calibration.current_person(frame, person)
            return replace(calibration.background, person=person, person_cover=None, calibration=calibration)
        objects, depth = self.estimator.infer(frame)
        h, w = person.shape
        cover = np.zeros_like(person)
        for obj in objects:
            np.maximum(cover, obj.mask, out=cover)
        ground = fit_ground(frame, depth, (person, cover))
        rejected_regions = ()
        # Empty rooms can produce a human-shaped PPHumanSeg response on chair
        # backs or clothing. In a wide furniture shot, check for corroborating
        # person pixels before using that response as a body-size reference.
        if ground is not None and objects and person.max() >= 0.5:
            confirmed = _confirmed_people(person, self.estimator.person_boxes)
            if not np.array_equal(confirmed, person):
                if self.fixed_camera:
                    rejected_regions = _rejected_regions(person, confirmed)
                person = confirmed
                ground = fit_ground(frame, depth, (person, cover))
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
        if self.fixed_camera:
            objects, depth = _exclude_subject(objects, depth, person if person_cover is None else person_cover)
            cover.fill(0)
            for obj in objects:
                np.maximum(cover, obj.mask, out=cover)
        descriptions = []
        for obj in objects:
            ink = obj.mask >= .5
            x, y, width, height = cv2.boundingRect(ink.astype(np.uint8))
            box = [round(v, 3) for v in (x / w, y / h, (x + width) / w, (y + height) / h)]
            info = dict(label=obj.label, box=box, confidence=round(obj.confidence, 3),
                        center_x=round((x + width / 2) / w, 3), foot_y=box[3])
            descriptions.append(info)
            if ground is not None:
                baseline = ground.depth_at(info["center_x"], 0)
                slope = ground.depth_at(info["center_x"], 1) - baseline
                if slope > 0:
                    far, near = np.percentile(depth[ink], (10, 90))
                    info["behind_y"] = round(float(np.clip((far - .035 - baseline) / slope, 0, 2)), 3)
                    info["front_y"] = round(float(np.clip((near + .035 - baseline) / slope, 0, 2)), 3)
        scene = SceneAnalysis(person, objects, depth, ground, cover, descriptions, person_cover)
        if self.fixed_camera:
            reference = frame.copy() if frame.flags.writeable else frame
            reference.setflags(write=False)
            self.calibration = CameraCalibration(reference, scene, rejected_regions)
            return replace(scene, calibration=self.calibration)
        return scene

    def close(self):
        self.reset_scene()
        self.person.close()
        self.estimator = None
