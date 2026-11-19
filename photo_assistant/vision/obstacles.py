"""Find a substantial panel or pillar with supported straight boundaries in a camera frame."""

import cv2
import numpy as np


def _bounded_panel(contour, width, height):
    """Reject the open background and isolated lines, even when their bounding boxes are large."""
    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.015 * perimeter, True).reshape(-1, 2)
    if not 4 <= len(corners) <= 7:
        return False
    vertical_sides = horizontal_sides = 0
    for (x1, y1), (x2, y2) in zip(corners, np.roll(corners, -1, axis=0)):
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if dy >= 0.25 * height and dx < 0.25 * dy and 3 < mid_x < width - 4:
            vertical_sides += 1
        if dx >= 0.10 * width and dy < 0.35 * dx and 3 < mid_y < height - 4:
            horizontal_sides += 1
    return vertical_sides >= 1 and vertical_sides + horizontal_sides >= 2


def panel_mask(frame: np.ndarray, person: np.ndarray | None) -> np.ndarray | None:
    """Return a soft foreground mask when a coherent, approximately rectangular surface is visible."""
    h, w = frame.shape[:2]
    ratio = min(1.0, 320 / max(w, h))
    if min(w, h) * ratio < 20:
        return None
    small = cv2.resize(frame, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
    height, width = small.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 35, 85)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    barriers = cv2.dilate(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel), kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((barriers == 0).astype(np.uint8), connectivity=4)
    person_small = None if person is None else cv2.resize(person, (width, height)) >= 0.5
    # Inspect largest surfaces first. A vertical edge alone gives no evidence
    # for which side is an object and which side belongs to the background.
    for label in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        x, y, cw, ch, area = stats[label]
        if area < 0.035 * width * height:
            break
        if area > 0.62 * width * height or area < 0.76 * cw * ch or cw < 0.075 * width:
            continue
        region = labels[y:y + ch, x:x + cw] == label
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE, offset=(int(x), int(y)))
        contour = max(contours, key=cv2.contourArea)
        if ch < 0.3 * height or not _bounded_panel(contour, width, height):
            continue
        if person_small is not None and person_small[y:y + ch, x:x + cw][region].mean() > 0.05:
            continue
        colors = small[y:y + ch, x:x + cw][region]
        low, high = np.percentile(colors, [10, 90], axis=0)
        if np.max(high - low) > 65:
            continue
        mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)
        mask = cv2.dilate(mask, kernel)
        return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return None
