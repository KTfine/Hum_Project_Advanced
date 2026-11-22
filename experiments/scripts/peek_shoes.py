"""Measure shoe separation on the two level-room test plates independently of scene inference."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from experiments.scripts.co_photo_interaction import read_frame
from experiments.scripts.seat_placement import ReplayCamera
from photo_assistant.client import CanvasSource, PhotoCapture
from photo_assistant.avatar.frames import AvatarFrames
from photo_assistant.avatar.pose import AvatarPose, WHITE_DRAGON_CATALOG
from photo_assistant.protocol import StreamingCommandParser
from photo_assistant.vision.occlusion import PersonOcclusion
from photo_assistant.vision.scene import SceneSegmenter


# Manually traced from the original room plates at 640 x 480. A polygon covers
# one shoe below its trouser cuff. Coordinates are independent of inferred masks.
SHOES = {
    "left": (
        [(94,413),(98,411),(104,411),(109,413),(109,424),(106,432),(104,434),(92,434),
         (89,432),(90,426),(93,421)],
        [(128,413),(132,411),(141,410),(145,412),(145,418),(150,425),(151,431),
         (149,433),(137,433),(133,430),(132,426),(128,423)]),
    "right": (
        [(475,409),(480,407),(486,408),(490,410),(490,419),(488,426),(485,430),
         (474,430),(471,428),(471,422),(474,416)],
        [(508,409),(513,407),(520,407),(523,409),(525,415),(529,421),(529,427),
         (526,429),(516,429),(512,426),(511,422),(507,418)]),
}
# Full-body contours from the original level-room hand tracings, reduced to
# the 640 x 480 camera coordinates used for both the reference and test shots.
PERSON_OUTLINE = {
    "left": [(122,122),(131,126),(137,134),(138,144),(137,152),(133,161),(134,171),(145,175),(156,183),
             (161,198),(164,263),(162,282),(160,292),(156,301),(153,301),(149,305),(147,406),(152,422),
             (152,430),(145,432),(133,432),(130,427),(126,415),(122,409),(119,330),(115,332),(109,410),
             (107,423),(104,431),(99,434),(89,433),(89,427),(94,411),(89,411),(90,329),(88,306),
             (84,303),(82,296),(80,282),(76,265),(77,209),(80,189),(84,181),(102,172),(107,168),
             (108,161),(104,151),(103,144),(103,136),(107,127),(114,123)],
    "right": [(498,123),(508,126),(513,132),(515,141),(513,152),(509,159),(509,168),(519,172),(533,180),
              (538,193),(541,262),(538,278),(534,293),(530,300),(526,301),(525,405),(529,417),(530,427),
              (525,430),(511,430),(507,423),(504,409),(498,406),(496,331),(493,334),(488,421),(484,429),
              (476,430),(471,428),(471,419),(476,408),(473,405),(469,315),(469,303),(463,298),(460,290),
              (455,258),(457,202),(461,187),(468,179),(484,171),(487,168),(487,161),(483,153),
              (480,136),(483,128),(488,125)],
}
# Annotated on the current 1024 x 1536 peek-right artwork, including the heel,
# toe box and ankle straps. The artwork digest prevents use with another revision.
ART_SHOES = (
    [(353,1312),(361,1307),(370,1315),(393,1313),(400,1306),(408,1314),(403,1337),
     (406,1356),(405,1384),(411,1417),(422,1450),(424,1478),(418,1487),(402,1492),
     (374,1492),(349,1485),(341,1471),(342,1430),(331,1429),(321,1424),(325,1403),
     (340,1376),(343,1361),(347,1340)],
    [(486,1309),(492,1304),(502,1315),(523,1313),(532,1307),(541,1312),(537,1337),
     (543,1352),(547,1366),(557,1377),(576,1418),(573,1426),(588,1452),(590,1478),
     (584,1486),(562,1491),(534,1490),(515,1486),(504,1474),(500,1456),(488,1458),
     (484,1451),(483,1411),(470,1407),(462,1400),(468,1383),(476,1365),(478,1341)],
)
ART_SHA256 = "fc4cbbc26009eeb8b895772b3fd224be68f729c4de290a7b0ea72e73e4da0c78"


def region(shape, polygon):
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.asarray(polygon, np.int32)], 1)
    return result != 0


def nearest_distance(distance, mask):
    return round(float(distance[mask].min()), 2) if mask.any() else None


def render_pair(plates, directory, prefix):
    """Replay the existing peek command after a fresh full-body entrance on each plate."""
    directory.mkdir(parents=True, exist_ok=False)
    for side in SHOES:
        frame = read_frame(plates / f"person-{side}.png")
        height, width = frame.shape[:2]
        camera = ReplayCamera(frame)
        worker = PersonOcclusion(SceneSegmenter, max_frame_age=120)
        canvas = CanvasSource(WHITE_DRAGON_CATALOG, width, height, 95, threading.Event(), preview=False,
                              scene_layers={"person": worker}, infer_obstacle=True, project_scene=True)
        try:
            canvas.start()
            canvas.configure_view(camera, show_character=True, entrance=canvas.entrance_state(camera))
            worker.submit(frame)
            worker.mask_for_capture(frame)
            _, entrance = canvas.frame_with_placement(frame, include_character=True)
            peek, = StreamingCommandParser().feed(side, "<peek>")
            if canvas.apply_command(peek) is None:
                raise RuntimeError(f"No feasible peek position on the {side} plate")
            _, placement = canvas.frame_with_placement(frame, include_character=True)
            name = f"{prefix}-{side}"
            PhotoCapture(camera, directory, canvas).save().rename(directory / f"{name}.png")
            info = dict(entrance=entrance, after=placement, state=asdict(canvas.controller.state_at()))
            (directory / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            canvas.close()
            worker.close()


def measure(frame, shot, row, side, frames):
    place = row["after"]
    entry = frames.catalog.entry(AvatarPose.from_tags(row["state"]["pose"]))
    art = frames.frame(AvatarPose.from_tags(row["state"]["pose"]))
    digest = hashlib.sha256(entry.path.read_bytes()).hexdigest()
    if entry.path.name != "peek-right.png" or digest != ART_SHA256:
        raise ValueError("The shoe annotations require the level-room peek-right artwork")
    height, width = frame.shape[:2]
    if (width, height) != (640, 480):
        raise ValueError("The test plates must be uncropped and rendered at 640 x 480")
    human = np.logical_or.reduce([region((height, width), p) for p in SHOES[side]])
    # This measures the shortest distance, including diagonals, between the
    # actual shoe regions, regardless of where their sprite anchors are.
    distance = cv2.distanceTransform((~human).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    size = place["standing_height_ratio"] * height
    matrix = cv2.getRotationMatrix2D(tuple(entry.anchor_px), -row["state"]["rotation"], size / entry.standing_height_px)
    matrix[:, 2] += np.rint(np.array([place["x"], place["y"]]) * [width, height]) - entry.anchor_px
    changed = cv2.absdiff(frame, shot).max(axis=2) >= 7
    annotated = shot.copy()
    shoes = []
    native = []
    for polygon in ART_SHOES:
        mask = region(art.shape[:2], polygon) & (art[:, :, 3] >= 155)
        projected = cv2.warpAffine(mask.astype(np.uint8) * 255, matrix, (width, height)) >= 128
        native.append(projected)
        visible = projected & changed
        shoes.append(dict(projected_px=int(projected.sum()), visible_px=int(visible.sum()),
                          overlap_px=int((projected & human).sum()),
                          visible_overlap_px=int((visible & human).sum()),
                          clearance_px=nearest_distance(distance, projected),
                          visible_clearance_px=nearest_distance(distance, visible)))
        contours, _ = cv2.findContours(projected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (210, 60, 255), 1)
    cv2.polylines(annotated, [np.array(p, np.int32) for p in SHOES[side]], True, (60, 245, 40), 1)
    yy, xx = np.nonzero(human | native[0] | native[1])
    bounds = np.s_[max(0, yy.min()-8):min(height, yy.max()+9), max(0, xx.min()-8):min(width, xx.max()+9)]
    crop = np.vstack([cv2.resize(image[bounds], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                      for image in (shot, annotated)])
    face_mask = np.zeros(art.shape[:2], np.uint8)
    l, t, r, b = map(int, entry.face_box_px)
    face_mask[t:b,l:r] = art[t:b,l:r,3]
    face = cv2.warpAffine(face_mask, matrix, (width, height)) >= 155
    feet = native[0] | native[1]
    # Exclude one border pixel to allow for uncertainty in the hand tracing.
    person_core = cv2.erode(region((height, width), PERSON_OUTLINE[side]).astype(np.uint8),
                            np.ones((3, 3), np.uint8)).astype(bool)
    hidden = {}
    # Native art y=810 is hip level for the annotated image (digest checked
    # above). Both denominators count only opaque, on-screen sprite pixels.
    for key, bottom_row in (("below_chin", art.shape[0]), ("chin_to_hip", 810)):
        band = np.zeros(art.shape[:2], np.uint8)
        band[b:bottom_row] = art[b:bottom_row, :, 3]
        projected = cv2.warpAffine(band, matrix, (width, height)) >= 155
        pixels = int((projected & person_core & ~changed).sum())
        hidden[key] = dict(hidden_by_person_px=pixels, projected_px=int(projected.sum()),
                           fraction=round(float(pixels / projected.sum()), 4))
    required = min(cv2.boundingRect(np.array(p))[2] for p in SHOES[side]) / 4
    clear = all(shoe["clearance_px"] is not None and shoe["clearance_px"] >= required for shoe in shoes)
    return dict(shoes=shoes, required_clearance_px=required, clear_of_human_shoes=clear,
                shoe_sole_setback_px=int(np.where(human)[0].max()-np.where(feet)[0].max()),
                body_occlusion=hidden,
                face_visible=round(float(changed[face].mean()), 3),
                changed_human_shoe_pixels=int((changed & human).sum()),
                art_sha256=digest), crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shots", type=Path, help="Directory containing PREFIX-left/right.png and matching .json state")
    parser.add_argument("prefix")
    parser.add_argument("output", type=Path)
    parser.add_argument("--plates", type=Path, default=Path("experiments/runtime/interaction-20261120"))
    parser.add_argument("--check", action="store_true", help="Exit nonzero if an annotated shoe is too close")
    parser.add_argument("--replay", action="store_true", help="Render the two source plates into a new shots directory")
    args = parser.parse_args()
    if args.replay:
        render_pair(args.plates, args.shots, args.prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = AvatarFrames(WHITE_DRAGON_CATALOG)
    results = {}
    for side in SHOES:
        source = read_frame(args.plates / f"person-{side}.png")
        name = f"{args.prefix}-{side}"
        shot = cv2.imread(str(args.shots / f"{name}.png"))
        row = json.loads((args.shots / f"{name}.json").read_text(encoding="utf-8"))
        results[side], inspection = measure(source, shot, row, side, frames)
        cv2.imwrite(str(args.output / f"{name}-inspection.png"), inspection)
    data = json.dumps(results, ensure_ascii=False, indent=2)
    (args.output / f"{args.prefix}-shoes.json").write_text(data, encoding="utf-8")
    print(data)
    if args.check and not all(m["clear_of_human_shoes"] for m in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
