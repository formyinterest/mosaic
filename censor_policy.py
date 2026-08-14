"""Production YOLO COCO-17 orientation and conservative censor ROI policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from yolo_pose_estimator import BoundingBox, PersonPose


class BodyOrientation(str, Enum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CensorRegion:
    name: str
    polygon: tuple[tuple[int, int], ...] | None
    confidence: float
    omit_reason: str | None = None

    @property
    def bbox(self):
        if not self.polygon:
            return None
        points = np.asarray(self.polygon)
        return tuple(map(int, (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())))


@dataclass(frozen=True, slots=True)
class CensorResult:
    person_index: int
    bbox: tuple[int, int, int, int]
    detector_confidence: float
    orientation_raw: BodyOrientation
    orientation_smoothed: BodyOrientation
    front_score: float
    side_score: float
    back_score: float
    face: CensorRegion
    eye: CensorRegion
    chest: CensorRegion
    buttocks: CensorRegion
    average_keypoint_confidence: float
    inference_ms: float


YOLO_INDEX = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
              "left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12,
              "left_knee": 13, "right_knee": 14}


class OrientationSmoother:
    """Physical transition state machine with separate enter/hold/exit rules."""

    def __init__(self, size: int = 5) -> None:
        self.state = BodyOrientation.UNKNOWN
        self.candidate = BodyOrientation.UNKNOWN
        self.candidate_frames = 0

    def update(self, raw: BodyOrientation, front: float = 0.0,
               side: float = 0.0, back: float = 0.0) -> BodyOrientation:
        # A body cannot physically jump FRONT <-> BACK without passing SIDE.
        allowed = {
            BodyOrientation.UNKNOWN: {BodyOrientation.FRONT, BodyOrientation.SIDE, BodyOrientation.BACK},
            BodyOrientation.FRONT: {BodyOrientation.FRONT, BodyOrientation.SIDE},
            BodyOrientation.SIDE: {BodyOrientation.FRONT, BodyOrientation.SIDE, BodyOrientation.BACK},
            BodyOrientation.BACK: {BodyOrientation.BACK, BodyOrientation.SIDE},
        }
        desired = raw if raw in allowed[self.state] else BodyOrientation.SIDE
        required = 1
        if desired == BodyOrientation.BACK:
            required = 3 if self.state != BodyOrientation.BACK else 1
            if back < (.72 if self.state == BodyOrientation.SIDE else .80):
                desired = self.state
        elif self.state == BodyOrientation.BACK and desired == BodyOrientation.SIDE:
            required = 2
            if max(side, front) < .56:
                desired = BodyOrientation.BACK
        elif desired in (BodyOrientation.FRONT, BodyOrientation.SIDE) and desired != self.state:
            required = 2

        if desired == self.state:
            self.candidate = desired; self.candidate_frames = 0
            return self.state
        if desired != self.candidate:
            self.candidate = desired; self.candidate_frames = 1
        else:
            self.candidate_frames += 1
        if self.candidate_frames >= required:
            self.state = desired; self.candidate_frames = 0
        return self.state


def _clamp(points: Iterable[tuple[float, float]], width: int, height: int):
    return tuple((int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))) for x, y in points)


def _rect(x1, y1, x2, y2, width, height):
    return _clamp(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), width, height)


def _points(person: PersonPose, minimum: float = .35):
    return {name: (person.keypoints[index].x, person.keypoints[index].y, person.keypoints[index].confidence)
            for name, index in YOLO_INDEX.items() if person.keypoints[index].confidence >= minimum}


def _midpoint(a, b):
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return a[:2] if a else (b[:2] if b else None)


def _distance(a, b):
    return math.dist(a[:2], b[:2]) if a and b else 0.0


def classify_orientation(points, bbox: BoundingBox, all_points=None):
    raw_points = all_points or points
    face_names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
    face = [points[name] for name in face_names if name in points]
    shoulders = _distance(points.get("left_shoulder"), points.get("right_shoulder"))
    hips = _distance(points.get("left_hip"), points.get("right_hip"))
    box_width = max(bbox.x2 - bbox.x1, 1.0)
    shoulder_center = _midpoint(points.get("left_shoulder"), points.get("right_shoulder"))
    face_above = bool(face and shoulder_center and np.mean([point[1] for point in face]) < shoulder_center[1])

    weights = {"nose": .30, "left_eye": .20, "right_eye": .20, "left_ear": .15, "right_ear": .15}
    face_visibility = sum(weights[name] * float(raw_points[name][2]) for name in face_names if name in raw_points)
    face_visibility = min(1.0, face_visibility)
    narrow = max(0.0, 1.0 - shoulders / (box_width * .55))

    front_order = back_order = order_weight = 0.0
    for left_name, right_name in (("left_shoulder", "right_shoulder"),
                                  ("left_hip", "right_hip"), ("left_knee", "right_knee")):
        left, right = raw_points.get(left_name), raw_points.get(right_name)
        if not left or not right:
            continue
        confidence = min(float(left[2]), float(right[2]))
        delta = float(right[0] - left[0]) / box_width
        if confidence < .25 or abs(delta) < .025:
            continue
        order_weight += confidence
        if delta < 0:  # person's left projects to image-right when front-facing
            front_order += confidence
        else:
            back_order += confidence
    if order_weight:
        front_order /= order_weight; back_order /= order_weight

    unilateral_face = sum(1 for name in face_names if name in points) in (1, 2)
    front = min(1.0, face_visibility*.70 + (.12 if face_above else 0) + front_order*.30)
    side = min(1.0, narrow*.58 + (.27 if unilateral_face else .04) + (1-max(front_order, back_order))*.12)
    # Shoulder width only measures side-vs-non-side; it is deliberately absent here.
    bilateral_torso = min(1.0, sum(raw_points.get(name, (0, 0, 0))[2] for name in
                                   ("left_shoulder", "right_shoulder", "left_hip", "right_hip")) / 2.4)
    back = min(1.0, (1-face_visibility)*.48 + back_order*.42 + bilateral_torso*.10)
    if bbox.confidence >= .50 and back >= .72 and back > front + .18:
        value = BodyOrientation.BACK
    elif front >= .55 and front >= side + .12:
        value = BodyOrientation.FRONT
    elif side >= .52:
        value = BodyOrientation.SIDE
    else:
        value = BodyOrientation.UNKNOWN
    return value, front, side, back


def _regions(points, bbox: BoundingBox, width: int, height: int, orientation: BodyOrientation):
    face_points = [points[name] for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear") if name in points]
    # An ear by itself frequently survives on a cropped profile even when the
    # face is entirely outside the frame (or hidden by hair/a hand).  It is not
    # enough evidence to draw an invented face-sized fallback rectangle.
    # Require a nose or eye before rendering a face ROI; ears can still expand
    # that ROI once a primary facial landmark is present.
    primary_face_points = [points[name] for name in ("nose", "left_eye", "right_eye") if name in points]
    shoulders = _midpoint(points.get("left_shoulder"), points.get("right_shoulder"))
    hips = _midpoint(points.get("left_hip"), points.get("right_hip"))
    shoulder_width = _distance(points.get("left_shoulder"), points.get("right_shoulder"))
    hip_width = _distance(points.get("left_hip"), points.get("right_hip"))
    box_width, box_height = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1

    eye_points = [points[name] for name in ("left_eye", "right_eye") if name in points]
    left_eye, right_eye = points.get("left_eye"), points.get("right_eye")
    eye_span = _distance(left_eye, right_eye) if left_eye and right_eye else 0.0

    if primary_face_points:
        # A visible face is censored regardless of the body/torso orientation
        # classification (e.g. looking back over the shoulder while the torso
        # faces away) -- face visibility is judged purely on its own keypoints.
        xs, ys = [p[0] for p in face_points], [p[1] for p in face_points]
        anchor = max(shoulder_width * .34, box_width * .11, 22.0)
        if left_eye and right_eye and eye_span > 1:
            # Rotate the box with head tilt using the eye line, same as the
            # eye/chest/buttocks boxes -- project onto the tilted axes and pad
            # relative to eye_span (a reliable in-frame face-scale measure)
            # rather than `anchor`, which is derived from shoulder/bbox width
            # and gets thrown off by limbs stretched away from the body (e.g.
            # a crouching pose), undersizing the box on exactly those frames.
            unit = (np.array(right_eye[:2]) - np.array(left_eye[:2])) / eye_span
            normal = np.array([-unit[1], unit[0]])
            pts = np.array([[p[0], p[1]] for p in face_points])
            along, across = pts @ unit, pts @ normal
            a_min, a_max = along.min() - eye_span*.55, along.max() + eye_span*.55
            # normal's sign flips arbitrarily with head roll direction, so
            # across.min()/max() don't reliably mean "toward the forehead" /
            # "toward the chin". The nose always sits below the eye line
            # anatomically -- use it to pin the bigger pad to the mouth/chin
            # side (COCO has no mouth/chin keypoint to anchor to directly)
            # and the smaller pad to the forehead side.
            nose = points.get("nose")
            eye_mid_across = float((np.array(left_eye[:2]) + np.array(right_eye[:2])) @ normal) / 2
            chin_is_min_side = nose is not None and float(np.array(nose[:2]) @ normal) < eye_mid_across
            pad_min, pad_max = (eye_span*1.6, eye_span*.75) if chin_is_min_side else (eye_span*.75, eye_span*1.6)
            c_min, c_max = across.min() - pad_min, across.max() + pad_max
            corners = [unit*a + normal*c for a, c in ((a_min, c_min), (a_max, c_min), (a_max, c_max), (a_min, c_max))]
            face = CensorRegion("face", _clamp(corners, width, height), float(np.mean([p[2] for p in face_points])))
        else:
            face = CensorRegion("face", _rect(min(xs)-anchor*.48, min(ys)-anchor*.60,
                                               max(xs)+anchor*.48, max(ys)+anchor*.72, width, height),
                                float(np.mean([p[2] for p in face_points])))
    else:
        face = CensorRegion("face", None, 0, "no_reliable_face_keypoint")
    if left_eye and right_eye and eye_span > 1:
        # Rotate the band with head tilt instead of a fixed horizontal box,
        # same idea as the chest/buttocks boxes following the spine angle.
        p_left, p_right = np.array(left_eye[:2]), np.array(right_eye[:2])
        unit = (p_right - p_left) / eye_span
        normal = np.array([-unit[1], unit[0]])
        center = (p_left + p_right) / 2
        eye_width, eye_height = eye_span * 2.0, eye_span * .5
        corners = [center + unit*a*eye_width/2 + normal*l*eye_height/2 for a, l in ((-1,-1),(1,-1),(1,1),(-1,1))]
        eye = CensorRegion("eye", _clamp(corners, width, height), float(np.mean([p[2] for p in eye_points])))
    elif eye_points:
        # Only one eye visible (e.g. side profile) -> no eye-line direction to
        # rotate with; fall back to an axis-aligned box at the face-anchor scale.
        xs, ys = [p[0] for p in eye_points], [p[1] for p in eye_points]
        anchor = max(shoulder_width * .34, box_width * .11, 22.0)
        pad_x, pad_y = anchor * .5, anchor * .25
        eye = CensorRegion("eye", _rect(min(xs)-pad_x, min(ys)-pad_y,
                                         max(xs)+pad_x, max(ys)+pad_y, width, height),
                           float(np.mean([p[2] for p in eye_points])))
    else:
        eye = CensorRegion("eye", None, 0, "no_reliable_eye_keypoint")

    if shoulders and hips and orientation != BodyOrientation.BACK:
        shoulder, hip = np.array(shoulders), np.array(hips)
        spine = hip - shoulder
        torso = float(np.linalg.norm(spine))
        if torso > 1:
            unit, normal = spine / torso, np.array([-spine[1], spine[0]]) / torso
            nose = points.get("nose")
            if nose and abs(float(np.dot(normal, np.array(nose[:2])-shoulder))) > torso*.08 and np.dot(normal, np.array(nose[:2])-shoulder) < 0:
                normal = -normal
            center = shoulder + spine*.34
            chest_width, chest_height = max(shoulder_width*1.24, torso*.72), max(torso*.52, box_height*.22)
            if orientation in (BodyOrientation.SIDE, BodyOrientation.UNKNOWN):
                center += normal*torso*.10
                chest_width *= 1.08
            corners = [center + normal*a*chest_width/2 + unit*l*chest_height/2 for a, l in ((-1,-1),(1,-1),(1,1),(-1,1))]
            chest = CensorRegion("chest", _clamp(corners, width, height),
                                  float(np.mean([points[n][2] for n in ("left_shoulder","right_shoulder","left_hip","right_hip") if n in points])))
        else:
            chest = CensorRegion("chest", None, 0, "zero_torso_length")
    elif orientation == BodyOrientation.BACK:
        chest = CensorRegion("chest", None, 0, "confirmed_back")
    elif hips:
        # Shoulders are unreliable (frame cropped at chest height, common in
        # low camera-angle shots) but the hip line is known: anchor the guess
        # box just above the hips instead of a bbox-wide ratio that ignores
        # where the torso actually sits in frame.
        hip_y = hips[1]
        estimated_torso = max(hip_width * 1.8, box_height * .30)
        top = max(bbox.y1, hip_y - estimated_torso)
        chest = CensorRegion("chest", _rect(bbox.x1+box_width*.12, top,
                                             bbox.x2-box_width*.12, hip_y-estimated_torso*.12, width, height),
                              .15, "torso_bbox_fallback")
    elif shoulders:
        # Mirror of the hips-only case above: hips are unreliable/out of frame
        # (common in close-up or upper-body-only framing, e.g. a webcam shot
        # cropped at the chest) but the shoulder line is known. Anchor a box
        # sized from the shoulders themselves just below the shoulder line --
        # a bbox-width/height ratio balloons on a close-up crop where the
        # detector's own bbox reaches for stretched-out arms/hands, covering
        # far more than the actual chest.
        # Ratios fit against two upper-body-only reference crops (least-
        # squares grid search over manually annotated chest boxes): chest
        # width/height/vertical-offset all scale most consistently off
        # shoulder_width, not box_height, which swings with how much of the
        # arms the detector's bbox happened to reach for.
        shoulder_x, shoulder_y = shoulders
        chest_width, chest_height = shoulder_width * 1.1, shoulder_width * .62
        center_y = shoulder_y + shoulder_width * .6
        chest = CensorRegion("chest", _rect(shoulder_x-chest_width/2, center_y-chest_height/2,
                                             shoulder_x+chest_width/2, center_y+chest_height/2, width, height),
                              .15, "torso_bbox_fallback")
    else:
        chest = CensorRegion("chest", _rect(bbox.x1+box_width*.12, bbox.y1+box_height*.18,
                                             bbox.x2-box_width*.12, bbox.y1+box_height*.58, width, height), .15, "torso_bbox_fallback")

    left_knee, right_knee = points.get("left_knee"), points.get("right_knee")
    both_knees = bool(left_knee and right_knee)
    knee = _midpoint(left_knee, right_knee)
    if hips and (shoulders or both_knees):
        hip = np.array(hips)
        if shoulders:
            spine_origin = np.array(shoulders)
            spine = hip - spine_origin
        else:
            # Shoulders are out of frame (common when the camera sits low and
            # only shows hip-down) but both hips and both knees are reliably
            # tracked: extend the same downward body axis one segment further
            # with hip->knee instead of collapsing to a generic bbox ratio.
            spine_origin = hip
            spine = np.array(knee) - hip
        torso = float(np.linalg.norm(spine))
        if torso > 1:
            unit, normal = spine/torso, np.array([-spine[1], spine[0]])/torso
            nose = points.get("nose")
            if nose and abs(float(np.dot(normal, np.array(nose[:2])-spine_origin))) > torso*.08 and np.dot(normal, np.array(nose[:2])-spine_origin) < 0:
                normal = -normal
            # Only trust the knee-based size clamp when both knees are visible;
            # a single low-confidence knee can sit almost anywhere in a bent or
            # foreshortened pose and would otherwise collapse the box to a sliver.
            knee_distance = math.dist(hips, knee) if both_knees else torso
            # Cover the full pelvic zone, not only the hip landmark line.
            # Pose hips sit near the centre of the pelvis and are routinely
            # above the lowest exposed area in partial/back-facing views.
            # The larger, downward-biased ROI keeps the lower pelvis and
            # groin protected even when orientation classification is wrong.
            if orientation == BodyOrientation.BACK:
                # Rear view: bias below the hip landmarks and size primarily
                # from pelvis/knee anatomy instead of arm-inflated person bbox.
                butt_height = min(max(torso*.72, knee_distance*.58), knee_distance*.88)
                butt_width = max(hip_width*1.70, box_width*.58, torso*.76)
                center = hip + unit*torso*.17
            else:
                butt_height = min(max(torso*.78, knee_distance*.65), knee_distance*.95)
                butt_width = max(hip_width*1.55, box_width*.70, torso*.78)
                center = hip + unit*torso*.06
            if orientation in (BodyOrientation.SIDE, BodyOrientation.UNKNOWN):
                center -= normal*butt_width*.18
                butt_width *= 1.04
            corners = [center + normal*a*butt_width/2 + unit*l*butt_height/2 for a, l in ((-1,-1),(1,-1),(1,1),(-1,1))]
            buttocks = CensorRegion("buttocks", _clamp(corners, width, height),
                                    float(np.mean([points[n][2] for n in ("left_hip","right_hip") if n in points])))
        else:
            buttocks = CensorRegion("buttocks", None, 0, "zero_torso_length")
    elif hips:
        buttocks = CensorRegion("buttocks", _rect(bbox.x1+box_width*.16, bbox.y1+box_height*.58,
                                                   bbox.x2-box_width*.16, bbox.y1+box_height*.80, width, height), .15, "hip_fallback")
    else:
        # No hip signal at all -- common in an upper-body-only/close-up crop
        # where the buttocks simply aren't in frame. The bbox-height-ratio
        # guess above assumes a full standing figure and, lacking that, lands
        # its box on whatever IS in frame instead (e.g. right on the chest).
        # Omit rather than invent a region with nothing anatomical behind it.
        buttocks = CensorRegion("buttocks", None, 0, "no_reliable_hip_keypoint")
    return face, eye, chest, buttocks


def build_censor_results(persons: list[PersonPose], dimensions, latency_ms: float,
                         smoothers: dict[int, OrientationSmoother],
                         orientation_predictions=None,
                         track_ids: list[int] | None = None) -> list[CensorResult]:
    height, width = dimensions[:2]
    results = []
    for index, person in enumerate(sorted(persons, key=lambda item: item.bbox.x1)):
        points = _points(person)
        all_points = {name: (person.keypoints[keypoint_index].x,
                             person.keypoints[keypoint_index].y,
                             person.keypoints[keypoint_index].confidence)
                      for name, keypoint_index in YOLO_INDEX.items()}
        if orientation_predictions is not None and index < len(orientation_predictions):
            prediction = orientation_predictions[index]
            raw = prediction.orientation
            front, side, back = prediction.front_score, prediction.side_score, prediction.back_score
        else:
            raw, front, side, back = classify_orientation(points, person.bbox, all_points)
        track_id = track_ids[index] if track_ids is not None else index
        smoothed = smoothers.setdefault(track_id, OrientationSmoother()).update(raw, front, side, back)
        face, eye, chest, buttocks = _regions(points, person.bbox, width, height, smoothed)
        box = person.bbox
        results.append(CensorResult(track_id, tuple(map(int, (box.x1, box.y1, box.x2, box.y2))), box.confidence,
                                    raw, smoothed, front, side, back, face, eye, chest, buttocks,
                                    float(np.mean([p.confidence for p in person.keypoints])), latency_ms))
    return results
