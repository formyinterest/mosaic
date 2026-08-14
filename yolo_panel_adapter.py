"""Map YOLO COCO-17 poses to the overlay's chest and hips panels.

YOLO pose has reliable 2-D joint locations but no MediaPipe-style world/depth
coordinates.  The adapter deliberately exposes that boundary: it produces the
2-D torso panels that can be derived from shoulders and hips, and never
pretends that it can make MediaPipe-only front/back or pitch decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from yolo_pose_estimator import PersonPose, PoseEstimation


NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 0, 5, 6, 11, 12
KEYPOINT_CONFIDENCE = 0.5


@dataclass(frozen=True, slots=True)
class PanelPose:
    """Overlay panel data for one YOLO-detected person, in absolute pixels."""

    person_index: int
    bbox: tuple[float, float, float, float]
    chest: tuple[float, float, float, float] | None  # center_x, center_y, width, height
    hips: tuple[float, float, float, float] | None
    debug: dict[str, object]


class _EMASmoother:
    def __init__(self, alpha: float, patience: int = 12) -> None:
        self.alpha = alpha
        self.patience = patience
        self.value: float | tuple[float, float] | None = None
        self.misses = 0

    def update(self, value: float | tuple[float, float] | None):
        if value is None:
            self.misses += 1
            if self.misses < self.patience:
                return self.value
            self.value = None
            return None
        self.misses = 0
        if self.value is None:
            self.value = value
        elif isinstance(value, tuple):
            previous = self.value
            assert isinstance(previous, tuple)
            self.value = tuple(self.alpha * v + (1 - self.alpha) * p for v, p in zip(value, previous))
        else:
            previous = self.value
            assert not isinstance(previous, tuple)
            self.value = self.alpha * value + (1 - self.alpha) * previous
        return self.value


class _PanelSmoothers:
    def __init__(self) -> None:
        self.chest_position = _EMASmoother(0.3)
        self.chest_width = _EMASmoother(0.15)
        self.chest_height = _EMASmoother(0.15)
        self.hips_position = _EMASmoother(0.3)
        self.hips_width = _EMASmoother(0.15)
        self.hips_height = _EMASmoother(0.15)


class YoloPanelAdapter:
    """Convert every person in a YOLO result into supported overlay panels."""

    def __init__(self) -> None:
        self._smoothers: dict[int, _PanelSmoothers] = {}
        print("[YOLO pose] 지원: 모든 사람의 bbox, nose/shoulder/hip 2D 좌표, chest/hips 패널")
        print(
            "[YOLO pose] 미지원(비활성화): MediaPipe 3D z 기반 앞/뒤 얼굴 판정 및 pitch 보정; "
            "얼굴 패널은 출력하지 않습니다."
        )

    @staticmethod
    def _point(person: PersonPose, index: int, sx: float, sy: float) -> tuple[float, float, float] | None:
        keypoint = person.keypoints[index]
        if keypoint.confidence < KEYPOINT_CONFIDENCE:
            return None
        return (keypoint.x * sx, keypoint.y * sy, keypoint.confidence)

    @staticmethod
    def _center(
        left: tuple[float, float, float] | None,
        right: tuple[float, float, float] | None,
    ) -> tuple[float, float] | None:
        if left is not None and right is not None:
            return ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
        if left is not None:
            return left[:2]
        if right is not None:
            return right[:2]
        return None

    @staticmethod
    def _absolute(point: tuple[float, float, float] | None, monitor: dict[str, int]):
        if point is None:
            return None
        return (monitor["left"] + int(point[0]), monitor["top"] + int(point[1]), point[2])

    def process(
        self,
        estimation: PoseEstimation,
        monitor: dict[str, int],
        scale_x: float,
        scale_y: float,
    ) -> tuple[PanelPose, ...]:
        # Ordering by horizontal center gives each on-screen person a stable
        # smoother slot for the common side-by-side use case.
        persons = sorted(estimation.persons, key=lambda person: (person.bbox.x1 + person.bbox.x2) * 0.5)
        panels = tuple(
            self._process_person(slot, person, monitor, scale_x, scale_y)
            for slot, person in enumerate(persons)
        )
        for slot in set(self._smoothers) - set(range(len(persons))):
            smoothers = self._smoothers[slot]
            smoothers.chest_position.update(None)
            smoothers.chest_width.update(None)
            smoothers.chest_height.update(None)
            smoothers.hips_position.update(None)
            smoothers.hips_width.update(None)
            smoothers.hips_height.update(None)
        return panels

    def _process_person(
        self,
        slot: int,
        person: PersonPose,
        monitor: dict[str, int],
        sx: float,
        sy: float,
    ) -> PanelPose:
        nose = self._point(person, NOSE, sx, sy)
        left_shoulder = self._point(person, LEFT_SHOULDER, sx, sy)
        right_shoulder = self._point(person, RIGHT_SHOULDER, sx, sy)
        left_hip = self._point(person, LEFT_HIP, sx, sy)
        right_hip = self._point(person, RIGHT_HIP, sx, sy)
        shoulders = self._center(left_shoulder, right_shoulder)
        hips = self._center(left_hip, right_hip)
        hips_estimated = False

        if hips is None and shoulders is not None:
            if left_shoulder is not None and right_shoulder is not None:
                shoulder_width = math.dist(left_shoulder[:2], right_shoulder[:2])
                torso_guess = shoulder_width * 1.5
            else:
                torso_guess = monitor["height"] * 0.15
            hips = (shoulders[0], shoulders[1] + torso_guess)
            hips_estimated = True

        chest = hips_panel = None
        smoothers = self._smoothers.setdefault(slot, _PanelSmoothers())
        if shoulders is not None and hips is not None:
            shoulders_vec = np.array(shoulders, dtype=float)
            hips_vec = np.array(hips, dtype=float)
            spine = hips_vec - shoulders_vec
            torso_length = float(np.linalg.norm(spine))
            if torso_length > 0:
                normal = np.array([-spine[1], spine[0]], dtype=float) / torso_length
                # Nose is usable as a 2-D orientation hint only; it is not a
                # substitute for MediaPipe z/front-back classification.
                if nose is not None and np.dot(normal, np.array(nose[:2]) - shoulders_vec) < 0:
                    normal = -normal
                shoulder_width = (
                    math.dist(left_shoulder[:2], right_shoulder[:2])
                    if left_shoulder is not None and right_shoulder is not None
                    else 0.0
                )
                turn_ratio = max(0.0, min(1.0, 1.0 - shoulder_width / (torso_length * 0.45))) if shoulder_width else 1.0
                chest_center = shoulders_vec + spine * 0.2 + normal * (torso_length * 0.3 * turn_ratio)
                spine_unit = spine / torso_length
                hips_center = hips_vec + spine_unit * (torso_length * 0.15) - normal * (torso_length * 0.28 * turn_ratio)

                # Pitch is intentionally fixed at zero: YOLO COCO-17 has no z.
                chest_position = smoothers.chest_position.update(tuple(chest_center))
                chest_width = smoothers.chest_width.update(torso_length * 0.50)
                chest_height = smoothers.chest_height.update(torso_length * 0.35)
                hips_position = smoothers.hips_position.update(None if hips_estimated else tuple(hips_center))
                hips_width = smoothers.hips_width.update(None if hips_estimated else torso_length * 0.85)
                hips_height = smoothers.hips_height.update(None if hips_estimated else torso_length * 0.55)
                if chest_position is not None and chest_width is not None and chest_height is not None:
                    chest = (
                        monitor["left"] + chest_position[0],
                        monitor["top"] + chest_position[1],
                        chest_width,
                        chest_height,
                    )
                if hips_position is not None and hips_width is not None and hips_height is not None:
                    hips_panel = (
                        monitor["left"] + hips_position[0],
                        monitor["top"] + hips_position[1],
                        hips_width,
                        hips_height,
                    )
        else:
            smoothers.chest_position.update(None)
            smoothers.chest_width.update(None)
            smoothers.chest_height.update(None)
            smoothers.hips_position.update(None)
            smoothers.hips_width.update(None)
            smoothers.hips_height.update(None)

        box = person.bbox
        debug = {
            "nose": self._absolute(nose, monitor),
            "l_shoulder": self._absolute(left_shoulder, monitor),
            "r_shoulder": self._absolute(right_shoulder, monitor),
            "l_hip": self._absolute(left_hip, monitor),
            "r_hip": self._absolute(right_hip, monitor),
            "shoulder_center": None if shoulders is None else (monitor["left"] + shoulders[0], monitor["top"] + shoulders[1]),
            "hip_center": None if hips is None else (monitor["left"] + hips[0], monitor["top"] + hips[1]),
            "hip_is_estimated": hips_estimated,
            "capability": "YOLO-2D: chest/hips supported; face/front-back/pitch disabled",
        }
        return PanelPose(
            person_index=slot + 1,
            bbox=(box.x1 * sx, box.y1 * sy, box.x2 * sx, box.y2 * sy),
            chest=chest,
            hips=hips_panel,
            debug=debug,
        )
