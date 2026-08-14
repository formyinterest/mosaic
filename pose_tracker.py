"""Lightweight person tracker for stable orientation state IDs."""

from __future__ import annotations

from dataclasses import dataclass
import math

from yolo_pose_estimator import PersonPose


@dataclass(slots=True)
class _Track:
    bbox: tuple[float, float, float, float]
    misses: int = 0


class PersonTracker:
    def __init__(self, *, maximum_misses: int = 20) -> None:
        self.maximum_misses = maximum_misses
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0.0, x2-x1)*max(0.0, y2-y1)
        area_a, area_b = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]), max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1])
        return intersection/max(area_a+area_b-intersection, 1.0)

    @staticmethod
    def _center_distance(a, b) -> float:
        ac = ((a[0]+a[2])*.5, (a[1]+a[3])*.5); bc = ((b[0]+b[2])*.5, (b[1]+b[3])*.5)
        diagonal = max(math.hypot(a[2]-a[0], a[3]-a[1]), 1.0)
        return math.dist(ac, bc)/diagonal

    def update(self, persons: list[PersonPose]) -> list[tuple[int, PersonPose]]:
        boxes = [(p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2) for p in persons]
        candidates = []
        for track_id, track in self._tracks.items():
            for index, box in enumerate(boxes):
                iou = self._iou(track.bbox, box); distance = self._center_distance(track.bbox, box)
                if iou >= .15 or distance <= .55:
                    candidates.append((iou-distance*.25, track_id, index))
        assigned_tracks, assigned_people, matches = set(), set(), {}
        for _, track_id, index in sorted(candidates, reverse=True):
            if track_id in assigned_tracks or index in assigned_people:
                continue
            assigned_tracks.add(track_id); assigned_people.add(index); matches[index] = track_id
        for index in range(len(persons)):
            if index not in matches:
                matches[index] = self._next_id
                self._tracks[self._next_id] = _Track(boxes[index])
                self._next_id += 1
        for track_id, track in list(self._tracks.items()):
            matched_index = next((i for i, value in matches.items() if value == track_id), None)
            if matched_index is None:
                track.misses += 1
                if track.misses > self.maximum_misses:
                    del self._tracks[track_id]
            else:
                track.bbox = boxes[matched_index]; track.misses = 0
        return [(matches[index], person) for index, person in enumerate(persons)]
