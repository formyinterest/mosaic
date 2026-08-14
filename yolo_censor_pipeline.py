"""GPU-only YOLO inference plus transparent face/chest/buttocks censorship."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from yolo_pose_estimator import YoloPoseEstimator

import cv2
import numpy as np

from censor_policy import CensorResult, OrientationSmoother, build_censor_results
from orientation_classifier import PersonOrientationClassifier
from pose_tracker import PersonTracker
from orientation_debug import OrientationDebugRecorder
from mosaic_renderer import mosaic_region, mosaic_frame, union_mask


@dataclass(frozen=True, slots=True)
class CensorFrame:
    overlay_bgra: np.ndarray
    results: tuple[CensorResult, ...]
    inference_ms: float
    total_ms: float


class YoloCensorPipeline:
    def __init__(
        self,
        model_path: str | Path,
        *,
        mosaic_ratio: int = 24,
        mode: str = "mosaic",
        synchronized_frame: bool = False,
        orientation_model_path: str | Path | None = None,
        enabled_regions: frozenset[str] | None = None,
        exclude_eye_from_face: bool = False,
        invert: bool = False,
    ) -> None:
        self.estimator = YoloPoseEstimator(model_path)
        self.mosaic_ratio = max(2, int(mosaic_ratio))
        self.mode = mode
        self.synchronized_frame = synchronized_frame
        self.enabled_regions = frozenset(enabled_regions) if enabled_regions is not None else frozenset(
            ("face", "chest", "buttocks")
        )
        self.exclude_eye_from_face = exclude_eye_from_face
        self.invert = invert
        self.orientation_classifier = PersonOrientationClassifier(
            orientation_model_path or Path("models") / "orientation.onnx"
        )
        self.person_tracker = PersonTracker()
        self.orientation_debug = OrientationDebugRecorder()
        self.smoothers: dict[int, OrientationSmoother] = {}
        self.last_stage_ms: dict[str, float] = {}

    def warmup(self, width: int, height: int, count: int = 30) -> None:
        blank = np.zeros((height, width, 3), np.uint8)
        for _ in range(count):
            self.estimator.estimate(blank)

    @property
    def execution_provider(self) -> str:
        return self.estimator.execution_provider

    @property
    def orientation_classifier_enabled(self) -> bool:
        return self.orientation_classifier.enabled

    def process(self, frame_bgr: np.ndarray) -> CensorFrame:
        started = time.perf_counter()
        pose_started = started
        estimation = self.estimator.estimate(frame_bgr)
        t_pose = time.perf_counter()
        inference_ms = (t_pose - pose_started) * 1000
        persons = sorted(estimation.persons, key=lambda item: item.bbox.x1)
        tracked = self.person_tracker.update(persons)
        track_ids = [track_id for track_id, _ in tracked]
        persons = [person for _, person in tracked]
        t_track = time.perf_counter()
        predictions = self.orientation_classifier.classify(frame_bgr, persons, track_ids)
        t_orient = time.perf_counter()
        results = build_censor_results(
            persons, frame_bgr.shape, inference_ms, self.smoothers, predictions, track_ids
        )
        self.orientation_debug.record(frame_bgr, results)
        polygons = [region.polygon for result in results for region in (result.face, result.eye, result.chest, result.buttocks)
                    if region.polygon and region.name in self.enabled_regions]
        mask = union_mask(frame_bgr.shape, polygons)
        if self.exclude_eye_from_face and "face" in self.enabled_regions:
            for result in results:
                if result.eye.polygon:
                    cv2.fillPoly(mask, [np.array(result.eye.polygon, np.int32)], 0)
        t_mask = time.perf_counter()

        self.last_stage_ms = {
            "pose": (t_pose - pose_started) * 1000,
            "track": (t_track - t_pose) * 1000,
            "orient": (t_orient - t_track) * 1000,
            "censor+mask": (t_mask - t_orient) * 1000,
        }

        if self.synchronized_frame:
            # Composite in plain contiguous BGR first, convert to BGRA once
            # at the end -- not the other way around. Doing the BGRA
            # conversion up front and mutating its BGR-slice view per region
            # meant every per-region write had to land in a channel-sliced
            # (non-contiguous) array; even a cheap blanket slice-assign into
            # that view measured 1-3ms *per region* on a large box, now the
            # single biggest cost in this path. A contiguous BGR buffer lets
            # every region call cv2 in-place directly (mosaic_region, the
            # same fast path the non-synchronized branch below already
            # uses), with the one BGR2BGRA conversion paying a flat ~2ms
            # regardless of how many regions there are.
            rendered = frame_bgr.copy()
            t_copy = time.perf_counter()
            if self.mode == "black":
                rendered[mask == 0 if self.invert else mask > 0] = 0
            elif self.invert:
                # Invert: censor everything EXCEPT the selected region instead
                # of just the region itself. Pixelate the whole frame, then
                # restore the original pixels back inside `mask` (the part
                # that should stay visible).
                rendered = mosaic_frame(frame_bgr, self.mosaic_ratio)
                rendered[mask > 0] = frame_bgr[mask > 0]
            else:
                # mosaic_region pixelates straight from each polygon's own
                # bounding rect -- it never looks at `mask`, so the eye
                # cutout above only matters for black-fill/alpha compositing.
                # Restore the original eye pixels here so it also applies to
                # the pixelated (mosaic) fill.
                for polygon in polygons:
                    mosaic_region(rendered, polygon, self.mosaic_ratio)
                if self.exclude_eye_from_face and "face" in self.enabled_regions:
                    for result in results:
                        if result.eye.polygon:
                            eye_mask = np.zeros(frame_bgr.shape[:2], np.uint8)
                            cv2.fillPoly(eye_mask, [np.array(result.eye.polygon, np.int32)], 255)
                            rendered[eye_mask > 0] = frame_bgr[eye_mask > 0]
                self.last_stage_ms["region_count"] = len(polygons)
            t_regions = time.perf_counter()
            overlay = cv2.cvtColor(rendered, cv2.COLOR_BGR2BGRA)
            finished = time.perf_counter()
            self.last_stage_ms["render_copy"] = (t_copy - t_mask) * 1000
            self.last_stage_ms["render_regions"] = (t_regions - t_copy) * 1000
            self.last_stage_ms["render_convert"] = (finished - t_regions) * 1000
            self.last_stage_ms["render"] = (finished - t_mask) * 1000
            self.last_stage_ms["total"] = (finished - started) * 1000
            return CensorFrame(
                overlay,
                tuple(results),
                inference_ms,
                (finished - started) * 1000,
            )

        effective_mask = cv2.bitwise_not(mask) if self.invert else mask
        overlay = np.zeros((*frame_bgr.shape[:2], 4), np.uint8)
        if self.mode == "black":
            overlay[:, :, 3] = effective_mask
        else:
            if self.invert:
                rendered = mosaic_frame(frame_bgr, self.mosaic_ratio)
            else:
                rendered = frame_bgr.copy()
                for polygon in polygons:
                    mosaic_region(rendered, polygon, self.mosaic_ratio)
            masked_bgr = np.zeros_like(frame_bgr)
            cv2.copyTo(rendered, effective_mask, masked_bgr)
            overlay[:, :, :3] = masked_bgr
            overlay[:, :, 3] = effective_mask
        finished = time.perf_counter()
        self.last_stage_ms["render"] = (finished - t_mask) * 1000
        self.last_stage_ms["total"] = (finished - started) * 1000
        return CensorFrame(overlay, tuple(results), inference_ms, (finished - started) * 1000)
