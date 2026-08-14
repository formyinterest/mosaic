"""YOLO11 pose inference through ONNX Runtime DirectML on Windows Radeon GPUs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import onnxruntime as ort
import cv2
import numpy as np


COCO_POSE_KEYPOINT_COUNT: Final = 17


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float; y1: float; x2: float; y2: float; confidence: float


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    x: float; y: float; confidence: float


@dataclass(frozen=True, slots=True)
class PersonPose:
    bbox: BoundingBox
    keypoints: tuple[PoseKeypoint, ...]
    def __post_init__(self) -> None:
        if len(self.keypoints) != COCO_POSE_KEYPOINT_COUNT:
            raise ValueError("A YOLO COCO pose must contain 17 keypoints")


@dataclass(frozen=True, slots=True)
class PoseEstimation:
    persons: tuple[PersonPose, ...]
    image_width: int
    image_height: int


class YoloPoseEstimator:
    """Run the exported YOLO11n-pose model with DirectML, falling back to CPU."""

    def __init__(self, model_path: str | Path, *, confidence: float = .25, iou: float = .7) -> None:
        self._confidence, self._iou = confidence, iou
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO ONNX model not found: {path}")
        available = ort.get_available_providers()
        preferred = ("DmlExecutionProvider", "CPUExecutionProvider")
        providers = [provider for provider in preferred if provider in available]
        if not providers:
            raise RuntimeError(
                "ONNX Runtime has no supported execution provider; "
                f"available providers: {available}"
            )
        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input = self._session.get_inputs()[0].name

    @property
    def execution_provider(self) -> str:
        return self._session.get_providers()[0]

    @staticmethod
    def _letterbox(image: np.ndarray, size: int = 480):
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = round(w * scale), round(h * scale)
        padded = np.full((size, size, 3), 114, np.uint8)
        left, top = (size - nw) // 2, (size - nh) // 2
        padded[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (size, size), swapRB=True)
        return blob, scale, left, top

    def estimate(self, image_bgr: np.ndarray) -> PoseEstimation:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be an HxWx3 image")
        height, width = image_bgr.shape[:2]
        blob, scale, left, top = self._letterbox(image_bgr)
        output = self._session.run(None, {self._input: blob})[0][0].T
        candidates = output[output[:, 4] >= self._confidence]
        if not len(candidates):
            return PoseEstimation((), width, height)
        boxes = []
        for row in candidates:
            x, y, bw, bh = row[:4]
            boxes.append([float(x - bw / 2), float(y - bh / 2), float(bw), float(bh)])
        kept = cv2.dnn.NMSBoxes(boxes, candidates[:, 4].tolist(), self._confidence, self._iou)
        people = []
        for index in np.asarray(kept).reshape(-1):
            row = candidates[int(index)]
            x, y, bw, bh = row[:4]
            x1, y1 = (x - bw / 2 - left) / scale, (y - bh / 2 - top) / scale
            x2, y2 = (x + bw / 2 - left) / scale, (y + bh / 2 - top) / scale
            points = []
            for point in row[5:].reshape(COCO_POSE_KEYPOINT_COUNT, 3):
                px, py, score = point
                points.append(PoseKeypoint(float((px - left) / scale), float((py - top) / scale), float(score)))
            people.append(PersonPose(BoundingBox(float(np.clip(x1, 0, width)), float(np.clip(y1, 0, height)), float(np.clip(x2, 0, width)), float(np.clip(y2, 0, height)), float(row[4])), tuple(points)))
        return PoseEstimation(tuple(people), width, height)
