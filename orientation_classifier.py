"""Optional crop-based ONNX front/side/back orientation classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort
import cv2
import numpy as np

from censor_policy import BodyOrientation
from yolo_pose_estimator import PersonPose


LABELS = (BodyOrientation.FRONT, BodyOrientation.SIDE, BodyOrientation.BACK)


@dataclass(frozen=True, slots=True)
class OrientationPrediction:
    orientation: BodyOrientation
    front_score: float
    side_score: float
    back_score: float
    confidence: float


class PersonOrientationClassifier:
    """Run an optional NCHW 224x224 ONNX classifier (front, side, back)."""

    def __init__(self, model_path: str | Path, *, ema_alpha: float = .35,
                 minimum_confidence: float = .58) -> None:
        self.path = Path(model_path)
        self.ema_alpha = float(ema_alpha)
        self.minimum_confidence = float(minimum_confidence)
        self._ema: dict[int, np.ndarray] = {}
        self._session: ort.InferenceSession | None = None
        self._input = ""
        if self.path.is_file():
            available = ort.get_available_providers()
            providers = [name for name in ("DmlExecutionProvider", "CPUExecutionProvider") if name in available]
            if not providers:
                raise RuntimeError(f"No supported ONNX execution provider: {available}")
            self._session = ort.InferenceSession(str(self.path), providers=providers)
            self._input = self._session.get_inputs()[0].name

    @property
    def enabled(self) -> bool:
        return self._session is not None

    @property
    def execution_provider(self) -> str | None:
        return None if self._session is None else self._session.get_providers()[0]

    @staticmethod
    def _crop(frame: np.ndarray, person: PersonPose) -> np.ndarray:
        height, width = frame.shape[:2]
        box = person.bbox
        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        x1, y1 = max(0, int(box.x1 - bw*.05)), max(0, int(box.y1 - bh*.04))
        x2, y2 = min(width, int(box.x2 + bw*.05)), min(height, int(box.y2 + bh*.04))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Empty person crop")
        scale = min(224/crop.shape[1], 224/crop.shape[0])
        resized = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
        canvas = np.full((224, 224, 3), 114, np.uint8)
        left, top = (224-resized.shape[1])//2, (224-resized.shape[0])//2
        canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        rgb = (rgb-np.array((.485,.456,.406), np.float32))/np.array((.229,.224,.225), np.float32)
        return np.transpose(rgb, (2, 0, 1))

    def classify(self, frame: np.ndarray, persons: list[PersonPose],
                 track_ids: list[int] | None = None) -> list[OrientationPrediction] | None:
        if self._session is None or not persons:
            return None
        batch = np.stack([self._crop(frame, person) for person in persons]).astype(np.float32)
        output = np.asarray(self._session.run(None, {self._input: batch})[0], np.float32).reshape(len(persons), -1)
        if output.shape[1] != 3:
            raise ValueError(f"Orientation model must output N x 3, got {output.shape}")
        shifted = output-output.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions = []
        keys = track_ids or list(range(len(persons)))
        for slot, current in enumerate(probabilities):
            key = keys[slot]
            previous = self._ema.get(key)
            smoothed = current if previous is None else self.ema_alpha*current+(1-self.ema_alpha)*previous
            self._ema[key] = smoothed
            best = int(np.argmax(smoothed)); confidence = float(smoothed[best])
            orientation = LABELS[best] if confidence >= self.minimum_confidence else BodyOrientation.UNKNOWN
            predictions.append(OrientationPrediction(orientation, *map(float, smoothed), confidence))
        for stale in set(self._ema)-set(keys):
            del self._ema[stale]
        return predictions
