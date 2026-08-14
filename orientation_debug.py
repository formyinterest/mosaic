"""Bounded review-frame recorder for orientation threshold tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import cv2
import numpy as np


class OrientationDebugRecorder:
    def __init__(self, directory: str | None = None, *, maximum_files: int = 500) -> None:
        configured = directory or os.environ.get("MOSAIC_ORIENTATION_DEBUG_DIR")
        self.directory = Path(configured) if configured else None
        self.maximum_files = maximum_files
        self._count = 0
        self._last_saved: dict[int, float] = {}
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, frame: np.ndarray, results) -> None:
        if self.directory is None or self._count >= self.maximum_files:
            return
        now = time.monotonic()
        for result in results:
            confidence = max(result.front_score, result.side_score, result.back_score)
            review = result.orientation_raw != result.orientation_smoothed or confidence < .62
            if not review or now-self._last_saved.get(result.person_index, 0) < .75:
                continue
            x1, y1, x2, y2 = result.bbox
            pad_x, pad_y = int((x2-x1)*.05), int((y2-y1)*.04)
            crop = frame[max(0,y1-pad_y):min(frame.shape[0],y2+pad_y),
                         max(0,x1-pad_x):min(frame.shape[1],x2+pad_x)]
            if not crop.size:
                continue
            stamp = time.time_ns()
            name = f"track{result.person_index}_{stamp}.jpg"
            cv2.imwrite(str(self.directory/name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            metadata = {
                "file": name, "track_id": result.person_index,
                "raw": result.orientation_raw.value,
                "smoothed": result.orientation_smoothed.value,
                "front": result.front_score, "side": result.side_score,
                "back": result.back_score,
                "detector_confidence": result.detector_confidence,
            }
            with (self.directory/"labels.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata, ensure_ascii=False)+"\n")
            self._last_saved[result.person_index] = now
            self._count += 1
            if self._count >= self.maximum_files:
                break
