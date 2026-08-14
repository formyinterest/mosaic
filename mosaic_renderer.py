"""Pixel mosaic implementations used by production candidates and regression."""

from __future__ import annotations

import cv2
import numpy as np


def _prepare(frame: np.ndarray, polygon: tuple[tuple[int, int], ...], ratio: int):
    points = np.asarray(polygon, np.int32)
    x, y, width, height = cv2.boundingRect(points)
    if width < 2 or height < 2:
        return None
    patch = frame[y:y + height, x:x + width]
    local_points = points - np.array([x, y], dtype=np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [local_points], 255)
    reduced = cv2.resize(
        patch,
        (max(1, width // ratio), max(1, height // ratio)),
        interpolation=cv2.INTER_AREA,
    )
    pixelated = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
    return patch, pixelated, mask


def mosaic_region_baseline(frame: np.ndarray, polygon: tuple[tuple[int, int], ...], ratio: int) -> None:
    """Frozen baseline: NumPy boolean-index compositing."""
    prepared = _prepare(frame, polygon, ratio)
    if prepared is None:
        return
    patch, pixelated, mask = prepared
    patch[mask > 0] = pixelated[mask > 0]


def mosaic_region_copyto(frame: np.ndarray, polygon: tuple[tuple[int, int], ...], ratio: int) -> None:
    """Candidate optimization: OpenCV masked copy without boolean temporaries."""
    prepared = _prepare(frame, polygon, ratio)
    if prepared is None:
        return
    patch, pixelated, mask = prepared
    cv2.copyTo(pixelated, mask, patch)


# Adopted production renderer. Keep the frozen baseline above only for exact
# regression and benchmark comparisons.
mosaic_region = mosaic_region_copyto


def mosaic_frame(frame: np.ndarray, ratio: int) -> np.ndarray:
    """Pixelate the entire frame (used for "invert" censoring, where almost
    everything gets mosaicked). A synthetic full-frame polygon through
    mosaic_region would hit cv2.boundingRect's inclusive-corner convention,
    which computes a box 1px taller/wider than the frame for a rect whose
    corner sits exactly at (width, height) -- cv2.copyTo then silently no-ops
    on the resulting shape mismatch instead of raising. Skip the polygon
    machinery entirely and just resize the whole frame down and back up."""
    height, width = frame.shape[:2]
    reduced = cv2.resize(frame, (max(1, width // ratio), max(1, height // ratio)), interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)


def union_mask(shape: tuple[int, int, int], polygons) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for polygon in polygons:
        if polygon:
            cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
    return mask


def _demo() -> None:
    """The adopted cv2.copyTo renderer must byte-match the boolean-index
    baseline it replaced."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    polygons = [
        ((20, 20), (150, 20), (150, 120), (20, 120)),
        ((200, 150), (380, 150), (380, 280), (200, 280)),
    ]
    expected = frame.copy()
    for polygon in polygons:
        mosaic_region_baseline(expected, polygon, 12)

    actual = frame.copy()
    for polygon in polygons:
        mosaic_region(actual, polygon, 12)

    assert np.array_equal(expected, actual), "mosaic_region diverged from mosaic_region_baseline"
    print("mosaic_renderer self-check OK")


if __name__ == "__main__":
    _demo()
