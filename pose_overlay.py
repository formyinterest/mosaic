"""Compatibility entry point; YOLO censorship now lives in the main application."""

from __future__ import annotations

import runpy
from pathlib import Path

from yolo_censor_overlay import YoloCensorOverlay, YoloCensorWorker
from yolo_censor_pipeline import YoloCensorPipeline


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("mosaic.py")), run_name="__main__")
