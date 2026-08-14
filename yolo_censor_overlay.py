"""Qt screen-capture worker and transparent YOLO censorship overlay."""

from __future__ import annotations

import ctypes
import platform
import sys
import time
from pathlib import Path

from yolo_censor_pipeline import YoloCensorPipeline

import cv2
import mss
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QWidget

WDA_EXCLUDEFROMCAPTURE = 0x00000011


def default_model_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "models" / "yolo11n-pose.onnx"
    return Path(__file__).resolve().parent / "models" / "yolo11n-pose.onnx"


def default_orientation_model_path() -> Path:
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    return base / "models" / "orientation.onnx"


class YoloCensorWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    statistics = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, monitor: dict[str, int], *, ratio: int, mode: str, target_fps: float = 60) -> None:
        super().__init__()
        self.monitor = dict(monitor)
        self.ratio = ratio
        self.mode = mode
        self.target_fps = target_fps
        self._running = False

    def run(self) -> None:
        self._running = True
        try:
            # Display the full captured frame together with its censor regions.
            # A transparent regions-only overlay mixes an older inference frame
            # with the current desktop and makes moving video visibly tear.
            pipeline = YoloCensorPipeline(
                default_model_path(),
                mosaic_ratio=self.ratio,
                mode=self.mode,
                synchronized_frame=True,
                orientation_model_path=default_orientation_model_path(),
            )
            pipeline.warmup(self.monitor["width"], self.monitor["height"])
            print(f"[YOLO] provider={pipeline.execution_provider}")
            orientation_provider = pipeline.orientation_classifier.execution_provider
            print(f"[YOLO] orientation={orientation_provider or 'pose-heuristic-fallback'}")
            count, accumulated = 0, 0.0
            with mss.mss() as capture:
                while self._running:
                    started = time.perf_counter()
                    shot = capture.grab(self.monitor)
                    bgra = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
                    frame_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                    pipeline.mosaic_ratio = max(2, int(self.ratio))
                    pipeline.mode = self.mode
                    output = pipeline.process(frame_bgr)
                    height, width = output.overlay_bgra.shape[:2]
                    image = QImage(output.overlay_bgra.data, width, height, width * 4, QImage.Format_ARGB32).copy()
                    self.frame_ready.emit(image)
                    count += 1
                    accumulated += output.total_ms
                    if count % 60 == 0:
                        self.statistics.emit({"fps": 60000.0 / accumulated, "persons": len(output.results),
                                              "inference_ms": output.inference_ms, "total_ms": output.total_ms})
                        accumulated = 0.0
                    remaining = 1.0 / self.target_fps - (time.perf_counter() - started)
                    if remaining > .001:
                        self.msleep(int(remaining * 1000))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self.wait(5000)


class YoloCensorOverlay(QWidget):
    failed = pyqtSignal(str)

    def __init__(self, monitor: dict[str, int], *, ratio: int = 24, mode: str = "mosaic") -> None:
        super().__init__()
        self.monitor = dict(monitor)
        self._image = QImage()
        self.worker = YoloCensorWorker(monitor, ratio=ratio, mode=mode)
        self.worker.frame_ready.connect(self._accept_frame)
        self.worker.statistics.connect(self._log_statistics)
        self.worker.failed.connect(self.failed)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setGeometry(monitor["left"], monitor["top"], monitor["width"], monitor["height"])

    def start(self) -> None:
        self.show()
        if platform.system() == "Windows":
            try:
                ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), WDA_EXCLUDEFROMCAPTURE)
            except Exception as exc:
                print(f"[YOLO] capture exclusion failed: {exc}")
        self.worker.start()

    def set_options(self, mode: str, ratio: int) -> None:
        self.worker.mode = mode
        self.worker.ratio = max(2, int(ratio))

    def _accept_frame(self, image: QImage) -> None:
        self._image = image
        self.update()

    @staticmethod
    def _log_statistics(stats) -> None:
        print(f"[YOLO] fps={stats['fps']:.1f} persons={stats['persons']} inference={stats['inference_ms']:.1f}ms total={stats['total_ms']:.1f}ms")

    def paintEvent(self, event) -> None:
        if not self._image.isNull():
            QPainter(self).drawImage(self.rect(), self._image)

    def closeEvent(self, event) -> None:
        self.worker.stop()
        super().closeEvent(event)
