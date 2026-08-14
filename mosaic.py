import sys
import os

# A windowed (console-less) PyInstaller build has no attached console, so
# sys.stdout/sys.stderr are None. Every print()/traceback in this codebase
# would otherwise raise AttributeError the first time it fires.
if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w")
    sys.stdout = sys.stdout or _devnull
    sys.stderr = sys.stderr or _devnull

import time
import random
import platform
import ctypes
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from yolo_censor_overlay import YoloCensorOverlay, default_model_path

import cv2
import numpy as np
import mss
import keyboard

from PyQt5.QtWidgets import (
    QApplication, QWidget, QSystemTrayIcon, QMenu,
    QAction, qApp, QStyle, QInputDialog, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QButtonGroup, QRadioButton, QCheckBox,
    QGroupBox, QFrame, QComboBox, QLineEdit, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QCursor

# 전부 왼손만으로 누를 수 있도록 Ctrl+Alt+[왼손 쪽 키](QWERTY 기준 1-5, Q~T, A~G, Z~B)로
# 통일했다. 다이얼로그 안에서 슬라이더 조절/중첩 메뉴 이동은 여전히 방향키(오른손)가
# 필요 -- Tab/Space만으로 되는 포커스 이동·버튼 확정과 달리 이 두 개는 남은 제약이다.
HOTKEY: str = "ctrl+alt+s"          # 전체 모자이크 ON/OFF
HOTKEY_NEW: str = "ctrl+alt+w"      # 새 영역 추가 (Window)
HOTKEY_PEEK: str = "ctrl+alt+f"     # 랜덤 잠깐 해제 ON/OFF
HOTKEY_MON: str = "ctrl+alt+d"      # 모니터 전체 모자이크 (Display)
HOTKEY_YOLO: str = "ctrl+alt+g"     # YOLO 자동 검열 토글 (GPU 인체 탐지)
HOTKEY_MENU: str = "ctrl+alt+q"     # 트레이 메뉴 열기 (Quick menu) -- 세션 프리셋/YOLO
                                     # 모드·강도·모니터 선택/확대창 제어/종료 등 메뉴에만
                                     # 있는 옵션은 전부 이걸로 진입한다
HOTKEY_NEXT_WIN: str = "ctrl+alt+c"  # 다음 창 설정 열기 (Cycle) -- 6~9번 창은 숫자키가
                                     # 오른손이라 직접 못 누르므로 이걸로 순환 접근
# 창별 설정 단축키: ctrl+alt+1 ~ ctrl+alt+9 (1~5는 왼손, 6~9는 위 HOTKEY_NEXT_WIN으로 순환)
HOTKEY_WIN: List[str] = [f"ctrl+alt+{i}" for i in range(1, 10)]
MAX_MOSAIC_WINDOWS: int = 9

GWL_EXSTYLE: int = -20
WS_EX_TRANSPARENT: int = 0x00000020
WDA_EXCLUDEFROMCAPTURE: int = 0x00000011

MIN_WIDTH: int = 80
MIN_HEIGHT: int = 60

PEEK_TIME_RATIO: float = 2.3  # 모자이크 중 체감 시간 배율 (표시 1초 = 실제 2.5초)


ZOOM_DEBUG_LOG: str = "mosaic_zoom_debug.log"


SESSION_PRESETS_FILE: str = "mosaic_session_presets.json"

DEFAULT_SESSION_PRESETS: Dict[str, Dict[str, Any]] = {
    "기본 세션": {
        "wait_minimum": 3,
        "wait_maximum": 10,
        "show_minimum": 10,
        "show_maximum": 20,
        "mosaic_ratio": 30,
        "mode": "mosaic",
    }
}


def session_presets_path() -> Path:
    """Return the local preset file path for source and bundled runs."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / SESSION_PRESETS_FILE
    return Path(__file__).resolve().with_name(SESSION_PRESETS_FILE)


def zoom_debug(message: str) -> None:
    """Append lightweight zoom diagnostics for fullscreen capture issues."""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(ZOOM_DEBUG_LOG, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


DXCAM_AVAILABLE: bool = False
try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False


class DXCamRegistry:
    """dxcam 인스턴스를 모니터별로 관리하고 스레드 간 공유하는 레지스트리 클래스."""
    _instances: Dict[int, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def get_camera(cls, screen_index: int, target_fps: int = 240) -> Optional[Any]:
        if not DXCAM_AVAILABLE:
            return None
        if screen_index not in cls._instances:
            # 여러 모자이크 창(=여러 QThread)이 같은 모니터를 동시에 캡처 시작할 수
            # 있음 -- 락 없이 두 스레드가 동시에 같은 output_idx로 dxcam.create를
            # 부르면 DXGI Desktop Duplication이 멈춰버려서 잠금으로 직렬화한다.
            with cls._lock:
                if screen_index not in cls._instances:
                    try:
                        # 성능 극대화를 위해 BGRA 형식 사용 (QImage.Format_ARGB32와 1-1 매핑)
                        camera = dxcam.create(output_idx=screen_index, output_color="BGRA")
                        camera.start(target_fps=target_fps, video_mode=True)
                        cls._instances[screen_index] = camera
                    except Exception as e:
                        print(f"[DXCamRegistry] Failed to create dxcam for screen {screen_index}: {e}")
                        cls._instances[screen_index] = None
        return cls._instances[screen_index]

    @classmethod
    def stop_all(cls) -> None:
        for camera in cls._instances.values():
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass
        cls._instances.clear()


class CaptureWorker(QThread):
    """화면을 캡처하고 모자이크 처리를 수행하는 스레드 워커 클래스.

    Attributes:
        frame_ready (pyqtSignal): 처리가 완료된 QImage 프레임을 전달하는 시그널.
    """
    frame_ready = pyqtSignal(QImage)

    def __init__(self, mosaic_ratio: int = 24) -> None:
        """CaptureWorker 초기화 메서드.

        Args:
            mosaic_ratio (int): 모자이크 처리 강도. 기본값 24.
        """
        super().__init__()
        self.mosaic_ratio: int = max(2, mosaic_ratio)
        self._monitor_area: Optional[Dict[str, int]] = None
        self._is_running: bool = False
        self.fade_ratio: float = float(mosaic_ratio)  # 실제 렌더링에 사용되는 강도 (fade 도중 변화)
        self._screen_index: int = 0
        self._local_bounds: Tuple[int, int, int, int] = (0, 0, 1, 1)  # (top, left, width, height)
        self.target_fps: float = 60.0

    def set_monitor(self, monitor_area: Dict[str, int]) -> None:
        """mss 캡처용 모니터 영역을 설정합니다."""
        self._monitor_area = monitor_area

    def set_capture_target(self, screen_index: int, local_bounds: Tuple[int, int, int, int], monitor_area: Dict[str, int]) -> None:
        """dxcam 및 mss 캡처 대상을 설정합니다.

        Args:
            screen_index (int): dxcam 모니터 인덱스.
            local_bounds (Tuple[int, int, int, int]): dxcam 크롭 영역 (top, left, width, height).
            monitor_area (Dict[str, int]): mss fallback용 절대 좌표 영역.
        """
        self._screen_index = screen_index
        self._local_bounds = local_bounds
        self._monitor_area = monitor_area
    
    def set_target_fps(self, fps: float) -> None:
        """목표 프레임 레이트를 설정합니다."""
        self.target_fps = max(1.0, float(fps))

    def set_mosaic_ratio(self, ratio: int) -> None:
        """모자이크 강도를 설정합니다.

        Args:
            ratio (int): 변경할 모자이크 비율 값.
        """
        self.mosaic_ratio = max(2, int(ratio))

    def run(self) -> None:
        """캡처 스레드의 메인 루프를 실행합니다."""
        self._is_running = True

        with mss.mss() as screen_capture:
            while self._is_running:
                # 스레드 경합 방지를 위해 로컬 변수로 복사
                screen_idx = self._screen_index
                top, left, w, h = self._local_bounds
                monitor = self._monitor_area
                target_fps = self.target_fps
                target_milliseconds = 1000.0 / target_fps

                if monitor is None or monitor.get("width", 0) < 1 or monitor.get("height", 0) < 1:
                    self.msleep(16)
                    continue

                start_time: float = time.perf_counter()
                try:
                    frame = None
                    if DXCAM_AVAILABLE:
                        # 모니터 주사율 또는 120 FPS 중 큰 값으로 dxcam 카메라 연동
                        camera = DXCamRegistry.get_camera(screen_idx, target_fps=max(int(target_fps), 120))
                        if camera is not None:
                            raw_frame = camera.get_latest_frame()
                            if raw_frame is not None:
                                f_h, f_w = raw_frame.shape[:2]
                                t_val = max(0, min(top, f_h))
                                b_val = max(0, min(top + h, f_h))
                                l_val = max(0, min(left, f_w))
                                r_val = max(0, min(left + w, f_w))
                                if b_val > t_val and r_val > l_val:
                                    frame = raw_frame[t_val:b_val, l_val:r_val]

                    if frame is None:
                        # mss fallback
                        raw_capture = screen_capture.grab(monitor)
                        frame = np.frombuffer(raw_capture.raw, dtype=np.uint8).reshape(
                            raw_capture.height, raw_capture.width, 4
                        )
                    height, width = frame.shape[:2]
                    current_ratio: int = max(2, int(self.fade_ratio))
                    
                    small_image = cv2.resize(
                        frame,
                        (max(1, width // current_ratio), max(1, height // current_ratio)),
                        interpolation=cv2.INTER_AREA
                    )
                    mosaic_image = cv2.resize(
                        small_image,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # BGRA 데이터를 불필요한 스왑/복사 없이 ARGB32로 직접 매핑
                    qt_image: QImage = QImage(
                        mosaic_image.data,
                        width,
                        height,
                        width * 4,
                        QImage.Format_ARGB32
                    ).copy()
                    
                    self.frame_ready.emit(qt_image)
                except Exception:
                    pass

                elapsed_milliseconds: float = (time.perf_counter() - start_time) * 1000.0
                remaining_milliseconds: float = target_milliseconds - elapsed_milliseconds
                
                if remaining_milliseconds > 1.0:
                    self.msleep(int(remaining_milliseconds))

    def stop(self) -> None:
        """스레드를 중지하고 자원을 정리합니다."""
        self._is_running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)
        try:
            self.frame_ready.disconnect()
        except Exception:
            pass


class YoloWindowWorker(QThread):
    """단일 모자이크 창의 사각형 범위 안에서만 YOLO 인체 탐지를 수행하는 스레드 워커.

    CaptureWorker와 동일한 공개 인터페이스(set_capture_target/set_target_fps/
    set_mosaic_ratio/stop/frame_ready)를 제공해 MosaicApp이 두 워커를 구분 없이
    다룰 수 있도록 한다. 탐지된 얼굴/가슴/엉덩이 영역에는 모자이크를, 나머지는
    같은 캡처 프레임에서 그대로 가져온 배경을 합성해 하나의 불투명 이미지로
    그린다(synchronized_frame=True). 배경을 진짜 투명 처리해 그 아래 실시간
    바탕화면이 비치게 하면, 그 배경은 프레임 지연이 전혀 없는데 모자이크
    영역은 캡처+추론 지연만큼 뒤처져 있어 움직이는 화면에서 서로 어긋나
    보인다 -- 전체 모니터용 YoloCensorOverlay가 이미 같은 이유로
    synchronized_frame=True를 쓰고 있음.
    """
    frame_ready = pyqtSignal(QImage)

    def __init__(self, mosaic_ratio: int = 24) -> None:
        super().__init__()
        self.mosaic_ratio: int = max(2, mosaic_ratio)
        self.fade_ratio: float = float(self.mosaic_ratio)  # CaptureWorker와의 인터페이스 호환용 (미사용)
        self._monitor_area: Optional[Dict[str, int]] = None
        self._is_running: bool = False
        self.target_fps: float = 30.0
        self.enabled_regions: frozenset = frozenset(("face", "chest", "buttocks"))
        self.fill_mode: str = "mosaic"
        self.exclude_eye_from_face: bool = False
        self.invert: bool = False
        self._screen_index: int = 0
        self._local_bounds: Tuple[int, int, int, int] = (0, 0, 1, 1)

    def set_monitor(self, monitor_area: Dict[str, int]) -> None:
        self._monitor_area = monitor_area

    def set_capture_target(self, screen_index: int, local_bounds: Tuple[int, int, int, int], monitor_area: Dict[str, int]) -> None:
        """CaptureWorker와 동일하게 dxcam 우선, mss 폴백으로 캡처한다."""
        self._screen_index = screen_index
        self._local_bounds = local_bounds
        self._monitor_area = monitor_area

    def set_target_fps(self, fps: float) -> None:
        self.target_fps = max(1.0, float(fps))

    def set_mosaic_ratio(self, ratio: int) -> None:
        self.mosaic_ratio = max(2, int(ratio))

    def set_regions(self, regions) -> None:
        """검열할 부위 집합(face/chest/buttocks 중 일부)을 설정합니다."""
        self.enabled_regions = frozenset(regions)

    def set_fill_mode(self, mode: str) -> None:
        """탐지 영역 채우기 방식 설정 ('mosaic' 또는 'black')."""
        self.fill_mode = mode

    def set_exclude_eye_from_face(self, enabled: bool) -> None:
        self.exclude_eye_from_face = enabled

    def set_invert(self, enabled: bool) -> None:
        """켜면 선택된 부위 대신 그 부위를 제외한 나머지 전체가 검열 대상이 됩니다."""
        self.invert = enabled

    def run(self) -> None:
        self._is_running = True
        try:
            from yolo_censor_pipeline import YoloCensorPipeline
            from yolo_censor_overlay import default_model_path, default_orientation_model_path

            pipeline = YoloCensorPipeline(
                default_model_path(),
                mosaic_ratio=self.mosaic_ratio,
                mode=self.fill_mode,
                synchronized_frame=True,
                orientation_model_path=default_orientation_model_path(),
                enabled_regions=self.enabled_regions,
                exclude_eye_from_face=self.exclude_eye_from_face,
                invert=self.invert,
            )
        except Exception:
            self._is_running = False
            return

        profile_enabled = os.environ.get("MOSAIC_PROFILE") == "1"
        profile_totals: Dict[str, float] = {}
        profile_frames = 0
        profile_window_start = time.perf_counter()

        with mss.mss() as screen_capture:
            while self._is_running:
                screen_idx = self._screen_index
                top, left, w, h = self._local_bounds
                monitor = self._monitor_area
                target_fps = self.target_fps
                target_milliseconds = 1000.0 / target_fps

                if monitor is None or monitor.get("width", 0) < 1 or monitor.get("height", 0) < 1:
                    self.msleep(16)
                    continue

                start_time: float = time.perf_counter()
                try:
                    t_capture_start = time.perf_counter()
                    frame_bgra = None
                    frame_source = "none"
                    if DXCAM_AVAILABLE:
                        camera = DXCamRegistry.get_camera(screen_idx, target_fps=max(int(target_fps), 120))
                        if camera is not None:
                            raw_frame = camera.get_latest_frame()
                            if raw_frame is not None:
                                f_h, f_w = raw_frame.shape[:2]
                                t_val = max(0, min(top, f_h))
                                b_val = max(0, min(top + h, f_h))
                                l_val = max(0, min(left, f_w))
                                r_val = max(0, min(left + w, f_w))
                                if b_val > t_val and r_val > l_val:
                                    frame_bgra = raw_frame[t_val:b_val, l_val:r_val]
                                    frame_source = "dxcam"
                            else:
                                frame_source = "dxcam-none-frame"
                        else:
                            frame_source = "dxcam-none-camera"

                    if frame_bgra is None:
                        raw_capture = screen_capture.grab(monitor)
                        frame_bgra = np.frombuffer(raw_capture.raw, dtype=np.uint8).reshape(
                            raw_capture.height, raw_capture.width, 4
                        )
                        if frame_source == "none":
                            frame_source = "mss"
                    # cv2's cvtColor drops the alpha channel with a real
                    # SIMD memcpy path; np.ascontiguousarray on a
                    # channel-sliced view does an element-by-element gather
                    # instead and measured ~5x slower on a 1080x1920 frame.
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                    t_capture_end = time.perf_counter()

                    pipeline.mosaic_ratio = max(2, int(self.mosaic_ratio))
                    pipeline.enabled_regions = self.enabled_regions
                    pipeline.mode = self.fill_mode
                    pipeline.exclude_eye_from_face = self.exclude_eye_from_face
                    pipeline.invert = self.invert
                    output = pipeline.process(frame_bgr)

                    height, width = output.overlay_bgra.shape[:2]
                    t_qimage_start = time.perf_counter()
                    qt_image: QImage = QImage(
                        output.overlay_bgra.data, width, height, width * 4, QImage.Format_ARGB32
                    ).copy()
                    t_qimage_end = time.perf_counter()
                    self.frame_ready.emit(qt_image)

                    if profile_enabled:
                        stage_ms = {
                            "capture": (t_capture_end - t_capture_start) * 1000.0,
                            **pipeline.last_stage_ms,
                            "qimage": (t_qimage_end - t_qimage_start) * 1000.0,
                            "loop_total": (time.perf_counter() - start_time) * 1000.0,
                        }
                        for key, value in stage_ms.items():
                            profile_totals[key] = profile_totals.get(key, 0.0) + value
                        profile_frames += 1
                        if time.perf_counter() - profile_window_start >= 2.0:
                            averages = ", ".join(
                                f"{key}={value / profile_frames:.1f}ms" for key, value in profile_totals.items()
                            )
                            fps = profile_frames / (time.perf_counter() - profile_window_start)
                            print(f"[MOSAIC_PROFILE] fps={fps:.1f} source={frame_source} {averages}")
                            profile_totals = {}
                            profile_frames = 0
                            profile_window_start = time.perf_counter()
                except Exception:
                    pass

                elapsed_milliseconds: float = (time.perf_counter() - start_time) * 1000.0
                remaining_milliseconds: float = target_milliseconds - elapsed_milliseconds
                if remaining_milliseconds > 1.0:
                    self.msleep(int(remaining_milliseconds))

    def stop(self) -> None:
        """스레드를 중지하고 자원을 정리합니다."""
        self._is_running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)
        try:
            self.frame_ready.disconnect()
        except Exception:
            pass


class ZoomWorker(QThread):
    """Capture a source rectangle and emit the raw frame for fullscreen zoom."""
    frame_ready = pyqtSignal(QImage)

    def __init__(self) -> None:
        super().__init__()
        self._is_running: bool = False
        self._screen_index: int = 0
        self._local_bounds: Tuple[int, int, int, int] = (0, 0, 1, 1)
        self._monitor_area: Dict[str, int] = {"top": 0, "left": 0, "width": 1, "height": 1}
        self.target_fps: float = 60.0
        self._debug_frame_count: int = 0

    def set_capture_target(self, screen_index: int, local_bounds: Tuple[int, int, int, int], monitor_area: Dict[str, int]) -> None:
        self._screen_index = screen_index
        self._local_bounds = local_bounds
        self._monitor_area = monitor_area
        zoom_debug(f"zoom worker target screen={screen_index} local={local_bounds} monitor={monitor_area}")

    def set_target_fps(self, fps: float) -> None:
        self.target_fps = max(1.0, float(fps))

    def run(self) -> None:
        self._is_running = True
        with mss.mss() as screen_capture:
            while self._is_running:
                screen_idx = self._screen_index
                top, left, w, h = self._local_bounds
                monitor = dict(self._monitor_area)
                target_milliseconds = 1000.0 / self.target_fps
                start_time = time.perf_counter()

                try:
                    frame = None
                    frame_source = "none"
                    if DXCAM_AVAILABLE:
                        camera = DXCamRegistry.get_camera(screen_idx, target_fps=max(int(self.target_fps), 120))
                        if camera is not None:
                            raw_frame = camera.get_latest_frame()
                            if raw_frame is not None:
                                f_h, f_w = raw_frame.shape[:2]
                                t_val = max(0, min(top, f_h))
                                b_val = max(0, min(top + h, f_h))
                                l_val = max(0, min(left, f_w))
                                r_val = max(0, min(left + w, f_w))
                                if b_val > t_val and r_val > l_val:
                                    frame = raw_frame[t_val:b_val, l_val:r_val]
                                    frame_source = "dxcam"

                    if frame is not None and self._debug_frame_count < 30:
                        dxcam_mean = float(frame[:, :, :3].mean())
                        if dxcam_mean < 2.0:
                            raw_capture = screen_capture.grab(monitor)
                            fallback_frame = np.frombuffer(raw_capture.raw, dtype=np.uint8).reshape(
                                raw_capture.height, raw_capture.width, 4
                            )
                            fallback_mean = float(fallback_frame[:, :, :3].mean())
                            zoom_debug(
                                f"dxcam black candidate mean={dxcam_mean:.2f}; "
                                f"mss comparison mean={fallback_mean:.2f}"
                            )
                            if fallback_mean > dxcam_mean + 2.0:
                                frame = fallback_frame
                                frame_source = "mss-after-black-dxcam"

                    if frame is None:
                        raw_capture = screen_capture.grab(monitor)
                        frame = np.frombuffer(raw_capture.raw, dtype=np.uint8).reshape(
                            raw_capture.height, raw_capture.width, 4
                        )
                        frame_source = "mss"

                    height, width = frame.shape[:2]
                    self._debug_frame_count += 1
                    if self._debug_frame_count <= 5 or self._debug_frame_count % 60 == 0:
                        rgb_mean = float(frame[:, :, :3].mean()) if height > 0 and width > 0 else -1.0
                        alpha_mean = float(frame[:, :, 3].mean()) if frame.shape[2] > 3 else -1.0
                        zoom_debug(
                            f"zoom frame #{self._debug_frame_count} source={frame_source} size={width}x{height} "
                            f"rgb_mean={rgb_mean:.2f} alpha_mean={alpha_mean:.2f}"
                        )
                    frame = np.ascontiguousarray(frame)
                    qt_image = QImage(
                        frame.tobytes(),
                        width,
                        height,
                        width * 4,
                        QImage.Format_ARGB32
                    ).copy()
                    self.frame_ready.emit(qt_image)
                except Exception as exc:
                    zoom_debug(f"zoom worker exception: {type(exc).__name__}: {exc}")

                remaining_milliseconds = target_milliseconds - ((time.perf_counter() - start_time) * 1000.0)
                if remaining_milliseconds > 1.0:
                    self.msleep(int(remaining_milliseconds))

    def stop(self) -> None:
        self._is_running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)
        try:
            self.frame_ready.disconnect()
        except Exception:
            pass


class ZoomFullscreenWindow(QWidget):
    """Fullscreen live zoom view for a selected MosaicApp region."""
    ZOOM_STEPS: List[float] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]

    def __init__(self, manager: "WindowManager", source_window: "MosaicApp", screen_index: int, screen_geometry: QRect) -> None:
        super().__init__()
        self.manager = manager
        self.source_window = source_window
        self.screen_index = screen_index
        self.screen_geometry = QRect(screen_geometry)
        self.initial_base_rect = QRect(source_window.geometry())
        self.base_rect = QRect(self.initial_base_rect)
        self.view_rect = QRect(self.base_rect)
        self.zoom_index = 0
        self._pixmap: Optional[QPixmap] = None
        self._middle_drag_position: Optional[Any] = None
        self._cover_pan_x: int = 0
        self._cover_pan_y: int = 0
        self._inner_mosaic_rects: List[QRect] = []
        self._selected_inner_mosaic: int = -1
        self._inner_drag_index: int = -1
        self._inner_drag_mode: Optional[str] = None
        self._inner_drag_start_position: Optional[Any] = None
        self._inner_drag_start_rect: Optional[QRect] = None
        self._inner_mosaic_ratio: int = 24
        self._inner_mosaic_active: bool = False
        self._overlay_until: float = 0.0
        self._input_passthrough: bool = False
        self._source_was_visible: bool = source_window.isVisible()

        self._clamp_base_rect()
        self.view_rect = QRect(self.base_rect)

        self.worker = ZoomWorker()
        self.worker.frame_ready.connect(self._on_frame)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setGeometry(self.screen_geometry)
        QTimer.singleShot(0, self._apply_capture_exclusion)
        zoom_debug(f"zoom window init screen={screen_index} screen_geo={screen_geometry} base={self.base_rect}")

    @property
    def zoom_factor(self) -> float:
        return self.ZOOM_STEPS[self.zoom_index]

    def start_zoom(self) -> None:
        if self.source_window.is_running:
            self.source_window.stop_mosaic()
        self.source_window.hide()
        self._apply_capture_exclusion()
        self._apply_capture_target()
        zoom_debug(f"zoom start base={self.base_rect} view={self.view_rect} factor={self.zoom_factor:g}")
        self.worker.start()
        self.show()
        self.raise_()
        self._apply_input_passthrough()
        self.manager._rebuild_context_menu()

    def _get_hwnd(self) -> int:
        return int(self.winId())

    def _apply_capture_exclusion(self) -> None:
        try:
            result = ctypes.windll.user32.SetWindowDisplayAffinity(self._get_hwnd(), WDA_EXCLUDEFROMCAPTURE)
            zoom_debug(f"SetWindowDisplayAffinity exclude result={result} hwnd={self._get_hwnd()}")
        except Exception as exc:
            zoom_debug(f"SetWindowDisplayAffinity exclude failed: {exc}")
        self._apply_input_passthrough()

    def set_input_passthrough(self, enabled: bool) -> None:
        self._input_passthrough = enabled
        self._apply_input_passthrough()
        self.manager._rebuild_context_menu()

    def _apply_input_passthrough(self) -> None:
        try:
            hwnd = self._get_hwnd()
            extended_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if self._input_passthrough:
                extended_style |= WS_EX_TRANSPARENT
            else:
                extended_style &= ~WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, extended_style)
        except Exception as exc:
            zoom_debug(f"input passthrough style failed: {exc}")

    def _on_frame(self, qt_image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(qt_image)
        if self.worker._debug_frame_count <= 5 or self.worker._debug_frame_count % 60 == 0:
            zoom_debug(f"paint frame accepted pixmap={self._pixmap.width()}x{self._pixmap.height()} view={self.view_rect}")
        self.update()

    def _display_rect(self) -> QRect:
        target = self.rect()
        if self.view_rect.width() < 1 or self.view_rect.height() < 1:
            return target

        source_ratio = self.view_rect.width() / max(1, self.view_rect.height())
        target_ratio = target.width() / max(1, target.height())
        if source_ratio > target_ratio:
            display_height = target.height()
            display_width = max(1, int(display_height * source_ratio))
        else:
            display_width = target.width()
            display_height = max(1, int(display_width / source_ratio))

        left = (target.width() - display_width) // 2 + self._cover_pan_x
        top = (target.height() - display_height) // 2 + self._cover_pan_y
        rect = QRect(left, top, display_width, display_height)
        return self._clamp_display_rect(rect)

    def _clamp_display_rect(self, rect: QRect) -> QRect:
        target = self.rect()
        left_min = min(0, target.width() - rect.width())
        left_max = 0
        top_min = min(0, target.height() - rect.height())
        top_max = 0
        left = max(left_min, min(rect.left(), left_max))
        top = max(top_min, min(rect.top(), top_max))
        self._cover_pan_x = left - (target.width() - rect.width()) // 2
        self._cover_pan_y = top - (target.height() - rect.height()) // 2
        return QRect(left, top, rect.width(), rect.height())

    def _move_cover_pan(self, delta_x: int, delta_y: int) -> None:
        self._cover_pan_x -= delta_x
        self._cover_pan_y -= delta_y
        self._display_rect()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._pixmap is not None:
            display = self._display_rect()
            scaled_pixmap = self._pixmap.scaled(
                display.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            self._paint_censored_frame(painter, scaled_pixmap, display)
            self._paint_inner_mosaics(painter, scaled_pixmap, display)
        else:
            painter.setPen(QColor(220, 220, 220))
            painter.setFont(QFont("Malgun Gothic", 16, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "캡처 대기 중...")

        if time.monotonic() < self._overlay_until:
            self._paint_zoom_overlay(painter)

    def _paint_zoom_overlay(self, painter: QPainter) -> None:
        text = f"{self.zoom_factor:g}x"
        painter.setFont(QFont("Malgun Gothic", 16, QFont.Bold))
        metrics = painter.fontMetrics()
        padding = 12
        width = metrics.horizontalAdvance(text) + padding * 2
        height = metrics.height() + padding
        rect = QRect((self.width() - width) // 2, 28, width, height)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect, Qt.AlignCenter, text)

    def _paint_censored_frame(self, painter: QPainter, scaled_pixmap: QPixmap, display: QRect) -> None:
        """Zoomed view was showing the raw capture with no censoring at all;
        mirror the source window's main mosaic/black mode over the whole frame."""
        if not self._inner_mosaic_active or self.manager.is_global_peek_showing():
            painter.drawPixmap(display, scaled_pixmap)
            return

        if self.source_window._mode == "black":
            painter.fillRect(display, QColor(0, 0, 0, 255))
            return

        # ponytail: "yolo" mode normally censors only detected regions, but
        # replaying that per-region overlay here would need the pose pipeline
        # wired into the zoom window too. Blanket mosaic instead (over-censor,
        # never under-censor); revisit if per-region YOLO censoring in the
        # zoom view is actually needed.
        #
        # Pixelate self._pixmap (the native-resolution capture) BEFORE
        # scaling up to the zoomed display size, not scaled_pixmap (which is
        # already stretched to display size). mosaic_ratio divides whichever
        # size it's given -- dividing the zoomed-up size made the same ratio
        # look progressively finer/weaker the more you zoomed in, down to
        # nearly uncensored at high zoom.
        mosaic_ratio = max(2, int(self.source_window.mosaic_ratio))
        small_width = max(1, self._pixmap.width() // mosaic_ratio)
        small_height = max(1, self._pixmap.height() // mosaic_ratio)
        small_pixmap = self._pixmap.scaled(
            small_width,
            small_height,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
        mosaic_pixmap = small_pixmap.scaled(
            display.size(),
            Qt.IgnoreAspectRatio,
            Qt.FastTransformation
        )
        painter.drawPixmap(display, mosaic_pixmap)

    def _paint_inner_mosaics(self, painter: QPainter, scaled_pixmap: QPixmap, display: QRect) -> None:
        if not self._inner_mosaic_active:
            return
        if self.manager.is_global_peek_showing():
            return

        mosaic_ratio = max(2, int(self.source_window.mosaic_ratio))
        render_mode = self.source_window._mode

        for index, rect in enumerate(self._inner_mosaic_rects):
            target = rect.intersected(display)
            if target.width() < 2 or target.height() < 2:
                continue

            source = QRect(
                target.x() - display.x(),
                target.y() - display.y(),
                target.width(),
                target.height()
            )
            if render_mode == "black":
                painter.fillRect(target, QColor(0, 0, 0, 255))
            else:
                patch = scaled_pixmap.copy(source)
                small_width = max(1, target.width() // mosaic_ratio)
                small_height = max(1, target.height() // mosaic_ratio)
                small_patch = patch.scaled(
                    small_width,
                    small_height,
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation
                )
                mosaic_patch = small_patch.scaled(
                    target.size(),
                    Qt.IgnoreAspectRatio,
                    Qt.FastTransformation
                )
                painter.drawPixmap(target, mosaic_patch)

            border_color = QColor(255, 220, 60, 230) if index == self._selected_inner_mosaic else QColor(255, 80, 80, 190)
            pen = QPen(border_color, 2, Qt.DashLine)
            pen.setDashPattern([6, 3])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(target.adjusted(1, 1, -1, -1))

    def set_inner_mosaic_active(self, active: bool) -> None:
        self._inner_mosaic_active = active
        self.update()

    def sync_source_options(self) -> None:
        self.update()

    def add_inner_mosaic_region(self, center_position: Optional[Any] = None) -> None:
        display = self._display_rect().intersected(self.rect())
        if display.width() < MIN_WIDTH or display.height() < MIN_HEIGHT:
            return

        if center_position is None:
            center = display.center()
        else:
            center = center_position
            if hasattr(center_position, "x") and hasattr(center_position, "y") and not self.rect().contains(center_position):
                center = self.mapFromGlobal(center_position)
            if not display.contains(center):
                center = display.center()

        width = max(MIN_WIDTH, min(display.width() // 4, 420))
        height = max(MIN_HEIGHT, min(display.height() // 4, 260))
        rect = QRect(center.x() - width // 2, center.y() - height // 2, width, height)
        rect = self._clamp_inner_mosaic_rect(rect)
        self._inner_mosaic_rects.append(rect)
        self._selected_inner_mosaic = len(self._inner_mosaic_rects) - 1
        zoom_debug(f"inner mosaic add rect={rect}")
        self.update()

    def _clamp_inner_mosaic_rect(self, rect: QRect) -> QRect:
        display = self._display_rect().intersected(self.rect())
        width = max(MIN_WIDTH, min(rect.width(), display.width()))
        height = max(MIN_HEIGHT, min(rect.height(), display.height()))
        left = max(display.left(), min(rect.left(), display.right() - width + 1))
        top = max(display.top(), min(rect.top(), display.bottom() - height + 1))
        return QRect(left, top, width, height)

    def _hit_inner_mosaic(self, position: Any) -> Tuple[int, Optional[str]]:
        margin = 10
        for index in range(len(self._inner_mosaic_rects) - 1, -1, -1):
            rect = self._inner_mosaic_rects[index]
            if not rect.adjusted(-margin, -margin, margin, margin).contains(position):
                continue

            near_left = abs(position.x() - rect.left()) <= margin
            near_right = abs(position.x() - rect.right()) <= margin
            near_top = abs(position.y() - rect.top()) <= margin
            near_bottom = abs(position.y() - rect.bottom()) <= margin

            if near_top and near_left:
                return index, "top-left"
            if near_top and near_right:
                return index, "top-right"
            if near_bottom and near_left:
                return index, "bottom-left"
            if near_bottom and near_right:
                return index, "bottom-right"
            if near_left:
                return index, "left"
            if near_right:
                return index, "right"
            if near_top:
                return index, "top"
            if near_bottom:
                return index, "bottom"
            if rect.contains(position):
                return index, "move"
        return -1, None

    def _resize_inner_mosaic(self, start_rect: QRect, mode: str, delta_x: int, delta_y: int) -> QRect:
        left = start_rect.left()
        top = start_rect.top()
        width = start_rect.width()
        height = start_rect.height()

        if "left" in mode:
            new_width = max(MIN_WIDTH, width - delta_x)
            left += width - new_width
            width = new_width
        if "right" in mode:
            width = max(MIN_WIDTH, width + delta_x)
        if "top" in mode:
            new_height = max(MIN_HEIGHT, height - delta_y)
            top += height - new_height
            height = new_height
        if "bottom" in mode:
            height = max(MIN_HEIGHT, height + delta_y)

        return self._clamp_inner_mosaic_rect(QRect(left, top, width, height))

    def _show_zoom_overlay(self) -> None:
        self._overlay_until = time.monotonic() + 1.0
        self.update()
        QTimer.singleShot(1000, self.update)

    def _point_to_source(self, position: Any) -> Tuple[float, float, float, float]:
        display = self._display_rect()
        if not display.contains(position):
            rel_x = 0.5
            rel_y = 0.5
        else:
            rel_x = (position.x() - display.x()) / max(1, display.width())
            rel_y = (position.y() - display.y()) / max(1, display.height())
        source_x = self.view_rect.x() + rel_x * self.view_rect.width()
        source_y = self.view_rect.y() + rel_y * self.view_rect.height()
        return source_x, source_y, rel_x, rel_y

    def wheelEvent(self, event: Any) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return

        next_index = self.zoom_index + (1 if delta > 0 else -1)
        next_index = max(0, min(len(self.ZOOM_STEPS) - 1, next_index))
        if next_index == self.zoom_index:
            return

        focus_x, focus_y, rel_x, rel_y = self._point_to_source(event.pos())
        self.zoom_index = next_index
        self._cover_pan_x = 0
        self._cover_pan_y = 0
        self._set_view_around(focus_x, focus_y, rel_x, rel_y)
        self._show_zoom_overlay()
        event.accept()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.RightButton:
            index, _ = self._hit_inner_mosaic(event.pos())
            if index >= 0:
                removed = self._inner_mosaic_rects.pop(index)
                zoom_debug(f"inner mosaic remove by right click rect={removed}")
                if self._selected_inner_mosaic == index:
                    self._selected_inner_mosaic = -1
                elif self._selected_inner_mosaic > index:
                    self._selected_inner_mosaic -= 1
                self.update()
            else:
                self.add_inner_mosaic_region(event.pos())
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            index, mode = self._hit_inner_mosaic(event.pos())
            if index >= 0 and mode is not None:
                self._selected_inner_mosaic = index
                self._inner_drag_index = index
                self._inner_drag_mode = mode
                self._inner_drag_start_position = event.pos()
                self._inner_drag_start_rect = QRect(self._inner_mosaic_rects[index])
                self.update()
                event.accept()
                return
            self._selected_inner_mosaic = -1
            self.update()

        if event.button() == Qt.MiddleButton:
            self._middle_drag_position = event.globalPos()
            event.accept()
            return

    def mouseMoveEvent(self, event: Any) -> None:
        if self._inner_drag_index >= 0 and self._inner_drag_start_position is not None and self._inner_drag_start_rect is not None:
            delta_x = event.pos().x() - self._inner_drag_start_position.x()
            delta_y = event.pos().y() - self._inner_drag_start_position.y()
            if self._inner_drag_mode == "move":
                moved_rect = QRect(self._inner_drag_start_rect)
                moved_rect.translate(delta_x, delta_y)
                self._inner_mosaic_rects[self._inner_drag_index] = self._clamp_inner_mosaic_rect(moved_rect)
            else:
                self._inner_mosaic_rects[self._inner_drag_index] = self._resize_inner_mosaic(
                    self._inner_drag_start_rect,
                    self._inner_drag_mode,
                    delta_x,
                    delta_y
                )
            self.update()
            event.accept()
            return

        if self._middle_drag_position is None or not (event.buttons() & Qt.MiddleButton):
            index, mode = self._hit_inner_mosaic(event.pos())
            if mode in ("left", "right"):
                self.setCursor(Qt.SizeHorCursor)
            elif mode in ("top", "bottom"):
                self.setCursor(Qt.SizeVerCursor)
            elif mode in ("top-left", "bottom-right"):
                self.setCursor(Qt.SizeFDiagCursor)
            elif mode in ("top-right", "bottom-left"):
                self.setCursor(Qt.SizeBDiagCursor)
            elif mode == "move":
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        current_position = event.globalPos()
        delta_x = self._middle_drag_position.x() - current_position.x()
        delta_y = self._middle_drag_position.y() - current_position.y()
        self._middle_drag_position = current_position

        display = self._display_rect()
        source_dx = int(round(delta_x * self.view_rect.width() / max(1, display.width())))
        source_dy = int(round(delta_y * self.view_rect.height() / max(1, display.height())))
        before_view = QRect(self.view_rect)
        self._move_base_and_view(source_dx, source_dy)
        if self.view_rect == before_view:
            self._move_cover_pan(-delta_x, -delta_y)
        else:
            self._cover_pan_x = 0
            self._cover_pan_y = 0
        event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            self._inner_drag_index = -1
            self._inner_drag_mode = None
            self._inner_drag_start_position = None
            self._inner_drag_start_rect = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.MiddleButton:
            self._middle_drag_position = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.MiddleButton:
            self.zoom_index = 0
            self.base_rect = QRect(self.initial_base_rect)
            self.view_rect = QRect(self.base_rect)
            self._cover_pan_x = 0
            self._cover_pan_y = 0
            self._apply_capture_target()
            self._show_zoom_overlay()
            event.accept()

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
        elif event.key() == Qt.Key_M:
            self.add_inner_mosaic_region()
            event.accept()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if 0 <= self._selected_inner_mosaic < len(self._inner_mosaic_rects):
                removed = self._inner_mosaic_rects.pop(self._selected_inner_mosaic)
                zoom_debug(f"inner mosaic remove rect={removed}")
                self._selected_inner_mosaic = -1
                self.update()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _set_view_around(self, focus_x: float, focus_y: float, rel_x: float, rel_y: float) -> None:
        zoom = self.zoom_factor
        new_width = max(1, int(round(self.base_rect.width() / zoom)))
        new_height = max(1, int(round(self.base_rect.height() / zoom)))
        new_left = int(round(focus_x - rel_x * new_width))
        new_top = int(round(focus_y - rel_y * new_height))
        self.view_rect = QRect(new_left, new_top, new_width, new_height)
        self._clamp_view_rect()
        self._apply_capture_target()

    def _move_base_and_view(self, delta_x: int, delta_y: int) -> None:
        self.view_rect.translate(delta_x, delta_y)
        self._clamp_view_rect()
        self._apply_capture_target()

    def _clamp_base_rect(self) -> None:
        left = max(self.screen_geometry.left(), min(self.base_rect.left(), self.screen_geometry.right() - self.base_rect.width() + 1))
        top = max(self.screen_geometry.top(), min(self.base_rect.top(), self.screen_geometry.bottom() - self.base_rect.height() + 1))
        self.base_rect.moveTo(left, top)

    def _clamp_view_rect(self) -> None:
        bounds = self.base_rect.intersected(self.screen_geometry)
        if bounds.width() < 1 or bounds.height() < 1:
            bounds = QRect(self.screen_geometry.left(), self.screen_geometry.top(), 1, 1)
        width = max(1, min(self.view_rect.width(), bounds.width()))
        height = max(1, min(self.view_rect.height(), bounds.height()))
        left = max(bounds.left(), min(self.view_rect.left(), bounds.right() - width + 1))
        top = max(bounds.top(), min(self.view_rect.top(), bounds.bottom() - height + 1))
        self.view_rect = QRect(left, top, width, height)

    def _apply_capture_target(self) -> None:
        target = QRect(self.view_rect)
        target = target.intersected(self.screen_geometry)
        if target.width() < 1 or target.height() < 1:
            target = QRect(self.screen_geometry.left(), self.screen_geometry.top(), 1, 1)

        local_bounds = (
            max(0, target.top() - self.screen_geometry.top()),
            max(0, target.left() - self.screen_geometry.left()),
            max(1, target.width()),
            max(1, target.height()),
        )
        monitor_bounds = {
            "top": target.top(),
            "left": target.left(),
            "width": max(1, target.width()),
            "height": max(1, target.height()),
        }

        target_fps = 60
        screens = QApplication.screens()
        if 0 <= self.screen_index < len(screens):
            target_fps = int(screens[self.screen_index].refreshRate())
            if target_fps < 30:
                target_fps = 60

        self.worker.set_capture_target(self.screen_index, local_bounds, monitor_bounds)
        self.worker.set_target_fps(target_fps)

    def closeEvent(self, event: Any) -> None:
        zoom_debug(f"zoom close base={self.base_rect} view={self.view_rect} factor={self.zoom_factor:g}")
        self.worker.stop()
        self.source_window.setGeometry(self.initial_base_rect)
        if self._source_was_visible and self.source_window in self.manager.windows:
            self.source_window.show()
            self.source_window.raise_()
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(self._get_hwnd(), 0)
        except Exception:
            pass
        self.manager.remove_zoom_window(self)
        event.accept()


class MosaicApp(QWidget):
    """모자이크 표시 및 설정 가능한 투명 창 인터페이스 클래스.

    Attributes:
        BORDER (int): 가장자리 사이즈 리사이즈 영역의 두께.
        MARGIN (int): 마우스 리사이즈 판정을 위한 마진 범위.
    """
    BORDER: int = 3
    MARGIN: int = 8

    _CURSOR_MAP: Dict[str, Any] = {
        "left": Qt.SizeHorCursor,
        "right": Qt.SizeHorCursor,
        "top": Qt.SizeVerCursor,
        "bottom": Qt.SizeVerCursor,
        "top-left": Qt.SizeFDiagCursor,
        "bottom-right": Qt.SizeFDiagCursor,
        "top-right": Qt.SizeBDiagCursor,
        "bottom-left": Qt.SizeBDiagCursor,
    }

    def __init__(self, manager: "WindowManager", offset: int = 0) -> None:
        """MosaicApp 초기화.

        Args:
            manager (WindowManager): 관리자 객체 참조.
            offset (int): 새 창 띄울 때의 위치 오프셋.
        """
        super().__init__()
        self.manager: "WindowManager" = manager
        self.mosaic_ratio: int = 30
        self.is_running: bool = False
        self._mosaic_pixmap: Optional[QPixmap] = None
        self._drag_position: Optional[Any] = None
        self._resize_direction: Optional[str] = None
        self._resize_start_geometry: Optional[Any] = None
        self._resize_start_global_position: Optional[Any] = None

        # 랜덤 피크 상태 속성
        self._peek_enabled: bool = False
        self._peek_wait_minimum: int = 3
        self._peek_wait_maximum: int = 10
        self._peek_show_minimum: int = 10
        self._peek_show_maximum: int = 20
        self._peek_phase: str = "idle"  # idle / waiting / showing
        self._peek_fire_at: float = 0.0
        self._peek_next_milliseconds: int = 0
        
        self._peek_timer: QTimer = QTimer(self)
        self._peek_timer.setSingleShot(True)
        self._peek_timer.timeout.connect(self._on_peek_tick)
        
        self._tick_timer: QTimer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self.update)
        
        # 페이드 인/아웃 관련
        self._fade_milliseconds: int = 1000
        self._fade_step: int = 50
        self._fade_direction: int = 0  # +1=in, -1=out, 0=idle
        self._fade_timer: QTimer = QTimer(self)
        self._fade_timer.setInterval(self._fade_step)
        self._fade_timer.timeout.connect(self._on_fade_tick)

        self._mode: str = "mosaic"
        self._yolo_enabled: bool = False
        self._yolo_regions: frozenset = frozenset(("face", "chest", "buttocks"))
        self._yolo_exclude_eye_from_face: bool = False
        self._yolo_invert: bool = False

        # FPS 표시 속성
        self.show_fps: bool = False
        self._frame_times: List[float] = []
        self._current_fps: float = 0.0

        self.worker: CaptureWorker = CaptureWorker(self.mosaic_ratio)
        self.worker.frame_ready.connect(self._on_frame)

        self._monitor_timer: QTimer = QTimer(self)
        self._monitor_timer.timeout.connect(self._push_monitor)

        self._build_window(offset)

    def _build_window(self, offset: int) -> None:
        """창 속성 및 초기 기하학적 형태를 설정합니다.

        Args:
            offset (int): 위치 오프셋 값.
        """
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        self.setGeometry(100 + offset, 100 + offset, 400, 300)
        # Deferred to the next event-loop tick because winId()/GetWindowLongW
        # need the native window handle to actually exist first. Reads
        # is_running at fire time rather than hardcoding transparent=False --
        # _mosaic_monitor() calls start_mosaic() synchronously right after
        # construction, so a hardcoded False would fire *after* start_mosaic
        # already turned click-through on and silently turn it back off.
        QTimer.singleShot(0, lambda: self._apply_ex_style(self.is_running))

    def _get_hwnd(self) -> int:
        """현재 창의 핸들(HWND)을 가져옵니다.

        Returns:
            int: 윈도우 핸들 ID.
        """
        return int(self.winId())

    def _apply_ex_style(self, transparent: bool = False) -> None:
        """클릭 투과 등의 EX_STYLE을 윈도우 창에 적용합니다.

        Args:
            transparent (bool): 투명화 여부 플래그.
        """
        hwnd = self._get_hwnd()
        extended_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        
        if transparent:
            extended_style |= WS_EX_TRANSPARENT
        else:
            extended_style &= ~WS_EX_TRANSPARENT
            
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, extended_style)

    def paintEvent(self, event: Any) -> None:
        """창의 페인트 이벤트를 처리합니다.

        Args:
            event (Any): 페인트 이벤트 객체.
        """
        painter = QPainter(self)

        if self.manager.is_global_peek_showing():
            self._paint_showing_phase(painter)
        elif self.is_running:
            self._paint_running_phase(painter)
        else:
            self._paint_idle_phase(painter)

        if self.show_fps and self.is_running:
            self._paint_fps(painter)

    def _paint_fps(self, painter: QPainter) -> None:
        """현재 FPS 표시 오버레이를 그립니다."""
        fps_text = f"FPS: {self._current_fps:.1f}"
        painter.setFont(QFont("Malgun Gothic", 9, QFont.Bold))
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.horizontalAdvance(fps_text)
        text_height = font_metrics.height()
        margin = 4
        
        # 우상단에 표시
        rect_x = self.width() - text_width - margin * 4
        rect_y = margin * 2
        
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect_x, rect_y, text_width + margin * 2, text_height + margin * 2, 4, 4)
        
        painter.setPen(QColor(0, 255, 0))  # 밝은 초록색
        painter.drawText(rect_x + margin, rect_y + margin + font_metrics.ascent(), fps_text)

    def _paint_showing_phase(self, painter: QPainter) -> None:
        """해제 중 페이즈의 화면 렌더링입니다.

        Args:
            painter (QPainter): QPainter 인스턴스.
        """
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        if not (self._peek_enabled and self._peek_next_milliseconds > 0):
            return

        seconds_left = max(0, int(self._peek_fire_at - time.monotonic()) + 1)
        seconds_next_mosaic = round(self._peek_next_milliseconds / PEEK_TIME_RATIO / 1000)
        text = f"🔓 {seconds_left}, 다음 딸딸이: {seconds_next_mosaic}"
        
        painter.setFont(QFont("Malgun Gothic", 11, QFont.Bold))
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        text_height = font_metrics.height()
        margin = 8
        
        rect_x = (self.width() - text_width) // 2 - margin
        rect_y = self.height() - text_height - margin * 4
        
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect_x, rect_y, text_width + margin * 2, text_height + margin * 2, 6, 6)
        
        painter.setPen(QColor(100, 210, 255))
        painter.drawText(rect_x + margin, rect_y + margin + font_metrics.ascent(), text)

    def _paint_running_phase(self, painter: QPainter) -> None:
        """실행 중(모자이크 적용) 페이즈의 화면 렌더링입니다.

        Args:
            painter (QPainter): QPainter 인스턴스.
        """
        if self._mode == "black" and not self._yolo_enabled:
            # 인체 탐지가 꺼져 있을 때만 전체를 단색으로 채운다 -- 켜져 있으면
            # 탐지 영역만 검정으로 합성된 _mosaic_pixmap을 그려야 한다(아래).
            painter.fillRect(self.rect(), QColor(0, 0, 0, 255))
        elif self._mosaic_pixmap:
            painter.drawPixmap(self.rect(), self._mosaic_pixmap)
        else:
            # 워커가 아직 첫 프레임을 만들기 전(시작 직후, 모드 전환 직후)에는
            # 원본 화면이 그대로 비치지 않도록 검정으로 가려 둔다.
            painter.fillRect(self.rect(), QColor(0, 0, 0, 255))

        if not (self._peek_enabled and self._peek_phase == "waiting"):
            return
        
        if self._fade_direction != 0:
            return
        else:
            real_left_seconds = max(0.0, self._peek_fire_at - time.monotonic())
            
        seconds_left = max(0, int(real_left_seconds / PEEK_TIME_RATIO) + 1)
        text = f"{seconds_left}"
        
        painter.setFont(QFont("Malgun Gothic", 11, QFont.Bold))
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        text_height = font_metrics.height()
        margin = 8
        
        rect_x = (self.width() - text_width) // 2 - margin
        rect_y = self.height() - text_height - margin * 4
        
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect_x, rect_y, text_width + margin * 2, text_height + margin * 2, 6, 6)
        
        painter.setPen(QColor(255, 220, 60))
        painter.drawText(rect_x + margin, rect_y + margin + font_metrics.ascent(), text)

    def _paint_idle_phase(self, painter: QPainter) -> None:
        """비활성 선택 상태의 화면 렌더링입니다.

        Args:
            painter (QPainter): QPainter 인스턴스.
        """
        painter.fillRect(self.rect(), QColor(0, 0, 0, 1))
        border_size = self.BORDER
        
        dash_pen = QPen(QColor(255, 60, 60, 220), border_size, Qt.DashLine)
        dash_pen.setDashPattern([6, 3])
        painter.setPen(dash_pen)
        painter.drawRect(border_size // 2, border_size // 2, self.width() - border_size, self.height() - border_size)
        
        painter.setPen(QColor(255, 80, 80, 200))
        painter.setFont(QFont("Malgun Gothic", 9))
        painter.drawText(
            self.rect(), Qt.AlignCenter,
            f"드래그: 이동  /  테두리: 리사이즈\n[{HOTKEY.upper()}]  모자이크 ON/OFF"
        )

    def start_mosaic(self) -> None:
        """모자이크 처리를 시작합니다."""
        if self.is_running:
            return
            
        self.is_running = True
        self._mosaic_pixmap = None
        self._apply_ex_style(transparent=True)
        
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(self._get_hwnd(), WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass
            
        self._push_monitor()
        self._monitor_timer.start(200)
        self._sync_worker_to_state()

        self.update()

        if self._peek_enabled:
            self.manager.ensure_global_peek_running()

    def _needs_worker(self) -> bool:
        """검정 단색 채우기(비-인체 탐지)는 캡처가 필요 없다 -- 그 경우만 워커가 불필요."""
        return self._yolo_enabled or self._mode != "black"

    def _ensure_worker_for_mode(self) -> None:
        """self._yolo_enabled 여부에 맞는 워커 타입인지 확인하고, 다르면 교체합니다."""
        desired_cls = YoloWindowWorker if self._yolo_enabled else CaptureWorker
        if isinstance(self.worker, desired_cls):
            return
        self.worker.stop()
        self.worker = desired_cls(self.mosaic_ratio)
        if isinstance(self.worker, YoloWindowWorker):
            self.worker.set_regions(self._yolo_regions)
            self.worker.set_fill_mode(self._mode)
            self.worker.set_exclude_eye_from_face(self._yolo_exclude_eye_from_face)
            self.worker.set_invert(self._yolo_invert)
        self.worker.frame_ready.connect(self._on_frame)
        self._mosaic_pixmap = None

    def _sync_worker_to_state(self) -> None:
        """실행 중일 때, 현재 _mode/_yolo_enabled에 맞춰 워커를 정리/교체/시작합니다."""
        if not self.is_running:
            return
        if not self._needs_worker():
            self.worker.stop()
            self.worker = CaptureWorker(self.mosaic_ratio)
            self.worker.frame_ready.connect(self._on_frame)
            self._mosaic_pixmap = None
            return
        self._ensure_worker_for_mode()
        if isinstance(self.worker, YoloWindowWorker):
            self.worker.set_fill_mode(self._mode)
        if not self.worker.isRunning():
            self._push_monitor()
            self.worker.fade_ratio = float(self.mosaic_ratio)
            self.worker.start()

    def stop_mosaic(self) -> None:
        """모자이크 처리를 중단하고 일반 창으로 복귀합니다."""
        if not self.is_running:
            return
            
        self._peek_timer.stop()
        self._tick_timer.stop()
        self._fade_timer.stop()
        self._peek_phase = "idle"
        self.is_running = False
        self._monitor_timer.stop()
        
        self.worker.stop()
        self.worker = CaptureWorker(self.mosaic_ratio)
        self.worker.frame_ready.connect(self._on_frame)
        
        self._apply_ex_style(transparent=False)
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(self._get_hwnd(), 0)
        except Exception:
            pass
            
        self._mosaic_pixmap = None
        self.manager.ensure_global_peek_running()
        self.raise_()
        self.update()

    def toggle(self) -> None:
        """모자이크 상태를 토글(시작/정지)합니다."""
        if self.is_running:
            self.stop_mosaic()
        else:
            self.start_mosaic()

    def _next_wait_time_ms(self) -> int:
        """다음 모자이크 대기 유지 시간을 계산합니다.

        Returns:
            int: 대기 시간 밀리초.
        """
        base_time = random.uniform(self._peek_wait_minimum, self._peek_wait_maximum)
        return int(base_time * PEEK_TIME_RATIO * 1000)

    def _next_show_time_ms(self) -> int:
        """다음 모자이크 해제 유지 시간을 계산합니다.

        Returns:
            int: 해제 시간 밀리초.
        """
        return int(random.uniform(self._peek_show_minimum, self._peek_show_maximum) * 1000)

    def _on_peek_tick(self) -> None:
        """피킹 타이머 틱 이벤트 발생 시 상태를 전이합니다."""
        if self._peek_phase == "waiting":
            self._tick_timer.start()
            self._peek_next_milliseconds = self._next_wait_time_ms()
            show_milliseconds = self._next_show_time_ms()
            self._peek_fire_at = time.monotonic() + show_milliseconds / 1000.0
            self._peek_timer.start(show_milliseconds)
            
            if self._peek_enabled:
                self._start_fade(-1)
            else:
                self._peek_phase = "showing"
                self._stop_capture_worker()
                self.update()
                
        elif self._peek_phase == "showing":
            next_wait_milliseconds = self._peek_next_milliseconds if self._peek_next_milliseconds else self._next_wait_time_ms()
            self._peek_phase = "waiting"
            
            self._push_monitor()
            self._monitor_timer.start(200)
            self.worker = CaptureWorker(self.mosaic_ratio)
            self.worker.fade_ratio = 2.0
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.start()
            
            self.update()
            self._peek_fire_at = time.monotonic() + next_wait_milliseconds / 1000.0
            self._peek_timer.start(next_wait_milliseconds)
            self._tick_timer.start()
            
            if self._peek_enabled:
                self._start_fade(+1)

    def _start_fade(self, direction: int) -> None:
        """페이드 인/아웃 타이머를 시작합니다.

        Args:
            direction (int): +1 (Fade In) 혹은 -1 (Fade Out).
        """
        self._fade_direction = direction
        if direction == +1:
            self.worker.fade_ratio = 2.0
        else:
            self.worker.fade_ratio = float(self.mosaic_ratio)
        self._fade_timer.start()

    def _on_fade_tick(self) -> None:
        """페이드 애니메이션의 스텝 타이머 틱을 처리합니다."""
        total_steps = self._fade_milliseconds / self._fade_step
        delta_ratio = (self.mosaic_ratio - 2.0) / total_steps
        
        if self._fade_direction == +1:
            self.worker.fade_ratio = min(self.worker.fade_ratio + delta_ratio, float(self.mosaic_ratio))
            if self.worker.fade_ratio >= self.mosaic_ratio:
                self.worker.fade_ratio = float(self.mosaic_ratio)
                self._fade_timer.stop()
                self._fade_direction = 0
                
        elif self._fade_direction == -1:
            self.worker.fade_ratio = max(self.worker.fade_ratio - delta_ratio, 2.0)
            
        if self.worker.fade_ratio <= 2.0:
            self._fade_timer.stop()
            self._fade_direction = 0
            self._peek_phase = "showing"
            self._stop_capture_worker()
            self._mosaic_pixmap = None
            self.update()

    def _stop_capture_worker(self) -> None:
        """진행 중인 캡처 워커와 모니터 타이머를 중단하고 새로 생성합니다."""
        self._monitor_timer.stop()
        self.worker.stop()
        self.worker = CaptureWorker(self.mosaic_ratio)
        self.worker.frame_ready.connect(self._on_frame)

    def set_peek_enabled(self, enabled: bool) -> None:
        """랜덤 잠깐 해제 기능 플래그를 설정합니다. 실제 타이머는 WindowManager가 통합 관리합니다."""
        self._peek_enabled = enabled
        self._peek_timer.stop()
        self._tick_timer.stop()
        self._fade_timer.stop()
        self._peek_phase = "idle"
        self._fade_direction = 0
        self.manager.on_window_peek_option_changed()
        self.update()

    def set_ratio(self, ratio: int) -> None:
        """선택된 윈도우의 모자이크 강도 셋업.

        Args:
            ratio (int): 새로운 모자이크 비율.
        """
        self.mosaic_ratio = ratio
        self.worker.set_mosaic_ratio(ratio)
        self.manager.sync_zoom_options(self)

    def set_yolo_regions(self, regions) -> None:
        """인체 탐지 모드에서 검열할 부위(face/chest/buttocks 중 일부)를 설정합니다."""
        self._yolo_regions = frozenset(regions)
        if isinstance(self.worker, YoloWindowWorker):
            self.worker.set_regions(self._yolo_regions)

    def set_yolo_exclude_eye_from_face(self, enabled: bool) -> None:
        """얼굴 검열 시 눈 영역만 제외할지 설정합니다."""
        self._yolo_exclude_eye_from_face = enabled
        if isinstance(self.worker, YoloWindowWorker):
            self.worker.set_exclude_eye_from_face(enabled)

    def set_yolo_invert(self, enabled: bool) -> None:
        """켜면 선택된 부위 대신 나머지 전체가 검열 대상이 됩니다."""
        self._yolo_invert = enabled
        if isinstance(self.worker, YoloWindowWorker):
            self.worker.set_invert(enabled)

    def set_yolo_enabled(self, enabled: bool) -> None:
        """인체 탐지 사용 여부를 설정합니다. 채우기 방식(모자이크/검정)은 그대로 _mode를 따릅니다."""
        if enabled == self._yolo_enabled:
            return
        self._yolo_enabled = enabled
        self._sync_worker_to_state()
        self.manager.sync_zoom_options(self)
        self.update()

    def set_mode(self, mode: str) -> None:
        """채우기 방식 설정 (모자이크 / 단색 검정). 인체 탐지 사용 여부는 set_yolo_enabled로 별도 설정합니다.

        Args:
            mode (str): 'mosaic' 또는 'black'.
        """
        if mode == self._mode:
            return

        self._mode = mode
        self._sync_worker_to_state()
        self.manager.sync_zoom_options(self)
        self.update()

    def _on_frame(self, qt_image: QImage) -> None:
        """프레임 준비 완료 시 픽스맵을 업데이트합니다.

        Args:
            qt_image (QImage): 처리된 화면 QImage 데이터.
        """
        self._mosaic_pixmap = QPixmap.fromImage(qt_image)
        
        if self.show_fps:
            current_time = time.perf_counter()
            self._frame_times.append(current_time)
            # 최근 1초 동안의 타임스탬프만 유지
            while self._frame_times and current_time - self._frame_times[0] > 1.0:
                self._frame_times.pop(0)
            
            if len(self._frame_times) > 1:
                self._current_fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])
            else:
                self._current_fps = 0.0
        else:
            self._frame_times.clear()
            self._current_fps = 0.0
            
        self.update()

    def _push_monitor(self) -> None:
        """현재 윈도우 객체의 절대좌표 크기를 워커에 주입합니다."""
        rect = self.geometry()
        
        # 윈도우 중심점이 위치한 모니터 인덱스 및 주사율 감지
        desktop = QApplication.desktop()
        window_center = rect.center()
        screen_index = desktop.screenNumber(window_center)
        
        screens = QApplication.screens()
        if 0 <= screen_index < len(screens):
            screen = screens[screen_index]
            screen_geo = screen.geometry()
            target_fps = int(screen.refreshRate())
            # 비정상적인 주사율(예: 0, 음수)인 경우 60 FPS 폴백
            if target_fps < 30:
                target_fps = 60
        else:
            screen_geo = rect
            target_fps = 60
            screen_index = 0
            
        # 해당 모니터 영역 내에서의 상대적인 로컬 좌표 크롭 범위 계산
        local_left = max(0, rect.x() - screen_geo.x())
        local_top = max(0, rect.y() - screen_geo.y())
        local_right = min(screen_geo.width(), rect.x() + rect.width() - screen_geo.x())
        local_bottom = min(screen_geo.height(), rect.y() + rect.height() - screen_geo.y())
        
        local_width = max(1, local_right - local_left)
        local_height = max(1, local_bottom - local_top)
        
        local_bounds = (local_top, local_left, local_width, local_height)
        
        monitor_bounds = {
            "top": rect.y(), "left": rect.x(),
            "width": rect.width(), "height": rect.height(),
        }
        
        self.worker.set_capture_target(screen_index, local_bounds, monitor_bounds)
        self.worker.set_target_fps(target_fps)

    def _get_window_edge(self, position: Any) -> Optional[str]:
        """주어진 마우스 위치가 창의 가장자리인지 판별합니다.

        Args:
            position (Any): 현재 마우스 이벤트 위치 객체 (QPoint).

        Returns:
            Optional[str]: 가장자리의 이름 ('top-left' 등) 혹은 일반 영역이면 None.
        """
        margin = self.MARGIN
        pos_x, pos_y, box_width, box_height = position.x(), position.y(), self.width(), self.height()
        
        is_left = pos_x < margin
        is_right = pos_x > box_width - margin
        is_top = pos_y < margin
        is_bottom = pos_y > box_height - margin
        
        if is_top and is_left: return "top-left"
        if is_top and is_right: return "top-right"
        if is_bottom and is_left: return "bottom-left"
        if is_bottom and is_right: return "bottom-right"
        if is_top: return "top"
        if is_bottom: return "bottom"
        if is_left: return "left"
        if is_right: return "right"
        
        return None

    def mousePressEvent(self, event: Any) -> None:
        """창 위에서 마우스 클릭 시 이벤트를 처리합니다."""
        if self.is_running or event.button() != Qt.LeftButton:
            return
            
        edge_direction = self._get_window_edge(event.pos())
        if edge_direction:
            self._resize_direction = edge_direction
            self._resize_start_geometry = self.geometry()
            self._resize_start_global_position = event.globalPos()
            self._drag_position = None
        else:
            self._resize_direction = None
            self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
            
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        """마우스 이동/드래그 이벤트를 처리합니다."""
        if self.is_running:
            return
            
        if not (event.buttons() & Qt.LeftButton):
            edge_direction = self._get_window_edge(event.pos())
            self.setCursor(self._CURSOR_MAP[edge_direction] if edge_direction else Qt.ArrowCursor)
            return
            
        if self._resize_direction:
            delta_x = event.globalPos().x() - self._resize_start_global_position.x()
            delta_y = event.globalPos().y() - self._resize_start_global_position.y()
            start_geo = self._resize_start_geometry
            
            pos_x, pos_y = start_geo.x(), start_geo.y()
            box_width, box_height = start_geo.width(), start_geo.height()
            current_direction = self._resize_direction
            
            if "right" in current_direction:
                box_width = max(MIN_WIDTH, box_width + delta_x)
            if "bottom" in current_direction:
                box_height = max(MIN_HEIGHT, box_height + delta_y)
            if "left" in current_direction:
                new_width = max(MIN_WIDTH, box_width - delta_x)
                pos_x += box_width - new_width
                box_width = new_width
            if "top" in current_direction:
                new_height = max(MIN_HEIGHT, box_height - delta_y)
                pos_y += box_height - new_height
                box_height = new_height
                
            self.setGeometry(pos_x, pos_y, box_width, box_height)
            
        elif self._drag_position is not None:
            self.move(event.globalPos() - self._drag_position)
            
        event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:
        """마우스 드롭 및 클릭 해제 이벤트를 처리합니다."""
        if event.button() == Qt.LeftButton:
            self._drag_position = None
            self._resize_direction = None
            self.setCursor(Qt.ArrowCursor)
            self._clamp_bounds_to_screen()
            event.accept()

    def _clamp_bounds_to_screen(self) -> None:
        """창이 화면 밖으로 완전히 나가지 않도록 위치를 보정합니다."""
        current_geometry = self.geometry()
        clamped_x, clamped_y = self._clamp_position(
            current_geometry.x(), current_geometry.y(),
            current_geometry.width(), current_geometry.height()
        )
        
        if clamped_x != current_geometry.x() or clamped_y != current_geometry.y():
            self.move(clamped_x, clamped_y)

    @staticmethod
    def _clamp_position(left: int, top: int, width: int, height: int) -> Tuple[int, int]:
        """전체 가상 화면 내에서 창이 최소 여백 픽셀 이상 보이도록 위치를 제한합니다.

        Args:
            left (int): 현재 좌측 X 좌표.
            top (int): 현재 상단 Y 좌표.
            width (int): 창의 폭.
            height (int): 창의 높이.

        Returns:
            Tuple[int, int]: 보정된 제한 X, Y 좌표 튜플.
        """
        desktop = QApplication.desktop()
        safe_margin = 40
        num_screens = desktop.screenCount()
        
        min_x = min(desktop.screenGeometry(i).left() for i in range(num_screens))
        min_y = min(desktop.screenGeometry(i).top() for i in range(num_screens))
        max_x = max(desktop.screenGeometry(i).right() for i in range(num_screens))
        max_y = max(desktop.screenGeometry(i).bottom() for i in range(num_screens))
        
        clamped_left = max(min_x - width + safe_margin, min(left, max_x - safe_margin))
        clamped_top = max(min_y - height + safe_margin, min(top, max_y - safe_margin))
        
        return clamped_left, clamped_top

    def closeEvent(self, event: Any) -> None:
        """창의 종료 이벤트를 처리하여 타이머 자원을 정리합니다."""
        self._peek_timer.stop()
        self._tick_timer.stop()
        self._fade_timer.stop()
        self._monitor_timer.stop()
        self.worker.stop()
        event.accept()


class PeekStatusOverlay(QWidget):
    """Bottom monitor overlay for the unified random peek countdown."""

    def __init__(self, manager: "WindowManager", screen_geometry: QRect) -> None:
        super().__init__()
        self.manager = manager
        self.screen_geometry = QRect(screen_geometry)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        height = 72
        self.setGeometry(
            self.screen_geometry.left(),
            self.screen_geometry.bottom() - height + 1,
            self.screen_geometry.width(),
            height
        )
        QTimer.singleShot(0, self._apply_click_through)

    def _apply_click_through(self) -> None:
        try:
            hwnd = int(self.winId())
            extended_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            extended_style |= WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, extended_style)
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    def paintEvent(self, event: Any) -> None:
        text = self.manager.global_peek_status_text()
        if not text:
            return

        painter = QPainter(self)
        painter.setFont(QFont("Malgun Gothic", 14, QFont.Bold))
        metrics = painter.fontMetrics()
        padding_x = 18
        padding_y = 10
        width = metrics.horizontalAdvance(text) + padding_x * 2
        height = metrics.height() + padding_y
        rect = QRect((self.width() - width) // 2, self.height() - height - 10, width, height)
        painter.setBrush(QColor(0, 0, 0, 165))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 8, 8)
        painter.setPen(QColor(255, 235, 90) if self.manager._global_peek_phase == "waiting" else QColor(100, 210, 255))
        painter.drawText(rect, Qt.AlignCenter, text)


class WindowManager(QObject):
    """다중 모자이크 창과 트레이 아이콘을 통합 제어하는 관리자 클래스.

    Attributes:
        windows (List[MosaicApp]): 생성된 모자이크 어플리케이션(창) 리스트.
    """
    _signal_toggle = pyqtSignal()
    _signal_new = pyqtSignal()
    _signal_peek = pyqtSignal()
    _signal_monitor = pyqtSignal()
    _signal_yolo = pyqtSignal()
    _signal_window = pyqtSignal(int)
    _signal_menu = pyqtSignal()
    _signal_next_window = pyqtSignal()

    def __init__(self) -> None:
        """WindowManager 초기화."""
        super().__init__()
        self.windows: List[MosaicApp] = []
        self.zoom_windows: List[ZoomFullscreenWindow] = []
        self.peek_overlays: List[PeekStatusOverlay] = []
        self.yolo_overlays: Dict[int, YoloCensorOverlay] = {}
        self._yolo_mode: str = "mosaic"
        self._yolo_ratio: int = 24
        self._default_ratio: int = 30
        self._session_presets: Dict[str, Dict[str, Any]] = self._load_session_presets()
        initial_session = next(iter(self._session_presets.values()))
        self._active_session_preset_name: Optional[str] = None
        self._session_wait_minimum: int = initial_session["wait_minimum"]
        self._session_wait_maximum: int = initial_session["wait_maximum"]
        self._session_show_minimum: int = initial_session["show_minimum"]
        self._session_show_maximum: int = initial_session["show_maximum"]
        self._global_peek_enabled: bool = False
        self._global_peek_phase: str = "idle"
        self._global_peek_fire_at: float = 0.0
        self._global_peek_next_milliseconds: int = 0
        self._global_peek_timer: QTimer = QTimer(self)
        self._global_peek_timer.setSingleShot(True)
        self._global_peek_timer.timeout.connect(self._on_global_peek_tick)
        self._global_tick_timer: QTimer = QTimer(self)
        self._global_tick_timer.setInterval(1000)
        self._global_tick_timer.timeout.connect(self._update_peek_overlays)
        self._next_window_cycle_index: int = 0

        self._signal_toggle.connect(self._toggle_all)
        self._signal_new.connect(self.add_window)
        self._signal_peek.connect(self._toggle_peek)
        self._signal_monitor.connect(self._pick_monitor_mosaic)
        self._signal_yolo.connect(self._toggle_yolo_at_cursor)
        self._signal_window.connect(self._open_window_settings)
        self._signal_menu.connect(self._open_tray_menu_via_keyboard)
        self._signal_next_window.connect(self._open_next_window_settings)

        keyboard.add_hotkey(HOTKEY, lambda: self._signal_toggle.emit())
        keyboard.add_hotkey(HOTKEY_NEW, lambda: self._signal_new.emit())
        keyboard.add_hotkey(HOTKEY_PEEK, lambda: self._signal_peek.emit())
        keyboard.add_hotkey(HOTKEY_MON, lambda: self._signal_monitor.emit())
        keyboard.add_hotkey(HOTKEY_YOLO, lambda: self._signal_yolo.emit())
        keyboard.add_hotkey(HOTKEY_MENU, lambda: self._signal_menu.emit())
        keyboard.add_hotkey(HOTKEY_NEXT_WIN, lambda: self._signal_next_window.emit())

        for index, key in enumerate(HOTKEY_WIN):
            keyboard.add_hotkey(key, lambda mapped_index=index: self._signal_window.emit(mapped_index))

        self._build_tray_icon()

    @staticmethod
    def _normalize_session_preset(raw_preset: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_preset, dict):
            return None

        try:
            wait_minimum = int(raw_preset["wait_minimum"])
            wait_maximum = int(raw_preset["wait_maximum"])
            show_minimum = int(raw_preset["show_minimum"])
            show_maximum = int(raw_preset["show_maximum"])
            mosaic_ratio = int(raw_preset["mosaic_ratio"])
        except (KeyError, TypeError, ValueError):
            return None

        mode = raw_preset.get("mode")
        if (
            mode not in ("mosaic", "black")
            or not 1 <= wait_minimum <= wait_maximum
            or not 1 <= show_minimum <= show_maximum
            or not 2 <= mosaic_ratio <= 64
        ):
            return None

        return {
            "wait_minimum": wait_minimum,
            "wait_maximum": wait_maximum,
            "show_minimum": show_minimum,
            "show_maximum": show_maximum,
            "mosaic_ratio": mosaic_ratio,
            "mode": mode,
        }

    def _load_session_presets(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw_presets = json.loads(session_presets_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_presets = None

        presets: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_presets, dict):
            for name, raw_preset in raw_presets.items():
                normalized = self._normalize_session_preset(raw_preset)
                if isinstance(name, str) and name.strip() and normalized:
                    presets[name.strip()] = normalized

        if presets:
            return presets
        return {name: dict(preset) for name, preset in DEFAULT_SESSION_PRESETS.items()}

    def _save_session_presets(self) -> bool:
        try:
            session_presets_path().write_text(
                json.dumps(self._session_presets, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except OSError:
            return False

    def _apply_session_preset(self, name: str) -> bool:
        preset = self._session_presets.get(name)
        if not preset:
            return False

        self._session_wait_minimum = preset["wait_minimum"]
        self._session_wait_maximum = preset["wait_maximum"]
        self._session_show_minimum = preset["show_minimum"]
        self._session_show_maximum = preset["show_maximum"]
        self._default_ratio = preset["mosaic_ratio"]
        self._active_session_preset_name = name

        for target_window in self.windows:
            target_window.set_mode(preset["mode"])
            target_window.set_ratio(preset["mosaic_ratio"])

        return True

    def _start_session_preset(self, name: str) -> None:
        if not self.windows and not self.zoom_windows:
            self.tray.showMessage("세션", "먼저 가릴 영역을 추가하세요.", QSystemTrayIcon.Warning, 1800)
            return
        if not self._apply_session_preset(name):
            return

        self._stop_global_peek()
        zoom_sources = self._zoom_source_windows()
        for target_window in self.windows:
            if target_window not in zoom_sources and not target_window.is_running:
                target_window.start_mosaic()
        for zoom_window in self.zoom_windows:
            zoom_window.set_inner_mosaic_active(True)
            zoom_window.source_window.is_running = True
        for target_window in self.windows:
            target_window.set_peek_enabled(True)

        self.ensure_global_peek_running()
        self._rebuild_context_menu()
        self.tray.showMessage("세션", f"{name} 세션을 시작했습니다.", QSystemTrayIcon.Information, 1800)

    def _stop_active_session(self) -> None:
        self._stop_global_peek()
        for target_window in self.windows:
            target_window.set_peek_enabled(False)
        self._active_session_preset_name = None
        self._rebuild_context_menu()
        self.tray.showMessage("세션", "세션을 종료하고 가림 상태를 유지합니다.", QSystemTrayIcon.Information, 1800)

    def sync_zoom_options(self, source_window: MosaicApp) -> None:
        for zoom_window in self.zoom_windows:
            if zoom_window.source_window is source_window:
                zoom_window.sync_source_options()

    def _zoom_source_windows(self) -> set:
        return {zoom_window.source_window for zoom_window in self.zoom_windows}

    def is_global_peek_showing(self) -> bool:
        return self._global_peek_enabled and self._global_peek_phase == "showing"

    def _is_any_mosaic_active(self) -> bool:
        zoom_sources = self._zoom_source_windows()
        return any(window.is_running for window in self.windows if window not in zoom_sources) or any(
            zoom_window._inner_mosaic_active for zoom_window in self.zoom_windows
        )

    def _has_peek_enabled_target(self) -> bool:
        return any(window._peek_enabled for window in self.windows) or any(
            zoom_window.source_window._peek_enabled for zoom_window in self.zoom_windows
        )

    def on_window_peek_option_changed(self) -> None:
        self._global_peek_enabled = self._has_peek_enabled_target()
        if self._global_peek_enabled and self._is_any_mosaic_active():
            self.ensure_global_peek_running()
        else:
            self._stop_global_peek()
        self._rebuild_context_menu()

    def ensure_global_peek_running(self) -> None:
        self._global_peek_enabled = self._has_peek_enabled_target()
        if not self._global_peek_enabled or not self._is_any_mosaic_active():
            self._stop_global_peek()
            return
        if self._global_peek_phase == "idle":
            self._start_global_peek_waiting()
        self._sync_global_peek_visuals()

    def _next_global_wait_time_ms(self) -> int:
        return int(random.uniform(self._session_wait_minimum, self._session_wait_maximum) * PEEK_TIME_RATIO * 1000)

    def _next_global_show_time_ms(self) -> int:
        return int(random.uniform(self._session_show_minimum, self._session_show_maximum) * 1000)

    def _start_global_peek_waiting(self, wait_milliseconds: Optional[int] = None) -> None:
        if wait_milliseconds is None:
            wait_milliseconds = self._next_global_wait_time_ms()
        self._global_peek_phase = "waiting"
        self._global_peek_next_milliseconds = self._next_global_wait_time_ms()
        self._global_peek_fire_at = time.monotonic() + wait_milliseconds / 1000.0
        self._global_peek_timer.start(wait_milliseconds)
        self._global_tick_timer.start()
        self._sync_global_peek_visuals()

    def _on_global_peek_tick(self) -> None:
        if not self._global_peek_enabled or not self._is_any_mosaic_active():
            self._stop_global_peek()
            return

        if self._global_peek_phase == "waiting":
            show_milliseconds = self._next_global_show_time_ms()
            self._global_peek_phase = "showing"
            self._global_peek_fire_at = time.monotonic() + show_milliseconds / 1000.0
            self._global_peek_timer.start(show_milliseconds)
        elif self._global_peek_phase == "showing":
            next_wait = self._global_peek_next_milliseconds or self._next_global_wait_time_ms()
            self._start_global_peek_waiting(next_wait)
            return
        self._sync_global_peek_visuals()

    def _stop_global_peek(self) -> None:
        self._global_peek_timer.stop()
        self._global_tick_timer.stop()
        self._global_peek_phase = "idle"
        self._global_peek_fire_at = 0.0
        self._global_peek_next_milliseconds = 0
        self._sync_global_peek_visuals()

    def global_peek_status_text(self) -> str:
        if not (self._global_peek_enabled and self._global_peek_phase in ("waiting", "showing")):
            return ""
        seconds_left = max(0, int(self._global_peek_fire_at - time.monotonic()) + 1)
        if self._global_peek_phase == "waiting":
            return f"모자이크 해제까지 {max(0, int(seconds_left / PEEK_TIME_RATIO) + 1)}초"
        next_seconds = round(self._global_peek_next_milliseconds / PEEK_TIME_RATIO / 1000)
        return f"해제 중 {seconds_left}초 · 다음 모자이크 {next_seconds}초"

    def _sync_global_peek_visuals(self) -> None:
        self._ensure_peek_overlays()
        active = self._global_peek_enabled and self._global_peek_phase in ("waiting", "showing") and self._is_any_mosaic_active()
        for overlay in self.peek_overlays:
            overlay.setVisible(active)
            overlay.update()
        for window in self.windows:
            window.update()
        for zoom_window in self.zoom_windows:
            zoom_window.update()

    def _update_peek_overlays(self) -> None:
        self._sync_global_peek_visuals()

    def _ensure_peek_overlays(self) -> None:
        desktop = QApplication.desktop()
        if len(self.peek_overlays) == desktop.screenCount():
            return
        for overlay in self.peek_overlays:
            overlay.close()
        self.peek_overlays = []
        for screen_index in range(desktop.screenCount()):
            overlay = PeekStatusOverlay(self, desktop.screenGeometry(screen_index))
            overlay.hide()
            self.peek_overlays.append(overlay)

    def _build_tray_icon(self) -> None:
        """시스템 트레이 아이콘 및 컨텍스트 메뉴를 초기 구축합니다."""
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(qApp.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray.setToolTip("화면 모자이크")
        self.tray.activated.connect(self._on_tray_click)
        self._tray_menu = QMenu()
        self._rebuild_context_menu()
        self.tray.setContextMenu(self._tray_menu)
        self.tray.show()
        QTimer.singleShot(500, self._show_startup_notification)

    def _show_startup_notification(self) -> None:
        self.tray.showMessage(
            "모자이크",
            f"실행되었습니다.  [{HOTKEY_MENU.upper()}]로 메뉴 열기",
            QSystemTrayIcon.Information,
            1800
        )

    def _rebuild_context_menu(self) -> None:
        """창 구성 변경 시 시스템 트레이 컨텍스트 메뉴를 다시 작성합니다."""
        self._tray_menu.clear()
        menu_ref = self._tray_menu

        action_toggle = QAction(f"ON / OFF  [{HOTKEY.upper()}]", menu_ref)
        action_toggle.triggered.connect(self._toggle_all)
        menu_ref.addAction(action_toggle)

        yolo_marker = "✔ " if self.yolo_overlays else "    "
        action_yolo_quick = QAction(
            f"{yolo_marker}YOLO 자동 검열  [{HOTKEY_YOLO.upper()}]",
            menu_ref,
        )
        action_yolo_quick.triggered.connect(self._toggle_yolo_at_cursor)
        menu_ref.addAction(action_yolo_quick)
        
        action_new_window = QAction(f"새 영역 추가  [{HOTKEY_NEW.upper()}]", menu_ref)
        action_new_window.triggered.connect(self.add_window)
        menu_ref.addAction(action_new_window)

        session_menu = menu_ref.addMenu("세션 프리셋")
        if self._active_session_preset_name:
            action_stop_session = QAction("세션 종료 (가림 유지)", session_menu)
            action_stop_session.triggered.connect(self._stop_active_session)
            session_menu.addAction(action_stop_session)
            session_menu.addSeparator()

        for preset_name in sorted(self._session_presets):
            marker = "▶ " if preset_name == self._active_session_preset_name else ""
            action_start_session = QAction(f"{marker}{preset_name} 적용 및 시작", session_menu)
            action_start_session.triggered.connect(
                lambda _, name=preset_name: self._start_session_preset(name)
            )
            session_menu.addAction(action_start_session)

        session_menu.addSeparator()
        action_manage_sessions = QAction("프리셋 관리...", session_menu)
        action_manage_sessions.triggered.connect(self._open_session_preset_manager)
        session_menu.addAction(action_manage_sessions)

        is_peek_active = self._global_peek_enabled or bool(self.windows and self.windows[0]._peek_enabled)
        peek_checkbox = "✔ " if is_peek_active else "    "
        action_peek = QAction(f"{peek_checkbox}랜덤 잠깐 해제  [{HOTKEY_PEEK.upper()}]", menu_ref)
        action_peek.triggered.connect(self._toggle_peek)
        menu_ref.addAction(action_peek)

        is_fps_active = bool(self.windows and self.windows[0].show_fps)
        fps_checkbox = "✔ " if is_fps_active else "    "
        action_fps_toggle = QAction(f"{fps_checkbox}모든 창 FPS 표시", menu_ref)
        action_fps_toggle.triggered.connect(self._toggle_all_fps)
        menu_ref.addAction(action_fps_toggle)

        yolo_menu = menu_ref.addMenu("YOLO 인체 영역 검열 (GPU)")
        mode_mosaic = QAction(f"{'✔ ' if self._yolo_mode == 'mosaic' else '    '}모자이크", yolo_menu)
        mode_mosaic.triggered.connect(lambda: self._set_yolo_mode("mosaic"))
        yolo_menu.addAction(mode_mosaic)
        mode_black = QAction(f"{'✔ ' if self._yolo_mode == 'black' else '    '}검정 덮기", yolo_menu)
        mode_black.triggered.connect(lambda: self._set_yolo_mode("black"))
        yolo_menu.addAction(mode_black)
        ratio_action = QAction(f"모자이크 강도: {self._yolo_ratio}", yolo_menu)
        ratio_action.triggered.connect(self._ask_yolo_ratio)
        yolo_menu.addAction(ratio_action)
        yolo_menu.addSeparator()
        desktop_for_yolo = QApplication.desktop()
        for screen_index in range(desktop_for_yolo.screenCount()):
            geometry = desktop_for_yolo.screenGeometry(screen_index)
            marker = "✔ " if screen_index in self.yolo_overlays else "    "
            action = QAction(f"{marker}모니터 {screen_index + 1} ({geometry.width()}×{geometry.height()})", yolo_menu)
            action.triggered.connect(lambda _, index=screen_index: self._toggle_yolo_monitor(index))
            yolo_menu.addAction(action)

        if self.zoom_windows:
            passthrough_active = self.zoom_windows[-1]._input_passthrough
            passthrough_checkbox = "✔ " if passthrough_active else "    "
            action_zoom_passthrough = QAction(f"{passthrough_checkbox}확대 창 입력 통과", menu_ref)
            action_zoom_passthrough.triggered.connect(self._toggle_zoom_input_passthrough)
            menu_ref.addAction(action_zoom_passthrough)

            action_add_inner_mosaic = QAction("확대 창 내부 모자이크 영역 추가", menu_ref)
            action_add_inner_mosaic.triggered.connect(lambda: self.zoom_windows[-1].add_inner_mosaic_region())
            action_add_inner_mosaic.setEnabled(not passthrough_active)
            menu_ref.addAction(action_add_inner_mosaic)

            action_close_zoom = QAction("확대 창 모두 닫기", menu_ref)
            action_close_zoom.triggered.connect(self._close_all_zoom_windows)
            menu_ref.addAction(action_close_zoom)

        menu_ref.addSeparator()
        
        desktop_environment = QApplication.desktop()
        for screen_index in range(desktop_environment.screenCount()):
            screen_geo = desktop_environment.screenGeometry(screen_index)
            is_primary = (screen_index == 0)
            shortcut_label = f"  [{HOTKEY_MON.upper()}]" if is_primary else ""
            display_label = f"    모니터 {screen_index + 1}  ({screen_geo.width()}×{screen_geo.height()}){shortcut_label}"
            
            action_monitor = QAction(display_label, menu_ref)
            action_monitor.triggered.connect(lambda _, index=screen_index: self._mosaic_monitor(index))
            menu_ref.addAction(action_monitor)

        if self.windows:
            menu_ref.addSeparator()
            for window_index, target_window in enumerate(self.windows):
                hotkey_label = f"  [CTRL+ALT+{window_index + 1}]" if window_index < 9 else ""
                submenu = menu_ref.addMenu(f"  창 {window_index + 1}{hotkey_label}")
                
                check_mosaic = "✔ " if target_window._mode == "mosaic" else "    "
                action_mosaic = QAction(f"{check_mosaic}모자이크", submenu)
                action_mosaic.triggered.connect(lambda _, form=target_window: self._set_window_mode(form, "mosaic"))
                submenu.addAction(action_mosaic)
                
                check_black = "✔ " if target_window._mode == "black" else "    "
                action_black = QAction(f"{check_black}검정 덮기", submenu)
                action_black.triggered.connect(lambda _, form=target_window: self._set_window_mode(form, "black"))
                submenu.addAction(action_black)

                check_yolo = "✔ " if target_window._yolo_enabled else "    "
                action_yolo = QAction(f"{check_yolo}인체 탐지 (GPU)", submenu)
                action_yolo.triggered.connect(lambda _, form=target_window: self._toggle_window_yolo(form))
                submenu.addAction(action_yolo)

                submenu.addSeparator()
                
                action_ratio = QAction(f"모자이크 강도: {target_window.mosaic_ratio}", submenu)
                action_ratio.triggered.connect(lambda _, form=target_window: self._ask_ratio(form))
                submenu.addAction(action_ratio)

                action_zoom_window = QAction("이 영역 전체화면 확대", submenu)
                action_zoom_window.triggered.connect(lambda _, form=target_window: self._zoom_window(form))
                submenu.addAction(action_zoom_window)
                
                action_close_window = QAction("이 창 닫기", submenu)
                action_close_window.triggered.connect(lambda _, form=target_window: self.remove_window(form))
                submenu.addAction(action_close_window)

        menu_ref.addSeparator()
        action_quit = QAction("전체 종료", menu_ref)
        action_quit.triggered.connect(qApp.quit)
        menu_ref.addAction(action_quit)

    def _on_tray_click(self, activation_reason: Any) -> None:
        """트레이 아이콘 클릭 이벤트를 라우팅합니다."""
        if activation_reason == QSystemTrayIcon.DoubleClick:
            self._toggle_all()

    def _open_tray_menu_via_keyboard(self) -> None:
        """트레이 아이콘을 마우스로 우클릭하지 않고도 컨텍스트 메뉴를 연다.

        세션 프리셋 관리, YOLO 전역 모드/강도/모니터 선택, 확대 창 제어,
        FPS 표시, 보조 모니터 전체 모자이크, 전체 종료는 이 메뉴에만 있는
        옵션이라 메뉴 자체를 못 열면 마우스 없이는 손댈 수 없었다. 메뉴가
        뜨면 방향키/Enter(또는 항목에 포커스된 뒤 Space)로 탐색 가능한 건
        Qt 기본 동작 -- 단, 아래 두 가지를 챙겨야 실제로 그 기본 동작이 먹힌다.

        같은 단축키를 다시 누르면 무조건 닫히는 토글로 만들었다: 전역
        단축키는 OS 후킹이라 우리 프로세스를 포그라운드로 만들지 않고
        발동되므로, popup()으로 띄운 메뉴는 화면엔 보여도 Windows가 실제
        키 입력은 이전에 포커스 있던 창으로 계속 보내는 경우가 있다(포커스
        탈취 방지 정책). 방향키/Esc가 안 먹히는 경우의 탈출구가 이 토글.
        """
        if self._tray_menu.isVisible():
            self._tray_menu.close()
            return

        self._rebuild_context_menu()
        self._tray_menu.popup(QCursor.pos())

        # Alt를 한 번 눌렀다 떼면 Windows의 포그라운드 탈취 방지 잠금이
        # 풀리는 잘 알려진 우회법 -- 그 직후의 SetForegroundWindow라야
        # 실제로 포커스가 넘어가서 메뉴 안에서 방향키 탐색이 가능해진다.
        keyboard.send("alt")
        try:
            ctypes.windll.user32.SetForegroundWindow(int(self._tray_menu.winId()))
        except Exception:
            pass
        self._tray_menu.activateWindow()

    def _open_next_window_settings(self) -> None:
        """생성된 창을 순서대로 순환하며 설정 다이얼로그를 연다.

        창 6~9번은 Ctrl+Alt+6~9 숫자키가 오른손 쪽이라 왼손만으로는 직접
        못 누른다 -- 이 단축키를 반복해서 눌러 1번부터 순서대로 도달한다.
        """
        if not self.windows:
            self.tray.showMessage("모자이크", "생성된 창이 없습니다.", QSystemTrayIcon.Warning, 1500)
            return
        self._next_window_cycle_index %= len(self.windows)
        self._open_window_settings(self._next_window_cycle_index)
        self._next_window_cycle_index = (self._next_window_cycle_index + 1) % len(self.windows)

    def add_window(self) -> None:
        """새로운 모자이크 창 인스턴스를 하나 추가 배치합니다."""
        if len(self.windows) >= MAX_MOSAIC_WINDOWS:
            self.tray.showMessage(
                "모자이크",
                f"모자이크 창은 최대 {MAX_MOSAIC_WINDOWS}개까지 사용할 수 있습니다.",
                QSystemTrayIcon.Warning,
                1800,
            )
            return
        offset: int = len(self.windows) * 30
        new_window = MosaicApp(manager=self, offset=offset)
        new_window.set_ratio(self._default_ratio)
        self.windows.append(new_window)
        new_window.show()
        self._rebuild_context_menu()
        self.tray.showMessage(
            "모자이크",
            f"영역 추가 - 현재 {len(self.windows)}개",
            QSystemTrayIcon.Information, 1500
        )

    def _set_yolo_mode(self, mode: str) -> None:
        self._yolo_mode = mode
        for overlay in self.yolo_overlays.values():
            overlay.set_options(mode, self._yolo_ratio)
        self._rebuild_context_menu()

    def _ask_yolo_ratio(self) -> None:
        value, accepted = QInputDialog.getInt(None, "YOLO 모자이크 강도", "강도", self._yolo_ratio, 2, 64, 1)
        if accepted:
            self._yolo_ratio = value
            for overlay in self.yolo_overlays.values():
                overlay.set_options(self._yolo_mode, value)
            self._rebuild_context_menu()

    def _toggle_yolo_at_cursor(self) -> None:
        """Toggle YOLO on the monitor containing the mouse cursor."""
        if self.yolo_overlays:
            for active_index in list(self.yolo_overlays):
                self._stop_yolo_monitor(active_index)
            self._rebuild_context_menu()
            return

        desktop = QApplication.desktop()
        screen_index = desktop.screenNumber(QCursor.pos())
        if screen_index < 0 or screen_index >= desktop.screenCount():
            screen_index = desktop.primaryScreen()
        self._toggle_yolo_monitor(screen_index)

    def _stop_yolo_monitor(self, screen_index: int) -> bool:
        overlay = self.yolo_overlays.pop(screen_index, None)
        if overlay is None:
            return False
        overlay.close()
        overlay.deleteLater()
        return True

    @staticmethod
    def _capture_monitor_for_screen(screen_index: int) -> Dict[str, Any]:
        """Match a Qt screen to MSS by geometry, not provider-specific order."""
        desktop = QApplication.desktop()
        if screen_index < 0 or screen_index >= desktop.screenCount():
            raise IndexError(f"Invalid screen index: {screen_index}")
        geometry = desktop.screenGeometry(screen_index)
        expected = (geometry.left(), geometry.top(), geometry.width(), geometry.height())
        with mss.mss() as capture:
            for candidate in capture.monitors[1:]:
                actual = (
                    candidate["left"], candidate["top"],
                    candidate["width"], candidate["height"],
                )
                if actual == expected:
                    return dict(candidate)
        raise RuntimeError(f"Could not match Qt screen geometry to MSS monitor: {expected}")

    def _toggle_yolo_monitor(self, screen_index: int) -> None:
        if self._stop_yolo_monitor(screen_index):
            self._rebuild_context_menu()
            return
        try:
            # One application owns one retained model/worker. Switching the
            # target monitor closes the previous worker instead of loading a
            # second copy of the model and duplicating VRAM.
            for active_index in list(self.yolo_overlays):
                self._stop_yolo_monitor(active_index)
            monitor = self._capture_monitor_for_screen(screen_index)
            overlay = YoloCensorOverlay(monitor, ratio=self._yolo_ratio, mode=self._yolo_mode)
            overlay.failed.connect(lambda message, index=screen_index: self._on_yolo_failure(index, message))
            self.yolo_overlays[screen_index] = overlay
            overlay.start()
            self.tray.showMessage("YOLO 검열", f"모니터 {screen_index + 1} GPU 검열 시작", QSystemTrayIcon.Information, 1800)
        except Exception as exc:
            self.tray.showMessage("YOLO 검열 오류", str(exc), QSystemTrayIcon.Critical, 4000)
        self._rebuild_context_menu()

    def _on_yolo_failure(self, screen_index: int, message: str) -> None:
        overlay = self.yolo_overlays.pop(screen_index, None)
        if overlay is not None:
            overlay.hide()
        self.tray.showMessage("YOLO 검열 오류", message, QSystemTrayIcon.Critical, 5000)
        self._rebuild_context_menu()

    def remove_window(self, target_window: MosaicApp) -> None:
        """특정 모자이크 창을 닫고 관리 목록에서 삭제합니다.

        Args:
            target_window (MosaicApp): 삭제할 윈도우 인스턴스.
        """
        for zoom_window in list(self.zoom_windows):
            if zoom_window.source_window is target_window:
                zoom_window.close()
        if target_window in self.windows:
            self.windows.remove(target_window)
        target_window.close()
        self.ensure_global_peek_running()
        self._rebuild_context_menu()

    def _zoom_window(self, target_window: MosaicApp) -> None:
        """Open a fullscreen zoom view for the selected region window."""
        desktop_environment = QApplication.desktop()
        window_center = target_window.geometry().center()
        screen_index = desktop_environment.screenNumber(window_center)
        if screen_index < 0 or screen_index >= desktop_environment.screenCount():
            screen_index = 0

        for existing_zoom in list(self.zoom_windows):
            if existing_zoom.source_window is target_window:
                existing_zoom.close()

        screen_geometry = desktop_environment.screenGeometry(screen_index)
        zoom_window = ZoomFullscreenWindow(self, target_window, screen_index, screen_geometry)
        was_running = target_window.is_running
        self.zoom_windows.append(zoom_window)
        zoom_window.start_zoom()
        zoom_window.set_inner_mosaic_active(was_running)
        target_window.is_running = was_running
        self.ensure_global_peek_running()
        self._rebuild_context_menu()
        self.tray.showMessage(
            "화면 모자이크",
            f"창 {self.windows.index(target_window) + 1} 전체화면 확대 시작",
            QSystemTrayIcon.Information, 1500
        )

    def remove_zoom_window(self, zoom_window: ZoomFullscreenWindow) -> None:
        if zoom_window in self.zoom_windows:
            self.zoom_windows.remove(zoom_window)
        self.ensure_global_peek_running()
        self._rebuild_context_menu()

    def _toggle_zoom_input_passthrough(self) -> None:
        if not self.zoom_windows:
            return
        new_state = not self.zoom_windows[-1]._input_passthrough
        for zoom_window in self.zoom_windows:
            zoom_window.set_input_passthrough(new_state)
        self.tray.showMessage(
            "모자이크",
            f"확대 창 입력 통과: {'켜짐' if new_state else '꺼짐'}",
            QSystemTrayIcon.Information,
            1500
        )

    def _close_all_zoom_windows(self) -> None:
        for zoom_window in list(self.zoom_windows):
            zoom_window.close()
        self._rebuild_context_menu()

    def _toggle_all(self) -> None:
        """관리 중인 모든 일반 영역과 확대 내부 모자이크를 공통 ON/OFF합니다."""
        if not self.windows and not self.zoom_windows:
            return

        any_running = self._is_any_mosaic_active()
        zoom_sources = self._zoom_source_windows()
        for target_window in self.windows:
            if target_window in zoom_sources:
                if any_running:
                    target_window.is_running = False
                continue
            if any_running:
                target_window.stop_mosaic()
            else:
                target_window.start_mosaic()

        for zoom_window in self.zoom_windows:
            zoom_window.set_inner_mosaic_active(not any_running)
            zoom_window.source_window.is_running = not any_running

        if any_running:
            self._stop_global_peek()
        else:
            self.ensure_global_peek_running()
        self._rebuild_context_menu()

    def _pick_monitor_mosaic(self) -> None:
        """마우스 커서가 있는 모니터 전체에 모자이크를 적용합니다."""
        desktop_environment = QApplication.desktop()
        cursor_point = QCursor.pos()

        for screen_index in range(desktop_environment.screenCount()):
            screen_geo = desktop_environment.screenGeometry(screen_index)
            if screen_geo.contains(cursor_point):
                self._mosaic_monitor(screen_index)
                return

        self._mosaic_monitor(0)

    def _mosaic_monitor(self, monitor_index: int) -> None:
        """특정 모니터 전체 영역에 창을 배치하고 즉시 모자이크 처리를 실행합니다.

        Args:
            monitor_index (int): 시스템상 모니터 인덱스.
        """
        desktop_environment = QApplication.desktop()
        if monitor_index >= desktop_environment.screenCount():
            return
            
        screen_geometry = desktop_environment.screenGeometry(monitor_index)
        full_window = MosaicApp(manager=self, offset=0)
        full_window.set_ratio(self._default_ratio)
        full_window.setGeometry(screen_geometry)
        self.windows.append(full_window)
        full_window.show()
        full_window.start_mosaic()
        self._rebuild_context_menu()
        self.tray.showMessage(
            "모자이크",
            f"모니터 {monitor_index + 1} 전체 모자이크 시작",
            QSystemTrayIcon.Information, 1500
        )

    def _open_session_preset_manager(self) -> None:
        dialog = QDialog()
        dialog.setWindowTitle("세션 프리셋 관리")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        dialog.setMinimumWidth(390)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("저장된 프리셋"))
        preset_combo = QComboBox()
        selection_layout.addWidget(preset_combo, 1)
        button_new = QPushButton("새 프리셋")
        selection_layout.addWidget(button_new)
        layout.addLayout(selection_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("프리셋 이름"))
        name_input = QLineEdit()
        name_layout.addWidget(name_input, 1)
        layout.addLayout(name_layout)

        mode_group = QGroupBox("가림 방식")
        mode_layout = QHBoxLayout(mode_group)
        radio_mosaic = QRadioButton("모자이크")
        radio_black = QRadioButton("검정 덮기")
        mode_buttons = QButtonGroup(dialog)
        mode_buttons.addButton(radio_mosaic)
        mode_buttons.addButton(radio_black)
        mode_layout.addWidget(radio_mosaic)
        mode_layout.addWidget(radio_black)
        layout.addWidget(mode_group)

        timing_group = QGroupBox("전환 시간 (초)")
        timing_layout = QFormLayout(timing_group)
        spin_wait_minimum = QSpinBox()
        spin_wait_maximum = QSpinBox()
        spin_show_minimum = QSpinBox()
        spin_show_maximum = QSpinBox()
        for spinbox in (
            spin_wait_minimum,
            spin_wait_maximum,
            spin_show_minimum,
            spin_show_maximum,
        ):
            spinbox.setRange(1, 3600)
        timing_layout.addRow("가림 대기 최소", spin_wait_minimum)
        timing_layout.addRow("가림 대기 최대", spin_wait_maximum)
        timing_layout.addRow("해제 유지 최소", spin_show_minimum)
        timing_layout.addRow("해제 유지 최대", spin_show_maximum)
        layout.addWidget(timing_group)

        strength_group = QGroupBox("모자이크 강도")
        strength_layout = QFormLayout(strength_group)
        spin_ratio = QSpinBox()
        spin_ratio.setRange(2, 64)
        strength_layout.addRow("강도", spin_ratio)
        layout.addWidget(strength_group)

        button_layout = QHBoxLayout()
        button_save = QPushButton("저장")
        button_save.setDefault(True)  # Tab으로 포커스만 옮기면 Space/Enter로 바로 확정
        button_start = QPushButton("저장 후 시작")
        button_delete = QPushButton("삭제")
        button_close = QPushButton("닫기")
        button_layout.addWidget(button_save)
        button_layout.addWidget(button_start)
        button_layout.addWidget(button_delete)
        button_layout.addStretch()
        button_layout.addWidget(button_close)
        layout.addLayout(button_layout)

        def load_values(name: str, preset: Dict[str, Any]) -> None:
            name_input.setText(name)
            radio_mosaic.setChecked(preset["mode"] == "mosaic")
            radio_black.setChecked(preset["mode"] == "black")
            spin_wait_minimum.setValue(preset["wait_minimum"])
            spin_wait_maximum.setValue(preset["wait_maximum"])
            spin_show_minimum.setValue(preset["show_minimum"])
            spin_show_maximum.setValue(preset["show_maximum"])
            spin_ratio.setValue(preset["mosaic_ratio"])

        def selected_name() -> str:
            return preset_combo.currentText().strip()

        def refresh_combo(selected: Optional[str] = None) -> None:
            preset_combo.blockSignals(True)
            preset_combo.clear()
            preset_combo.addItems(sorted(self._session_presets))
            if selected:
                preset_combo.setCurrentText(selected)
            preset_combo.blockSignals(False)
            current_name = selected_name()
            if current_name in self._session_presets:
                load_values(current_name, self._session_presets[current_name])

        def current_values() -> Optional[Dict[str, Any]]:
            preset = self._normalize_session_preset({
                "wait_minimum": spin_wait_minimum.value(),
                "wait_maximum": spin_wait_maximum.value(),
                "show_minimum": spin_show_minimum.value(),
                "show_maximum": spin_show_maximum.value(),
                "mosaic_ratio": spin_ratio.value(),
                "mode": "mosaic" if radio_mosaic.isChecked() else "black",
            })
            if not preset:
                QMessageBox.warning(dialog, "세션 프리셋", "최소 시간은 최대 시간보다 클 수 없습니다.")
            return preset

        def save_current() -> Optional[str]:
            name = name_input.text().strip()
            if not name:
                QMessageBox.warning(dialog, "세션 프리셋", "프리셋 이름을 입력하세요.")
                return None
            preset = current_values()
            if not preset:
                return None

            self._session_presets[name] = preset
            if not self._save_session_presets():
                QMessageBox.critical(dialog, "세션 프리셋", "프리셋 파일을 저장하지 못했습니다.")
                return None
            refresh_combo(name)
            self._rebuild_context_menu()
            return name

        def on_selection_changed(_: int) -> None:
            name = selected_name()
            if name in self._session_presets:
                load_values(name, self._session_presets[name])

        def create_new() -> None:
            preset_combo.blockSignals(True)
            preset_combo.setCurrentIndex(-1)
            preset_combo.blockSignals(False)
            name_input.clear()
            load_values("", next(iter(DEFAULT_SESSION_PRESETS.values())))
            name_input.setFocus()

        def delete_current() -> None:
            name = selected_name()
            if name not in self._session_presets:
                return
            answer = QMessageBox.question(
                dialog,
                "세션 프리셋",
                f"{name} 프리셋을 삭제할까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            del self._session_presets[name]
            self._save_session_presets()
            if self._active_session_preset_name == name:
                self._active_session_preset_name = None
            refresh_combo()
            if not self._session_presets:
                create_new()
            self._rebuild_context_menu()

        def save_and_start() -> None:
            name = save_current()
            if name:
                self._start_session_preset(name)
                dialog.accept()

        preset_combo.currentIndexChanged.connect(on_selection_changed)
        button_new.clicked.connect(create_new)
        button_save.clicked.connect(save_current)
        button_start.clicked.connect(save_and_start)
        button_delete.clicked.connect(delete_current)
        button_close.clicked.connect(dialog.reject)

        refresh_combo(self._active_session_preset_name)
        if preset_combo.currentIndex() < 0:
            create_new()
        dialog.exec_()

    def _toggle_peek(self) -> None:
        """모든 창에 대해 통합 랜덤 잠깐 해제 옵션을 켜거나 끕니다."""
        if not self.windows and not self.zoom_windows:
            return

        new_active_state = not self._global_peek_enabled
        for target_window in self.windows:
            target_window.set_peek_enabled(new_active_state)
        for zoom_window in self.zoom_windows:
            zoom_window.source_window.set_peek_enabled(new_active_state)

        self._global_peek_enabled = new_active_state
        if new_active_state and self._is_any_mosaic_active():
            self.ensure_global_peek_running()
        else:
            self._stop_global_peek()

        self._rebuild_context_menu()
        status_message = "켜짐" if new_active_state else "꺼짐"
        self.tray.showMessage(
            "모자이크",
            f"랜덤 잠깐 해제: {status_message}",
            QSystemTrayIcon.Information, 1500
        )

    def _toggle_all_fps(self) -> None:
        """모든 창들의 현재 FPS 표시 여부를 한꺼번에 변경합니다."""
        if not self.windows:
            return
            
        new_active_state = not self.windows[0].show_fps
        for target_window in self.windows:
            target_window.show_fps = new_active_state
            if not new_active_state:
                target_window._frame_times.clear()
                target_window._current_fps = 0.0
            
        self._rebuild_context_menu()
        status_message = "켜짐" if new_active_state else "꺼짐"
        self.tray.showMessage(
            "모자이크",
            f"FPS 표시: {status_message}",
            QSystemTrayIcon.Information, 1500
        )

    def _ask_ratio(self, target_window: MosaicApp) -> None:
        """특정 창의 모자이크 강도를 팝업창을 사용하여 변경합니다.

        Args:
            target_window (MosaicApp): 강도를 변경할 대상 윈도우 인스턴스.
        """
        new_ratio_value, bool_ok = QInputDialog.getInt(
            None,
            "모자이크 강도",
            "강도  (2=약함  64=강함):",
            value=target_window.mosaic_ratio,
            min=2,
            max=64,
            step=1
        )
        if bool_ok:
            target_window.set_ratio(new_ratio_value)
            self._default_ratio = new_ratio_value
            self._rebuild_context_menu()

    def _open_window_settings(self, index: int) -> None:
        """선택한 단일 창의 설정 제어용 다이얼로그 UI를 실행합니다.

        Args:
            index (int): 조작 대상 창의 배열 인덱스.
        """
        if index >= len(self.windows):
            self.tray.showMessage("모자이크", f"창 {index + 1}이(가) 없습니다.", QSystemTrayIcon.Warning, 1500)
            return

        target_window = self.windows[index]
        dialog = QDialog()
        dialog.setWindowTitle(f"창 {index + 1}  설정")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        dialog.setMinimumWidth(320)
        
        main_layout = QVBoxLayout(dialog)
        main_layout.setSpacing(10)

        # 헤더 조립
        title_label = QLabel(f"<b>창 {index + 1}</b>  설정  <small>(Ctrl+Alt+{index + 1})</small>")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        separator_line_upper = QFrame()
        separator_line_upper.setFrameShape(QFrame.HLine)
        main_layout.addWidget(separator_line_upper)

        # 처리 모드 설정 (채우기 방식 -- 인체 탐지에도 그대로 적용됨)
        group_mode = QGroupBox("처리 모드")
        layout_mode = QHBoxLayout(group_mode)
        radio_btn_mosaic = QRadioButton("모자이크")
        radio_btn_black = QRadioButton("검정 덮기")
        radio_btn_mosaic.setChecked(target_window._mode == "mosaic")
        radio_btn_black.setChecked(target_window._mode == "black")

        button_group = QButtonGroup(dialog)
        button_group.addButton(radio_btn_mosaic)
        button_group.addButton(radio_btn_black)

        layout_mode.addWidget(radio_btn_mosaic)
        layout_mode.addWidget(radio_btn_black)
        main_layout.addWidget(group_mode)

        # 인체 탐지 사용 여부 (채우기 방식은 위 처리 모드를 그대로 따름)
        checkbox_yolo_enabled = QCheckBox("인체 탐지 사용 (GPU)")
        checkbox_yolo_enabled.setChecked(target_window._yolo_enabled)
        main_layout.addWidget(checkbox_yolo_enabled)

        # 인체 탐지 검열 부위 선택 (인체 탐지 사용 시에만 의미가 있음)
        # 얼굴+눈 체크 조합: 얼굴만=눈 제외한 얼굴, 얼굴+눈=얼굴 전체, 눈만=눈만
        group_regions = QGroupBox("인체 탐지 검열 부위")
        layout_regions = QHBoxLayout(group_regions)
        checkbox_face = QCheckBox("얼굴")
        checkbox_eye = QCheckBox("눈")
        checkbox_chest = QCheckBox("가슴")
        checkbox_buttocks = QCheckBox("엉덩이")
        has_face = "face" in target_window._yolo_regions
        checkbox_face.setChecked(has_face)
        if has_face:
            checkbox_eye.setChecked(not target_window._yolo_exclude_eye_from_face)
        else:
            checkbox_eye.setChecked("eye" in target_window._yolo_regions)
        checkbox_chest.setChecked("chest" in target_window._yolo_regions)
        checkbox_buttocks.setChecked("buttocks" in target_window._yolo_regions)
        checkbox_invert = QCheckBox("반전 (선택 부위 대신 나머지 전체)")
        checkbox_invert.setChecked(target_window._yolo_invert)
        layout_regions.addWidget(checkbox_face)
        layout_regions.addWidget(checkbox_eye)
        layout_regions.addWidget(checkbox_chest)
        layout_regions.addWidget(checkbox_buttocks)
        layout_regions.addWidget(checkbox_invert)
        main_layout.addWidget(group_regions)

        group_regions.setEnabled(target_window._yolo_enabled)
        checkbox_yolo_enabled.toggled.connect(group_regions.setEnabled)

        # 모자이크 강도 설정
        group_ratio = QGroupBox("모자이크 강도  (2 = 약함, 64 = 강함)")
        layout_ratio_vertical = QVBoxLayout(group_ratio)
        layout_ratio_horizontal = QHBoxLayout()
        
        slider_ratio = QSlider(Qt.Horizontal)
        slider_ratio.setRange(2, 64)
        slider_ratio.setValue(target_window.mosaic_ratio)
        
        label_ratio_text = QLabel(str(target_window.mosaic_ratio))
        label_ratio_text.setFixedWidth(28)
        
        slider_ratio.valueChanged.connect(lambda val: label_ratio_text.setText(str(val)))
        layout_ratio_horizontal.addWidget(slider_ratio)
        layout_ratio_horizontal.addWidget(label_ratio_text)
        layout_ratio_vertical.addLayout(layout_ratio_horizontal)
        main_layout.addWidget(group_ratio)

        # 잠금 해제 (피킹) 모드 설정
        group_peek = QGroupBox("랜덤 잠깐 해제  (대기 3~10초, 해제 10~20초 랜덤 고정)")
        layout_peek = QVBoxLayout(group_peek)
        checkbox_peek = QCheckBox("활성화")
        checkbox_peek.setChecked(target_window._peek_enabled)
        layout_peek.addWidget(checkbox_peek)
        main_layout.addWidget(group_peek)

        # FPS 표시 설정
        group_fps = QGroupBox("성능 표시")
        layout_fps = QVBoxLayout(group_fps)
        checkbox_fps = QCheckBox("현재 FPS 표시")
        checkbox_fps.setChecked(target_window.show_fps)
        layout_fps.addWidget(checkbox_fps)
        main_layout.addWidget(group_fps)

        # 버튼 구역 조립
        separator_line_lower = QFrame()
        separator_line_lower.setFrameShape(QFrame.HLine)
        main_layout.addWidget(separator_line_lower)
        
        layout_buttons = QHBoxLayout()
        button_apply = QPushButton("적용")
        button_apply.setDefault(True)  # Tab으로 포커스만 옮기면 Space/Enter로 바로 확정
        button_zoom_window = QPushButton("전체화면 확대")
        button_close_window = QPushButton("이 창 닫기")
        button_cancel = QPushButton("취소")
        
        layout_buttons.addWidget(button_apply)
        layout_buttons.addWidget(button_zoom_window)
        layout_buttons.addWidget(button_close_window)
        layout_buttons.addStretch()
        layout_buttons.addWidget(button_cancel)
        main_layout.addLayout(layout_buttons)

        def apply_settings_to_window() -> None:
            chosen_mode = "mosaic" if radio_btn_mosaic.isChecked() else "black"
            target_window.set_mode(chosen_mode)
            target_window.set_yolo_enabled(checkbox_yolo_enabled.isChecked())

            face_checked, eye_checked = checkbox_face.isChecked(), checkbox_eye.isChecked()
            selected_regions = set()
            if checkbox_chest.isChecked():
                selected_regions.add("chest")
            if checkbox_buttocks.isChecked():
                selected_regions.add("buttocks")
            # 얼굴+눈 둘 다: 얼굴 전체(눈 포함). 얼굴만: 눈 제외. 눈만: 눈만.
            if face_checked:
                selected_regions.add("face")
                exclude_eye = not eye_checked
            elif eye_checked:
                selected_regions.add("eye")
                exclude_eye = False
            else:
                exclude_eye = False
            target_window.set_yolo_regions(frozenset(selected_regions))
            target_window.set_yolo_exclude_eye_from_face(exclude_eye)
            target_window.set_yolo_invert(checkbox_invert.isChecked())
            target_window.set_ratio(slider_ratio.value())
            self._default_ratio = slider_ratio.value()
            target_window.set_peek_enabled(checkbox_peek.isChecked())
            target_window.show_fps = checkbox_fps.isChecked()
            if not target_window.show_fps:
                target_window._frame_times.clear()
                target_window._current_fps = 0.0
            self._rebuild_context_menu()

        def apply_settings_callback() -> None:
            apply_settings_to_window()
            dialog.accept()

        def zoom_window_callback() -> None:
            apply_settings_to_window()
            dialog.accept()
            self._zoom_window(target_window)

        def close_window_callback() -> None:
            """다이얼로그 종료와 동시에 대상 윈도우 앱도 종료시킵니다."""
            dialog.accept()
            self.remove_window(target_window)

        button_apply.clicked.connect(apply_settings_callback)
        button_zoom_window.clicked.connect(zoom_window_callback)
        button_close_window.clicked.connect(close_window_callback)
        button_cancel.clicked.connect(dialog.reject)

        dialog.exec_()

    def _set_window_mode(self, target_window: MosaicApp, mode_str: str) -> None:
        """타겟 창의 채우기 방식을 세팅하는 헬퍼 메서드.

        Args:
            target_window (MosaicApp): 대상 창 인터페이스.
            mode_str (str): 변경할 모드 문자열 기반 설정 ('mosaic' 또는 'black').
        """
        target_window.set_mode(mode_str)
        self._rebuild_context_menu()

    def _toggle_window_yolo(self, target_window: MosaicApp) -> None:
        """타겟 창의 인체 탐지 사용 여부를 토글하는 헬퍼 메서드."""
        target_window.set_yolo_enabled(not target_window._yolo_enabled)
        self._rebuild_context_menu()

    def cleanup(self) -> None:
        """애플리케이션 종료 시 전역 자원(키보드 핫키 등)과 창들을 해제합니다."""
        for global_hotkey in (HOTKEY, HOTKEY_NEW, HOTKEY_PEEK, HOTKEY_MON, HOTKEY_YOLO, HOTKEY_MENU, HOTKEY_NEXT_WIN):
            try:
                keyboard.remove_hotkey(global_hotkey)
            except Exception:
                pass
                
        for local_hotkey in HOTKEY_WIN:
            try:
                keyboard.remove_hotkey(local_hotkey)
            except Exception:
                pass
                
        self._stop_global_peek()
        for overlay in list(self.yolo_overlays.values()):
            overlay.close()
        self.yolo_overlays.clear()
        for overlay in list(self.peek_overlays):
            overlay.close()
        self.peek_overlays.clear()

        for zoom_window in list(self.zoom_windows):
            zoom_window.close()

        for running_window in list(self.windows):
            running_window.close()
            
        self.tray.hide()

        # dxcam 인스턴스 전역 정리
        DXCamRegistry.stop_all()

        # Windows 시스템 타이머 정밀도 설정 복원
        if platform.system() == "Windows":
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        from yolo_censor_pipeline import YoloCensorPipeline

        test_frame = np.zeros((640, 640, 3), np.uint8)
        test_pipeline = YoloCensorPipeline(default_model_path())
        test_pipeline.warmup(640, 640, count=1)
        test_output = test_pipeline.process(test_frame)
        print(f"provider={test_pipeline.execution_provider}")
        print(f"model={default_model_path()}")
        print(f"overlay={test_output.overlay_bgra.shape}")
        print("YOLO_BUILD_SELF_TEST=PASS")
        sys.exit(0)

    if platform.system() == "Windows":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

        # 여러 인스턴스가 동시에 뜨면 같은 전역 단축키를 서로 가로채고,
        # dxcam이 같은 모니터에 대해 동시에 DXGI Desktop Duplication을 두 번
        # 만들려다 응답 없음 상태에 빠지는 문제가 있었다. 프로세스 시작
        # 시점에 이름 있는 뮤텍스로 막아 원천 차단한다. 핸들은 이 프로세스가
        # 종료될 때 Windows가 알아서 정리하므로 별도 해제 코드는 필요 없다.
        _SINGLE_INSTANCE_MUTEX_NAME = "MosaicOnnxCensorApp_SingleInstanceMutex"
        ERROR_ALREADY_EXISTS = 183
        MB_ICONINFORMATION = 0x40
        MB_SETFOREGROUND = 0x10000  # 그냥 띄우면 이 프로세스가 포그라운드가 아니라서
                                     # 안 눌리는 창으로 뜰 수 있음 -- 이 창만은 강제로 앞에 오도록.
        ctypes.windll.kernel32.CreateMutexW(None, False, _SINGLE_INSTANCE_MUTEX_NAME)
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            ctypes.windll.user32.MessageBoxW(0, "이미 실행 중입니다.", "모자이크", MB_ICONINFORMATION | MB_SETFOREGROUND)
            sys.exit(0)

    pyqt_app = QApplication(sys.argv)
    pyqt_app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "오류", "시스템 트레이를 사용할 수 없습니다.")
        sys.exit(1)

    shared_manager = WindowManager()

    if "--auto-yolo" in sys.argv:
        # Skips the manual "hover a monitor + press ctrl+alt+g" step so
        # MOSAIC_PROFILE runs can be scripted end to end. Also cross-process:
        # ctrl+alt+g (HOTKEY_YOLO) is a global hotkey, so pressing it while another
        # mosaic.py instance owns it fires both -- picking a fixed monitor here avoids
        # that ambiguity during testing. Quits itself after N seconds via
        # pyqt_app.quit() (not a hard kill) so aboutToQuit -> cleanup() ->
        # DXCamRegistry.stop_all() actually run and release the DXGI capture
        # session cleanly for the next run.
        auto_seconds = 20.0
        auto_monitor_index = None  # 1-based, matches the tray menu's "모니터 N" label
        for _arg in sys.argv:
            if _arg.startswith("--auto-yolo-seconds="):
                auto_seconds = float(_arg.split("=", 1)[1])
            elif _arg.startswith("--auto-yolo-monitor="):
                auto_monitor_index = int(_arg.split("=", 1)[1])

        shared_manager.add_window()
        auto_window = shared_manager.windows[-1]
        screens = QApplication.screens()
        if auto_monitor_index is not None and 1 <= auto_monitor_index <= len(screens):
            target_screen = screens[auto_monitor_index - 1]
        else:
            target_screen = next((s for s in screens if s.geometry().height() > s.geometry().width()), screens[0])
        auto_window.setGeometry(target_screen.geometry())
        auto_window.set_yolo_enabled(True)
        auto_window.start_mosaic()
        print(f"[auto-yolo] window covering screen geometry={target_screen.geometry()} "
              f"for {auto_seconds:.0f}s, then clean quit")
        QTimer.singleShot(int(auto_seconds * 1000), pyqt_app.quit)

    pyqt_app.aboutToQuit.connect(shared_manager.cleanup)
    sys.exit(pyqt_app.exec_())
