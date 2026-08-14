"""Detects whether a monitor currently has Windows HDR (Advanced Color) enabled.

dxcam captures via DXGI Desktop Duplication, which reads the compositor's
backbuffer directly. On a monitor with HDR/Advanced Color turned on, that
backbuffer carries different tone mapping than the legacy SDR path GDI BitBlt
(used by mss) reads from, so dxcam frames come back measurably brighter than
mss frames for identical on-screen content -- confirmed empirically (dxcam
BGR mean ~6/255 higher than mss for the same region) and by inspecting the
installed dxcam package, which copies the mapped DXGI surface bytes as-is
with no HDR/color-space handling. The only reliable mitigation without
patching dxcam itself is to skip dxcam capture on monitors where this
mismatch exists and let the caller fall back to mss.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

QDC_ONLY_ACTIVE_PATHS = 2
DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME = 1
DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO = 9
MONITOR_DEFAULTTONEAREST = 2


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DISPLAYCONFIG_PATH_SOURCE_INFO(ctypes.Structure):
    _fields_ = [
        ("adapterId", _LUID), ("id", wintypes.UINT),
        ("modeInfoIdx", wintypes.UINT), ("statusFlags", wintypes.UINT),
    ]


class _DISPLAYCONFIG_PATH_TARGET_INFO(ctypes.Structure):
    _fields_ = [
        ("adapterId", _LUID), ("id", wintypes.UINT), ("modeInfoIdx", wintypes.UINT),
        ("outputTechnology", wintypes.UINT), ("rotation", wintypes.UINT), ("scaling", wintypes.UINT),
        ("refreshRateNumerator", wintypes.UINT), ("refreshRateDenominator", wintypes.UINT),
        ("scanLineOrdering", wintypes.UINT), ("targetAvailable", wintypes.BOOL), ("statusFlags", wintypes.UINT),
    ]


class _DISPLAYCONFIG_PATH_INFO(ctypes.Structure):
    _fields_ = [
        ("sourceInfo", _DISPLAYCONFIG_PATH_SOURCE_INFO),
        ("targetInfo", _DISPLAYCONFIG_PATH_TARGET_INFO),
        ("flags", wintypes.UINT),
    ]


class _DISPLAYCONFIG_DEVICE_INFO_HEADER(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.UINT), ("size", wintypes.UINT),
        ("adapterId", _LUID), ("id", wintypes.UINT),
    ]


class _DISPLAYCONFIG_SOURCE_DEVICE_NAME(ctypes.Structure):
    _fields_ = [
        ("header", _DISPLAYCONFIG_DEVICE_INFO_HEADER),
        ("viewGdiDeviceName", wintypes.WCHAR * 32),
    ]


class _DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO(ctypes.Structure):
    _fields_ = [
        ("header", _DISPLAYCONFIG_DEVICE_INFO_HEADER),
        ("value", wintypes.UINT),
        ("colorEncoding", wintypes.UINT),
        ("bitsPerColorChannel", wintypes.UINT),
    ]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def _gdi_device_name_for_rect(left: int, top: int, right: int, bottom: int) -> str | None:
    rect = wintypes.RECT(left, top, right, bottom)
    hmonitor = ctypes.windll.user32.MonitorFromRect(ctypes.byref(rect), MONITOR_DEFAULTTONEAREST)
    if not hmonitor:
        return None
    info = _MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(_MONITORINFOEXW)
    if not ctypes.windll.user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
        return None
    return info.szDevice


def _query_display_paths() -> list:
    path_count, mode_count = wintypes.UINT(0), wintypes.UINT(0)
    if ctypes.windll.user32.GetDisplayConfigBufferSizes(
        QDC_ONLY_ACTIVE_PATHS, ctypes.byref(path_count), ctypes.byref(mode_count)
    ) != 0:
        return []
    paths = (_DISPLAYCONFIG_PATH_INFO * path_count.value)()
    modes = (ctypes.c_byte * max(1, mode_count.value * 64))()
    if ctypes.windll.user32.QueryDisplayConfig(
        QDC_ONLY_ACTIVE_PATHS, ctypes.byref(path_count), paths,
        ctypes.byref(mode_count), modes, None,
    ) != 0:
        return []
    return list(paths[: path_count.value])


def is_advanced_color_enabled_for_rect(left: int, top: int, right: int, bottom: int) -> bool:
    """Best-effort check: does the monitor covering this screen rect (in
    virtual-desktop pixel coordinates) currently have Windows HDR/Advanced
    Color turned on? Returns False (dxcam stays enabled) on any failure,
    matching pre-existing behavior when the check can't be performed.
    """
    try:
        device_name = _gdi_device_name_for_rect(left, top, right, bottom)
        if not device_name:
            return False
        for path in _query_display_paths():
            source_info = _DISPLAYCONFIG_SOURCE_DEVICE_NAME()
            source_info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME
            source_info.header.size = ctypes.sizeof(source_info)
            source_info.header.adapterId = path.sourceInfo.adapterId
            source_info.header.id = path.sourceInfo.id
            if ctypes.windll.user32.DisplayConfigGetDeviceInfo(ctypes.byref(source_info)) != 0:
                continue
            if source_info.viewGdiDeviceName != device_name:
                continue

            color_info = _DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO()
            color_info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO
            color_info.header.size = ctypes.sizeof(color_info)
            color_info.header.adapterId = path.targetInfo.adapterId
            color_info.header.id = path.targetInfo.id
            if ctypes.windll.user32.DisplayConfigGetDeviceInfo(ctypes.byref(color_info)) != 0:
                continue
            return bool((color_info.value >> 1) & 1)
        return False
    except Exception:
        return False
