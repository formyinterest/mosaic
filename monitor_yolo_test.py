"""Run a bounded YOLO integration test against a portrait monitor."""

from __future__ import annotations

import argparse
import json
import time

from yolo_censor_pipeline import YoloCensorPipeline

import cv2
import mss
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--monitor", type=int)
    parser.add_argument("--ratio", type=int, default=24, help="mosaic_ratio")
    parser.add_argument("--mode", default="mosaic", choices=("mosaic", "black"))
    parser.add_argument("--output", default="monitor_yolo_test.png")
    args = parser.parse_args()

    with mss.mss() as capture:
        monitors = capture.monitors
        monitor_index = args.monitor
        if monitor_index is None:
            monitor_index = next(
                (index for index, item in enumerate(monitors[1:], 1) if item["height"] > item["width"]),
                1,
            )
        monitor = monitors[monitor_index]
        # synchronized_frame=True matches the live app (mosaic.py's
        # YoloWindowWorker) -- that path composites into an opaque frame
        # instead of an alpha-masked overlay, and it's the one actually
        # worth profiling since it's what the user sees.
        pipeline = YoloCensorPipeline(
            "models/yolo11n-pose.onnx",
            mosaic_ratio=args.ratio,
            mode=args.mode,
            synchronized_frame=True,
        )
        pipeline.warmup(monitor["width"], monitor["height"], count=2)

        persons = []
        stage_totals: dict[str, float] = {}
        last_frame = last_overlay = None
        started = window_started = time.perf_counter()
        window_frames = 0
        for frame_index in range(args.frames):
            t_capture_start = time.perf_counter()
            shot = capture.grab(monitor)
            bgra = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
            last_frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            capture_ms = (time.perf_counter() - t_capture_start) * 1000

            output = pipeline.process(last_frame)
            last_overlay = output.overlay_bgra
            persons.append(len(output.results))

            for key, value in {"capture": capture_ms, **pipeline.last_stage_ms}.items():
                stage_totals[key] = stage_totals.get(key, 0.0) + value
            window_frames += 1

            now = time.perf_counter()
            if now - window_started >= 2.0 or frame_index == args.frames - 1:
                averages = ", ".join(f"{key}={value / window_frames:.1f}ms" for key, value in stage_totals.items())
                fps = window_frames / (now - window_started)
                print(f"[frame {frame_index + 1}/{args.frames}] fps={fps:.1f} {averages}")
                stage_totals = {}
                window_frames = 0
                window_started = now

        elapsed = time.perf_counter() - started
        preview = last_overlay[:, :, :3]
        cv2.imwrite(args.output, preview)
        summary = {
            "monitor": monitor_index,
            "resolution": [monitor["width"], monitor["height"]],
            "provider": pipeline.execution_provider,
            "frames": args.frames,
            "capture_fps": round(args.frames / elapsed, 2),
            "persons_min": min(persons),
            "persons_max": max(persons),
            "frames_with_person": sum(value > 0 for value in persons),
            "preview": args.output,
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
