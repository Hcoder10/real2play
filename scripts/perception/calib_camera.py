#!/usr/bin/env python3
"""Capture board views, calibrate intrinsics, and solve measured flat-board extrinsics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2p.perception.camera import CameraModel, open_camera
from r2p.perception.aruco import BoardConfig, calibrate_intrinsics, calibrate_extrinsics, footprint_overlay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Bench JSON including measured board placement")
    parser.add_argument("--intrinsic-views", type=Path, help="Directory of 8-12 tilted board PNG/JPG views")
    parser.add_argument("--intrinsics", type=Path, help="Reuse known intrinsics instead of fitting them")
    parser.add_argument("--flat", type=Path, help="Flat board frame; otherwise capture from configured camera")
    parser.add_argument("--capture", type=Path, help="Interactive capture directory (SPACE saves, q finishes)")
    parser.add_argument("--out", type=Path, default=Path("config/camera.json"))
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    board = BoardConfig.from_dict(cfg)
    if args.capture:
        args.capture.mkdir(parents=True, exist_ok=True)
        index = len(list(args.capture.glob("view_*.png")))
        with open_camera(cfg) as camera:
            print("Tilt the board between 8-12 captures. SPACE saves, q finishes.")
            try:
                while True:
                    frame = camera.read()
                    cv2.imshow("calibration capture", frame)
                    key = cv2.waitKey(20) & 255
                    if key == ord("q"):
                        break
                    if key == 32:
                        cv2.imwrite(str(args.capture / f"view_{index:02d}.png"), frame)
                        index += 1
                        print(f"Saved view {index}")
            finally:
                cv2.destroyAllWindows()
        return
    report = {}
    if args.intrinsics:
        intrinsic = CameraModel.load(args.intrinsics)
    elif args.intrinsic_views:
        paths = sorted(p for p in args.intrinsic_views.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        frames = [cv2.imread(str(p)) for p in paths]
        if any(f is None for f in frames):
            raise ValueError("Could not decode an intrinsic calibration view")
        intrinsic, report["intrinsics"] = calibrate_intrinsics(frames, board)
    else:
        parser.error("Supply --intrinsic-views or --intrinsics (or --capture to collect views)")
    if args.flat:
        frame = cv2.imread(str(args.flat))
        if frame is None:
            raise FileNotFoundError(args.flat)
    else:
        with open_camera(cfg) as camera:
            frame = camera.read()
    cam, report["extrinsics"] = calibrate_extrinsics(frame, intrinsic, board)
    cam.save(args.out, calibration=report)
    overlay = args.out.with_name(args.out.stem + "_footprint.png")
    cv2.imwrite(str(overlay), footprint_overlay(frame, cam, cfg.get("base_footprint_size_m", [.10, .08])))
    print(json.dumps(report, indent=2))
    print(f"Calibration: {args.out}; check base footprint and +x/+y axes: {overlay}")


if __name__ == "__main__":
    main()
