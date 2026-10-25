"""Read-only calibrated overlay of the real camera and model collision hulls."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
import cv2
import numpy as np

from r2p.robot.config import load_config, load_joint_map, config_path


def overlay_model(frame, arm, q, cam):
    from r2p.frames import BASE_ORIGIN_MJCF
    result = frame.copy()
    arm.fk(q)
    for body, vertices, kind in arm.parts:
        xyz = vertices @ arm.d.xmat[body].reshape(3, 3).T + arm.d.xpos[body] - BASE_ORIGIN_MJCF
        uv = cam.project(xyz)
        if not np.isfinite(uv).all():
            continue
        hull = cv2.convexHull(np.clip(uv, -10000, 10000).astype(np.int32))
        color = (0, 255, 255) if kind == "finger" else (60, 240, 60)
        cv2.polylines(result, [hull], True, color, 1, cv2.LINE_AA)
    for xyz in arm.tips(q):
        uv = tuple(np.rint(cam.project(xyz)).astype(int))
        cv2.circle(result, uv, 4, (0, 0, 255), -1)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--frames", type=int, default=0, help="zero = until q/Esc")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    from r2p.model import Arm
    from r2p.robot.io import MockRobot
    from r2p.robot.daemon import DaemonRobot
    from r2p.perception import CameraModel, open_camera
    cfg, arm = load_config(args.config), Arm()
    jm = load_joint_map(cfg)
    robot = MockRobot(arm=arm, joint_map=jm) if args.mock else DaemonRobot(cfg["daemon_url"], jm)
    cam = CameraModel.look_at(image_size=(640,360)) if args.mock else CameraModel.load(config_path(cfg["camera"]["calibration"]))
    camera = None if args.mock else open_camera(cfg["camera"])
    try:
        index = 0
        while args.frames <= 0 or index < args.frames:
            frame = np.full((cam.height, cam.width, 3), 220, np.uint8) if camera is None else camera.read()
            image = overlay_model(frame, arm, jm.to_model(robot.read()), cam)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.out), image)
                if args.frames <= 0:
                    break
            else:
                cv2.imshow("Real2Play joint-map mirror (read only)", image)
                if cv2.waitKey(33) & 255 in (27, ord("q")):
                    break
            index += 1
    finally:
        if camera:
            camera.close()
        robot.close()
        if not args.out:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
