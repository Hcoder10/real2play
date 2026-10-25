"""Save cap tracking appearance using a camera GET and one SMALL-role box call.

No arm IO. --boxes provides {piece: [x0,y0,x1,y1]} pixel boxes for zero requests.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from r2p.robot.config import config_path, load_config
from r2p.robot.rehearsal import placement_task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="r1c1")
    parser.add_argument("--cap", type=int, choices=range(3), default=0)
    parser.add_argument("--supply-xy", type=float, nargs=2)
    parser.add_argument("--config")
    parser.add_argument("--boxes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from r2p.perception import CameraModel, TaskPerceiver, open_camera
    cfg = load_config(args.config)
    task = placement_task(args.cell, args.cap, supply_xy=args.supply_xy)
    detector = None
    if args.boxes:
        boxes = json.loads(args.boxes.read_text(encoding="utf-8"))
        detector = lambda *a, **kw: boxes
    camera = open_camera(cfg["camera"])
    try:
        cam = CameraModel.load(config_path(cfg["camera"]["calibration"]))
        with TaskPerceiver(cam, task, detector=detector) as perceiver:
            result = perceiver.setup(camera.read())
            piece = next(o for o in task["objects"] if o["role"] == "manipulated")
            if not result["poses"].get(piece["id"], {}).get("seen") or piece["id"] not in perceiver.models:
                raise ValueError("selected cap has no reliable visual appearance fit")
            perceiver.save_setup(args.out)
            print(json.dumps({"motion": False, "setup": str(args.out), "piece": result["poses"][piece["id"]]}))
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
