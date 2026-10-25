"""Plan, collect through an explicit adapter, point, and fit visual joint calibration."""
import _bootstrap  # noqa: F401
import argparse
import importlib
import json
from pathlib import Path


def main(argv=None):
    from r2p import gemini
    from r2p.sysid import visual
    from r2p.model import JointMap
    from r2p.perception import CameraModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "collect", "point", "fit"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--poses")
    parser.add_argument("--plan")
    parser.add_argument("--adapter", help="Explicit module:factory returning {mover, capture}; collection only")
    parser.add_argument("--camera", default="config/camera.json")
    parser.add_argument("--joint-map", default="config/joint_map.json")
    parser.add_argument("--candidate")
    parser.add_argument("--force", action="store_true", help="Re-point already completed observations (cache still applies)")
    parser.add_argument("--gate-px", type=float, default=5.)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.stage == "plan":
        gemini.atomic_json(out / "plan.json", visual.plan())
    elif args.stage == "collect":
        if not args.adapter or not args.plan:
            parser.error("collect requires --adapter module:factory and --plan; there is no default robot transport")
        module, factory = args.adapter.split(":", 1)
        adapter = getattr(importlib.import_module(module), factory)()
        try:
            visual.collect(json.loads(Path(args.plan).read_text()), mover=adapter["mover"],
                           capture=adapter["capture"], out_dir=out)
        finally:
            if "close" in adapter:
                adapter["close"]()
    else:
        if not args.poses:
            parser.error("point/fit require --poses")
        camera = CameraModel.load(args.camera)
        mapping = JointMap.load(Path(args.joint_map), args.candidate)
        if args.stage == "point":
            visual.point(args.poses, out_path=out / "pointed.json", camera=camera, joint_map=mapping, force=args.force)
        else:
            result = visual.fit(args.poses, camera, mapping, gate_px=args.gate_px)
            visual.report(result, out)
            print(json.dumps({key: result[key] for key in ("accepted", "offset_deg", "loo_median_px", "loo_p90_px")}))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
