"""Infer a task from the whole episode and, with calibration, fit its scene."""
import _bootstrap  # noqa: F401
import argparse
import json


def main(argv=None):
    from r2p.real2sim import task_from_video

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_or_video")
    parser.add_argument("--instruction")
    parser.add_argument("--camera")
    parser.add_argument("--out-root", default="tasks")
    parser.add_argument("--out-live")
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-twin", action="store_true")
    args = parser.parse_args(argv)
    result = task_from_video(args.session_or_video, instruction=args.instruction, out_root=args.out_root,
                             camera=args.camera, retries=args.retries, fit_twin=not args.no_twin,
                             twin_options={"iterations": args.iters, "out_dir": args.out_live})
    print(json.dumps({key: result[key] for key in ("task_path", "scene_path", "scene_status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
