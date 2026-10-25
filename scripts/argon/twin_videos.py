"""Build timestamp-aligned real | sim-before | sim-after videos for a saved run."""
import argparse
import json

import _bootstrap  # noqa: F401
from r2p.real2sim.twin_videos import build_videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--continuous-sim", action="store_true", help="Render extra standalone sim frames between camera captures (slower)")
    args = parser.parse_args(argv)
    result = build_videos(args.run, fps=args.fps, width=args.width, continuous_sim=args.continuous_sim)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
