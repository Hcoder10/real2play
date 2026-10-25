"""python -m r2p.tasks validate|sample|check"""
import argparse
import glob
import json
from pathlib import Path
import numpy as np

from . import check, check_window, load_task, sample_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    validate_parser = subs.add_parser("validate")
    validate_parser.add_argument("tasks", nargs="+")
    sample_parser = subs.add_parser("sample")
    sample_parser.add_argument("task")
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--out", type=Path)
    check_parser = subs.add_parser("check")
    check_parser.add_argument("task")
    check_parser.add_argument("state", type=Path)
    check_parser.add_argument("--hz", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            failures = 0
            for pattern in args.tasks:
                for path in sorted(glob.glob(pattern)) or [pattern]:
                    try:
                        task = load_task(path)
                        print(f"ok {task['name']} ({path})")
                    except (OSError, ValueError) as exc:
                        failures += 1
                        print(f"ERROR {path}: {exc}")
            return int(failures > 0)
        task = load_task(args.task)
        if args.action == "sample":
            payload = sample_scene(task, np.random.default_rng(args.seed))
        else:
            state = json.loads(args.state.read_text(encoding="utf-8-sig"))
            payload = check_window(task, state, hz=args.hz) if isinstance(state, list) else check(task, state)
        output = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if args.action == "sample" and args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        return 0 if args.action == "sample" or payload["ok"] else 1
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
