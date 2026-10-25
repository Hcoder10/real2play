"""Author a task and checker vectors from a goal and demo summary, with local gates."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _managed import run_stage, task_prompt, task_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--demo-summary", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        files = task_sources()
        result = run_stage("author_task", task_prompt(args.goal, args.demo_summary, files), args.out,
                           dry_run=args.dry_run, goal=args.goal, source_files=files)
    except Exception as exc:
        parser.exit(1, f"Task authoring failed: {exc}\nSee {args.out / 'result.json'}\n")
    print(json.dumps({k: result[k] for k in ("passed", "workspace", "requests_used", "deployment_ready")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
