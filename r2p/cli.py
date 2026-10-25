"""Command line entry point. Running status never imports robot or ML modules."""

import argparse
import json
import os
from pathlib import Path

from .pipeline import Pipeline, RunOptions, STAGES
from .pipeline.storage import read_json


def parser():
    result = argparse.ArgumentParser(prog="python -m r2p", description="Real2Play: experiment, twin, play, learn")
    subs = result.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run", help="Run or resume a task pipeline")
    run.add_argument("--task", required=True, help="Task name, task.json, or an instruction (with --video)")
    run.add_argument("--video", help="Recorded session directory or video for task inference")
    run.add_argument("--rounds", type=int, default=1)
    modes = run.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Synthetic bench; Gemini cache-only by default")
    modes.add_argument("--real", action="store_true", help="Execute robot steps on the configured bench")
    run.add_argument("--main-model", default=os.getenv("R2P_MODEL_MAIN", "gemini-4-argon"))
    run.add_argument("--small-model", default=os.getenv("R2P_MODEL_SMALL", "gemini-3.8-flash"))
    run.add_argument("--allow-fallback", action="store_true", default=os.getenv("R2P_ALLOW_FALLBACK") == "1")
    cache = run.add_mutually_exclusive_group()
    cache.add_argument("--cache-only", dest="cache_only", action="store_true")
    cache.add_argument("--no-cache-only", dest="cache_only", action="store_false")
    run.set_defaults(cache_only=None)
    run.add_argument("--force", action="append", default=[], metavar="rN:STAGE")
    run.add_argument("--no-cascade", action="store_true", help="Keep downstream cached stages when forcing")
    run.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu")
    run.add_argument("--traces", type=int, help="Expert episodes (dry: 2; other modes: 200)")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--steps", type=int, help="Training steps (dry: 5; other modes: 10000)")
    run.add_argument("--eval-n", type=int, help="Paired scenes (dry: 2; other modes: 100)")
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--incumbent", help="Export to compare against (default: exports/<task>_v1)")
    run.add_argument("--lease", default="runs/lease.json")
    run.add_argument("--twin-iters", type=int, default=4)
    run.add_argument("--perception-setup", help="Saved C2 perception setup required for real bench attempts")
    run.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    status = subs.add_parser("status", help="Show persisted pipeline progress")
    status.add_argument("task", nargs="?")
    status.add_argument("--json", action="store_true")
    status.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "status":
        manifests = []
        for path in sorted((args.root / "runs").glob("*/manifest.json")):
            if args.task and path.parent.name not in {args.task, args.task + "-dry"}:
                continue
            try:
                manifests.append(read_json(path))
            except (ValueError, OSError) as exc:
                print(f"BLOCKED {path}: {type(exc).__name__}")
        if args.json:
            print(json.dumps(manifests, indent=2))
        elif not manifests:
            print("No runs yet. Start with: python -m r2p run --task <name> --dry-run")
        else:
            for manifest in manifests:
                label = "SIM DRY RUN" if manifest.get("synthetic") else manifest.get("mode", "plan").upper()
                print(f"{manifest.get('run', manifest['task'])} [{label}] {manifest.get('status', 'UNKNOWN')}")
                for name, row in manifest.get("rounds", {}).items():
                    for stage in STAGES:
                        detail = row.get("stages", {}).get(stage)
                        if not detail:
                            continue
                        print(f"  {name}:{stage} {detail['status']} {detail.get('message', '')}")
        return 0
    values = vars(args).copy()
    values.pop("command")
    values["mode"] = "dry-run" if values.pop("dry_run") else "real" if values["real"] else "plan"
    values.pop("real")
    if values["cache_only"] is None:
        values["cache_only"] = values["mode"] != "real" or os.getenv("R2P_CACHE_ONLY") == "1"
    values["force"] = tuple(values["force"])
    try:
        return 0 if Pipeline(RunOptions(**values)).run() else 2
    except (ValueError, RuntimeError, OSError) as exc:
        from .pipeline.stages import safe_error
        print(f"BLOCKED: {safe_error(exc)}")
        return 2
