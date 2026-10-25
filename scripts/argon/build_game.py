"""Generate a game in Antigravity, then apply and verify in an isolated local workspace."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _managed import ROOT, game_context, game_prompt, game_sources, read_json, run_stage
from r2p.tasks import load_task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_on_cell")
    parser.add_argument("--physics", type=Path, default=ROOT / "config/physics/so101_bench.json")
    parser.add_argument("--camera", type=Path, default=ROOT / "config/camera.json")
    parser.add_argument("--verification", type=Path, help="Passed twin evidence + input_sha256 for task/physics/board/camera")
    parser.add_argument("--allow-provisional-twin", action="store_true", help="Candidate-only rehearsal; never claims verified real fidelity")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Fake agent; real local Luau and parity gates")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.verification and not args.allow_provisional_twin:
        parser.error("Live game builds need --verification (or --allow-provisional-twin for candidate-only rehearsal)")
    context = game_context(load_task(args.task), read_json(args.physics), read_json(args.camera),
                           read_json(args.verification) if args.verification else None)
    try:
        files = game_sources(context)
        result = run_stage("build_game", game_prompt(context, files), args.out, dry_run=args.dry_run,
                           context=context, source_files=files)
    except Exception as exc:
        parser.exit(1, f"Game build failed: {exc}\nSee {args.out / 'result.json'}\n")
    print(json.dumps({k: result[k] for k in ("passed", "workspace", "requests_used", "deployment_ready")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
