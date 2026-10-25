"""Rehearse the exact guarded live runner on all measured cells and three caps."""
import _bootstrap  # noqa: F401
import argparse
import json
import math
from pathlib import Path

from r2p.robot.config import ROOT, load_config, load_joint_map
from r2p.robot.rehearsal import capture_profile, placement_task, rehearse, digest
from r2p.sim.bench import load_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export")
    parser.add_argument("--config")
    parser.add_argument("--bench-profile", default=str(ROOT/"scripts/robot/K_BENCH_PROFILE.json"))
    parser.add_argument("--capture-profile", type=Path, help="GET /state only; save current owner bounds/rest and active map")
    parser.add_argument("--out", type=Path, default=ROOT/"out/robot/k_rehearsal.json")
    parser.add_argument("--cell", choices=[f"r{i}c{j}" for i in range(3) for j in range(3)])
    parser.add_argument("--cap", type=int, choices=range(3))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--transport", choices=("direct", "armd"), default="direct")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.capture_profile:
        profile = capture_profile(cfg["daemon_url"], load_joint_map(cfg), cfg)
        args.capture_profile.parent.mkdir(parents=True, exist_ok=True)
        args.capture_profile.write_text(json.dumps(profile, indent=2)+"\n", encoding="utf-8")
        print(f"Read-only bench profile: {args.capture_profile}")
        if not args.export:
            return 0
        args.bench_profile = args.capture_profile
    if not args.export:
        parser.error("--export is required unless capturing a read-only profile")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be finite and positive")
    from r2p.policy import load
    from r2p.ttt import CELL_IDS
    from r2p.tasks import load_task
    profile = load_profile(args.bench_profile)
    measured = load_task("place_on_cell")
    rows = []
    for cell in [args.cell] if args.cell else CELL_IDS:
        for cap in [args.cap] if args.cap is not None else range(3):
            policy = load(args.export)
            task = placement_task(cell, cap, measured=measured)
            record = args.record/f"{cell}_cap{cap}" if args.record else None
            row = rehearse(policy, task, profile, cfg=cfg, timeout_s=args.timeout, record=record, transport=args.transport)
            row["cap"] = cap
            rows.append(row)
            print(f"{cell} cap {cap}: {row['outcome']}, error={row['placement_error_mm']:.2f} mm, "
                  f"clipped={row['clipped_ticks']}, resets={row['commanded_q_resets']}, "
                  f"limits={row['requested_limit_ticks']}, sends={row['sent_commands']}", flush=True)
    report = {"schema": "r2p_policy_rehearsal/1", "export": args.export,
              "export_config_sha256": digest(policy.config), "bench_profile": profile,
              "policy_eval": policy.config.get("eval"), "policy_smoke_only": policy.config.get("smoke_only", False),
              "measured_task": measured, "measured_task_sha256": digest(measured),
              "successes": sum(r["outcome"] == "goal_reached" and r["success"] for r in rows),
              "n": len(rows), "episodes": rows, "synthetic": True,
              "observation_source": "physical_twin_truth", "physical_success_demonstrated": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"{report['successes']}/{len(rows)} goals reached. Report: {args.out}")
    return 0 if report["successes"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
