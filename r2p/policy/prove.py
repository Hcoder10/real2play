"""CPU fit proof on exact training scenes, never a deployment qualification.

python -m r2p.policy.prove --traces DIR --out DIR --steps 16000
Use --reuse to evaluate an already trained proof export on every input scene.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import time

from r2p.traces import read_trace
from . import load
from .data import Augmentation
from .eval import rollout
from .train import train


@lru_cache(maxsize=1)
def _worker_policy(export):
    return load(export)


def _replay(item):
    export, path = item
    meta, _ = read_trace(path)
    if not meta["success"]:
        raise ValueError(f"Proof trace is not an expert success: {path}")
    result = rollout(_worker_policy(export), meta["scene"], seed=meta["seed"])
    return dict(result, seed=meta["seed"], trace=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--eval-scenes", type=int)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--require-rate", type=float, default=.5)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    paths = sorted(args.traces.glob("*.jsonl"))
    if len(paths) != 200:
        raise ValueError(f"The proof requires exactly 200 traces, got {len(paths)}")
    task = read_trace(paths[0])[0]["task"]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if not args.reuse:
        augmentation = Augmentation(joint_offset_std=0, position_std=0,
                                    stale_probability=0, aim_radial_m=0,
                                    aim_lateral_m=0, smooth_ticks=1)
        train(task, args.traces, "P2_cpu200", steps=args.steps, device="cpu",
              seeds=(0,), width=256, blocks=2, batch_size=128, threads=2,
              augmentation=augmentation, out_dir=args.out/"export",
              eval_n=1, eval_max_ticks=20)
    policy = load(args.out/"export")
    outcomes = []
    jobs = [(str(args.out/"export"), path) for path in paths[:args.eval_scenes]]
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    results = pool.map(_replay, jobs) if pool else map(_replay, jobs)
    for result in results:
        outcomes.append(result)
        print(json.dumps({"scene": len(outcomes), "seed": result["seed"],
                          "success": result["success"], "ticks": result["ticks"]}), flush=True)
        report = {"training_traces": len(paths), "evaluated_training_scenes": len(outcomes),
                  "success": sum(r["success"] for r in outcomes)/len(outcomes),
                  "noise": "clean", "seconds": time.perf_counter()-started,
                  "weights_sha256": policy.config["weights_sha256"],
                  "training": {k: v for k, v in policy.config["training"].items()
                               if k not in ("traces", "losses")}, "outcomes": outcomes}
        (args.out/"proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if pool:
        pool.shutdown()
    print(json.dumps({k: v for k, v in report.items() if k != "outcomes"}), flush=True)
    config_path = args.out/"export/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training_scene_eval"] = report
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if report["success"] <= args.require_rate:
        raise SystemExit(f"CPU proof failed: {report['success']:.1%} <= {args.require_rate:.1%}")


if __name__ == "__main__":
    main()
