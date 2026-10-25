"""Teacher-forced chunk errors and closed-loop replay of an exact recorded scene.

python -m r2p.policy.diagnose --export NAME --trace episode.jsonl --out report.json
No hardware, model API, or GPU. Prediction metrics are not task success metrics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from r2p.traces import read_trace
from . import load
from .data import Augmentation, TraceDataset
from .eval import rollout


def teacher_forced(policy, trace):
    augmentation = Augmentation(**policy.config.get("training", {}).get("augmentation", {}))
    dataset = TraceDataset(policy.config["task_spec"], trace, policy.config, augmentation)
    errors = []
    for start in range(0, len(dataset.features), 128):
        prediction = policy.predict_features(dataset.features[start:start+128])
        labels = dataset.labels[start:start+128].reshape(-1, 15, 6)
        errors.append(np.abs(prediction-labels))
    error = np.concatenate(errors)
    return {"ticks": len(error), "chunk_mae_rad": float(error.mean()),
            "first_mae_rad": float(error[:, 0].mean()),
            "first_p95_rad": float(np.quantile(error[:, 0], .95)),
            "first_per_joint_mae_rad": error[:, 0].mean(axis=0).tolist()}


def diagnose(export, trace):
    policy = load(export)
    meta, ticks = read_trace(trace)
    phases = defaultdict(list)
    # Use the exact scene in the trace, including object sizes and board pose.
    report = {"export": str(export), "trace": str(trace), "trace_seed": meta.get("seed"),
              "trace_success": meta["success"], "trace_target_space": meta.get("target_space"),
              "teacher_forced": teacher_forced(policy, trace)}
    if policy.config.get("physics") != meta.get("physics_config"):
        raise ValueError("Exact-scene replay requires the trace's physics configuration")
    samples = []

    def observe(row):
        phases[row["command"]].append(row["tick"])
        if row["tick"] % 100 == 0:
            samples.append({"tick": row["tick"], "command": row["command"],
                            "q": row["q"].tolist(), "q_input": row["q_input"].tolist(),
                            "target": row["target"].tolist(), "resets": row["resets"],
                            "objects": row["state"]["objects"]})

    report["closed_loop"] = rollout(policy, meta["scene"], seed=meta.get("seed") or 0, on_step=observe)
    report["command_ticks"] = {k: len(v) for k, v in phases.items()}
    report["samples"] = samples
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.export, args.trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
