"""CPU regression: fitting expert chunks must also permit physical task success.

This deliberately overfits twenty diverse scenes, without observation noise.
It is an implementation gate, not a held-out or hardware qualification.
"""
import json
from pathlib import Path

from r2p.policy import load
from r2p.policy.data import Augmentation
from r2p.policy.diagnose import teacher_forced
from r2p.policy.eval import rollout
from r2p.policy.train import train
from r2p.sim.traces import gen_traces
from r2p.traces import read_trace


def test_twenty_trace_fit_replays_and_completes_training_scenes(tmp_path):
    # Freeze the task independently of ongoing real-bench board surveys.
    task = json.loads((Path(__file__).parent/"fixtures/place_on_cell.json").read_text())
    paths = gen_traces(task, 20, tmp_path/"traces", workers=1, seed=200001, dart=.01)
    aug = Augmentation(joint_offset_std=0, position_std=0, stale_probability=0,
                       aim_radial_m=0, aim_lateral_m=0, smooth_ticks=1)
    export = train(task, tmp_path/"traces", "twenty_trace_regression", steps=8000,
                   device="cpu", seeds=(0,), width=256, blocks=2, batch_size=128,
                   augmentation=aug, out_dir=tmp_path/"export", eval_n=1, eval_max_ticks=20)
    policy = load(export)
    errors = teacher_forced(policy, tmp_path/"traces")
    assert errors["chunk_mae_rad"] < .02, errors
    assert errors["first_mae_rad"] < .02, errors
    outcomes = []
    for path in paths[:5]:
        meta, _ = read_trace(path)
        outcomes.append(rollout(policy, meta["scene"], seed=meta["seed"]))
    assert sum(row["success"] for row in outcomes) > 0, outcomes
    print(json.dumps({"teacher_forced": errors, "training_scene_outcomes": outcomes}))
