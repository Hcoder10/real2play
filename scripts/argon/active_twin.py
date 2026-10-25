"""Active twin campaign. Sim and plan-only modes never send actuator commands."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import _bootstrap  # noqa: F401
from r2p import gemini
from r2p.frames import ROOT
from r2p.perception import CameraModel, TaskPerceiver
from r2p.real2sim.active import (ArmdMover, SimMover, Validator, bench_scene, experiment_menu,
                                 heldout_menu, materialize, initial_parameters, offline_client, run_campaign)
from r2p.sim.twin import load_physics
from r2p.tasks import load_task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mover", choices=("sim", "armd"), default="sim")
    parser.add_argument("--budget", type=int, default=6, help="Total executed experiments, including 3 held-out")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scene", type=Path, help="Surveyed/reset scene required for hardware execution")
    parser.add_argument("--camera", type=Path, default=ROOT/"config/camera.json")
    parser.add_argument("--perception-setup", type=Path)
    parser.add_argument("--armd-url", default="http://127.0.0.1:8793")
    parser.add_argument("--camera-url", default="http://127.0.0.1:8792")
    parser.add_argument("--argon", choices=("fake", "live"), default=None)
    parser.add_argument("--request-budget", type=int, default=30, help="Hard wire-request budget for live Argon")
    parser.add_argument("--max-nfev", type=int, default=9)
    args = parser.parse_args(argv)
    task = load_task("place_on_cell")
    scene = json.loads(args.scene.read_text(encoding="utf-8")) if args.scene else bench_scene(task)
    physics = load_physics()
    if args.mover == "sim":
        hidden = {**initial_parameters(scene), "stiffness_scale": 1.35, "cap_friction": .45}
        truth_physics, truth_scene = materialize(physics, scene, hidden)
        mover = SimMover(scene, truth_physics=truth_physics, truth_scene=truth_scene)
        q0 = mover.q0
    else:
        if not args.plan_only and (args.scene is None or args.perception_setup is None):
            parser.error("armd execution requires --scene and --perception-setup (fresh vision/reset checks)")
        camera = CameraModel.load(args.camera)
        perceiver = None
        if args.perception_setup and not args.plan_only:
            perceiver = TaskPerceiver(camera, task).load_setup(args.perception_setup)
        mover = ArmdMover(scene, camera, url=args.armd_url, camera_url=args.camera_url, perceiver=perceiver)
        q0 = mover.measured_q(mover.initial_state)
    out = args.out or ROOT/"runs/active_twin"/args.run
    if args.plan_only:
        result = {"schema": "r2p_active_plan/1", "mover": args.mover, "plan_only": True,
                  "budget": args.budget, "frame": "spec", "state_requests_only": True,
                  "runtime_compatible": args.mover == "sim" or mover.modern,
                  "note": "Legacy armd min_z is current mesh height, not a safety setting. Its Kin already returns spec coordinates. Streaming telemetry is required for execution.",
                  "experiments": [{"split": split, "experiment": asdict_no_none(e),
                                   "safety": mover.validator.verdict(e, scene, q0)}
                                  for split, menu in (("heldout", heldout_menu(scene)), ("train", experiment_menu(scene))) for e in menu]}
        gemini.atomic_json(out/"plan.json", result)
        print(json.dumps(result, indent=2))
        return 0
    mode = args.argon or ("fake" if args.mover == "sim" else "live")
    client = offline_client() if mode == "fake" else gemini.Gemini(max_requests=args.request_budget, retries=1)
    try:
        result = run_campaign(mover, budget=args.budget, out_dir=out, client=client, max_nfev=args.max_nfev)
        print(json.dumps({"run": str(out), "before": result["before"]["error"], "after": result["heldout"]["error"],
                          "parameters": result["parameters"], "candidate": result["candidate_physics"],
                          "activated": False}, indent=2))
    finally:
        if getattr(mover, "perceiver", None) is not None:
            mover.perceiver.close()
    return 0


def asdict_no_none(experiment):
    from dataclasses import asdict
    return {k: v for k, v in asdict(experiment).items() if v is not None}


if __name__ == "__main__":
    raise SystemExit(main())
