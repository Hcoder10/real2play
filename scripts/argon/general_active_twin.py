"""Task-derived active twin campaign. Defaults to CPU sim and zero live requests."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import _bootstrap  # noqa: F401
from r2p import gemini
from r2p.frames import ROOT
from r2p.perception import CameraModel, TaskPerceiver
from r2p.real2sim.active import ArmdMover, Safety
from r2p.real2sim.experiments import Validator, apply_owner_limits, experiment_menu, json_experiment, offline_client
from r2p.real2sim.general_active import SimMover, materialize, parameter_space, run_campaign
from r2p.sim.twin import load_physics
from r2p.tasks import load_task, sample_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--run", default="general-"+time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=7)
    parser.add_argument("--mover", choices=("sim", "armd"), default="sim")
    parser.add_argument("--argon", choices=("fake", "live"), default="fake")
    parser.add_argument("--request-budget", type=int, default=15)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--camera", type=Path, default=ROOT/"config/camera.json")
    parser.add_argument("--perception-setup", type=Path, help="Optional appearance sensor; default live sensor is MAIN Argon")
    parser.add_argument("--owner-limits", type=Path, help="Saved calibrated {joint_map,lo,hi} if daemon state omits them")
    args = parser.parse_args(argv)
    if not 1 <= args.request_budget <= 15:
        parser.error("request budget must be 1..15")
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("run must be a single directory name")
    task = load_task(args.task)
    scene = json.loads(args.scene.read_text()) if args.scene else sample_scene(task, np.random.default_rng(args.seed))
    physics = load_physics()
    client = offline_client() if args.argon == "fake" else gemini.Gemini(max_requests=args.request_budget, retries=0)
    if args.mover == "sim":
        values, _ = parameter_space(task, scene, physics)
        truth_physics, truth_scene = materialize(physics, scene, {**values, "stiffness_scale": 1.3})
        mover = SimMover(task, scene, truth_physics=truth_physics, truth_scene=truth_scene)
    else:
        if args.scene is None:
            parser.error("armd requires --scene")
        if args.argon != "live" and not args.plan_only:
            parser.error("real experiments require --argon live (shared 15-request budget)")
        if not args.perception_setup and not args.plan_only and 3*args.budget-2+max(0,args.budget-4) > args.request_budget:
            parser.error("budget needs two MAIN surveys per experiment, training design calls and contact wrist reviews")
        # This explicit next-run entry point does not modify/restart the running
        # lead campaign. All requests remain owned by the existing daemon.
        from r2p.real2sim.experiment_recording import ArgonSurveyPerceiver, SurveyedMover
        import sys
        sys.path.insert(0, str(ROOT/"scripts/lead"))
        from measure_grid import measure
        camera = CameraModel.load(args.camera)
        perceiver = (TaskPerceiver(camera, task).load_setup(args.perception_setup) if args.perception_setup
                     else ArgonSurveyPerceiver(task, scene, camera, client))
        safety = Safety(reach=(.127, .32), joint_margin_deg=2, trajectory_speed_cap=8)
        base = ArmdMover(scene, camera, validator=Validator(task, safety), perceiver=perceiver)
        owner_state = json.loads(args.owner_limits.read_text()) if args.owner_limits else base.initial_state
        apply_owner_limits(base.validator, base.joint_map, owner_state)
        def survey(frame, q):
            if isinstance(perceiver, ArgonSurveyPerceiver):
                perceiver.scene = base.scene
                return perceiver.read(frame, q)["poses"]
            return perceiver.estimate(frame, q)["poses"]
        def sheet_geometry(frame):
            return measure(frame, camera)[0]
        mover = SurveyedMover(base, survey=survey, sheet_geometry=sheet_geometry,
                              final_observe=perceiver.read if isinstance(perceiver, ArgonSurveyPerceiver) else None)
    out = ROOT/"runs/active_twin"/args.run
    if args.plan_only:
        q0 = mover.q0 if args.mover == "sim" else mover.measured_q(mover.initial_state)
        result = {"task": task["name"], "plan_only": True,
                  "menu": [{"experiment": json_experiment(e), **mover.validator.verdict(e, scene, q0)}
                           for e in experiment_menu(task, scene)]}
        gemini.atomic_json(out/"plan.json", result)
    else:
        result = run_campaign(mover, task, out_dir=out, budget=args.budget, physics=physics,
                              client=client, max_nfev=args.max_nfev, videos=not args.no_videos)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
