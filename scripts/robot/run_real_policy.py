"""One learned placement via streaming armd or a released bus; default is no motion."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from r2p.robot.config import ROOT, config_path, load_config, load_joint_map
from r2p.robot.rehearsal import audit, make_session, placement_task, require_released_bus, validate_profile, streaming_robot
from r2p.sim.bench import load_profile


def checked_rest(cfg, profile, joint_map):
    """Read-only serial check AFTER daemon release, BEFORE torque priming."""
    from r2p.frames import JOINTS
    calibration_file = Path(cfg["calibration_dir"]).expanduser()/f"{cfg['robot_id']}.json"
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    half = np.array([(calibration[j]["range_max"]-calibration[j]["range_min"])/2*360/4095 for j in JOINTS[:5]])
    state = profile["owner_state"]
    if not (np.allclose(-half+1, state["lo"][:5], atol=.051, rtol=0)
            and np.allclose(half-1, state["hi"][:5], atol=.051, rtol=0)):
        raise ValueError("saved owner limits differ from the current servo calibration; recapture the profile")
    from r2p.robot.io import ReadOnlyBus
    with ReadOnlyBus(cfg) as bus:
        measured = joint_map.to_model(bus.read())
    rest = joint_map.to_model(profile["owner_state"]["rest"])
    if np.max(np.abs(measured-rest)) > np.deg2rad(4):
        raise ValueError("arm is not at the captured supported rest pose; refusing torque priming")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--cell", choices=[f"r{i}c{j}" for i in range(3) for j in range(3)])
    parser.add_argument("--cap", type=int, choices=range(3), default=0)
    parser.add_argument("--supply-xy", type=float, nargs=2, help="Fresh measured spec-frame cap centre, metres")
    parser.add_argument("--board", type=Path, help="JSON list of nine confirmed marks, or a {board: [...]} object")
    parser.add_argument("--bench-profile", default=str(ROOT/"scripts/robot/K_BENCH_PROFILE.json"))
    parser.add_argument("--config")
    parser.add_argument("--transport", choices=("direct", "armd"), default="direct")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--arm-released", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--perception-setup", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--record", type=Path, default=ROOT/"runs/robot/k_policy")
    args = parser.parse_args(argv)
    from r2p.policy import load
    from r2p import ttt
    cfg, profile = load_config(args.config), load_profile(args.bench_profile)
    joint_map = load_joint_map(cfg)
    validate_profile(profile, joint_map)
    policy = load(args.export)
    board = [""]*9
    if args.board:
        value = json.loads(args.board.read_text(encoding="utf-8"))
        board = ttt.validate_board(value["board"] if isinstance(value, dict) else value)
    task = placement_task(args.cell or "r1c1", args.cap, board=board, supply_xy=args.supply_xy)
    diagnostics = audit(policy, task, profile, joint_map, cfg)
    print(json.dumps({"motion": False, "joint_map": joint_map.name, "verified": joint_map.verified,
                      "export": args.export, "audit": diagnostics}, indent=2), flush=True)
    if not diagnostics["start_valid"]:
        print("Refused: export start_q exceeds the real envelope. Regenerate traces and retrain; do not clip start_q.")
        return 1
    if not args.real:
        return 0
    if policy.config.get("smoke_only"):
        parser.error("this export declares smoke_only; use an evaluated placement candidate for --real")
    if not joint_map.verified and not args.allow_unverified:
        parser.error("joint map is unverified; --allow-unverified explicitly selects half-speed execution")
    if args.perception_setup is None or not args.perception_setup.is_file():
        parser.error("live mode requires an existing --perception-setup for this cap and current camera")
    if args.timeout <= 0 or not np.isfinite(args.timeout):
        parser.error("--timeout must be finite and positive")
    if args.transport == "direct":
        require_released_bus(cfg, args.arm_released)
    else:
        streaming_robot(cfg, profile, joint_map)  # handshake performs only GET
    from r2p.perception import CameraModel, open_camera
    from r2p.robot.io import RealRobot
    from r2p.robot.operator import Operator
    from datetime import datetime
    camera = open_camera(cfg["camera"])
    robot = perceiver = None
    try:
        calibration = CameraModel.load(config_path(cfg["camera"]["calibration"]))
        decision = {"source": "operator", "model": None, "move": args.cell, "board": board}
        if args.cell is None:
            decision = ttt.decide(camera.read(), calibration, {"board": board}, "O",
                                  "white plastic bottle caps, open side up", "flat metallic coins")
            if decision["source"] != "gemini":
                raise ValueError("Argon did not provide a validated move; specify --cell explicitly to use an operator choice")
            if decision["move"] is None:
                print(json.dumps(decision))
                return 0
            board = decision["board"]
        cell = decision["move"]
        if board[ttt.cell_index(cell)]:
            raise ValueError("requested cell is occupied")
        task = placement_task(cell, args.cap, board=board, supply_xy=args.supply_xy)
        audit(policy, task, profile, joint_map, cfg)
        from r2p.robot.policy_vision import load_placement_setup
        perceiver = load_placement_setup(calibration, task, args.perception_setup)
        # Recheck ownership/streaming identity immediately before execution.
        # Preflight performs no daemon writes, including rest/release.
        if args.transport == "direct":
            require_released_bus(cfg, args.arm_released)
            checked_rest(cfg, profile, joint_map)
            initial_q = joint_map.to_model(profile["owner_state"]["rest"])
        else:
            robot = streaming_robot(cfg, profile, joint_map)
            initial_q = joint_map.to_model(robot.read())
        # Verify a current cap observation before priming torque or moving home.
        initial = perceiver.estimate(camera.read(), initial_q)["poses"]
        piece = next(o for o in task["objects"] if o["role"] == "manipulated")
        pose = initial.get(piece["id"], {})
        if not pose.get("seen") or pose.get("source") != "vision":
            raise ValueError("saved setup cannot currently see the selected cap")
        if np.linalg.norm(np.asarray(pose["xyz"])[:2]-np.asarray(piece["xyz"])[:2]) > .015:
            raise ValueError("tracked cap is not at the selected supply position; update --supply-xy/setup")
        record = args.record/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        print(json.dumps({"decision": decision, "record": str(record), "speed": .5 if not joint_map.verified else 1.}), flush=True)
        if args.transport == "direct":
            robot = RealRobot(cfg)
        with Operator(cfg["control_file"], keyboard=True) as operator:
            result = make_session(robot, policy, task, profile, cfg=cfg, camera=camera, perceiver=perceiver,
                                  realtime=True, operator=operator, record=record).run(timeout_s=args.timeout)
        calibration.save(record/"camera.json")
        (record/"decision.json").write_text(json.dumps(decision, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["outcome"] == "goal_reached" and result["success"] else 1
    finally:
        if perceiver is not None:
            perceiver.close()
        camera.close()
        if robot is not None:
            robot.close(torque_off=False)


if __name__ == "__main__":
    raise SystemExit(main())
