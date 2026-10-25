"""Run exported absolute-target policies through armd, or a CPU mock bench."""
import _bootstrap  # noqa: F401
import argparse
from datetime import datetime
import json
from pathlib import Path

from r2p.robot.config import load_config, load_joint_map, config_path, sag_gain
from r2p.robot.record import json_default


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", nargs="?", const="runs/robot")
    parser.add_argument("--perception-setup")
    parser.add_argument("--gravity-comp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.attempts < 1 or not 0 < args.speed <= 1:
        parser.error("attempts must be positive and speed in (0,1]")
    from r2p.model import Arm
    from r2p.tasks import load_task
    from r2p.robot.runner import Session
    from r2p.robot.operator import Operator
    from r2p.robot.dryrun import ScriptedPolicy, make_world
    from r2p.robot.policy import load_policy, policy_config
    cfg = load_config(args.config)
    joint_map = load_joint_map(cfg)
    if not args.dry_run and not joint_map.verified and not args.allow_unverified:
        parser.error("joint map is unverified; verify it or explicitly use --allow-unverified")
    if not args.dry_run and not args.perception_setup:
        parser.error("real runs require a saved --perception-setup")
    arm, task = Arm(), load_task(args.task)
    results = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with Operator(cfg["control_file"], keyboard=not args.dry_run) as operator:
        for attempt in range(args.attempts):
            policy = ScriptedPolicy(task, arm) if args.policy == "stub" else load_policy(args.policy)
            policy_cfg = policy_config(policy, None if args.policy == "stub" else args.policy)
            if policy_cfg.get("task") != task["name"]:
                parser.error("policy task does not match --task")
            camera = perceiver = world = None
            if args.dry_run:
                from r2p.robot.io import MockRobot
                from r2p.robot.gravity import GravitySag
                robot = MockRobot(arm=arm, joint_map=joint_map, q0=policy_cfg["start_q"], sag=GravitySag(arm, sag_gain(cfg)))
                world, camera, source = make_world(task, arm, args.seed + attempt)
            else:
                from r2p.robot.daemon import DaemonRobot
                from r2p.perception import CameraModel, TaskPerceiver, open_camera
                robot = DaemonRobot(cfg["daemon_url"], joint_map)
                camera = open_camera(cfg["camera"])
                cam_model = CameraModel.load(config_path(cfg["camera"]["calibration"]))
                perceiver = TaskPerceiver(cam_model, task, detector=None)
                perceiver.load_setup(args.perception_setup)
                source = "real"
            record = Path(args.record) / stamp / f"attempt_{attempt+1:02d}" if args.record else None
            try:
                session = Session(robot, policy, task, arm=arm, joint_map=joint_map, camera=camera if (not args.dry_run or args.record) else None,
                                  perceiver=perceiver, world=world, cfg=cfg, operator=operator,
                                  gravity_comp=args.gravity_comp, speed=min(args.speed, .5) if not args.dry_run and not joint_map.verified else args.speed,
                                  realtime=not args.dry_run, record=record, policy_cfg=policy_cfg)
                result = session.run(timeout_s=args.timeout)
                result.update(attempt=attempt+1, source=source, record=str(record) if record else None)
                results.append(result)
                print(json.dumps(result, default=json_default), flush=True)
                if result["outcome"] in ("operator", "load_stop", "error", "held_after_refusals", "start_refused"):
                    break
                if not args.dry_run and attempt+1 < args.attempts:
                    print("Reset the objects, then press g to start the next attempt; q quits.", flush=True)
                    import time
                    while True:
                        events = operator.poll()
                        if "quit" in events or "stop" in events:
                            return 1
                        if "torque_off" in events:
                            robot.torque_off()
                            return 1
                        if "go" in events:
                            break
                        time.sleep(1/30)
            finally:
                if camera is not None:
                    camera.close()
                robot.close(torque_off=False)
    successes = sum(r["outcome"] == "goal_reached" and r["success"] for r in results)
    print(f"goal_reached {successes}/{args.attempts}")
    return 0 if successes == args.attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
