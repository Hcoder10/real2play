"""Record a hand-guided follower with torque explicitly off at supported rest."""
import _bootstrap  # noqa: F401
import argparse
from datetime import datetime
from pathlib import Path
import time

from r2p.robot.config import load_config, load_joint_map
from r2p.robot.record import TraceWriter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--task", default="cube_in_bowl")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    from r2p.tasks import load_task
    from r2p.robot.daemon import DaemonRobot
    from r2p.robot.io import MockRobot
    from r2p.perception import open_camera
    cfg = load_config(args.config)
    jm, task = load_joint_map(cfg), load_task(args.task)
    robot = MockRobot(joint_map=jm) if args.mock else DaemonRobot(cfg["daemon_url"], jm)
    camera = None
    writer = None
    try:
        # This script's explicit hand-guided mode is the operator torque-off request.
        # armd refuses it unless the arm is at its recorded supported rest pose.
        robot.torque_off()
        if not args.mock:
            camera = open_camera(cfg["camera"])
        out = args.out or Path("runs/robot") / datetime.now().strftime("%Y%m%d_%H%M%S_%f_handguided")
        writer = TraceWriter(out, {"task": task, "task_name": task["name"], "source": "human",
                                  "player": "handguided", "rig": cfg["rig"], "success": None,
                                  "commands": [], "mode": "handguided", "torque": "off"})
        for tick in range(int(args.seconds*30)):
            start = time.perf_counter()
            owner = robot.read()
            q = jm.to_model(owner)
            frame = camera.read() if camera else None
            writer.submit({"t": tick/30, "q": q, "q_target": q, "owner_deg": owner,
                           "poses": {}, "grip_closed": bool(q[5] < .35), "command": "", "phase": "handguided"}, frame)
            time.sleep(max(0, 1/30 - (time.perf_counter()-start)))
        print(out)
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.close()
        if camera:
            camera.close()
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
