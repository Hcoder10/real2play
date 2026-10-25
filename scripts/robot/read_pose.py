"""Read-only stable captures and touch-based joint-offset fit. No motion commands."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import sys
import time

from r2p.robot.config import load_config, load_joint_map
from r2p.robot.calibration import fit_touch, stable_pose
from r2p.robot.record import json_default


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--samples", type=Path, help="offline fit from recorded touches JSON")
    parser.add_argument("--out", type=Path, default=Path("runs/touch_fit.json"))
    parser.add_argument("--control-file", type=Path, default=Path("runs/pose_control.json"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--direct", action="store_true", help="read-only serial access with armd stopped")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    from r2p.model import Arm
    from r2p.robot.io import MockRobot, ReadOnlyBus
    from r2p.robot.daemon import DaemonRobot
    cfg, arm = load_config(args.config), Arm()
    jm = load_joint_map(cfg)
    if args.samples:
        data = json.loads(args.samples.read_text(encoding="utf-8"))
        result = fit_touch(data["touches"] if isinstance(data, dict) else data, arm, jm)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return 0 if result["usable"] else 1
    robot = MockRobot(arm=arm, joint_map=jm) if args.mock else ReadOnlyBus(cfg) if args.direct else DaemonRobot(cfg["daemon_url"], jm)
    samples, captures, seq = [], [], -1
    try:
        if args.once:
            owner = stable_pose(robot.read)
            print(json.dumps({"owner_deg": owner, "model_rad": jm.to_model(owner), "tips": arm.tips(jm.to_model(owner))}, default=json_default))
            return 0
        print("Commands: folded | closed | open | touch <x> <y> | fit | done. Keep torque off and support the arm.", flush=True)
        # A tiny input thread supports Windows console lines and POSIX identically.
        import queue
        import threading
        lines = queue.Queue()
        def read_lines():
            for line in sys.stdin:
                lines.put(line.strip())
        threading.Thread(target=read_lines, daemon=True).start()
        while True:
            command = None
            try:
                command = lines.get_nowait()
            except queue.Empty:
                pass
            try:
                control = json.loads(args.control_file.read_text())
                if isinstance(control["seq"], int) and control["seq"] > seq:
                    seq, command = control["seq"], control["cmd"]
            except (OSError, KeyError, ValueError, TypeError):
                pass
            if not command:
                time.sleep(.05)
                continue
            if command == "done":
                break
            try:
                if command == "fit":
                    print(json.dumps(fit_touch(samples, arm, jm)), flush=True)
                    continue
                parts = command.split()
                if parts[0] not in ("folded", "closed", "open", "touch"):
                    raise ValueError("unknown pose command")
                owner = stable_pose(robot.read)
                capture = {"kind": parts[0], "owner_deg": owner.tolist()}
                if parts[0] == "touch":
                    if len(parts) != 3:
                        raise ValueError("touch requires x y in spec-frame metres")
                    capture["xyz"] = [float(parts[1]), float(parts[2]), 0]
                    samples.append(capture)
                captures.append(capture)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({"touches": samples, "captures": captures}, indent=2), encoding="utf-8")
                print(json.dumps(capture), flush=True)
            except ValueError as exc:
                print(f"REFUSED {exc}", flush=True)
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
