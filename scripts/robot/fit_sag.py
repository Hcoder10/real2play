"""Fit sag from eight supervised armd moves, or an offline --simulate K run."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import time
import numpy as np
from r2p.robot.calibration import SAG_POSES, fit_sag
from r2p.robot.record import json_default


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", type=float, metavar="K")
    parser.add_argument("--url", default="http://127.0.0.1:8793")
    parser.add_argument("--out", type=Path, default=Path("runs/sag_fit.json"))
    args = parser.parse_args(argv)
    from r2p.model import Arm
    from r2p.frames import HOME_Q
    from r2p.robot.kinematics import solve_ik
    from r2p.robot.gravity import GravitySag
    from r2p.robot.daemon import DaemonRobot
    arm, samples = Arm(), []
    client = DaemonRobot(args.url) if args.simulate is None else None
    sag = GravitySag(arm, args.simulate) if args.simulate is not None else None
    for index, (radius, azimuth, height) in enumerate(SAG_POSES):
        angle = np.deg2rad(azimuth)
        xyz = [radius*np.cos(angle), radius*np.sin(angle), height]
        # Elevated calibration poses may tilt the fingers; xyz daemon commands
        # remain strictly fingers-down. Send validated joint IK poses here.
        q = solve_ik(arm, xyz, HOME_Q, fingers_down=False)
        if client:
            input(f"Support/supervise the arm. ENTER for move {index+1}/8 to {np.round(xyz,3)}; Ctrl-C stops: ")
            state = client.request({"joints": q})
            time.sleep(1)
            measured = np.median([client.request()["q"] for _ in range(10)], axis=0)
            commanded = state["q_target"]
        else:
            commanded, measured = q, sag(q)
        samples.append({"xyz": xyz, "commanded": commanded, "measured": measured})
    result = {**fit_sag(samples, arm), "simulated": client is None, "samples": samples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
