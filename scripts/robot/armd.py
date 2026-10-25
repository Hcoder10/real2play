"""Start only with the arm supported at rest. This process owns the bus until exit."""
import _bootstrap  # noqa: F401
import argparse
import json

from r2p.robot.config import load_config, load_joint_map
from r2p.robot.daemon import ArmController, make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--port", help="override follower serial port")
    parser.add_argument("--http-port", type=int, default=8793)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.port:
        cfg["port"] = args.port
    joint_map = load_joint_map(cfg)
    if not args.mock and not joint_map.verified and not args.allow_unverified:
        parser.error("joint map is unverified; verify it or explicitly use --allow-unverified")
    from r2p.robot.io import MockRobot, RealRobot
    robot = MockRobot(joint_map=joint_map) if args.mock else RealRobot(cfg)
    controller = ArmController(robot, joint_map=joint_map, cfg=cfg)
    server = make_server(controller, ("127.0.0.1", args.http_port))
    print(json.dumps({"listening": f"http://127.0.0.1:{server.server_port}", "rest_q": controller.rest_q.tolist(),
                      "note": "Boot pose is supported rest; release is permitted only here."}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_requested.set()
        server.server_close()
        with controller.lock:
            controller._hold()
        robot.close(torque_off=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
