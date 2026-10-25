"""Argon reads the board; Session runs one place_on_cell policy per robot turn.

--dry-run renders C2's synthetic bench and uses an oracle board reader. It checks
the game/vision contract without claiming a physical placement-policy evaluation.
--oracle-only runs the same legal game when the synthetic bench is not installed.
"""
import _bootstrap  # noqa: F401
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from r2p import gemini, ttt


def publish(path, board, decision=None, **fields):
    gemini.atomic_json(path, {"schema": "r2p_ttt/1", "updated": time.time(), "board": board,
                             "argon_says": decision.get("why", "") if decision else "",
                             "model": decision.get("model") if decision else None,
                             "source": decision.get("source") if decision else None,
                             "move": decision.get("move") if decision else None, **fields})


def bench_supply(piece_height):
    from r2p.tasks import load_task
    obj = next(o for o in load_task("place_on_cell")["objects"] if o["role"] == "manipulated")
    return [*(obj.get("xyz", obj.get("xy"))[:2]), piece_height / 2]


def placement(args, cell, board=None):
    task = ttt.placement_task(cell, args.supply_xyz, piece_size=args.piece_size, board=board)
    piece = next(o for o in task["objects"] if o["role"] == "manipulated")
    piece["size_m"][2] = getattr(args, "piece_height", args.piece_size)
    return task


def check_plans(args):
    """Read configuration and solve IK; no camera, serial bus or Gemini access."""
    from r2p.model import Arm
    from r2p.robot.config import load_config, load_joint_map
    from r2p.robot.dryrun import ScriptedPolicy
    from r2p.tasks import encode_slots, sample_scene
    cfg, arm = load_config(args.config), Arm()
    joint_map = load_joint_map(cfg)
    plans = []
    for cell in ttt.CELL_IDS:
        task = placement(args, cell)
        policy = ScriptedPolicy(task, arm, cfg)
        scene = sample_scene(task, np.random.default_rng(args.seed))
        poses = {o["id"]: o for o in scene["objects"]}
        row = {"cell": cell, "supply_xyz": args.supply_xyz, "target_xy": ttt.geometry(task)["cells"][cell],
               "waypoints": [], "safe": True}
        try:
            previous = policy.comp(policy.start_q)
            for phase, target, dwell in policy.plan(encode_slots(task, poses)):
                sent = policy.comp(target)
                geometry = arm.geometry(sent)
                clear = policy.guard.path_clear(previous, sent)
                within_limits = bool(np.all(sent >= policy.guard.limits[:, 0]) and np.all(sent <= policy.guard.limits[:, 1]))
                row["waypoints"].append({"phase": phase, "q": target.tolist(), "q_servo": sent.tolist(),
                                         "owner_deg": joint_map.to_owner(sent).tolist(),
                                         "jaw_xyz": arm.tips(target)[2].tolist(), "geometry": geometry,
                                         "path_clear": clear, "within_limits": within_limits, "dwell_ticks": dwell})
                row["safe"] = row["safe"] and clear and within_limits
                previous = sent
        except ValueError as exc:
            row.update(safe=False, error=str(exc))
        plans.append(row)
        print(f"{cell}: {'CLEAR' if row['safe'] else 'BLOCKED'} target={row['target_xy']} "
              f"approach=0.120m carry/retreat=0.100m "
              f"min finger={min((w['geometry']['finger'] for w in row['waypoints']), default=float('nan')):.4f}m "
              f"{row.get('error', '')}")
    report = {"schema": "r2p_ttt_plan/1", "motion": False, "joint_map": joint_map.name,
              "joint_map_verified": joint_map.verified, "piece_size_m": [args.piece_size, args.piece_size, args.piece_height],
              "port": cfg.get("port"), "plans": plans, "safe": all(row["safe"] for row in plans)}
    gemini.atomic_json(args.check_out, report)
    print(f"No motion. Full IK, owner targets and clearance: {args.check_out}; map={joint_map.name}, verified={joint_map.verified}")
    return 0 if report["safe"] else 1


def mock_live(args):
    """Mock motion with live phone viewing only; no Gemini or hardware writes."""
    from r2p.model import Arm
    from r2p.perception import open_camera
    from r2p.robot import MockRobot, Session, Operator
    from r2p.robot.config import load_config, load_joint_map, sag_gain
    from r2p.robot.dryrun import ScriptedPolicy, make_world
    from r2p.robot.gravity import GravitySag
    from r2p.robot.publisher import Publisher
    cfg, arm = load_config(args.config), Arm()
    joint_map = load_joint_map(cfg)
    task = placement(args, args.cell)
    policy = ScriptedPolicy(task, arm, cfg)
    camera = open_camera(cfg["camera"])
    world, synthetic_camera, _ = make_world(task, arm, args.seed)
    robot = MockRobot(arm=arm, joint_map=joint_map, q0=policy.start_q, sag=GravitySag(arm, sag_gain(cfg)))
    class ViewingCamera:
        frame = None
        captured = 0
        def read(self):
            if self.frame is None or time.monotonic() - self.captured > .5:
                self.frame = camera.read()
                self.captured = time.monotonic()
            return self.frame
    publisher = Publisher(args.publish)
    try:
        result = Session(robot, policy, task, arm=arm, joint_map=joint_map, world=world, camera=ViewingCamera(),
                         cfg=cfg, realtime=False, operator=Operator(None, keyboard=False), publisher=publisher).run(timeout_s=args.timeout)
        board = [""] * 9
        if result["success"]:
            board[ttt.cell_index(args.cell)] = "O"
        decision = {"why": "Mock placement; phone camera is read-only viewing, not placement evidence",
                    "source": "mock", "model": None, "move": args.cell}
        publish(args.status, board, decision, status="mock_complete", synthetic=True, physical_placement=False,
                camera_source="real_phone_read_only", attempt=result)
        print(json.dumps(result))
        return 0 if result["outcome"] == "goal_reached" and result["success"] else 1
    finally:
        camera.close()
        if synthetic_camera:
            synthetic_camera.close()
        robot.close(torque_off=False)


def require_released_bus(cfg, released):
    import socket
    from urllib.parse import urlparse
    if not released:
        raise ValueError("--real requires --arm-released after the lead stops armd and rests the arm")
    parsed = urlparse(cfg["daemon_url"])
    try:
        connection = socket.create_connection((parsed.hostname, parsed.port or 8793), timeout=.5)
    except ConnectionRefusedError:
        return
    except OSError as exc:
        raise ValueError("Cannot verify armd is stopped; refusing serial access") from exc
    else:
        connection.close()
        raise ValueError("armd still owns the bench; refusing to open the serial port")


def oracle_board(world):
    """Read labelled synthetic object poses, with the same metric cell geometry."""
    board = [""] * 9
    centers = np.asarray(list(ttt.geometry()["cells"].values()))
    for obj in world.objects:
        if "ttt_mark" not in obj:
            continue
        xy = np.asarray(world.poses()[obj["id"]]["xyz"])[:2]
        index = int(np.argmin(np.linalg.norm(centers - xy, axis=1)))
        if np.linalg.norm(centers[index] - xy) > 0.02 or board[index]:
            raise ValueError("Synthetic board piece is off-grid or overlaps another piece")
        board[index] = obj["ttt_mark"]
    return ttt.validate_board(board)


def synth_frame(board, piece_size, seed=0):
    from r2p.perception import CameraModel
    from r2p.synthbench import World
    from r2p.tasks import load_task

    task = copy.deepcopy(load_task("place_on_cell"))
    task["objects"] = [obj for obj in task["objects"] if obj["role"] in ("manipulated", "target")]
    piece_id = next(obj["id"] for obj in task["objects"] if obj["role"] == "manipulated")
    task["goal"] = [{"pred": "at", "a": piece_id, "xy": "cell", "tol_m": 0.015}]
    task["randomize"] = {piece_id: {"x": [0.16, 0.16], "y": [-0.14, -0.14]},
                         "cell": {"x": [0.245, 0.245], "y": [0, 0]}, "min_sep_m": 0}
    for i, mark in enumerate(board):
        if mark:
            xy = ttt.geometry(task)["cells"][ttt.CELL_IDS[i]]
            task["objects"].append({"id": f"mark_{i}", "role": "obstacle", "shape": "box" if mark == "O" else "cylinder",
                                    "size_m": [piece_size] * (3 if mark == "O" else 2), "xyz": [*xy, piece_size / 2],
                                    "color": "brown" if mark == "O" else "blue", "fixed": True, "ttt_mark": mark})
            task["randomize"][f"mark_{i}"] = {"xy_choice": [xy]}
    task["limits"]["max_objects"] = max(task["limits"]["max_objects"], len(task["objects"]))
    cam = CameraModel.look_at(image_size=(640, 480))
    world = World(task, np.random.default_rng(seed), cam)
    try:
        return world.render(), cam, oracle_board(world)
    finally:
        world.close()


def dry_games(args):
    rng = np.random.default_rng(args.seed)
    results = []
    for game in range(args.games):
        board, history = [""] * 9, []
        decision = None
        while not ttt.terminal(board):
            human = int(rng.choice([i for i, mark in enumerate(board) if not mark]))
            board[human] = "X"
            if ttt.terminal(board):
                break
            if args.oracle_only:
                move = ttt.minimax(board, "O")
                decision = {"board": board.copy(), "move": ttt.CELL_IDS[move], "why": "Oracle board, perfect minimax",
                            "source": "oracle", "model": None}
            else:
                frame, cam, observed = synth_frame(board, args.piece_size, args.seed + game)
                best = ttt.minimax(observed, "O")
                fake = gemini.FakeGemini(responder=lambda **_: (json.dumps({"board": observed, "move": ttt.CELL_IDS[best],
                                                                            "why": "Oracle board, perfect minimax"}), "oracle/no-model"))
                decision = ttt.decide(frame, cam, history, "O", args.robot_desc, args.human_desc, ask=fake)
                decision["source"] = "oracle"
            history.append({"cell": ttt.CELL_IDS[human], "mark": "X"})
            move = ttt.cell_index(decision["move"])
            board[move] = "O"
            history.append({"cell": ttt.CELL_IDS[move], "mark": "O"})
            publish(args.status, board, decision, game=game + 1, status="playing", physical_placement=False)
        result = ttt.winner(board) or "draw"
        results.append(result)
        publish(args.status, board, decision, game=game + 1, status="done", result=result, physical_placement=False)
        print(f"Game {game + 1}: {result}; board={board}")
    print(f"Completed {len(results)} games; robot losses={results.count('X')}")
    return 0 if "X" not in results else 1


def confirm_human(history, robot_mark="O"):
    board = ttt.history_board(history)
    while True:
        value = input("Argon unavailable: confirm human cell (r0c0..r2c2), or q to stop: ").strip()
        if value == "q":
            raise KeyboardInterrupt
        try:
            i = ttt.cell_index(value)
            if board[i]:
                raise ValueError("Cell is occupied")
            proposed = board.copy()
            proposed[i] = "X" if robot_mark == "O" else "O"
            ttt.consistent_board(proposed, board, robot_mark)
            return proposed
        except ValueError as exc:
            print(str(exc))


def live_games(args):
    from r2p.perception import CameraModel, TaskPerceiver, open_camera
    from r2p.robot import Session, Operator
    from r2p.robot.config import load_config, load_joint_map, config_path
    from r2p.robot.io import RealRobot
    from r2p.robot.policy import load_policy, policy_config

    cfg = load_config(args.config)
    joint_map = load_joint_map(cfg)
    if not joint_map.verified and not getattr(args, "allow_unverified", False):
        raise ValueError("Live tic-tac-toe requires a verified joint map or explicit --allow-unverified after bench verification")
    require_released_bus(cfg, getattr(args, "arm_released", False))
    cam = CameraModel.load(args.camera or config_path(cfg["camera"]["calibration"]))
    camera = open_camera(cfg["camera"])
    try:
        robot = RealRobot(cfg)
    except BaseException:
        camera.close()
        raise
    try:
        with Operator(cfg["control_file"], keyboard=True) as operator:
            for game in range(args.games):
                history, board = [], [""] * 9
                decision = None
                while not ttt.terminal(board):
                    if input("Make your X move; replenish the measured supply spot; Enter to read, q to stop: ").strip() == "q":
                        return 0
                    frame = camera.read()
                    try:
                        decision = ttt.decide(frame, cam, history, "O", args.robot_desc, args.human_desc)
                    except ttt.BoardUnavailable:
                        observed = confirm_human(history)
                        move = ttt.minimax(observed, "O")
                        decision = {"board": observed, "move": ttt.CELL_IDS[move] if move is not None else None,
                                    "why": "Operator-confirmed board; local minimax", "source": "operator", "model": None}
                    board = decision["board"]
                    publish(args.status, board, decision, status="planning", game=game + 1)
                    history = {"board": board.copy()}
                    if decision["move"] is None:
                        break
                    task = placement(args, decision["move"], board)
                    # Projection of the measured supply spot seeds perception without a Gemini re-detection.
                    class KnownSetup:
                        def detect(self, image, task, *, role="small"):
                            from r2p.perception.geometry import silhouette_points, rotation
                            boxes = {}
                            for obj in task["objects"]:
                                if "xyz" not in obj:
                                    continue
                                pixels = cam.project(silhouette_points(obj) @ rotation(obj.get("yaw_deg", 0)).T + obj["xyz"])
                                lo, hi = pixels.min(axis=0) - 4, pixels.max(axis=0) + 4
                                boxes[obj["id"]] = [*lo, *hi]
                            return boxes
                    perceiver = TaskPerceiver(cam, task, detector=KnownSetup())
                    try:
                        perceiver.setup(frame)
                        from r2p.robot.dryrun import ScriptedPolicy
                        from r2p.model import Arm
                        from r2p.robot.publisher import Publisher
                        policy = ScriptedPolicy(task, Arm(), cfg, speed=.5 if not joint_map.verified else 1) if args.policy == "stub" else load_policy(args.policy)
                        policy_cfg = policy_config(policy, None if args.policy == "stub" else args.policy)
                        if policy_cfg.get("task") != "place_on_cell":
                            raise ValueError("Policy export must declare task place_on_cell")
                        record = Path(args.record) / f"game_{game+1}_turn_{board.count('O')+1}_{time.time_ns()}"
                        result = Session(robot, policy, task, camera=camera, perceiver=perceiver, cfg=cfg,
                                         joint_map=joint_map, operator=operator, record=record,
                                         policy_cfg=policy_cfg, publisher=Publisher(getattr(args, "publish", "out/robot/live")),
                                         speed=.5 if not joint_map.verified else 1).run(timeout_s=args.timeout)
                        cam.save(record / "camera.json")
                    finally:
                        perceiver.close()
                    if result["outcome"] != "goal_reached" or not result["success"]:
                        publish(args.status, board, decision, status="placement_failed", attempt=result)
                        return 1
                    board[ttt.cell_index(decision["move"])] = "O"
                    history = {"board": board.copy()}
                    publish(args.status, board, decision, status="playing", attempt=result)
                publish(args.status, board, decision, status="done", result=ttt.winner(board) or "draw")
    finally:
        camera.close()
        robot.close(torque_off=False)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--oracle-only", action="store_true")
    mode.add_argument("--frame-only", metavar="IMAGE")
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--check", action="store_true", help="Print IK and clearance for every cell; no motion or requests")
    mode.add_argument("--mock-live", action="store_true", help="MockRobot + read-only live phone viewing; no Gemini")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--piece-size", type=float, default=0.025, help="piece width in metres")
    parser.add_argument("--piece-height", type=float, default=0.014, help="piece height in metres")
    parser.add_argument("--robot-desc", default="brown cubes")
    parser.add_argument("--human-desc", default="blue discs")
    parser.add_argument("--camera", help="calibrated camera JSON")
    parser.add_argument("--history", help="JSON file containing prior confirmed moves or board")
    parser.add_argument("--status", default="out/ttt/status.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="stub", help="place_on_cell export or the scripted probe")
    parser.add_argument("--supply-xyz", nargs=3, type=float, help="measured piece centre x y z, spec metres")
    parser.add_argument("--record", default="runs/tictactoe")
    parser.add_argument("--config")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--check-out", default="out/ttt/check.json")
    parser.add_argument("--publish", default="out/robot/live")
    parser.add_argument("--cell", choices=ttt.CELL_IDS, default="r1c1")
    parser.add_argument("--arm-released", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args(argv)
    if args.games < 1 or not 0.01 <= args.piece_size <= 0.04 or not .005 <= args.piece_height <= .04:
        parser.error("games must be positive; piece-size must be 0.01..0.04 metres")
    if args.supply_xyz is None:
        args.supply_xyz = bench_supply(args.piece_height)
    if args.check:
        return check_plans(args)
    if args.mock_live:
        return mock_live(args)
    if args.real:
        if not args.policy or args.supply_xyz is None:
            parser.error("--real requires --policy and the measured --supply-xyz")
        return live_games(args)
    if args.frame_only:
        import cv2
        from r2p.perception import CameraModel
        if not args.camera:
            parser.error("--frame-only requires --camera")
        frame = cv2.imread(args.frame_only)
        if frame is None:
            parser.error("could not read the saved image")
        history = json.loads(Path(args.history).read_text()) if args.history else []
        result = ttt.decide(frame, CameraModel.load(args.camera), history, "O", args.robot_desc, args.human_desc)
        publish(args.status, result["board"], result, status="frame_only")
        print(json.dumps(result))
        return 0
    return dry_games(args)


if __name__ == "__main__":
    raise SystemExit(main())
