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
    from r2p.robot.daemon import DaemonRobot
    from r2p.robot.policy import load_policy, policy_config

    cfg = load_config(args.config)
    joint_map = load_joint_map(cfg)
    if not joint_map.verified:
        raise ValueError("Live tic-tac-toe requires a verified joint map")
    cam = CameraModel.load(args.camera or config_path(cfg["camera"]["calibration"]))
    camera = open_camera(cfg["camera"])
    robot = DaemonRobot(cfg["daemon_url"], joint_map)
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
                    task = ttt.placement_task(decision["move"], args.supply_xyz, piece_size=args.piece_size, board=board)
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
                        policy = load_policy(args.policy)
                        policy_cfg = policy_config(policy, args.policy)
                        if policy_cfg.get("task") != "place_on_cell":
                            raise ValueError("Policy export must declare task place_on_cell")
                        record = Path(args.record) / f"game_{game+1}_turn_{board.count('O')+1}_{time.time_ns()}"
                        result = Session(robot, policy, task, camera=camera, perceiver=perceiver, cfg=cfg,
                                         joint_map=joint_map, operator=operator, record=record,
                                         policy_cfg=policy_cfg).run(timeout_s=args.timeout)
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
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--piece-size", type=float, default=0.032, help="piece width in metres")
    parser.add_argument("--robot-desc", default="brown cubes")
    parser.add_argument("--human-desc", default="blue discs")
    parser.add_argument("--camera", help="calibrated camera JSON")
    parser.add_argument("--history", help="JSON file containing prior confirmed moves or board")
    parser.add_argument("--status", default="out/ttt/status.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", help="place_on_cell export for live Session")
    parser.add_argument("--supply-xyz", nargs=3, type=float, help="measured piece centre x y z, spec metres")
    parser.add_argument("--record", default="runs/tictactoe")
    parser.add_argument("--config")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args(argv)
    if args.games < 1 or not 0.01 <= args.piece_size <= 0.04:
        parser.error("games must be positive; piece-size must be 0.01..0.04 metres")
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
