"""One placement path shared by the physical twin rehearsal and direct live run."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from r2p.frames import JOINTS
from r2p.model import Arm
from r2p.tasks import check, commands, encode_slots, load_task, sample_scene
from .config import load_config, load_joint_map
from .gravity import GravityComp
from .limits import LimitedRobot, OwnerLimits
from .operator import Operator
from .policy import check_chunk
from .runner import Session

SUPPLY_XY = [[.1274, -.1272], [.1748, -.1245], [.2205, -.1248]]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def calibration_limits(cfg):
    path = Path(cfg["calibration_dir"]).expanduser()/f"{cfg['robot_id']}.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    half = np.array([(calibration[j]["range_max"]-calibration[j]["range_min"])/2*360/4095 for j in JOINTS[:5]])
    return OwnerLimits(np.r_[-half+1, 0], np.r_[half-1, 100])


def validate_stream_state(state, joint_map):
    from .io import vector
    if not {"owner_deg", "q", "q_target", "rest_q", "loads"} <= state.keys():
        raise ValueError("daemon is not the repo streaming protocol; legacy waypoint armd cannot run a 30 Hz policy")
    if not np.allclose(joint_map.to_model(vector(state["owner_deg"])), vector(state["q"]), atol=1e-6, rtol=0):
        raise ValueError("daemon joint map differs from the active policy map")
    vector(state["q_target"])
    vector(state["rest_q"])
    if not state.get("torque_enabled", False):
        raise ValueError("streaming daemon torque is off")
    return state


def capture_profile(url, joint_map, cfg=None):
    from r2p.sim.bench import make_profile
    with urlopen(url.rstrip("/")+"/state", timeout=5) as response:
        state = json.load(response)
    protocol = "legacy_waypoint"
    streaming = None
    if "owner_deg" in state:
        streaming = deepcopy(validate_stream_state(state, joint_map))
        limits = calibration_limits(cfg or load_config())
        # Match the older daemon's published tenth-degree bounds. The live
        # guard still reserves two full model degrees of margin.
        state = {"q": state["owner_deg"], "rest": joint_map.to_owner(state["rest_q"]).tolist(),
                 "lo": limits.lo.round(1).tolist(), "hi": limits.hi.round(1).tolist(),
                 "joint_map": joint_map.name}
        protocol = "repo_streaming"
    OwnerLimits.from_state(state)
    profile = make_profile(state, joint_map)
    profile.update(captured_at=datetime.now(timezone.utc).isoformat(), state_url=url.rstrip("/")+"/state",
                   state_read_only=True, daemon_protocol=protocol)
    if streaming is not None:
        profile["streaming_state"] = streaming
        profile["owner_limits_source"] = "local servo calibration, checked against EEPROM by the daemon owner"
    return profile


def validate_profile(profile, joint_map):
    actual = {"sign": joint_map.sign.tolist(), "offset": joint_map.offset.tolist(),
              "closed_rad": joint_map.gc, "open_rad": joint_map.go}
    if profile["joint_map"] != joint_map.name or profile["joint_map_parameters"] != actual:
        raise ValueError("bench profile is stale: active joint map differs; capture a new read-only profile")
    limits = OwnerLimits.from_state(profile["owner_state"])
    expected = limits.model_bounds(joint_map, Arm().range, profile["margin_deg"])
    if not np.allclose(expected, profile["model_limits"], atol=1e-9):
        raise ValueError("bench profile model limits do not match the calibrated owner limits")
    return limits


def placement_task(cell, cap=0, *, board=None, supply_xy=None, measured=None):
    """Freeze the CURRENT measured board, cap dimensions and requested supply."""
    from r2p.ttt import CELL_IDS, cell_index, validate_board
    task = deepcopy(measured) if measured is not None else load_task("place_on_cell")
    cell = CELL_IDS[cell_index(cell)]
    board = validate_board(board or [""]*9)
    piece = next(o for o in task["objects"] if o["role"] == "manipulated")
    supply = SUPPLY_XY[cap] if supply_xy is None else supply_xy
    piece.pop("xy", None)
    piece["xyz"] = [*supply, piece["size_m"][-1]/2]
    target = next(o for o in task["objects"] if o["id"] == "cell")
    target.pop("xy", None)
    target["xyz"] = [*task["board"]["cells"][cell], .0005]
    task["randomize"] = {"min_sep_m": 0}
    occupied = [(i, mark) for i, mark in enumerate(board) if mark]
    blockers = [o for o in task["objects"] if o["role"] == "obstacle"]
    for i, obj in enumerate(blockers):
        if i < min(len(occupied), len(blockers)):
            board_index = occupied[-len(blockers):][i][0]
            obj["xyz"] = [*task["board"]["cells"][CELL_IDS[board_index]], obj["size_m"][-1]/2]
            obj.pop("xy", None)
            obj.update(fixed=True, optional=False, p_present=1)
            task["randomize"][obj["id"]] = {"p_present": 1}
        else:
            obj.update(optional=True, p_present=0)
            task["randomize"][obj["id"]] = {"p_present": 0}
    task["target_cell"] = cell
    return task


def validate_export(policy, task):
    cfg = policy.config
    if cfg.get("task") != task["name"] or cfg.get("action") != "absolute_q" or cfg.get("horizon") != 15:
        raise ValueError("export must declare place_on_cell, absolute_q and horizon 15")
    if cfg.get("K") != task["limits"]["max_objects"] or cfg.get("object_ids") != [o["id"] for o in task["objects"]]:
        raise ValueError("export slot layout differs from the current measured task")
    for key in ("start_q", "obs_layout", "obs_const", "eval"):
        if key not in cfg:
            raise ValueError(f"export lacks required {key}")
    from .io import vector
    vector(cfg["start_q"])
    if cfg.get("obs_frame") != "spec_footprint":
        raise ValueError("export must use the spec footprint frame")


def audit(policy, task, profile, joint_map, cfg):
    """Read-only diagnostic, including the raw chunk before any safety projection."""
    validate_export(policy, task)
    limits = validate_profile(profile, joint_map)
    if policy.config.get("deployment"):
        trained = policy.config["deployment"]
        for key in ("joint_map_parameters", "model_limits", "roll_rad", "gripper_open_rad", "gripper_close_rad"):
            if trained.get(key) != profile[key]:
                raise ValueError(f"export deployment convention differs from current bench profile: {key}")
    arm = Arm()
    bounds = limits.model_bounds(joint_map, arm.range, cfg["guard"]["margin_deg"])
    start = np.asarray(policy.config["start_q"], float)
    from .config import sag_gain
    servo_start = GravityComp(arm, sag_gain(cfg))(start)
    scene = sample_scene(task, np.random.default_rng(0))
    slots = encode_slots(task, {o["id"]: o for o in scene["objects"]})
    chunk = check_chunk(policy.act({"q": start, "slots": slots, "command": commands(task)[0]}))
    raw_hits = limits.violations(joint_map.to_owner(chunk))
    start_hits = (servo_start < bounds[:, 0]) | (servo_start > bounds[:, 1])
    return {"start_q": start.tolist(), "start_owner": joint_map.to_owner(start).tolist(),
            "compensated_start_owner": joint_map.to_owner(servo_start).tolist(),
            "start_valid": not bool(np.any(start_hits)),
            "start_limit_joints": [j for j, bad in zip(JOINTS, start_hits) if bad],
            "first_chunk_owner_limit_targets": int(np.any(raw_hits, axis=1).sum()),
            "first_chunk_owner_limit_joints": raw_hits.sum(axis=0).tolist(),
            "first_chunk_owner_min": joint_map.to_owner(chunk).min(axis=0).tolist(),
            "first_chunk_owner_max": joint_map.to_owner(chunk).max(axis=0).tolist()}


class TwinRobot:
    """Owner-space facade around TaskTwin; physics provides sag/contact/latency."""
    def __init__(self, twin, joint_map, owner_limits):
        self.twin, self.joint_map, self.owner_limits = twin, joint_map, owner_limits
        self.target = twin.q()
        self.sent, self.limit_hits = [], 0
        self.torque_enabled = True

    def read(self):
        return self.joint_map.to_owner(self.twin.q())

    def write(self, owner):
        q = self.owner_limits.require(owner)
        self.target = self.joint_map.to_model(q)
        self.sent.append(q.copy())
        return q.copy()

    def step(self, dt):
        if self.torque_enabled:
            self.twin.step(self.target)

    def hold(self):
        return self.write(np.clip(self.read(), self.owner_limits.lo, self.owner_limits.hi))

    def loads(self):
        return {}  # no calibrated conversion from MuJoCo torque to servo load counts

    def torque_off(self):
        self.torque_enabled = False

    def close(self, torque_off=False):
        self.twin.close()


def twin_daemon_robot(robot, joint_map, cfg, state):
    """Exercise the actual server controller and client without HTTP or serial.

    Physics advances once per runner tick. The controller sees a bus without a
    step method, as on hardware. Network/inference scheduling jitter is not
    modeled; the daemon's nominal 30 Hz rate cap and accepted targets are.
    """
    from types import SimpleNamespace
    from .daemon import ArmController, DaemonRobot
    from .io import vector
    bus = SimpleNamespace(read=robot.read, write=robot.write, hold=robot.hold,
                          loads=robot.loads, torque_off=robot.torque_off)
    controller = ArmController(bus, joint_map=joint_map, cfg=cfg, realtime=False)
    controller.target, controller.rest_q = vector(state["q_target"]), vector(state["rest_q"])

    class InProcessDaemon(DaemonRobot):
        def __init__(self):
            super().__init__(joint_map=joint_map)
            self.daemon_clipped_commands = 0

        def request(self, command=None):
            if command is None:
                return controller.state()
            result = controller.command(command)
            if "joints" in command and not np.allclose(command["joints"], result["q_target"], atol=1e-9, rtol=0):
                self.daemon_clipped_commands += 1
            return result

        def step(self, dt):
            robot.step(dt)

    return InProcessDaemon()


def require_stream_handover(state, profile, joint_map):
    validate_stream_state(state, joint_map)
    OwnerLimits.from_state(profile["owner_state"]).require(joint_map.to_owner(state["q_target"]))
    if np.max(np.abs(np.asarray(state["q_target"])-np.asarray(state["q"]))) > np.deg2rad(4):
        raise ValueError("daemon has not settled to its held target; wait before the policy handover")
    if profile.get("daemon_protocol") != "repo_streaming":
        raise ValueError("capture and rehearse a streaming profile before live streaming")
    initial = joint_map.to_model(profile["owner_state"]["q"])
    if np.max(np.abs(initial-np.asarray(state["q"]))) > np.deg2rad(4):
        raise ValueError("daemon pose changed since rehearsal; recapture its read-only profile and rehearse")


def streaming_robot(cfg, profile, joint_map):
    """GET-only handshake. The returned object writes only when Session runs."""
    from .daemon import DaemonRobot
    robot = DaemonRobot(cfg["daemon_url"], joint_map)
    state = robot.request()
    require_stream_handover(state, profile, joint_map)
    calibrated = calibration_limits(cfg)
    saved = OwnerLimits.from_state(profile["owner_state"])
    if not (np.allclose(calibrated.lo, saved.lo, atol=.051, rtol=0) and np.allclose(calibrated.hi, saved.hi, atol=.051, rtol=0)):
        raise ValueError("saved owner limits differ from the current calibration")
    return robot


class TwinWorld:
    def __init__(self, twin, task, arm, threshold):
        self.twin, self.task, self.arm, self.threshold = twin, task, arm, threshold
        self.scene, self.seed = twin.scene, 0
        self.initial = twin.poses()

    def update(self, q, robot):
        pass  # robot.step advances physics exactly once, including object contacts

    def poses(self):
        return self.twin.poses()

    def truth_check(self):
        return check(self.task, {"objects": self.poses(), "initial": self.initial,
            "gripper_open": bool(self.twin.q()[5] > self.threshold and self.twin.commanded_q[5] > self.threshold),
            "site_xyz": self.arm.tips(self.twin.q())[2].tolist()})


def make_session(robot, policy, task, profile, *, cfg=None, **kwargs):
    cfg = cfg or load_config()
    joint_map = load_joint_map(cfg)
    limits = validate_profile(profile, joint_map)
    validate_export(policy, task)
    # Use the profile's captured margin in both rehearsal and live execution.
    if cfg["guard"]["margin_deg"] != profile["margin_deg"]:
        raise ValueError("guard margin differs from bench profile; recapture the profile")
    speed = kwargs.pop("speed", .5 if not joint_map.verified else 1.)
    return Session(LimitedRobot(robot, limits), policy, task, joint_map=joint_map, cfg=cfg,
                   owner_limits=limits, speed=speed, **kwargs)


def rehearse(policy, task, profile, *, cfg=None, timeout_s=60, record=None, transport="direct"):
    from r2p.sim.twin import TaskTwin, load_physics
    cfg = cfg or load_config()
    joint_map, arm = load_joint_map(cfg), Arm()
    limits = validate_profile(profile, joint_map)
    diagnostics = audit(policy, task, profile, joint_map, cfg)
    scene = sample_scene(task, np.random.default_rng(0))
    # Two other caps remain physical distractors, without changing policy slots.
    piece = next(o for o in scene["objects"] if o["role"] == "manipulated")
    for index, xy in enumerate(SUPPLY_XY):
        if np.linalg.norm(np.asarray(xy)-piece["xyz"][:2]) < .001:
            continue
        scene["objects"].append(dict(deepcopy(piece), id=f"supply_cap_{index}", role="obstacle", xyz=[*xy, piece["xyz"][2]]))
    physics = load_physics(cfg["rig"])
    if transport not in ("direct", "armd"):
        raise ValueError("transport must be direct or armd")
    if transport == "armd" and profile.get("daemon_protocol") != "repo_streaming":
        raise ValueError("streaming rehearsal requires a captured repo daemon profile")
    initial_owner = profile["owner_state"]["q" if transport == "armd" else "rest"]
    rest = joint_map.to_model(initial_owner)
    with TaskTwin(scene, physics=physics, q0=rest, joint_limits=limits.model_bounds(joint_map)) as twin:
        robot = TwinRobot(twin, joint_map, limits)
        interface = (twin_daemon_robot(robot, joint_map, cfg, profile["streaming_state"])
                     if transport == "armd" else robot)
        world = TwinWorld(twin, task, arm, policy.config.get("gripper_open_threshold_rad", .5))
        session = make_session(interface, policy, task, profile, cfg=cfg, arm=arm, world=world,
                               realtime=False, operator=Operator(None, keyboard=False), record=record)
        handover_error = None
        if transport == "armd" and diagnostics["start_valid"]:
            try:
                require_stream_handover(profile["streaming_state"], profile, joint_map)
            except ValueError as exc:
                handover_error = str(exc)
        if handover_error:
            result = {"outcome": "handover_refused", "success": False, "error": handover_error,
                      "clipped_ticks": 0, "clipped_fraction": 0, "commanded_q_resets": 0,
                      "requested_limit_ticks": 0, "requested_limit_joints": [0]*6,
                      "raw_chunk_owner_limit_targets": 0, "raw_chunk_owner_limit_joints": [0]*6,
                      "ticks": 0, "duration_s": 0., "refusals": 0, "start": session.start_stats,
                      "observation_source": "synthetic_truth", "truth_check": world.truth_check()}
        else:
            result = session.run(timeout_s=timeout_s)
        poses = twin.poses()
        error = float(np.linalg.norm(np.asarray(poses[piece["id"]]["xyz"])[:2]-poses["cell"]["xyz"][:2])*1000)
        return {**result, "cell": task["target_cell"], "supply_xy": piece["xyz"][:2],
                "placement_error_mm": error, "audit": diagnostics,
                "sent_owner_limit_hits": sum(int(np.any(limits.violations(q))) for q in robot.sent),
                "sent_commands": len(robot.sent), "final_poses": poses,
                "daemon_clipped_commands": getattr(interface, "daemon_clipped_commands", 0),
                "transport": transport, "initial_owner": initial_owner,
                "physics_sha256": digest(physics), "latency_ticks": physics.get("latency_ticks", 0),
                "task_sha256": digest(task), "synthetic": True, "load_abort_tested": False}


def require_released_bus(cfg, released):
    if not released:
        raise ValueError("--real requires --arm-released after the owner releases armd at supported rest")
    parsed = urlparse(cfg["daemon_url"])
    try:
        connection = socket.create_connection((parsed.hostname, parsed.port or 8793), timeout=.5)
    except ConnectionRefusedError:
        return
    except OSError as exc:
        raise ValueError("cannot verify armd is stopped; refusing serial access") from exc
    connection.close()
    raise ValueError("armd still owns the bench; refusing serial access")
