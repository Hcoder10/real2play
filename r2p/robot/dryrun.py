"""Scripted CPU smoke test and a truth-only fallback until synthbench is installed.

The fallback uses closing jaw contact and a rigid gripper-frame attachment; it
never moves objects directly from policy phase or scores a grasp by proximity.
"""
from copy import deepcopy
import importlib.util
import numpy as np

from r2p.frames import HOME_Q
from .kinematics import solve_ik


class TruthWorld:
    def __init__(self, task, arm, seed=0):
        from r2p.tasks import sample_scene
        self.task, self.arm, self.seed = task, arm, seed
        self.scene = sample_scene(task, np.random.default_rng(seed))
        if task["name"] != "cube_in_bowl":
            raise ValueError("truth-only fallback supports cube_in_bowl; install C2 synthbench for other tasks")
        rng = np.random.default_rng(seed)
        for obj in self.scene["objects"]:
            if obj["id"] == "cube":
                obj["xyz"] = [0.235 + rng.uniform(-0.008, 0.008), -0.055, obj["size_m"][2]/2]
                obj["yaw_deg"] = 0
            elif obj["id"] == "bowl":
                obj["xyz"] = [0.245, 0.075, obj["size_m"][1]/2]
        self.objects = {o["id"]: deepcopy(o) for o in self.scene["objects"]}
        self.held = None
        self.local = None
        self.local_rotation = None
        self.previous_gap = float("inf")
        self.q = HOME_Q.copy()

    def poses(self):
        return {k: {"xyz": list(o["xyz"]), "yaw_deg": o.get("yaw_deg", 0), "size_m": o["size_m"], "seen": True}
                for k, o in self.objects.items()}

    def update(self, q, robot):
        self.q = np.asarray(q).copy()
        a, b, mid = self.arm.tips(q)
        rotation = self.arm.d.body("gripper").xmat.reshape(3, 3).copy()
        gap = float(np.linalg.norm(b-a))
        if self.held:
            obj = self.objects[self.held]
            obj["xyz"] = (mid + rotation @ self.local).tolist()
            object_rotation = rotation @ self.local_rotation
            obj["yaw_deg"] = float(np.rad2deg(np.arctan2(object_rotation[1, 0], object_rotation[0, 0])))
            if q[5] > 0.65:
                xyz = np.asarray(obj["xyz"])
                floor = 0.0
                for bowl in self.objects.values():
                    if bowl["shape"] == "bowl" and np.linalg.norm(xyz[:2]-np.asarray(bowl["xyz"])[:2]) < bowl["size_m"][0]/2 - obj["size_m"][0]/2:
                        floor = bowl.get("wall_m", 0.005)
                obj["xyz"][2] = floor + obj["size_m"][2]/2
                self.held = None
                robot.set_grasp(None)
        elif gap < self.previous_gap and q[5] < 0.65:
            axis = (b-a)/gap
            for key, obj in self.objects.items():
                if obj["role"] != "manipulated":
                    continue
                p = np.asarray(obj["xyz"])
                width = obj["size_m"][0]
                along = np.dot(p-a, axis)
                lateral = p - (a + along*axis)
                # Both pads straddle the object's volume as the gap closes.
                if width-0.006 <= gap <= width+0.004 and 0 <= along <= gap and np.linalg.norm(lateral[:2]) < 0.012 and abs(lateral[2]) < obj["size_m"][2]/2:
                    self.held = key
                    self.local = rotation.T @ (p-mid)
                    yaw = np.deg2rad(obj.get("yaw_deg", 0))
                    c, s = np.cos(yaw), np.sin(yaw)
                    self.local_rotation = rotation.T @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    robot.set_grasp(q[5])
                    break
        self.previous_gap = gap

    def truth_check(self):
        from r2p.tasks import check
        return check(self.task, {"objects": self.poses(), "gripper_open": bool(self.q[5] > 0.5),
                                "site_xyz": self.arm.tips(self.q)[2].tolist()})


class ScriptedPolicy:
    """Conservative pick/place probe; absolute chunks respect the bench speed limits."""
    def __init__(self, task, arm, cfg=None, speed=1.0):
        from .config import load_config, sag_gain
        from .gravity import GravityComp
        from .guard import Guard
        if task["name"] not in ("cube_in_bowl", "place_on_cell"):
            raise ValueError("stub implements cube_in_bowl and place_on_cell")
        self.task, self.arm = task, arm
        self.cfg = cfg or load_config()
        self.speed = speed
        self.comp = GravityComp(arm, sag_gain(self.cfg))
        self.start_q = np.clip(HOME_Q, arm.range[:, 0]+np.deg2rad(2), arm.range[:, 1]-np.deg2rad(2))
        self.config = {"name": "stub", "task": task["name"], "start_q": self.start_q.tolist(),
                       "horizon": 15, "action": "absolute_q", "obs_const": {}, "eval": {"n": 0}}
        self.targets = None
        self.stage = 0
        self.dwell = 0
        self.done = False
        self.phase = "approach"
        self.command = "pick(cube)"
        self.guard = Guard(arm, self.cfg["guard"], start_q=self.start_q)

    def plan(self, slots):
        """Build the same nominal waypoints used by --check and execution."""
        ids = [o["id"] for o in self.task["objects"]]
        obj = next(o for o in self.task["objects"] if o["role"] == "manipulated")
        dest = "bowl" if self.task["name"] == "cube_in_bowl" else "cell"
        cube = np.asarray(slots[ids.index(obj["id"]), :3])
        target_xyz = np.asarray(slots[ids.index(dest), :3])
        width = min(slots[ids.index(obj["id"]), 5:7])
        # Aim just inside pad contact, instead of commanding the jaws through
        # the piece and triggering the mandated 4-degree observation reset.
        lo, hi = self.arm.range[5]
        for _ in range(30):
            grip = (lo + hi) / 2
            q = self.start_q.copy()
            q[5] = grip
            a, b, _ = self.arm.tips(q)
            if np.linalg.norm(a-b) < width:
                lo = grip
            else:
                hi = grip
        close = hi - np.deg2rad(.5)
        height = float(slots[ids.index(obj["id"]), 7])
        grasp_xyz = cube.copy()
        grasp_xyz[2] = max(.006, cube[2] + min(.006, height / 2 - .006))
        grasp_q = solve_ik(self.arm, grasp_xyz, self.start_q, grip=hi)
        grasp_open, grasp_close = grasp_q.copy(), grasp_q.copy()
        grasp_open[5], grasp_close[5] = .9, close
        lower = .065 if dest == "bowl" else max(.016, cube[2] + .006)
        carry = .14 if dest == "bowl" else .10
        # The nearest printed row is inside the arm's high-pose workspace.
        # Travel above its outer edge, then descend diagonally to the cell.
        carry_xy = target_xyz[:2].copy()
        retreat_xy = target_xyz[:2].copy()
        if dest == "cell":
            carry_xy[0] = max(carry_xy[0], .165)
            retreat_xy[0] = max(retreat_xy[0], .18)
        points = [("approach", [*cube[:2], .12], .9, 12),
                  ("descend", None, .9, 18), ("close", None, close, 24),
                  ("lift", [*cube[:2], carry], close, 12),
                  ("transport", [*carry_xy, carry], close, 12),
                  ("lower", [*target_xyz[:2], lower], close, 18),
                  ("release", None, .9, 24),
                  ("retreat", [*retreat_xy, carry], .9, 12)]
        targets, seed = [], self.start_q
        for phase, xyz, grip, dwell in points:
            if phase == "descend":
                target = grasp_open.copy()
            elif phase == "close":
                target = grasp_close.copy()
            elif phase == "release":
                target = seed.copy()
                target[5] = grip
            else:
                target = solve_ik(self.arm, xyz, seed, grip=grip,
                                  fingers_down=phase == "lower" and dest != "bowl")
            targets.append((phase, target.copy(), dwell))
            seed = target
        self.piece_id, self.destination = obj["id"], dest
        return targets

    def act(self, obs):
        if self.targets is None:
            self.targets = self.plan(obs["slots"])
        phase, target, dwell = self.targets[self.stage]
        self.phase = phase
        place = "place_in" if self.destination == "bowl" else "place_at"
        self.command = (f"pick({self.piece_id})" if self.stage < 4 else
                        f"{place}({self.piece_id},{self.destination})" if self.stage < 6 else
                        "release" if self.stage == 6 else "home")
        # The runner supplies commanded servo q; policy targets are nominal and
        # receive compensation exactly once, after temporal ensembling.
        current = np.asarray(obs["q"]).copy()
        current -= self.comp.offset(current)
        arm_close = np.max(np.abs(current - target)) < np.deg2rad(1)
        self.dwell = self.dwell + 1 if arm_close else 0
        if self.dwell >= dwell and self.stage < len(self.targets)-1:
            self.stage += 1
            self.dwell = 0
        chunk = []
        for _ in range(15):
            geo = self.arm.geometry(current)
            speed = self.cfg["guard"]["speed_deg_s"]
            if geo["finger"] < self.cfg["guard"]["table_z"] + self.cfg["guard"]["slow_zone_m"] + .015:
                speed = min(speed, self.cfg["guard"]["slow_speed_deg_s"])
            step = np.deg2rad(speed * self.speed * .65) / 30
            current = current + np.clip(target-current, -step, step)
            chunk.append(current.copy())
        return np.asarray(chunk)


def make_world(task, arm, seed=0):
    if importlib.util.find_spec("r2p.synthbench") is None:
        return TruthWorld(task, arm, seed), None, "mock_truth"
    from r2p.synthbench import World
    from r2p.perception import CameraModel
    cam = CameraModel.look_at(image_size=(640, 360))
    world = World(task, np.random.default_rng(seed), cam)
    return SynthAdapter(world, task, seed), SynthCameraAdapter(world), "synthbench_truth"


class SynthAdapter:
    def __init__(self, world, task, seed):
        self.world, self.task, self.seed = world, task, seed
        self.scene = getattr(world, "scene", None)
        self.previous_gap = float("inf")

    def pad_contact(self, q):
        a, b, _ = self.world.arm.tips(q)
        gap = float(np.linalg.norm(b-a))
        closing = gap < self.previous_gap
        self.previous_gap = gap
        if not closing or gap < 1e-8:
            return False
        axis = (b-a)/gap
        poses = self.world.poses()
        for obj in self.world.objects:
            if obj.get("fixed") or obj["role"] != "manipulated":
                continue
            from r2p.tasks import slot_size
            size = slot_size(obj["shape"], obj["size_m"], obj.get("axis", "z"))
            width = min(size[:2])
            p = np.asarray(poses[obj["id"]]["xyz"])
            along = float(np.dot(p-a, axis))
            lateral = p - (a + along*axis)
            if width-.006 <= gap <= width+.003 and 0 <= along <= gap and np.linalg.norm(lateral[:2]) < width/2 and abs(lateral[2]) < size[2]/2:
                return True
        return False

    def update(self, q, robot):
        command = np.asarray(q).copy()
        # C2's rendering fixture has a broad proximity latch. Admit that latch
        # only during measured closing pad contact, never on a servo goal alone.
        if self.world.held is None:
            self.world._closed = not self.pad_contact(command)
        self.world.step(command)
        held = getattr(self.world, "held", None)
        # The world owns contact and attachment; the servo shares its jaw floor.
        floor = self.world.q()[5] if held else None
        robot.set_grasp(floor if held else None)
        if held:
            robot.q[5] = max(robot.q[5], floor)

    def poses(self):
        return self.world.poses()

    def truth_check(self):
        return self.world.truth_check()


class SynthCameraAdapter:
    def __init__(self, world):
        self.world = world
        self.stride = 1
        self.ticks = 0

    def read(self):
        self.ticks += 1
        return self.world.render() if (self.ticks-1) % self.stride == 0 else None

    def close(self):
        if hasattr(self.world, "close"):
            self.world.close()
