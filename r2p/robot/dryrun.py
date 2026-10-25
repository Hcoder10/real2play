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
    """Absolute-target cube-in-bowl stub, with measured-observation feedback."""
    def __init__(self, task, arm):
        if task["name"] != "cube_in_bowl":
            raise ValueError("stub implements cube_in_bowl; use an exported policy for other tasks")
        self.task, self.arm = task, arm
        self.start_q = np.clip(HOME_Q, arm.range[:, 0]+np.deg2rad(2), arm.range[:, 1]-np.deg2rad(2))
        self.config = {"name": "stub", "task": task["name"], "start_q": self.start_q.tolist(),
                       "horizon": 15, "action": "absolute_q", "obs_const": {}, "eval": {"n": 0}}
        self.targets = None
        self.stage = 0
        self.dwell = 0
        self.done = False
        self.phase = "approach"
        self.command = "pick(cube)"

    def act(self, obs):
        if self.targets is None:
            ids = [o["id"] for o in self.task["objects"]]
            cube = np.asarray(obs["slots"][ids.index("cube"), :3])
            bowl = np.asarray(obs["slots"][ids.index("bowl"), :3])
            # Compute the grasp pitch pose at the object-width jaw opening.
            grasp_q = solve_ik(self.arm, cube + [0, 0, 0.006], self.start_q, grip=0.35)
            grasp_close = grasp_q.copy()
            grasp_close[5] = -0.12
            grasp_open = grasp_q.copy()
            grasp_open[5] = 0.9
            points = [("approach", [*cube[:2], .12], .9, 15),
                      ("descend", None, .9, 30), ("close", None, -.12, 230),
                      ("lift", [*cube[:2], .14], -.12, 15),
                      ("transport", [*bowl[:2], .14], -.12, 15),
                      ("lower", [*bowl[:2], .065], -.12, 15),
                      ("release", [*bowl[:2], .065], .9, 180),
                      ("retreat", [*bowl[:2], .14], .9, 15)]
            self.targets = []
            seed = self.start_q
            for phase, xyz, grip, dwell in points:
                if phase == "descend":
                    target = grasp_open
                elif phase == "close":
                    target = grasp_close
                else:
                    target = solve_ik(self.arm, xyz, seed, grip=0.35 if grip < 0 else grip,
                                      fingers_down=phase in ("lower", "release"))
                    target[5] = grip
                self.targets.append((phase, target.copy(), dwell))
                seed = target
        phase, target, dwell = self.targets[self.stage]
        self.phase = phase
        self.command = "pick(cube)" if self.stage < 4 else "place_in(cube,bowl)" if self.stage < 6 else "release" if self.stage == 6 else "home"
        arm_close = np.max(np.abs(np.asarray(obs["q"])[:5] - target[:5])) < np.deg2rad(4)
        self.dwell = self.dwell + 1 if arm_close else 0
        if self.dwell >= dwell and self.stage < len(self.targets)-1:
            self.stage += 1
            self.dwell = 0
        return np.tile(target, (15, 1))


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

    def update(self, q, robot):
        command = np.asarray(q).copy()
        command[5] = robot.target[5]
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

    def read(self):
        return self.world.render()

    def close(self):
        if hasattr(self.world, "close"):
            self.world.close()
