"""Task-derived experiment proposals and complete, conservative motion validation.

An experiment names a skeleton step, never an arbitrary contact point. Scene
coordinates are resolved anew after each survey. Importing this module does no IO.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from r2p import gemini
from r2p.frames import HZ
from r2p.sim.twin import extents
from r2p.tasks import load_task, validate
from .active import Plan, Safety, UnsafeExperiment, Validator as BenchValidator, _solve, start_q


@dataclass(frozen=True)
class TaskExperiment:
    kind: str
    step: int
    location: str = "target"
    speed: float = 8.0
    duration_s: float = 1.0
    height_m: float = .09
    amplitude_deg: float = 4.0
    joint: int | None = None
    offset_xy: tuple = (0., 0.)

    @classmethod
    def parse(cls, value):
        errors = gemini.schema_errors(value, EXPERIMENT_SCHEMA)
        if errors:
            raise UnsafeExperiment("; ".join(errors))
        result = cls(**value)
        if not all(math.isfinite(getattr(result, k)) for k in
                   ("speed", "duration_s", "height_m", "amplitude_deg")):
            raise UnsafeExperiment("Experiment numbers must be finite")
        return result

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


EXPERIMENT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["kind", "step"],
    "properties": {
        "kind": {"type": "string", "enum": ["hold_pose", "joint_sweep", "pick_lift_place", "push"]},
        "step": {"type": "integer", "minimum": 0},
        "location": {"type": "string", "enum": ["source", "target", "return"]},
        "joint": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
        "offset_xy": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
        **{k: {"type": "number", "minimum": lo, "maximum": hi} for k, lo, hi in (
            ("speed", 1, 12), ("duration_s", .5, 4), ("height_m", .06, .14), ("amplitude_deg", 1, 6))},
    },
}


def json_experiment(e):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(e).items() if v is not None}


def freeze_task(task, scene):
    task = load_task(task)
    errors = validate(task)
    if errors:
        raise ValueError("Invalid task: " + "; ".join(errors))
    if scene.get("task") != task["name"]:
        raise ValueError("Task and surveyed scene disagree")
    if scene.get("board"):
        task["board"] = deepcopy(scene["board"])
    elif task.get("board"):
        cells = {o["id"]: o["xyz"][:2] for o in scene["objects"] if o["id"] in task["board"]["cells"]}
        if len(cells) != len(task["board"]["cells"]):
            raise UnsafeExperiment("Survey is missing protected board cells")
        task["board"]["cells"] = cells
    return task


def motion_steps(task):
    placements = {s.get("obj") for s in task["skeleton"] if s["skill"].startswith("place_")}
    return [(i, s) for i, s in enumerate(task["skeleton"])
            if s["skill"] in ("place_in", "place_on", "place_at", "push_to")
            or s["skill"] == "pick" and s["obj"] not in placements]


def resolve(task, scene, e):
    """Constrain proposals to a task skill and its surveyed objects/target."""
    TaskExperiment.parse(json_experiment(e))
    steps = dict(motion_steps(task))
    if e.step not in steps:
        raise UnsafeExperiment("Experiment must reference a task motion step")
    step = steps[e.step]
    objects = {o["id"]: o for o in scene["objects"]}
    obj = objects.get(step["obj"])
    if not obj or obj.get("fixed") or obj.get("role") != "manipulated":
        raise UnsafeExperiment("Manipulated object absent or fixed")
    ref = step.get("target", step.get("xy"))
    target = objects.get(ref) if isinstance(ref, str) else None
    if isinstance(ref, str) and target is None:
        raise UnsafeExperiment("Task target absent from survey")
    xy = np.array(target["xyz"][:2] if target else ref if ref is not None else obj["xyz"][:2], float)
    if e.location in ("source", "return"):
        xy = np.array(obj["xyz"][:2], float)
    offset = np.asarray(e.offset_xy, float)
    if not np.isfinite(offset).all():
        raise UnsafeExperiment("Nonfinite offset")
    # Only a bowl task may deliberately probe near a wall. The object stays
    # inside the opening; the hand releases above the rim, never inside it.
    max_offset = 0.
    if step["skill"] == "place_in" and e.location == "target" and target and target["shape"] == "bowl":
        max_offset = max(0., target["size_m"][0]/2 - np.linalg.norm(extents(obj)[:2])/2 - .004)
    if np.linalg.norm(offset) > max_offset + 1e-8:
        raise UnsafeExperiment("Offset leaves the task target's safe interior")
    if e.kind == "push" and step["skill"] != "push_to":
        raise UnsafeExperiment("Push not required by this task skill")
    if e.kind == "pick_lift_place" and step["skill"] == "push_to":
        raise UnsafeExperiment("Pick/place not required by this task skill")
    if e.kind == "push" and e.location != "target":
        raise UnsafeExperiment("Push must go toward the task target")
    if e.kind == "joint_sweep" and (type(e.joint) is not int or not 0 <= e.joint < 5):
        raise UnsafeExperiment("A sweep needs a non-gripper joint")
    return step, obj, target, xy + offset


def experiment_menu(task, scene):
    """Task/environment-derived fallback; no task-name cases or generic pushes."""
    task = freeze_task(task, scene)
    result = []
    arm = BenchValidator(task).arm
    for i, step in motion_steps(task):
        e = TaskExperiment("hold_pose", i, "source")
        _, obj, target, destination = resolve(task, scene, TaskExperiment("hold_pose", i))
        result.extend([e, TaskExperiment("hold_pose", i)])
        # Sweep the joint with the largest change on this task's actual path.
        source_q = _solve(arm, np.r_[obj["xyz"][:2], .09], start_q(), .5)
        target_q = _solve(arm, np.r_[destination, .09], source_q, .5)
        joint = int(np.argmax(abs(target_q[:4] - source_q[:4])))
        result.append(TaskExperiment("joint_sweep", i, "source", joint=joint))
        if step["skill"] == "push_to":
            result.append(TaskExperiment("push", i))
        else:
            result.append(TaskExperiment("pick_lift_place", i))
            if step["skill"] == "place_in" and target and target["shape"] == "bowl":
                radius = max(0., target["size_m"][0]/2 - np.linalg.norm(extents(obj)[:2])/2 - .004)
                result.append(TaskExperiment("pick_lift_place", i, offset_xy=(0., radius)))
            result.append(TaskExperiment("pick_lift_place", i, "return"))
    return list({e.key: e for e in result}.values())


def excited_parameters(task, e):
    """Parameters eligible to fit, not a claim that each is identifiable."""
    step = task["skeleton"][e.step]
    if e.kind == "hold_pose":
        return ["stiffness_scale"]
    if e.kind == "joint_sweep":
        return ["stiffness_scale", "damping_scale"]
    keys = ["stiffness_scale", "damping_scale", f"object:{step['obj']}:mass_kg", f"object:{step['obj']}:friction"]
    if e.kind == "push":
        keys.append("table_friction")
    else:
        keys.append("contact_timeconst")
        if step["skill"] in ("place_in", "place_on") and e.location == "target":
            keys.append(f"object:{step['target']}:friction")
            definition = next(o for o in task["objects"] if o["id"] == step["target"])
            if not definition.get("fixed"):
                keys.append(f"object:{step['target']}:mass_kg")
    return keys


class Validator(BenchValidator):
    """Validate every absolute joint target, including low Cartesian segments."""
    def __init__(self, task, safety=None):
        super().__init__(load_task(task), safety or Safety(reach=(.127, .32)))

    def plan(self, experiment, scene, q0=None):
        e = experiment if isinstance(experiment, TaskExperiment) else TaskExperiment.parse(experiment)
        self.task = freeze_task(self.task, scene)
        step, obj, target, dest = resolve(self.task, scene, e)
        s = self.safety
        q = np.asarray(start_q() if q0 is None else q0, float).copy()
        if q.shape != (6,) or not np.isfinite(q).all() or e.speed > s.speed_cap:
            raise UnsafeExperiment("Invalid joints or excessive speed")
        qstart = q.copy()
        targets, phases, points = [], [], []
        source = np.asarray(obj["xyz"], float)
        self._allowed = [source[:2], dest] if e.kind == "pick_lift_place" else []
        travel = max(s.safe_z, e.height_m, max(o["xyz"][2]+extents(o)[2]/2+.035
                     for o in scene["objects"] if o.get("collide", True)))
        if travel > s.max_z:
            raise UnsafeExperiment("Objects exceed safe travel envelope")

        def joint(goal, phase, dwell=0.):
            nonlocal q
            n = max(1, math.ceil(np.max(abs(goal-q))/np.deg2rad(min(e.speed, s.trajectory_speed_cap))*HZ))
            for a in np.linspace(0, 1, n+1)[1:]:
                targets.append(q+a*(goal-q)); phases.append(phase)
            q = goal.copy()
            for _ in range(round(dwell*HZ)):
                targets.append(q.copy()); phases.append(phase)

        def move(p, phase, grip=None, dwell=0.):
            p = np.asarray(p, float)
            current = self.arm.tips(q)[2]
            # Dense Cartesian interpolation keeps every near-table approach
            # vertical; the small joint interpolation between samples is checked.
            n = max(1, math.ceil(np.linalg.norm(p-current)/.004))
            for a in np.linspace(0, 1, n+1)[1:]:
                point = current+a*(p-current)
                self._point(point)
                goal = _solve(self.arm, point, q, q[5] if grip is None else grip, margin_deg=s.joint_margin_deg)
                if np.linalg.norm(self.arm.tips(goal)[2]-point) > .003:
                    raise UnsafeExperiment("IK residual exceeds 3 mm")
                joint(goal, phase)
            points.append(p.tolist())
            joint(q, phase, dwell)

        initial = self.arm.tips(q)[2]
        self._point(initial)
        move([*initial[:2], max(initial[2], travel)], "raise")
        if e.kind in ("hold_pose", "joint_sweep"):
            move([*dest, travel], "approach")
            move([*dest, e.height_m], "hold", dwell=e.duration_s)
            if e.kind == "joint_sweep":
                center = q.copy()
                for sign in (1, -1, 0):
                    goal = center.copy(); goal[e.joint] += sign*np.deg2rad(e.amplitude_deg)
                    joint(goal, "sweep", e.duration_s/3)
        elif e.kind == "push":
            delta = dest-source[:2]
            distance = np.linalg.norm(delta)
            if not .005 <= distance <= .12:
                raise UnsafeExperiment("Task push distance must be 5..120 mm")
            direction = delta/distance
            radius = np.linalg.norm(extents(obj)[:2])/2
            behind = source[:2]-direction*(radius+s.push_standoff)
            z = max(s.push_contact_z, source[2])
            if z > source[2]+extents(obj)[2]/2-.002:
                raise UnsafeExperiment("Object too short for sag-safe push contact")
            move([*behind, travel], "approach", s.closed_grip)
            move([*behind, z], "lower")
            move([*(dest-direction*radius), z], "push", dwell=e.duration_s)
            end = self.arm.tips(q)[2]
            move([*end[:2], travel], "retreat")
        else:
            dims = extents(obj)
            if min(dims[:2]) > .055 or obj.get("grasp", "top") == "side":
                raise UnsafeExperiment("Object needs a separately calibrated grasp envelope")
            # Commanded jaw floor includes bench sag; no 8 mm contact commands.
            grasp = max(s.grasp_z, source[2]-dims[2]/2+s.grasp_z)
            if grasp > source[2]+dims[2]/2:
                raise UnsafeExperiment("Object too short for sag-safe grasp")
            release_z = s.place_z
            if target and e.location == "target" and target.get("collide", True):
                release_z += target["xyz"][2]+extents(target)[2]/2
                if step["skill"] == "place_in":
                    release_z += .01  # entire fingertip remains above the rim
            move([*source[:2], travel], "approach", s.open_grip)
            move([*source[:2], grasp], "lower")
            move([*source[:2], grasp], "close", s.closed_grip, .5)
            move([*source[:2], travel], "lift", dwell=.4)
            move([*dest, travel], "carry")
            move([*dest, release_z], "place")
            move([*dest, release_z], "release", s.open_grip, .5)
            move([*dest, travel], "retreat", dwell=e.duration_s)
        end = self.arm.tips(q)[2]
        move([*end[:2], max(travel, end[2])], "safe_finish", dwell=.5)
        if len(targets)/HZ > s.max_duration_s:
            raise UnsafeExperiment("Experiment exceeds duration cap")
        limits = self.arm.range + np.deg2rad([s.joint_margin_deg, -s.joint_margin_deg])
        for index, command in enumerate([qstart, *targets]):
            if np.any(command <= limits[:, 0]) or np.any(command >= limits[:, 1]):
                raise UnsafeExperiment("Joint limit")
            p = self.arm.tips(command)[2]
            self._point(p)
            geo = self.arm.geometry(command)
            if any(geo[k] < v for k, v in s.clearance.items()):
                raise UnsafeExperiment(f"Mesh/table clearance: {geo}")
            if s.joint_map is not None and s.owner_lo is not None:
                owner = s.joint_map.to_owner(command)
                if np.any(owner < s.owner_lo) or np.any(owner > s.owner_hi):
                    raise UnsafeExperiment("Calibrated owner joint limit")
            phase = "start" if index == 0 else phases[index-1]
            if p[2] < s.grid_free_z and phase in ("raise", "lower", "lift", "place", "retreat"):
                # Check verticality against the previous measured kinematic point.
                if index > 1:
                    previous = self.arm.tips(targets[index-2])[2]
                    if abs(p[2]-previous[2]) > 1e-6 and np.linalg.norm(p[:2]-previous[:2]) > .001:
                        raise UnsafeExperiment("Near-table approach is not vertical")
            for other in scene["objects"]:
                if other["id"] == obj["id"] or not other.get("collide", True):
                    continue
                top = other["xyz"][2]+extents(other)[2]/2
                is_destination = target and other["id"] == target["id"] and e.location == "target"
                # Target exemptions apply only to the planned vertical release,
                # above its top surface. Other targets remain obstacles.
                if is_destination and phase in ("place", "release", "retreat", "safe_finish") and p[2] >= top+s.min_z:
                    continue
                radius = np.linalg.norm(extents(other)[:2])/2+.025
                if p[2] < top+.025 and np.linalg.norm(p[:2]-np.asarray(other["xyz"][:2])) < radius:
                    raise UnsafeExperiment("Path intersects another object/target")
        return Plan(e, qstart, np.asarray(targets), phases, points)


def choose_experiment(client, task, scene, parameters, uncertainty, history, tried, validator, q0, *, attempts=2, wrist_parts=()):
    """MAIN proposes bounded new experiments; validation refusals go back to MAIN."""
    menu = experiment_menu(task, scene)
    available, refusals = [], []
    for e in menu:
        if e.key in tried:
            continue
        verdict = validator.verdict(e, scene, q0)
        if verdict["safe"]:
            available.append(e)
        else:
            refusals.append({"experiment": json_experiment(e), "reason": verdict["reason"]})
    tried_kinds = {e.kind for e in menu if e.key in tried}
    # Separate unloaded servo compliance/tracking before fitting contact data.
    preferred = "hold_pose" if "hold_pose" not in tried_kinds else "joint_sweep" if "joint_sweep" not in tried_kinds else None
    available.sort(key=lambda e: (e.kind != preferred if preferred else e.kind in ("hold_pose", "joint_sweep"),
                                 -sum(not uncertainty.get(k, {}).get("identified", False)
                                      for k in excited_parameters(task, e))))
    prompt = {"instruction": "Design one informative experiment for THIS task and surveyed environment. "
              "You may vary the typed numeric fields beyond the fallback menu. No unrelated motions. "
              "Only training residuals are supplied. Return one experiment JSON.",
              "task": task, "scene": scene, "parameters": parameters, "uncertainty": uncertainty,
              "training_residual_history": history, "fallback_menu": [json_experiment(e) for e in available],
              "safety_rules": {"frame": "spec metres, absolute MJCF radians", "vertical_near_table": True,
                               "minimum_contact_reach_m": validator.safety.reach[0],
                               "commanded_grasp_floor_m": validator.safety.grasp_z,
                               "commanded_push_floor_m": validator.safety.push_contact_z,
                               "measured_abort_floor_m": validator.safety.live_min_jaw_z,
                               "protected_targets": "Only own vertical pick/place spots exempt",
                               "sheet_guard": "survey before and compare after EVERY experiment"},
              "refusals": refusals}
    records = []
    for _ in range(attempts):
        try:
            answer, record = client.call([json.dumps(prompt), *wrist_parts], role="main", thinking="high",
                                         tag="general_experiment_design", schema=EXPERIMENT_SCHEMA)
            records.append(record)
            e = TaskExperiment.parse(gemini.parse_json(answer))
            if e.key in tried:
                raise UnsafeExperiment("Experiment already tried or frozen as held-out")
            validator.plan(e, scene, q0)
            return e, {"source": "main", "records": records, "refusals": prompt["refusals"]}
        except (gemini.JSONError, ValueError, TypeError, KeyError) as exc:
            prompt["refusals"].append({"reason": str(exc)})
    return (available[0] if available else None), {"source": "task_derived_fallback", "records": records,
                                                 "refusals": prompt["refusals"]}


def offline_client():
    def respond(**request):
        prompt = json.loads(request["parts"][0])
        return prompt["fallback_menu"][0] if prompt["fallback_menu"] else {"kind": "hold_pose", "step": 999}
    return gemini.FakeGemini(responder=respond)
