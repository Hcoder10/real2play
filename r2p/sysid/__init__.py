"""Plan recorded experiments, fit reference physics, and gate on held-out attempts.

Planning never moves hardware. Physics files are returned/written to an explicit
destination only after the held-out gate passes; an incumbent is never overwritten.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Parameter:
    path: str
    low: float
    high: float


def _get(config, path):
    value = config
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return float(value)


def _set(config, path, value):
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    key = int(keys[-1]) if isinstance(target, list) else keys[-1]
    target[key] = float(value)


def plan_experiments(residuals, *, q0=None, joint_ranges=None, duration_s=6, amplitude=0.12, count=3):
    """Target the worst normalized residual; emit absolute joint excitation or resettable object probes."""
    from r2p.frames import HOME_Q

    if duration_s <= 0 or not 0 < amplitude <= 0.3 or count < 1:
        raise ValueError("Invalid excitation duration/amplitude/count")
    q0 = np.asarray(HOME_Q if q0 is None else q0, dtype=float)
    if q0.shape != (6,) or not np.isfinite(q0).all():
        raise ValueError("q0 must be six finite absolute joint angles")
    if joint_ranges is None:
        from r2p.model import Arm
        joint_ranges = Arm().range
    ranges = np.asarray(joint_ranges, dtype=float)
    if ranges.shape != (6, 2) or np.any(q0 < ranges[:, 0]) or np.any(q0 > ranges[:, 1]):
        raise ValueError("q0 must lie within six calibrated joint ranges")
    candidates = []
    for key, row in residuals.items():
        row = row if isinstance(row, dict) else {"value": row}
        kind = row.get("kind", "grasp" if "grasp" in key else "push" if "object" in key or "friction" in key else "excitation")
        value = float(row.get("value", 0))
        scale = float(row.get("tolerance", 1))
        if not np.isfinite([value, scale]).all() or scale <= 0:
            raise ValueError("Residual values must be finite and tolerance positive")
        candidates.append((abs(value) / scale, key, kind, row))
    if not candidates:
        candidates = [(1, "servo", "excitation", {})]
    plans = []
    for rank, key, kind, row in sorted(candidates, reverse=True, key=lambda item: item[0])[:count]:
        plan = {"kind": kind, "residual": key, "priority": rank, "reason": f"Reduce {key}, {rank:.2f} times tolerance",
                "execute": False, "requires": ["operator-approved workspace", "load/velocity guards", "reset between attempts"],
                "split": "train; repeat independently for heldout"}
        if kind == "excitation":
            joint = int(row.get("joint", 1))
            if not 0 <= joint < 6:
                raise ValueError("Excitation joint index must be in 0..5")
            t = np.arange(int(duration_s * 30) + 1) / 30
            available = min(amplitude, q0[joint] - ranges[joint, 0], ranges[joint, 1] - q0[joint])
            # Smooth onset/offset, increasing frequency, positive and negative motion for friction identifiability.
            signal = np.sin(np.pi * t / duration_s) ** 2 * np.sin(2 * np.pi * (0.2 * t + 0.08 * t ** 2))
            q = np.tile(q0, (len(t), 1))
            q[:, joint] += available * signal
            plan.update(hz=30, joint=joint, q_target=q.tolist(), t=t.tolist(), parameters=["kp", "kv", "sag_gain", "latency_ticks"])
        elif kind in ("push", "grasp", "drop"):
            plan.update(object=row.get("object"), start_xyz=row.get("xyz"), distance_m=0.025,
                        repetitions=5, parameters=["object.friction", "object.mass_kg"] if kind == "push" else ["contact", "gripper.kp"],
                        reset="return object to measured initial pose before every trial")
        else:
            raise ValueError(f"Unknown experiment kind: {kind}")
        plans.append(plan)
    return plans


def default_parameters(config):
    from r2p.frames import JOINTS

    params = []
    for joint in JOINTS:
        params.extend([Parameter(f"joints.{joint}.kp", 0.5, 150), Parameter(f"joints.{joint}.kv", 0.01, 8),
                       Parameter(f"joints.{joint}.frictionloss", 0, 0.15)])
    # TaskTwin's sag comes from finite kp. sag_gain is the runner's compensation
    # model and is fitted separately from static gravity residuals below.
    params.append(Parameter("object.friction", 0.01, 2))
    return params


def _attempt(value):
    if isinstance(value, (str, Path)):
        from r2p.traces import read_trace
        meta, ticks = read_trace(value)
        value = {"meta": meta, "ticks": ticks, "id": str(Path(value).resolve())}
    if "ticks" not in value or not value["ticks"]:
        raise ValueError("Every identification attempt needs recorded ticks")
    result = dict(value)
    meta = result.get("meta", {})
    for key in ("grasp_outcome", "latency_ticks", "end_poses", "jaw_xyz"):
        if key not in result and key in meta:
            result[key] = meta[key]
    if "jaw_xyz" not in result and all("jaw_xyz" in row for row in result["ticks"]):
        result["jaw_xyz"] = [row["jaw_xyz"] for row in result["ticks"]]
    return result


def fit_sag(attempts, *, arm=None, initial=None, joints=(1, 2, 3), settle_s=0.4):
    """Fit q_target-q = k * gravity_torque on measured static holds, in MJCF radians."""
    if arm is None:
        from r2p.model import Arm
        arm = Arm()
    gains = np.broadcast_to(0.05 if initial is None else initial, (6,)).astype(float).copy()
    torque_rows, residual_rows = [], []
    for raw in attempts:
        rows = _attempt(raw)["ticks"]
        q = np.asarray([row["q"] for row in rows], float)
        target = np.asarray([row["q_target"] for row in rows], float)
        times = np.asarray([row["t"] for row in rows], float)
        for i in range(1, len(rows)):
            first = int(np.searchsorted(times, times[i] - settle_s, side="left"))
            if times[i] - times[first] < settle_s - 1e-6:
                continue
            if np.max(np.ptp(target[first:i+1, :5], axis=0)) > 0.004:
                continue
            if np.max(np.ptp(q[first:i+1, :5], axis=0)) > 0.004:
                continue
            torque_rows.append(np.asarray(arm.gravity_torque(q[i]), float))
            residual_rows.append(target[i] - q[i])
    measured = {}
    if torque_rows:
        torque, residual = np.asarray(torque_rows), np.asarray(residual_rows)
        for joint in joints:
            x, y = torque[:, joint], residual[:, joint]
            valid = abs(x) > 0.005
            if np.count_nonzero(valid) >= 3 and np.dot(x[valid], x[valid]) > 1e-6:
                gain = float(np.dot(x[valid], y[valid]) / np.dot(x[valid], x[valid]))
                if 0 <= gain <= 0.3:
                    gains[joint] = gain
                    measured[str(joint)] = {"k": gain, "n": int(valid.sum()),
                                             "rms_rad": float(np.sqrt(np.mean((y[valid] - gain*x[valid])**2)))}
    return {"sag_gain": gains.tolist(), "joints": measured, "identified": all(str(j) in measured for j in joints),
            "source": "measured_static_gravity_residual"}


def _physical_grasp(twin):
    """Opposing jaw contact in the physics engine; proximity alone never counts."""
    if hasattr(twin, "grasped"):
        return bool(twin.grasped())
    if not hasattr(twin, "m") or not hasattr(twin, "body_ids") or not twin.grip_closed():
        return False
    try:
        fixed = twin.m.geom("fixed_finger").id
        moving = twin.m.geom("moving_finger").id
    except KeyError:
        return False
    contacts = {bid: set() for bid in twin.body_ids.values()}
    for contact in twin.d.contact[:twin.d.ncon]:
        for jaw, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
            bid = int(twin.m.geom_bodyid[other])
            if jaw in (fixed, moving) and bid in contacts and contact.dist <= 0:
                contacts[bid].add(int(jaw))
    return any(len(jaws) == 2 for jaws in contacts.values())


def replay_attempt(attempt, physics, *, twin_factory=None, arm=None):
    if twin_factory is None:
        from r2p.sim import TaskTwin
        twin_factory = TaskTwin
    if arm is None:
        from r2p.model import Arm
        arm = Arm()
    rows = attempt["ticks"]
    scene = attempt.get("scene", attempt.get("meta", {}).get("scene"))
    if scene is None:
        from r2p.tasks import load_task
        task_spec = attempt.get("meta", {}).get("task")
        if task_spec is None or not rows[0].get("poses"):
            raise ValueError("Identification needs a measured initial scene or task plus initial perception poses")
        task_spec = load_task(task_spec)
        scene = {"version": 0, "task": task_spec["name"], "objects": []}
        for definition in task_spec["objects"]:
            pose = rows[0]["poses"].get(definition["id"])
            if pose is None:
                if task_spec.get("randomize", {}).get(definition["id"], {}).get("p_present", 1) >= 1:
                    raise ValueError("Initial scene lacks required object " + definition["id"])
                continue
            scene["objects"].append({**copy.deepcopy(definition), **copy.deepcopy(pose)})
    scene = copy.deepcopy(scene)
    for obj in scene.get("objects", []):
        if not obj.get("fixed", False):
            for key in ("friction", "mass_kg"):
                if key in physics.get("object", {}):
                    obj[key] = physics["object"][key]
            obj.update(physics.get("objects", {}).get(obj["id"], {}))
    twin = twin_factory(scene, physics=physics, q0=rows[0]["q"])
    q, jaws, closed, grasp_samples = [], [], [], []
    try:
        for index, row in enumerate(rows):
            if index:
                dt = row["t"] - rows[index - 1]["t"]
                n = int(round(dt * 30))
                if n < 1 or abs(n / 30 - dt) > 0.005:
                    raise ValueError("Attempts must align to the 30 Hz control clock")
                for _ in range(n):
                    twin.step(rows[index - 1]["q_target"])
            actual = np.asarray(twin.q(), dtype=float)
            q.append(actual)
            jaws.append(arm.tips(actual)[2])
            closed.append(bool(twin.grip_closed()))
            grasp_samples.append(_physical_grasp(twin))
        poses = twin.poses()
        # Use the shared success checker for the grasped predicate, not distance-to-jaw heuristics.
        task = attempt.get("meta", {}).get("task")
        outcome = None
        if isinstance(task, dict):
            from r2p.tasks import check
            outcome = bool(check(task, {"objects": poses, "site_xyz": np.asarray(jaws[-1]).tolist(),
                                        "gripper_open": not closed[-1],
                                        "initial": {o["id"]: {"xyz": o["xyz"], "yaw_deg": o.get("yaw_deg", 0)}
                                                    for o in scene["objects"]}})["ok"])
        return {"q": np.asarray(q), "jaw_xyz": np.asarray(jaws), "end_poses": poses, "outcome": outcome,
                "grasp_outcome": any(all(grasp_samples[i:i+3]) for i in range(max(0, len(grasp_samples)-2))),
                "latency_ticks": physics.get("latency_ticks", 0)}
    finally:
        if hasattr(twin, "close"):
            twin.close()


def _residual(attempt, prediction):
    measured = np.asarray([row["q"] for row in attempt["ticks"]], dtype=float)
    residuals = [(np.asarray(prediction["q"]) - measured).ravel()]
    jaw = attempt.get("jaw_xyz")
    if jaw is not None and prediction.get("jaw_xyz") is not None:
        residuals.append(((np.asarray(prediction["jaw_xyz"]) - jaw) * 20).ravel())
    targets = attempt.get("end_poses", attempt["ticks"][-1].get("poses", {}))
    for oid, pose in targets.items():
        actual = prediction.get("end_poses", {}).get(oid)
        if actual is None:
            raise ValueError(f"Replay omitted measured object {oid}")
        residuals.append((np.asarray(actual["xyz"]) - pose["xyz"]) * 10)
    return np.concatenate(residuals)


def validation_gate(attempts, physics, *, replay=None, jaw_rms_mm=2, object_end_mm=5,
                    grasp_agreement=0.95, timing_ticks=1):
    replay = replay or replay_attempt
    jaw_errors, pose_errors, agreements, timing_errors = [], [], [], []
    missing = set()
    for raw in attempts:
        attempt = _attempt(raw)
        prediction = replay(attempt, physics)
        target_jaw = attempt.get("jaw_xyz")
        if target_jaw is None:
            from r2p.model import Arm
            arm = Arm()
            target_jaw = [arm.tips(row["q"])[2] for row in attempt["ticks"]]
        if prediction.get("jaw_xyz") is None:
            missing.add("jaw")
        else:
            jaw_errors.extend(np.linalg.norm(np.asarray(prediction["jaw_xyz"]) - target_jaw, axis=1) * 1000)
        targets = attempt.get("end_poses", attempt["ticks"][-1].get("poses", {}))
        if not targets:
            missing.add("object_end_pose")
        for oid, target in targets.items():
            pose = prediction.get("end_poses", {}).get(oid)
            if pose is None:
                missing.add("object_end_pose")
            else:
                pose_errors.append(float(np.linalg.norm(np.asarray(pose["xyz"]) - target["xyz"]) * 1000))
        outcome = attempt.get("grasp_outcome", attempt.get("outcome"))
        predicted = prediction.get("grasp_outcome", prediction.get("outcome"))
        if outcome is None or predicted is None:
            missing.add("grasp_outcome")
        else:
            agreements.append(bool(outcome) == bool(predicted))
        # Measured onset is required; a fitted latency value is not validation evidence.
        expected_latency = attempt.get("latency_ticks")
        if expected_latency is None:
            q = np.asarray([row["q"] for row in attempt["ticks"]])
            p = np.asarray(prediction["q"])
            real_onset = np.flatnonzero(np.linalg.norm(q - q[0], axis=1) > 0.01)
            sim_onset = np.flatnonzero(np.linalg.norm(p - p[0], axis=1) > 0.01)
            if len(real_onset) and len(sim_onset):
                timing_errors.append(abs(int(real_onset[0]) - int(sim_onset[0])))
            else:
                missing.add("timing")
        else:
            timing_errors.append(abs(prediction["latency_ticks"] - expected_latency))
    metrics = {"jaw_rms_mm": float(np.sqrt(np.mean(np.square(jaw_errors)))) if jaw_errors else None,
               "object_end_mm": max(pose_errors) if pose_errors else None,
               "grasp_agreement": float(np.mean(agreements)) if agreements else None,
               "timing_ticks": max(timing_errors) if timing_errors else None}
    failed = ["missing:" + item for item in sorted(missing)]
    for key, limit in (("jaw_rms_mm", jaw_rms_mm), ("object_end_mm", object_end_mm), ("timing_ticks", timing_ticks)):
        if metrics[key] is None or not np.isfinite(metrics[key]) or metrics[key] >= limit:
            failed.append(key)
    if metrics["grasp_agreement"] is None or metrics["grasp_agreement"] < grasp_agreement:
        failed.append("grasp_agreement")
    return {"passed": not failed, "metrics": metrics, "failures": failed, "n": len(attempts),
            "thresholds": {"jaw_rms_mm": jaw_rms_mm, "object_end_mm": object_end_mm,
                           "grasp_agreement": grasp_agreement, "timing_ticks": timing_ticks}}


def fit_parameters(attempts, physics, *, heldout, parameters=None, latency_candidates=range(4), replay=None,
                   max_nfev=100, output=None):
    """Least squares on training attempts, discrete latency search, untouched held-out gate."""
    from r2p.sim.twin import load_physics

    train, test = [_attempt(a) for a in attempts], [_attempt(a) for a in heldout]
    if not train or not test:
        raise ValueError("Separate training and held-out attempts are mandatory")
    # Same trace content under a different id is still leakage.
    from r2p.gemini import _canonical, atomic_json
    fingerprints = {_canonical(a["ticks"]) for a in train}
    if any(_canonical(a["ticks"]) in fingerprints for a in test):
        raise ValueError("Training and held-out attempts must be disjoint")
    base = load_physics(physics)
    params = list(default_parameters(base) if parameters is None else parameters)
    if not params:
        raise ValueError("No identifiable parameters requested")
    replay = replay or replay_attempt
    initial = np.array([_get(base, param.path) for param in params])
    lower, upper = np.array([p.low for p in params]), np.array([p.high for p in params])
    if np.any(initial < lower) or np.any(initial > upper) or np.any(lower >= upper):
        raise ValueError("Parameter initial values/bounds are inconsistent")
    best = None
    evaluations = 0
    for latency in latency_candidates:
        if not isinstance(latency, int) or latency < 0:
            raise ValueError("Latency candidates must be nonnegative integers")

        def unpack(values):
            config = copy.deepcopy(base)
            config["latency_ticks"] = latency
            for param, value in zip(params, values):
                _set(config, param.path, value)
            return config

        def residual(values):
            nonlocal evaluations
            evaluations += 1
            config = unpack(values)
            return np.concatenate([_residual(a, replay(a, config)) for a in train])

        fit = least_squares(residual, initial, bounds=(lower, upper), x_scale="jac", max_nfev=max_nfev,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        cost = float(np.mean(fit.fun ** 2))
        if best is None or cost < best[0]:
            best = cost, unpack(fit.x), fit
    if best is None:
        raise ValueError("At least one latency candidate is required")
    cost, candidate, fitted = best
    sag_fit = None
    if parameters is None:
        sag_fit = fit_sag(train, initial=base.get("sag_gain", 0.05))
        candidate["sag_gain"] = sag_fit["sag_gain"]
    gate = validation_gate(test, candidate, replay=replay)
    if sag_fit is not None and not sag_fit["identified"]:
        gate["passed"] = False
        gate["failures"].append("sag_needs_static_holds")
    rank = int(np.linalg.matrix_rank(fitted.jac))
    identifiable = rank == len(params)
    if not identifiable:
        gate["passed"] = False
        gate["failures"].append("unidentifiable_parameters")
    result = {"physics": candidate, "train_mse": cost, "gate": gate, "identifiable": identifiable,
              "jacobian_rank": rank, "parameters": [p.path for p in params], "evaluations": evaluations,
              "optimizer_success": bool(fitted.success), "train_n": len(train), "heldout_n": len(test), "sag_fit": sag_fit}
    if not fitted.success:
        gate["passed"] = False
        gate["failures"].append("optimizer_not_converged")
    if output is not None:
        if not gate["passed"]:
            raise ValueError("Physics gate failed; candidate was not published: " + ", ".join(gate["failures"]))
        path = Path(output)
        if path.exists():
            raise FileExistsError("Choose a new versioned physics output; the incumbent is preserved")
        atomic_json(path, {**candidate, "identification": {k: v for k, v in result.items() if k != "physics"}})
    return result


fit = fit_parameters
plan = plan_experiments
