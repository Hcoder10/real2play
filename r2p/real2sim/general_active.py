"""General task-conditioned active twin campaigns, isolated from a running bench.

Only training observations enter identification and MAIN. Complete held-out
experiments gate changes. Candidates stay inside the run directory, inactive.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from r2p import gemini
from r2p.frames import HZ, JOINTS
from r2p.perception.geometry import fit_pose
from r2p.sim import TaskTwin
from r2p.sim.twin import extents, load_physics
from r2p.tasks import check_window
from .active import SimMover as BenchSimMover, _save_observation, _trace_row, keep_heldout
from .evaluate import CalibratedRenderer, physics_replay, mask_metrics
from .experiments import (Validator, choose_experiment, excited_parameters, experiment_menu,
                          freeze_task, json_experiment, offline_client, resolve)
from .session import SessionData


WRIST_REVIEW_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["grasp_confirmed", "slip", "release_confirmed", "evidence"], "properties": {
        **{k: {"type": ["boolean", "null"]} for k in ("grasp_confirmed", "slip", "release_confirmed")},
        "evidence": {"type": "string"}}}


def parameter_space(task, scene, physics):
    values = {"stiffness_scale": 1., "damping_scale": 1.,
              "table_friction": float(scene.get("table", {}).get("friction", .6)),
              "contact_timeconst": float(physics["contact"]["solref"][0])}
    bounds = {"stiffness_scale": (.6, 1.8), "damping_scale": (.5, 1.8),
              "table_friction": (.1, 1.2), "contact_timeconst": (.006, .04)}
    for obj in scene["objects"]:
        for field, default in (("mass_kg", .03), ("friction", .6)):
            if field == "mass_kg" and obj.get("fixed"):
                continue
            key = f"object:{obj['id']}:{field}"
            value = float(obj.get(field, default))
            values[key] = value
            configured = task.get("randomize", {}).get(obj["id"], {}).get(field)
            limits = (max(.0005, value*.4), max(.001, value*2.5)) if field == "mass_kg" else (.1, 1.2)
            if isinstance(configured, list) and configured[0] < configured[1]:
                limits = (min(limits[0], configured[0]), max(limits[1], configured[1]))
            bounds[key] = limits
    return values, bounds


def materialize(physics, scene, values):
    physics, scene = deepcopy(physics), deepcopy(scene)
    for name in JOINTS:
        physics["joints"][name]["kp"] *= values["stiffness_scale"]
        physics["joints"][name]["kv"] *= values["damping_scale"]
    physics["sag_gain"] = (np.asarray(physics["sag_gain"])/values["stiffness_scale"]).tolist()
    physics["contact"]["solref"][0] = values["contact_timeconst"]
    scene.setdefault("table", {})["friction"] = values["table_friction"]
    for obj in scene["objects"]:
        for field in ("mass_kg", "friction"):
            key = f"object:{obj['id']}:{field}"
            if key in values:
                obj[field] = values[key]
    return physics, scene


def replay_scene(observation, candidate_scene):
    """Each re-survey owns its initial positions; fitting changes materials only."""
    result = deepcopy(observation.session.meta["scene"])
    candidate = {o["id"]: o for o in candidate_scene["objects"]}
    for obj in result["objects"]:
        for key in ("friction", "mass_kg"):
            if key in candidate.get(obj["id"], {}):
                obj[key] = candidate[obj["id"]][key]
    result["table"] = deepcopy(candidate_scene.get("table", result.get("table", {})))
    return result


def samples(observation, physics, scene):
    session = observation.session
    indices = sorted(set(range(0, len(session.ticks), 15)) | {len(session.ticks)-1} |
                     {i for i, row in enumerate(session.ticks) if row["poses"]})
    sampled = SessionData(session.path, session.meta, session.ticks, indices,
                          [session.ticks[i]["t"] for i in indices], session.camera)
    snapshots = physics_replay(sampled, replay_scene(observation, scene), physics=physics, render=False)
    return [(session.ticks[index], snapshots[i]) for i, index in enumerate(indices)]


def residual(observations, physics, scene):
    values = []
    for observation in observations:
        pairs = samples(observation, physics, scene)
        for measured, predicted in pairs:
            values.extend((np.rad2deg(np.array(predicted["q"])-measured["q"])/.1).tolist())
        for measured, predicted in pairs:
            for oid, pose in measured["poses"].items():
                if oid in predicted["poses"]:
                    values.extend(((np.asarray(predicted["poses"][oid]["xyz"])-pose["xyz"])/.005).tolist())
    return np.asarray(values)


def require_contact_evidence(observation):
    if observation.experiment.kind not in ("push", "pick_lift_place"):
        return
    session = observation.session
    task = session.meta.get("task")
    required = ({task["skeleton"][observation.experiment.step]["obj"]} if isinstance(task, dict) else
                {o["id"] for o in session.meta["scene"]["objects"] if o.get("role") == "manipulated"})
    final = session.ticks[session.tick_index(len(session.frames)-1)]["poses"]
    if not required.issubset(final):
        raise ValueError("Final contact observation is missing the manipulated object; cannot identify/gate contact")


def fit_parameters(training, task, physics, scene, current, *, max_nfev=5):
    if not training or any(o.split != "train" for o in training):
        raise ValueError("Fitting requires training experiments only")
    for observation in training:
        require_contact_evidence(observation)
    _, bounds = parameter_space(task, scene, physics)
    keys = sorted(set(k for o in training for k in excited_parameters(task, o.experiment)) & bounds.keys())
    uncertainty = {k: {"std": None, "identified": False, "excited": k in keys, "bounds": list(b)}
                   for k, b in bounds.items()}
    lower = np.array([bounds[k][0] for k in keys])
    scale = np.array([bounds[k][1]-bounds[k][0] for k in keys])
    x0 = np.clip((np.array([current[k] for k in keys])-lower)/scale, 0, 1)
    cache = {}

    def objective(x):
        key = tuple(x)
        if key not in cache:
            values = {**current, **dict(zip(keys, lower+x*scale))}
            p, s = materialize(physics, scene, values)
            cache[key] = residual(training, p, s)
        return cache[key]

    initial = objective(x0)
    probes = []
    for j in range(len(keys)):
        trial = x0.copy(); trial[j] += .025 if trial[j] <= .975 else -.025
        probes.append((objective(trial)-initial)/(trial[j]-x0[j]))
    jac = np.column_stack(probes)
    active = np.flatnonzero(np.linalg.norm(jac, axis=0) > 1e-5)
    fitted = dict(current)
    rank, final_cost = 0, float(initial@initial)
    if len(active):
        def reduced(x):
            full = x0.copy(); full[active] = x
            return objective(full)
        result = least_squares(reduced, np.clip(x0[active], .0001, .9999), bounds=(0, 1),
                               diff_step=.01, max_nfev=max_nfev)
        rank = int(np.linalg.matrix_rank(result.jac, tol=1e-5))
        covariance = np.linalg.pinv(result.jac.T@result.jac)*max(1., float(np.mean(result.fun**2)))
        for i, j in enumerate(active):
            fitted[keys[j]] = float(lower[j]+result.x[i]*scale[j])
            std = float(np.sqrt(max(0, covariance[i, i]))*scale[j])
            uncertainty[keys[j]].update(std=std if rank == len(active) else None,
                identified=bool(rank == len(active) and std < scale[j]/2))
        final_cost = float(result.fun@result.fun)
    return fitted, uncertainty, {"eligible": keys, "sensitive": [keys[j] for j in active],
                                "jacobian_rank": rank, "evaluations": len(cache),
                                "cost_before": float(initial@initial), "cost_after": final_cost}


def score(observations, physics, scene):
    from r2p.model import Arm
    arm, rows, ious = Arm(), [], {}
    for observation in observations:
        require_contact_evidence(observation)
        pairs = samples(observation, physics, scene)
        joint, jaw, sag, positions = [], [], [], []
        for measured, predicted in pairs:
            joint.extend(np.rad2deg(np.asarray(measured["q"])-predicted["q"])[:5])
            delta = (arm.tips(measured["q"])[2]-arm.tips(predicted["q"])[2])*1000
            jaw.append(np.linalg.norm(delta)); sag.append(delta[2])
            for oid, pose in measured["poses"].items():
                positions.append(np.linalg.norm(np.asarray(pose["xyz"])-predicted["poses"][oid]["xyz"])*1000)
        snapshots = physics_replay(observation.session, replay_scene(observation, scene), physics=physics)
        local = []
        for i, snapshot in snapshots.items():
            for oid, mask in observation.session.masks.get(i, {}).items():
                value = mask_metrics(mask, snapshot["masks"].get(oid, np.zeros_like(mask)))["iou"]
                ious.setdefault(oid, []).append(value); local.append(value)
        if not local or not positions:
            raise ValueError("Missing independent mask/pose evidence; cannot gate candidate")
        rows.append({"experiment": json_experiment(observation.experiment), "session": str(observation.session.path),
                     "joint_deg": float(np.sqrt(np.mean(np.square(joint)))),
                     "jaw_mm": float(np.sqrt(np.mean(np.square(jaw)))),
                     "sag_mm": float(np.sqrt(np.mean(np.square(sag)))),
                     "object_mm": float(np.mean(positions)), "iou": float(np.mean(local))})
    result = {k: float(np.mean([r[k] for r in rows])) for k in ("joint_deg", "jaw_mm", "sag_mm", "object_mm", "iou")}
    result.update(error=result["joint_deg"]/.5+result["jaw_mm"]/2+result["object_mm"]/5+2*(1-result["iou"]),
                  objects={oid: {"iou": float(np.mean(v))} for oid, v in ious.items()}, rows=rows)
    return result


class SimMover(BenchSimMover):
    def __init__(self, task, scene, *, truth_physics=None, truth_scene=None, **kwargs):
        self.task = freeze_task(task, scene)
        self.truth_physics = load_physics(truth_physics or "so101_bench")
        self.truth_scene = deepcopy(truth_scene or scene)
        super().__init__(scene, truth_physics=self.truth_physics, truth_scene=self.truth_scene,
                         validator=kwargs.pop("validator", None) or Validator(self.task), **kwargs)

    def execute(self, experiment, destination, *, split="train"):
        plan = self.validator.plan(experiment, self.scene, self.q0)
        twin = TaskTwin(self.truth_scene, self.truth_physics, plan.q0)
        renderer = CalibratedRenderer(twin, self.camera)
        indices = set(np.linspace(0, len(plan.targets), 7, dtype=int))
        ticks, images, masks, states = [], [], {}, []
        initial_poses = twin.poses()
        try:
            for i in range(len(plan.targets)+1):
                target = plan.targets[min(i, len(plan.targets)-1)]
                poses = {}
                if i in indices:
                    frame, observed = renderer.render()
                    if i in (0, len(plan.targets)):
                        # Infer support height from silhouette fit candidates, not
                        # hidden simulator coordinates or an assumed successful grasp.
                        for obj in self.scene["objects"]:
                            mask = observed.get(obj["id"])
                            if mask is None or mask.sum() < 8 or obj.get("fixed"):
                                continue
                            bottoms = [0.]
                            if i and obj.get("role") == "manipulated":
                                for support in self.scene["objects"]:
                                    if support["id"] != obj["id"] and support.get("collide", True):
                                        bottoms.append(support.get("wall_m", .005) if support["shape"] == "bowl"
                                                       else support["xyz"][2]+extents(support)[2]/2)
                            fits = [fit_pose(self.camera, obj, mask, bottom=b) for b in bottoms]
                            pose = min(fits, key=lambda p: p["fit_px"])
                            if pose["fit_px"] < 6:
                                poses[obj["id"]] = {**pose, "seen": True, "source": "rendered_mask_fit"}
                    masks[len(images)] = observed; images.append((i/HZ, frame))
                ticks.append(_trace_row(i/HZ, twin.q(), target, poses, plan.phases[min(i, len(plan.phases)-1)]))
                if i >= len(plan.targets)-HZ:
                    states.append({"objects": twin.poses(), "gripper_open": not twin.grip_closed(),
                                   "site_xyz": self.validator.arm.tips(twin.q())[2].tolist(), "initial": initial_poses})
                if i < len(plan.targets):
                    twin.step(target)
            outcome = check_window(self.task, states)
        finally:
            renderer.close(); twin.close()
        return _save_observation(destination, experiment, plan, self.scene, self.camera, ticks, images, masks,
            source="bot", split=split, extra={"task": self.task, "task_name": self.task["name"],
                "observations": "joint sensors and rendered segmentation fits; sparse synthetic camera",
                "synthetic_task_check": outcome})


def run_campaign(mover, task, *, out_dir, budget=6, physics="so101_bench", client=None, max_nfev=5, videos=True):
    if type(budget) is not int or not 4 <= budget <= 30:
        raise ValueError("budget must be 4..30, including two complete held-out experiments")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    task = freeze_task(task, mover.scene)
    physics, scene = load_physics(physics), deepcopy(mover.scene)
    values, bounds = parameter_space(task, scene, physics)
    uncertainty = {k: {"identified": False, "std": None, "bounds": list(v)} for k, v in bounds.items()}
    client = client or offline_client()
    status = {"schema": "r2p_general_active/1", "task": task["name"], "synthetic": isinstance(mover, SimMover),
              "experiments_used": 0, "budget": budget, "iterations": [], "candidate_activated": False}
    gemini.atomic_json(out/"task.json", task)
    gemini.atomic_json(out/"prior_physics.json", physics)
    gemini.atomic_json(out/"prior_scene.json", scene)
    def publish(stage):
        status.update(status=stage, parameters=values, uncertainty=uncertainty)
        gemini.atomic_json(out/"status.json", status)
    try:
        publish("planning")
        menu = experiment_menu(task, scene)
        q0 = mover.q0 if hasattr(mover, "q0") else mover.measured_q(mover.request(mover.url+"/state"))
        verdicts = [{"experiment": json_experiment(e), **mover.validator.verdict(e, scene, q0)} for e in menu]
        gemini.atomic_json(out/"menu.json", verdicts)
        safe = [e for e, v in zip(menu, verdicts) if v["safe"]]
        holds = [e for e in safe if e.kind == "hold_pose"]
        contacts = [e for e in safe if e.kind in ("push", "pick_lift_place") and e.location == "target"]
        if not holds or not contacts:
            raise ValueError("No safe hold AND task-contact experiment; cannot claim a task twin")
        # Distinct whole command sequences; no holdout result enters a prompt.
        held_menu = [replace(holds[-1], duration_s=1.3), replace(contacts[0], duration_s=1.3)]
        for e in held_menu:
            mover.validator.plan(e, scene, q0)
        heldout, training, tried, history = [], [], {e.key for e in held_menu}, []
        publish("collecting_holdout")
        for i, e in enumerate(held_menu):
            heldout.append(mover.execute(e, out/f"heldout_{i:02d}", split="heldout"))
            status["experiments_used"] += 1
            publish("collecting_holdout")
        best_physics, best_scene = materialize(physics, scene, values)
        before = best = score(heldout, best_physics, best_scene)
        status.update(before=before, heldout=best)
        while status["experiments_used"] < budget:
            q0 = mover.q0 if hasattr(mover, "q0") else mover.measured_q(mover.request(mover.url+"/state"))
            wrist = mover.wrist_parts(training[-1]) if training and hasattr(mover, "wrist_parts") else ()
            e, choice = choose_experiment(client, task, mover.scene, values, uncertainty, history, tried,
                                         mover.validator, q0, wrist_parts=wrist)
            if e is None:
                status["stop_reason"] = "safe_task_menu_exhausted"; break
            tried.add(e.key)
            publish("executing")
            observation = mover.execute(e, out/f"train_{len(training)+1:02d}", split="train")
            training.append(observation); status["experiments_used"] += 1
            wrist_review = None
            if e.kind in ("push", "pick_lift_place") and hasattr(mover, "wrist_parts"):
                parts = mover.wrist_parts(observation)
                if parts:
                    answer, record = client.call([
                        "Review this TRAINING experiment's timed wrist frames. Report visible grasp, slip and release "
                        "evidence. Use null when occluded or not applicable; never infer success from the commands. "
                        +json.dumps(json_experiment(e)), *parts], role="main", thinking="high",
                        tag="general_wrist_review", schema=WRIST_REVIEW_SCHEMA)
                    wrist_review = {"observation": gemini.parse_json(answer), "record": record}
            publish("fitting")
            fitted, estimated, diagnostics = fit_parameters(training, task, physics, scene, values, max_nfev=max_nfev)
            proposed_physics, proposed_scene = materialize(physics, scene, fitted)
            candidate = score(heldout, proposed_physics, proposed_scene)
            verdict = keep_heldout(candidate, best)
            if verdict["kept"]:
                values, uncertainty = fitted, estimated
                best_physics, best_scene, best = proposed_physics, proposed_scene, candidate
            for key in uncertainty:
                uncertainty[key]["excited"] = estimated[key]["excited"]
            status["heldout"] = best
            history.append({**score(training[-1:], best_physics, best_scene), "wrist_review": wrist_review})
            row = {"experiment": json_experiment(e), "choice": choice, "fit": diagnostics, "verdict": verdict,
                   "estimated_parameters": fitted, "estimated_uncertainty": estimated, "heldout": candidate,
                   "wrist_review": wrist_review}
            status["iterations"].append(row)
            gemini.atomic_json(out/f"iter_{len(training):02d}.json", row)
            gemini.atomic_json(out/"residuals.json", {"training": history, "heldout": best})
            publish("evaluated")
        best_physics["active_twin"] = {"activated": False, "run": str(out), "task": task["name"],
                                       "parameters": values, "uncertainty": uncertainty}
        gemini.atomic_json(out/"candidate_physics.json", best_physics)
        gemini.atomic_json(out/"candidate_scene.json", best_scene)
        status.update(heldout=best, candidate_physics=str(out/"candidate_physics.json"))
        status["task_outcomes"] = {o.session.path.name: o.session.meta.get("synthetic_task_check")
                                   for o in [*heldout, *training]}
        lines = [f"# General active twin: {task['name']}", "",
                 "Synthetic experiment campaign; no real transfer claim." if status["synthetic"] else
                 "Recorded real experiments; candidate is inactive.", "",
                 "| Held-out metric | Before | Best |", "|---|---:|---:|"]
        lines += [f"| {k} | {before[k]:.6f} | {best[k]:.6f} |"
                  for k in ("error", "joint_deg", "jaw_mm", "sag_mm", "object_mm", "iou")]
        lines += ["", f"Experiments: {status['experiments_used']}/{budget}; two complete held-out sequences.",
                  "Task-derived menu and safety refusals: `menu.json`. Training choices and fit eligibility: `iter_*.json`.",
                  "Use `candidate_physics.json` together with `candidate_scene.json`; neither was activated.",
                  "Repeated held-out gating is model selection, not an independent transfer audit.",
                  "Synthetic camera evidence uses sparse rendered masks; poor/occluded silhouette fits are omitted.",
                  "", "Task outcomes (diagnostics, never optimizer input):", "```json",
                  json.dumps(status["task_outcomes"], indent=2), "```", "", "Parameters and uncertainty:",
                  "```json", json.dumps({"parameters": values, "uncertainty": uncertainty}, indent=2), "```"]
        (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        publish("complete")
        if videos:
            from .twin_videos import build_videos
            status["videos"] = build_videos(out)
            publish("complete")
        return status
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"; publish("failed")
        raise
