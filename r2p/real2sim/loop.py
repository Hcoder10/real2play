"""Render, critique, patch, hold-out score, keep best; publish every stage atomically."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from pathlib import Path

from r2p import gemini
from .anchors import apply_anchors, extract_anchors
from .detect import detect, initial_scene
from .evaluate import evaluate, observation_masks, overlay, physics_image, physics_replay
from .keep import keep_candidate
from .patch import PATCH_SCHEMA, apply_patch
from .session import load_session, split_frames


def atomic_image(path, image):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError("Could not encode iteration image")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded.tobytes())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class LiveRun:
    def __init__(self, directory, session):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "run.json").exists():
            raise FileExistsError("Use a fresh run directory to preserve earlier evidence")
        self.start = time.monotonic()
        self.records = []
        self.data = {"schema": "r2s_live/1", "run": self.directory.name, "status": "starting", "stage": "anchors",
                     "started": time.time(), "iters": [], "best_iter": None, "stop_reason": None,
                     "session": str(session.path), "session_kind": session.kind, "video": str(session.video) if session.video else None,
                     "calibration": session.camera.to_dict() if hasattr(session.camera, "to_dict") else {},
                     "anchors": [], "calls": {"by_model": {}, "served_by": {}}, "n_frames": len(session.frames)}
        self.save()

    def save(self, **updates):
        self.data.update(updates, updated=time.time(), elapsed_s=time.monotonic() - self.start)
        by_model, served = {}, {}
        for record in self.records:
            model = record.get("model", record.get("served_by", "unknown"))
            by_model[model] = by_model.get(model, 0) + (0 if record.get("cached") else record.get("attempts", 1))
            answerer = record.get("served_by", model)
            if not record.get("error"):
                served[answerer] = served.get(answerer, 0) + 1
        self.data["calls"] = {"requests": sum(by_model.values()), "cache_hits": sum(r.get("cached", False) for r in self.records),
                              "by_model": by_model, "served_by": served, "errors": sum("error" in r for r in self.records),
                              "latency_s": sum(r.get("latency", 0) for r in self.records)}
        gemini.atomic_json(self.directory / "run.json", self.data)

    def record(self, record):
        self.records.append({key: value for key, value in record.items() if key != "repair"})
        if record.get("repair"):
            self.records.append(record["repair"])

    def iteration(self, index, scene, metrics, verdict, patch, session, critic, snapshots, appearances):
        images = {}
        for j, frame_index in enumerate(critic):
            name = f"iter_{index}_overlay_{j}.jpg"
            atomic_image(self.directory / name, overlay(session.frame(frame_index), snapshots[frame_index]))
            images[f"overlay_{j}"] = name
        name = f"iter_{index}_physics.jpg"
        atomic_image(self.directory / name, physics_image(session, critic, snapshots, appearances))
        images["physics"] = name
        scene_name = f"iter_{index}_scene.json"
        gemini.atomic_json(self.directory / scene_name, scene)
        images["scene"] = scene_name
        best_iter = index if verdict["kept"] else self.data["best_iter"]
        entry = {"iter": index, "score": metrics["score"], **verdict, "metrics": metrics, "patch": patch,
                 "best_iter": best_iter, "images": images, "elapsed_s": time.monotonic() - self.start}
        gemini.atomic_json(self.directory / f"iter_{index}.json", entry)
        best_score = metrics["score"] if verdict["kept"] else self.data["iters"][best_iter]["score"]
        self.data["iters"].append({"iter": index, "file": f"iter_{index}.json", "score": metrics["score"],
                                   "kept": verdict["kept"], "verdict": verdict["verdict"],
                                   "best_so_far": best_score, "thumb": images["overlay_0"]})
        self.save(best_iter=best_iter)


def run_twin(source, *, task=None, out_dir=None, iterations=6, patience=2, ask=None, camera=None,
             anchors="auto", scene=None, detections=None, appearances=None, twin_factory=None,
             evaluator=None, replay=None, max_calls=12, budget_s=1800):
    session = load_session(source, camera=camera)
    if iterations < 0 or patience < 1:
        raise ValueError("iterations must be nonnegative and patience positive")
    if anchors not in ("auto", "hard", "soft"):
        raise ValueError("anchors must be auto, hard or soft")
    if task is not None:
        from r2p.tasks import load_task
        task = load_task(task)
    directory = Path(out_dir or Path("out/live") / (session.path.stem + "_" + time.strftime("%Y%m%d_%H%M%S")))
    live = LiveRun(directory, session)
    provider = ask or gemini.Gemini()
    owner = getattr(provider, "__self__", provider)

    def tracked_ask(*args, **kwargs):
        records = getattr(owner, "records", None)
        count = len(records) if records is not None else 0
        used = sum(r.get("attempts", 0 if r.get("cached") else 1) for r in live.records)
        if used >= max_calls and not isinstance(owner, gemini.Gemini):
            raise gemini.RequestLimit("Twin request budget exhausted")
        old_limit = getattr(owner, "max_requests", None)
        if isinstance(owner, gemini.Gemini):
            remaining_limit = owner.requests_used + max_calls - used
            owner.max_requests = min(old_limit, remaining_limit) if old_limit is not None else remaining_limit
        try:
            text, record = provider(*args, **kwargs)
            if records is None:
                live.record(record)
            return text, record
        finally:
            if records is not None:
                for record in records[count:]:
                    live.record(record)
            if isinstance(owner, gemini.Gemini):
                owner.max_requests = old_limit
            live.save()
    best_scene = best_metrics = None
    try:
        critic, heldout = split_frames(session)
        if session.camera is None:
            raise ValueError("Twin loop needs a calibrated camera.json; no guessed camera is used")
        facts = extract_anchors(session.ticks) if session.ticks else []
        live.save(status="running", stage="detect", critic_frames=critic, eval_frames=heldout)
        if scene is None:
            if detections is None:
                detections, record = detect(session, critic, task=task, ask=tracked_ask)
            scene, appearances = initial_scene(session, detections, task=task, first_index=critic[0])
        scene = apply_anchors(scene, facts, mode=anchors)
        appearances = appearances or {}
        observation_masks(session, appearances)
        enforced = bool(facts) and all(a.get("enforced", False) for a in facts)
        live.save(anchors=facts, anchors_enforced=enforced, detection=detections, stage="evaluate")
        replay_fn, evaluate_fn = replay or physics_replay, evaluator or evaluate

        def score(candidate):
            snapshots = replay_fn(session, candidate, indices=critic + heldout, twin_factory=twin_factory)
            metrics = evaluate_fn(session, candidate, heldout, appearances, snapshots, facts)
            return snapshots, metrics

        snapshots, best_metrics = score(scene)
        best_scene = copy.deepcopy(scene)
        peaks = {key: value["iou"] for key, value in best_metrics["objects"].items()}
        containers = [obj["id"] for obj in scene["objects"] if obj.get("role") == "container"]
        live.iteration(0, scene, best_metrics, {"kept": True, "verdict": "INITIAL", "reasons": []}, None,
                       session, critic, snapshots, appearances)
        gemini.atomic_json(directory / "scene_best.json", best_scene)
        stale, reason, previous_proposal = 0, "max_iterations", None
        for iteration in range(1, iterations + 1):
            if time.monotonic() - live.start >= budget_s:
                reason = "time_budget"
                break
            if sum(r.get("attempts", 0 if r.get("cached") else 1) for r in live.records) >= max_calls:
                reason = "call_budget"
                break
            live.save(stage=f"critique {iteration}")
            # Only critic frames and their physics tracks are ever sent to Gemini.
            prompt = ("Critique this calibrated digital twin. Left: real image with simulated outlines. Right: simulated image. "
                      "Return a JSON patch for object xyz (metres, base footprint frame x forward/y left/z up), "
                      "size_m, yaw_deg, mass_kg, friction. Do not change camera calibration or invent objects. "
                      "Use one objects array, why and done. An empty patch means done.\n"
                      + "Current scene: " + json.dumps(best_scene) + "\nMeasured anchors: " + json.dumps(facts))
            if previous_proposal is not None:
                prompt += "\nPrevious proposal and local validation verdict: " + json.dumps(previous_proposal)
            parts = [prompt] + [gemini.image_part(overlay(session.frame(i), snapshots[i])) for i in critic]
            parts.append(gemini.image_part(physics_image(session, critic, snapshots, appearances)))
            try:
                text, record = tracked_ask(parts, role="main", tag=f"real2sim:critique:{iteration}", schema=PATCH_SCHEMA)
                patch = gemini.parse_json(text)
            except gemini.GeminiError as exc:
                reason = ("quota" if isinstance(exc, gemini.QuotaError) else
                          "cache_miss" if isinstance(exc, gemini.CacheMiss) else
                          "call_budget" if isinstance(exc, gemini.RequestLimit) else
                          "invalid_reply" if isinstance(exc, gemini.JSONError) else "request_failed")
                live.save(error=str(exc))
                break
            try:
                candidate = apply_patch(best_scene, patch)
                candidate = apply_anchors(candidate, facts, mode=anchors, pin_pose=False)
                if not patch["objects"]:
                    reason = "critic_done"
                    break
                candidate_snapshots, metrics = score(candidate)
                verdict = keep_candidate(metrics, best_metrics, peaks, containers=containers, anchors_enforced=enforced)
            except (ValueError, gemini.JSONError) as exc:
                candidate, candidate_snapshots, metrics = copy.deepcopy(best_scene), snapshots, best_metrics
                verdict = {"kept": False, "verdict": "REJECTED", "reasons": ["invalid_patch: " + str(exc)]}
            live.iteration(iteration, candidate, metrics, verdict, patch, session, critic, candidate_snapshots, appearances)
            previous_proposal = {"patch": patch, "verdict": verdict}
            if verdict["kept"]:
                # Rejected proposals must never tighten the incumbent's gates.
                for oid, values in metrics["objects"].items():
                    peaks[oid] = max(peaks.get(oid, 0), values["iou"])
                best_scene, best_metrics, snapshots = candidate, metrics, candidate_snapshots
                gemini.atomic_json(directory / "scene_best.json", best_scene)
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                reason = "plateau"
                break
        live.save(status="done", stage="done", stop_reason=reason, best_scene="scene_best.json")
        return {"scene": best_scene, "metrics": best_metrics, "best_iter": live.data["best_iter"],
                "stop_reason": reason, "run_dir": str(directory), "run": live.data}
    except Exception as exc:
        live.save(status="failed", stage="failed", stop_reason=type(exc).__name__, error=gemini._safe(str(exc)))
        raise
