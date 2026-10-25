"""Timestamp-aligned real | sim-before | sim-after MP4 command replays (CPU)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

from r2p import gemini
from r2p.frames import HZ
from r2p.perception import CameraModel
from r2p.sim import TaskTwin
from .evaluate import CalibratedRenderer
from .session import load_session


def _scene(recorded, candidate):
    result = deepcopy(recorded)
    by_id = {o["id"]: o for o in candidate["objects"]}
    for obj in result["objects"]:
        for key in ("mass_kg", "friction", "size_m", "color", "wall_m"):
            if key in by_id.get(obj["id"], {}):
                obj[key] = deepcopy(by_id[obj["id"]][key])
    result["table"] = deepcopy(candidate.get("table", result.get("table", {})))
    return result


def render_experiment(session, before_physics, after_physics, before_scene, after_scene, destination, *, fps=5, width=320):
    if not 1 <= fps <= 30 or width < 64:
        raise ValueError("fps must be 1..30 and width >=64")
    if not session.ticks or not session.camera:
        raise ValueError("Replay needs recorded commands and camera calibration")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    original = session.camera
    scale = width/original.image_size[0]
    height = int(round(original.image_size[1]*scale/2))*2
    width = int(round(width/2))*2
    camera = original.resized((width, height))
    twins, renderers, writers, paths = [], [], {}, {}
    count = 0
    try:
        for physics, scene in ((before_physics, before_scene), (after_physics, after_scene)):
            twin = TaskTwin(scene, physics, session.ticks[0]["q"])
            twins.append(twin); renderers.append(CalibratedRenderer(twin, camera))
        for label in ("real", "sim_before", "sim_after", "comparison"):
            path = destination/f"{label}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (width*3 if label == "comparison" else width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open MP4 writer: {path}")
            writers[label] = writer; paths[label] = str(path)
        command_index = 0
        sim_t = session.ticks[0]["t"]
        start = max(sim_t, session.times[0])
        end = min(session.ticks[-1]["t"], session.times[-1])
        cursor, real = 0, session.frame(0)
        for t in np.arange(start, end+1e-7, 1/fps):
            while sim_t+1/HZ <= t+1e-7:
                while command_index+1 < len(session.ticks) and session.ticks[command_index+1]["t"] <= sim_t+1e-7:
                    command_index += 1
                for twin in twins:
                    twin.step(session.ticks[command_index]["q_target"])
                sim_t += 1/HZ
            while cursor+1 < len(session.times) and session.times[cursor+1] <= t+1e-7:
                cursor += 1; real = session.frame(cursor)
            panels = [cv2.resize(real, (width, height)), *[renderer.render()[0] for renderer in renderers]]
            for panel, label, name in zip(panels, ("real", "sim_before", "sim_after"),
                                        ("REAL" if session.meta.get("source") == "real" else "SYNTHETIC OBSERVED",
                                         "SIM BEFORE", "SIM AFTER")):
                cv2.rectangle(panel, (0, 0), (width, 24), (15, 15, 15), -1)
                cv2.putText(panel, f"{name}  {t:.2f}s", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
                writers[label].write(panel)
            writers["comparison"].write(np.concatenate(panels, axis=1)); count += 1
    finally:
        for writer in writers.values():
            writer.release()
        for renderer in renderers:
            renderer.close()
        for twin in twins:
            twin.close()
    if not count:
        raise ValueError("No overlapping camera/command timestamps")
    for path in paths.values():
        cap = cv2.VideoCapture(path)
        try:
            ok, frame = cap.read()
            if not ok or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                raise ValueError(f"MP4 verification failed: {path}")
        finally:
            cap.release()
    return {"files": paths, "frames": count, "fps": fps, "duration_s": count/fps,
            "alignment": "recorded timestamps; source frames held until next capture",
            "source_frames": len(session.frames), "source": session.meta.get("source")}


def build_videos(run, *, fps=5, width=320):
    run = Path(run)
    if not run.is_dir():
        raise FileNotFoundError(run)
    status_path = run/"status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    candidate = run/"candidate_physics.json"
    if not candidate.exists() and status.get("candidate_physics"):
        candidate = Path(status["candidate_physics"])
    required = [run/"prior_physics.json", run/"prior_scene.json", candidate, run/"candidate_scene.json"]
    sessions = sorted(p.parent for p in run.glob("*/trace.jsonl") if p.parent.name.startswith(("heldout_", "train_")))
    missing = [str(p) for p in required if not p.is_file()]
    manifest = {"schema": "r2p_twin_videos/1", "run": str(run), "experiments": [],
                "status": "waiting_for_campaign" if missing or not sessions else "rendering",
                "missing": missing, "recorded_sessions": len(sessions)}
    gemini.atomic_json(run/"videos/manifest.json", manifest)
    if missing or not sessions:
        return manifest
    before_physics, prior, after_physics, candidate_scene = [json.loads(p.read_text()) for p in required]
    for directory in sessions:
        session = load_session(directory)
        recorded = session.meta.get("scene", prior)
        result = render_experiment(session, before_physics, after_physics, _scene(recorded, prior),
                                   _scene(recorded, candidate_scene), run/"videos"/directory.name, fps=fps, width=width)
        manifest["experiments"].append({"experiment": directory.name, **result})
        gemini.atomic_json(run/"videos/manifest.json", manifest)
    manifest["status"] = "complete"
    gemini.atomic_json(run/"videos/manifest.json", manifest)
    return manifest
