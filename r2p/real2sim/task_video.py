"""Write a validated task from one whole episode, then fit its calibrated scene."""
from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path

from r2p import gemini
from .anchors import extract_anchors
from .session import load_session

TASK_REPLY_SCHEMA = {"type": "object", "required": ["task"], "properties": {
    "task": {"type": "object"}, "boxes": {"type": "array", "items": {"type": "object"}}}}

CONTRACT = """Write task schema argon_task/1: name (snake_case), instruction, objects, goal, skeleton,
randomize, limits {episode_s,max_objects}. Object fields: id, role, shape, size_m, color, optional xyz/yaw_deg,
mass_kg/friction/grasp. Roles: manipulated, container, target, obstacle, fixture. Shapes/dimensions:
box/plate=[x,y,z], cylinder/bowl=[diameter,height], sphere=[diameter], stick=[length,diameter]. All metres.
Frame is robot base FOOTPRINT centre on table: x forward, y left, z up. Goal predicates (key pred):
inside(a,b), on_top(a,b), at(a,xy,tol_m), near(a,b,d_m), lifted(a,h_m), grasped(a), upright(a,tol_deg),
yaw(a,deg,tol_deg), not_moved(a,tol_m), stacked(objs), gripper_open. Skeleton (key skill):
pick(obj), place_in(obj,target), place_on(obj,target), place_at(obj,xy), push_to(obj,xy), release, home, wait(s).
xy can be an object id. Use only observed objects and motions, with obstacle/not_moved goals where appropriate.
Return {task: {...}, boxes: [{id, box:[xmin,ymin,xmax,ymax]}]} with boxes normalized [0,1] in the FIRST image.
Infer the intended goal from the WHOLE video, even if the attempt fails. First/last images supplement the video.
Randomization ranges can be omitted; they are filled locally. Do not infer arbitrary physics precision from pixels.
"""


def fill_randomize(task):
    task = copy.deepcopy(task)
    ranges = task.setdefault("randomize", {})
    if not isinstance(ranges, dict):
        return task
    ranges.setdefault("min_sep_m", 0.015)
    for obj in task.get("objects", []):
        if not isinstance(obj, dict) or "id" not in obj:
            continue
        entry = ranges.setdefault(obj["id"], {})
        if not isinstance(entry, dict):
            continue
        if not any(key in entry for key in ("r", "x", "xy_choice")) and "on" not in obj:
            xy = obj.get("xyz", obj.get("xy", [0.24, 0]))[:2]
            radius = 0 if obj.get("fixed") or obj.get("role") == "fixture" else 0.025
            entry.update(x=[xy[0] - radius, xy[0] + radius], y=[xy[1] - radius, xy[1] + radius])
        entry.setdefault("yaw_deg", [obj.get("yaw_deg", 0) - 15, obj.get("yaw_deg", 0) + 15])
    return task


def _anchor_width(task, facts):
    manipulated = [obj for obj in task.get("objects", []) if obj.get("role") == "manipulated"]
    if len(manipulated) == 1 and facts and facts[0]["width_m"] > 0:
        obj, width = manipulated[0], facts[0]["width_m"]
        dim = 1 if obj.get("shape") in ("box", "plate", "stick") else 0
        if "size_m" in obj and len(obj["size_m"]) > dim and abs(obj["size_m"][dim] - width) > width * 0.25:
            obj["size_m"][dim] = width


def task_from_video(source, instruction=None, *, out_root="tasks", ask=None, camera=None, retries=2,
                    fit_twin=True, twin_runner=None, twin_options=None):
    from r2p.tasks import validate
    from .loop import run_twin

    if not 0 <= retries <= 2:
        raise ValueError("At most two task validation re-asks are permitted")
    session = load_session(source, camera=camera)
    if len(session.frames) == 1 and not instruction:
        raise ValueError("A single frame requires an instruction")
    facts = extract_anchors(session.ticks) if session.ticks else []
    ask = ask or gemini.call
    with tempfile.TemporaryDirectory(prefix="r2p-episode-") as temporary:
        parts = [CONTRACT + "\nInstruction: " + (instruction or "infer from episode")
                 + f"\nVideo time zero corresponds to recorded t={session.times[0]:.6f}s."
                 + "\nMeasured anchor facts: " + json.dumps(facts) + "\nJoint log (5 Hz):\n" + session.joint_table()]
        if len(session.frames) > 1:
            parts.append(gemini.video_part(session.video_path(Path(temporary) / "episode.mp4")))
        parts += [gemini.image_part(session.frame(0)), gemini.image_part(session.frame(len(session.frames) - 1))]
        original = parts[:]
        errors, records, answer = [], [], None
        for attempt in range(retries + 1):
            text = "No parseable reply"
            try:
                text, record = ask(parts, role="main", tag=f"task_from_video:{attempt}", schema=TASK_REPLY_SCHEMA)
                records.append(record)
                answer = gemini.parse_json(text)
                task = fill_randomize(answer["task"])
                _anchor_width(task, facts)
                errors = validate(task)
                if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", str(task.get("name", ""))):
                    errors.append("name must be a safe snake_case identifier, at most 64 characters")
                if facts and not any(obj.get("role") == "manipulated" for obj in task.get("objects", [])):
                    errors.append("a grasp was measured but no object has role manipulated")
                for obj in task.get("objects", []):
                    if any(size > 1 for size in obj.get("size_m", [])):
                        errors.append(f"{obj.get('id')}: size_m must be tabletop dimensions in metres")
                if not errors:
                    break
            except (gemini.JSONError, KeyError, ValueError, TypeError, AttributeError) as exc:
                errors = [str(exc)]
            parts = original + ["Previous reply: " + text + "\nExact task validation errors:\n"
                                + "\n".join(errors) + "\nCorrect these errors; retain the observed task."]
        if errors:
            raise ValueError("Task still invalid after validation re-asks: " + "; ".join(errors))
    task["source"] = {"kind": session.kind, "path": str(session.path), "model": records[-1]["served_by"]}
    destination = Path(out_root) / task["name"]
    gemini.atomic_json(destination / "task.json", task)
    result = {"task": task, "task_path": str(destination / "task.json"), "records": records,
              "scene_path": None, "scene_status": "not_requested"}
    if fit_twin:
        if session.camera is None:
            result["scene_status"] = "requires_calibrated_camera"
        elif len(session.frames) < 2:
            result["scene_status"] = "requires_heldout_frames"
        else:
            options = dict(twin_options or {})
            # First frame is always critic frame 0, so task-writer boxes can be reused safely.
            boxes = answer.get("boxes")
            if boxes:
                options.setdefault("detections", boxes)
            fitted = (twin_runner or run_twin)(session, task=task, ask=ask, **options)
            gemini.atomic_json(destination / "scene.json", fitted["scene"])
            result.update(scene_path=str(destination / "scene.json"), scene_status="fitted", twin=fitted)
            # Preserve the measured fitted pose in the training distribution without narrowing supplied ranges.
            for obj in fitted["scene"]["objects"]:
                entry = task["randomize"].get(obj["id"], {})
                for axis, coordinate in zip(("x", "y"), obj["xyz"][:2]):
                    if axis in entry and isinstance(entry[axis], list):
                        entry[axis] = [min(entry[axis][0], coordinate), max(entry[axis][1], coordinate)]
            gemini.atomic_json(destination / "task.json", task)
    return result
