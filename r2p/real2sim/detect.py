"""One SMALL request for a frame set; local masks and calibrated geometry thereafter."""
from __future__ import annotations

import copy
import json

import numpy as np

from r2p import gemini

DETECTION_SCHEMA = {"type": "object", "required": ["objects"], "properties": {"objects": {
    "type": "array", "items": {"type": "object", "required": ["id", "box"], "properties": {
        "id": {"type": "string"}, "box": {"type": "array", "minItems": 4, "maxItems": 4,
                                              "items": {"type": "number", "minimum": 0, "maximum": 1}},
        "shape": {"type": "string", "enum": ["box", "cylinder", "sphere", "bowl", "plate", "stick"]},
        "role": {"type": "string", "enum": ["manipulated", "container", "target", "obstacle", "fixture"]},
        "size_m": {"type": "array", "items": {"type": "number", "minimum": 0.001, "maximum": 1}},
        "color": {"type": "string"}}}}}}


def detect(session, indices, *, task=None, ask=None):
    if not indices:
        raise ValueError("Detection needs at least one critic frame")
    prompt = ("Locate the tabletop task objects in the FIRST image; later images provide identity context. "
              "Return objects with id and box [x_min,y_min,x_max,y_max] normalized to [0,1]. "
              "Exclude the robot and table. If objects are not specified, also supply shape, role, size_m in metres, "
              "and color. These are visual detections, not robot instructions.\n")
    if task:
        prompt += "Use these exact object ids and detect every present object: " + json.dumps(task["objects"])
    parts = [prompt] + [gemini.image_part(session.frame(i)) for i in indices]
    text, record = (ask or gemini.call)(parts, role="small", tag="real2sim:detect", schema=DETECTION_SCHEMA)
    answer = gemini.parse_json(text)
    objects = answer["objects"]
    seen = set()
    for obj in objects:
        if obj["id"] in seen:
            raise ValueError("Duplicate detected object id")
        seen.add(obj["id"])
        x0, y0, x1, y1 = obj["box"]
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"Invalid normalized box for {obj['id']}")
    if not objects:
        raise ValueError("No objects were detected")
    return objects, record


def initial_scene(session, detections, *, task=None, first_index=0):
    from r2p.perception.appearance import Appearance, seed_mask
    from r2p.perception.geometry import dimensions, fit_pose

    if session.camera is None:
        raise ValueError("A calibrated camera.json is required to fit metric scene geometry")
    image = session.frame(first_index)
    height, width = image.shape[:2]
    specs = {obj["id"]: obj for obj in task["objects"]} if task else {}
    found, appearances = [], {}
    for detection in detections:
        oid = detection["id"]
        if task and oid not in specs:
            raise ValueError(f"Detector returned an unknown task object: {oid}")
        obj = copy.deepcopy(specs.get(oid, {k: v for k, v in detection.items() if k != "box"}))
        if not all(key in obj for key in ("shape", "size_m", "role")):
            raise ValueError(f"Detection for {oid} lacks shape/size_m/role")
        box = np.asarray(detection["box"]) * [width, height, width, height]
        mask = seed_mask(image, box, obj.get("color"))
        appearances[oid] = Appearance.fit(image, mask)
        pose = fit_pose(session.camera, obj, mask)
        obj.update(xyz=np.asarray(pose["xyz"]).tolist(), yaw_deg=float(pose.get("yaw_deg", 0)))
        obj.setdefault("mass_kg", 0.03)
        obj.setdefault("friction", 0.6)
        obj.setdefault("fixed", obj["role"] in ("target", "fixture"))
        found.append(obj)
    if task:
        missing = set(specs) - {obj["id"] for obj in found}
        required = [oid for oid in missing if task.get("randomize", {}).get(oid, {}).get("p_present", 1) > 0]
        if required:
            raise ValueError("Missing task objects: " + ", ".join(sorted(required)))
    return {"version": 0, "task": task["name"] if task else "observed_task", "objects": found,
            "robot": {"type": "so101", "joint_convention": "mjcf_rad"},
            "provenance": {"source": str(session.path), "iteration": 0}}, appearances
