"""Strict scene patches; camera calibration is deliberately immutable."""
from __future__ import annotations

import copy
import math

PATCH_SCHEMA = {"type": "object", "required": ["objects", "why", "done"], "properties": {
    "objects": {"type": "array", "items": {"type": "object", "required": ["id"], "properties": {
        "id": {"type": "string"}, "xyz": {"type": "array", "minItems": 3, "maxItems": 3,
                                             "items": {"type": "number"}},
        "size_m": {"type": "array", "items": {"type": "number"}}, "yaw_deg": {"type": "number"},
        "friction": {"type": "number"}, "mass_kg": {"type": "number"}}, "additionalProperties": False}},
    "why": {"type": "string"}, "done": {"type": "boolean"}}, "additionalProperties": False}


def apply_patch(scene, patch):
    from r2p.gemini import schema_errors

    errors = schema_errors(patch, PATCH_SCHEMA)
    if errors:
        raise ValueError("; ".join(errors))
    candidate = copy.deepcopy(scene)
    objects = {obj["id"]: obj for obj in candidate["objects"]}
    seen = set()
    for change in patch["objects"]:
        oid = change["id"]
        if oid not in objects or oid in seen:
            raise ValueError(f"Unknown or repeated patch object: {oid}")
        seen.add(oid)
        obj = objects[oid]
        for key, value in change.items():
            if key == "id":
                continue
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values):
                raise ValueError(f"{oid}.{key} must be finite numeric values")
            if key == "xyz" and (len(value) != 3 or any(abs(v) > 1.5 for v in value) or value[2] < 0):
                raise ValueError(f"{oid}.xyz must be a metric tabletop position")
            if key == "size_m" and (len(value) != len(obj["size_m"]) or any(not 0.001 <= v <= 1 for v in value)):
                raise ValueError(f"{oid}.size_m has invalid shape or metre dimensions")
            if key in ("friction", "mass_kg") and not (0.001 <= value <= (3 if key == "friction" else 5)):
                raise ValueError(f"{oid}.{key} outside allowed physical range")
            if key == "yaw_deg":
                value = (value + 180) % 360 - 180
            obj[key] = value
    return candidate
