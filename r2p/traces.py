"""The r2p_trace/1 JSONL contract, shared by expert, human, bot and real runs.

q_target is always required: measured q or a perturbed q_exec is never substituted
for the training label. Upload joins are pure parsing/validation, with no network I/O.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import numpy as np

from .tasks import validate as validate_task
from .tasks.scene import pose_of
from .tasks.spec import number, vector

SCHEMA = "r2p_trace/1"
SOURCES = ("expert", "human", "bot", "real")
META_FIELDS = ("schema", "task", "task_name", "source", "player", "level", "seed", "scene",
               "success", "fail_reason", "commands", "rig")
TICK_FIELDS = ("t", "q", "q_target", "poses", "grip_closed", "command", "phase")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _plain(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def validate_trace(meta, ticks, *, hz=30) -> list[str]:
    """Validate a complete trace without modifying it; return all detected errors."""
    errors = []
    if not number(hz) or hz <= 0:
        return ["hz must be finite and positive"]
    if not isinstance(meta, dict):
        return ["meta must be an object"]
    for field in META_FIELDS:
        if field not in meta:
            errors.append(f"meta needs {field}")
    if meta.get("schema") != SCHEMA:
        errors.append(f"meta.schema must be {SCHEMA}")
    if meta.get("source") not in SOURCES:
        errors.append("meta.source must be expert, human, bot or real")
    task = meta.get("task")
    if isinstance(task, dict):
        errors.extend(f"meta.task: {error}" for error in validate_task(task))
        name = task.get("name")
    elif isinstance(task, str) and task:
        name = task
    else:
        name = None
        errors.append("meta.task must be a task dict or nonempty task name")
    if not isinstance(meta.get("task_name"), str) or not meta["task_name"]:
        errors.append("meta.task_name must be a nonempty string")
    elif name is not None and name != meta["task_name"]:
        errors.append("meta.task_name must match meta.task")
    if not isinstance(meta.get("success"), bool):
        errors.append("meta.success must be boolean")
    if meta.get("seed") is not None and (not isinstance(meta["seed"], int) or isinstance(meta["seed"], bool)):
        errors.append("meta.seed must be an integer or null")
    if not isinstance(meta.get("scene"), dict):
        errors.append("meta.scene must be an object")
    for field in ("fail_reason", "rig", "level"):
        if meta.get(field) is not None and not isinstance(meta[field], str):
            errors.append(f"meta.{field} must be a string or null")
    commands = meta.get("commands")
    if not isinstance(commands, list):
        errors.append("meta.commands must be a list")
    else:
        previous_end = -math.inf
        for i, segment in enumerate(commands):
            if not isinstance(segment, (tuple, list)) or len(segment) != 3:
                errors.append(f"meta.commands[{i}] must be [start,end,command]")
                continue
            start, end, command = segment
            if not number(start) or not number(end) or start < 0 or start > end:
                errors.append(f"meta.commands[{i}] has invalid times")
            else:
                if start < previous_end - 1e-8:
                    errors.append(f"meta.commands[{i}] overlaps or is out of order")
                previous_end = end
            if not isinstance(command, str) or not command:
                errors.append(f"meta.commands[{i}] needs a nonempty command")
    if not isinstance(ticks, (list, tuple)):
        return errors + ["ticks must be a list"]
    previous_t = None
    for i, tick in enumerate(ticks):
        label = f"tick[{i}]"
        if not isinstance(tick, dict):
            errors.append(f"{label} must be an object")
            continue
        for field in TICK_FIELDS:
            if field not in tick:
                errors.append(f"{label} needs {field}")
        t = tick.get("t")
        if not number(t) or t < 0:
            errors.append(f"{label}.t must be finite and nonnegative")
        else:
            if previous_t is not None and not math.isclose(t - previous_t, 1/hz, rel_tol=0, abs_tol=1e-5):
                errors.append(f"{label}.t must advance by 1/{hz:g} seconds")
            previous_t = t
        for field in ("q", "q_target"):
            if not vector(tick.get(field), 6):
                errors.append(f"{label}.{field} must contain six finite MJCF radians")
        if not isinstance(tick.get("grip_closed"), bool):
            errors.append(f"{label}.grip_closed must be boolean")
        for field in ("command", "phase"):
            if tick.get(field) is not None and not isinstance(tick[field], str):
                errors.append(f"{label}.{field} must be a string or null")
        poses = tick.get("poses")
        if not isinstance(poses, dict):
            errors.append(f"{label}.poses must be an object-id mapping")
        else:
            for oid, pose in poses.items():
                if not isinstance(oid, str) or not oid or not isinstance(pose, dict):
                    errors.append(f"{label}.poses must map nonempty ids to pose objects")
                    continue
                try:
                    pose_of(pose)
                except (ValueError, TypeError, KeyError, OverflowError):
                    errors.append(f"{label}.poses.{oid} has an invalid pose")
                if not number(pose.get("yaw_deg")):
                    errors.append(f"{label}.poses.{oid}.yaw_deg must be finite")
    try:
        _plain([meta, ticks])
    except (ValueError, TypeError, OverflowError):
        errors.append("trace must be finite JSON data")
    return errors


def _require_valid(meta, ticks):
    errors = validate_trace(meta, ticks)
    if errors:
        raise ValueError("Invalid trace:\n" + "\n".join(errors))


def _prepare(meta, ticks):
    meta = _plain(meta)
    if not isinstance(meta, dict):
        raise ValueError("meta must be an object")
    task = meta.get("task")
    name = task.get("name") if isinstance(task, dict) else task
    defaults = dict(schema=SCHEMA, task_name=name, player=None, level=name, seed=None,
                    scene={}, success=False, fail_reason=None, commands=[], rig=None)
    for key, value in defaults.items():
        meta.setdefault(key, value)
    ticks = _plain(list(ticks))
    for tick in ticks:
        if isinstance(tick, dict):
            tick.setdefault("command", None)
            tick.setdefault("phase", None)
    _require_valid(meta, ticks)
    return meta, ticks


def write_trace(path, meta, ticks):
    """Validate, then atomically replace path. Accept numpy values and tick iterables.

    Optional metadata gets explicit defaults; task and source must be supplied.
    command/phase default to null. All measured/target/pose/gripper fields are required.
    """
    meta, ticks = _prepare(meta, ticks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            for record in [{"meta": meta}, *ticks]:
                output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _records(text):
    records = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank trace line {i}")
        try:
            record = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid JSON at trace line {i}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Trace line {i} must be an object")
        records.append(record)
    if not records or set(records[0]) != {"meta"} or not isinstance(records[0]["meta"], dict):
        raise ValueError("Trace line 1 must contain only a meta object")
    return records


def read_trace(path):
    """Read a canonical trace; upload trailers must first go through join_upload_parts."""
    records = _records(Path(path).read_text(encoding="utf-8-sig"))
    if any("meta" in tick or "meta_final" in tick for tick in records[1:]):
        raise ValueError("Metadata may occur only on the first trace line")
    meta, ticks = records[0]["meta"], records[1:]
    _require_valid(meta, ticks)
    return meta, ticks


def join_upload_parts(parts, path=None):
    """Join Roblox upload chunks and merge a final {meta_final: {...}} trailer.

    parts is ordered text/bytes chunks, a mapping {zero_based_part: text}, or an
    iterable of {part: int, data: text} records. The Roblox HTTP representation
    {trace_id, part, final, meta?, ticks, meta_final?} is also accepted.
    Numbered parts may arrive out of
    order; duplicate/missing part numbers are errors. Chunks concatenate verbatim,
    including splits inside JSON lines. An optional path writes a canonical trace.
    Returns (meta, ticks). Identity fields cannot change in the final trailer.
    """
    if isinstance(parts, Mapping):
        pairs = list(parts.items())
    else:
        chunks = list(parts)
        if chunks and all(isinstance(part, dict) for part in chunks):
            pairs = [(part.get("part"), part if "ticks" in part else part.get("data")) for part in chunks]
        elif any(isinstance(part, dict) for part in chunks):
            raise ValueError("Upload parts must all use the same representation")
        else:
            pairs = list(enumerate(chunks))
    if any(not isinstance(index, int) or isinstance(index, bool) for index, _ in pairs):
        raise ValueError("Upload part numbers must be integers")
    pairs.sort(key=lambda pair: pair[0])
    if [index for index, _ in pairs] != list(range(len(pairs))):
        raise ValueError("Upload parts must be numbered contiguously from zero, without duplicates")
    if pairs and all(isinstance(part, dict) for _, part in pairs):
        trace_ids = [part.get("trace_id") for _, part in pairs]
        if any(value != trace_ids[0] for value in trace_ids):
            raise ValueError("Upload parts must belong to the same trace_id")
        converted = []
        for index, part in pairs:
            if "part" in part and part["part"] != index:
                raise ValueError("Payload part number disagrees with its index")
            if not isinstance(part.get("final"), bool) or part["final"] != (index == len(pairs)-1):
                raise ValueError("Only the last upload part must declare final=true")
            if not isinstance(part.get("ticks"), list):
                raise ValueError("Upload ticks must be a list")
            rows = []
            if index == 0:
                if not isinstance(part.get("meta"), dict):
                    raise ValueError("Part zero needs meta")
                rows.append({"meta": part["meta"]})
            elif "meta" in part:
                raise ValueError("Only part zero can contain meta")
            rows.extend(part["ticks"])
            if "meta_final" in part:
                if not part["final"]:
                    raise ValueError("Only the last part can contain meta_final")
                rows.append({"meta_final": part["meta_final"]})
            converted.append((index, "".join(json.dumps(row, default=_json_default, allow_nan=False) + "\n" for row in rows)))
        pairs = converted
    chunks = []
    for _, data in pairs:
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, bytes):
            raise ValueError("Upload data must be text or bytes")
        chunks.append(data)
    records = _records(b"".join(chunks).decode("utf-8-sig"))
    meta, ticks = records[0]["meta"], records[1:]
    # Even an abbreviated opening header has a task identity: a trailer cannot
    # change the task_name simply because its redundant spelling was omitted.
    task = meta.get("task")
    meta.setdefault("task_name", task.get("name") if isinstance(task, dict) else task)
    meta.setdefault("schema", SCHEMA)
    if ticks and "meta_final" in ticks[-1]:
        trailer = ticks.pop()
        if set(trailer) != {"meta_final"} or not isinstance(trailer["meta_final"], dict):
            raise ValueError("meta_final must be the final line's only field")
        for key in ("schema", "task", "task_name", "source", "player", "level", "seed", "scene", "rig"):
            if key in trailer["meta_final"] and key in meta and trailer["meta_final"][key] != meta[key]:
                raise ValueError(f"meta_final cannot change {key}")
        meta.update(trailer["meta_final"])
    if any("meta" in tick or "meta_final" in tick for tick in ticks):
        raise ValueError("Unexpected metadata inside upload ticks")
    meta, ticks = _prepare(meta, ticks)
    if path is not None:
        write_trace(path, meta, ticks)
    return meta, ticks


# Explicit aliases for callers whose operation is named after its source/format.
join_roblox_parts = join_upload_parts
join_trace_parts = join_upload_parts
