"""Deterministic skill segments for expert, real and Roblox/human r2p_trace/1 episodes."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from r2p import gemini
from r2p.real2sim.anchors import gripper_events


def _read(trace, task=None):
    if isinstance(trace, (str, Path)):
        from r2p.traces import read_trace
        meta, ticks = read_trace(trace)
    elif isinstance(trace, tuple) and len(trace) == 2:
        meta, ticks = trace
    elif isinstance(trace, dict):
        meta, ticks = trace.get("meta", {}), trace["ticks"]
    else:
        meta, ticks = {}, list(trace)
    if task is None:
        task = meta.get("task") or meta.get("task_name")
    if task is None:
        raise ValueError("A task is needed to label the trace's command vocabulary")
    from r2p.tasks import load_task
    return meta, list(ticks), load_task(task)


def _runs(ticks, labels):
    if not ticks:
        return []
    segments = []
    start, previous = ticks[0]["t"], labels[0]
    for row, command in zip(ticks[1:], labels[1:]):
        if command != previous:
            segments.append([float(start), float(row["t"]), previous])
            start, previous = row["t"], command
    segments.append([float(start), float(ticks[-1]["t"] + 1 / 30), previous])
    return segments


def _validate_segments(segments, ticks, vocab):
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a nonempty list")
    previous = float(ticks[0]["t"])
    for row in segments:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Each segment must be [t0,t1,command]")
        a, b, command = row
        if command not in vocab or not np.isfinite([a, b]).all() or b <= a or abs(a - previous) > 1e-6:
            raise ValueError("Segments must form a contiguous partition using only task commands")
        previous = b
    if abs(previous - (ticks[-1]["t"] + 1 / 30)) > 1e-6:
        raise ValueError("Segments must cover the entire trace")
    return segments


def segment(trace, task=None, *, refine=False, ask=None):
    """Return half-open [t0, t1, command] intervals; MAIN refinement is opt-in."""
    from r2p.tasks import commands

    meta, ticks, task = _read(trace, task)
    if not ticks:
        return []
    times = np.array([row["t"] for row in ticks], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Trace timestamps must be finite and strictly increasing")
    vocab = commands(task)
    # Valid existing labels are evidence and retain their exact boundaries.
    if all(row.get("command") in vocab for row in ticks):
        segments = _runs(ticks, [row["command"] for row in ticks])
    else:
        q = np.asarray([row["q_target"] for row in ticks], float)
        if q.shape != (len(ticks), 6) or not np.isfinite(q).all():
            raise ValueError("Each tick needs six finite q_target angles")
        speeds = np.r_[0.0, np.linalg.norm(np.diff(q[:, :5], axis=0), axis=1) / np.diff(times)]
        events = {event["index"]: event["kind"] for event in gripper_events(ticks)}
        pairs = []
        for index, step in enumerate(task["skeleton"]):
            if step["skill"] in ("pick", "push_to"):
                later = next((j for j in range(index + 1, len(vocab))
                              if task["skeleton"][j]["skill"] in ("place_in", "place_at", "place_on", "push_to")), index)
                pairs.append((vocab[index], vocab[later]))
        if not pairs:
            pairs = [(vocab[0], vocab[0])]
        cycle, state, release_index = 0, "pick", None
        labels = []
        for i, row in enumerate(ticks):
            if events.get(i) == "close":
                state = "carry"
            elif events.get(i) == "release":
                state, release_index = "release", i
            elif state == "release" and i > release_index and speeds[i] > 0.04:
                cycle += 1
                state = "pick" if cycle < len(pairs) else "home"
            pair = pairs[min(cycle, len(pairs) - 1)]
            phase = str(row.get("phase", "")).lower()
            command = row.get("command") if row.get("command") in vocab else None
            if command is None:
                if phase in ("home", "return", "return_home") and "home" in vocab:
                    command = "home"
                elif phase in ("release", "open", "drop") and "release" in vocab:
                    command = "release"
                elif any(word in phase for word in ("place", "transport", "carry", "lift", "lower")):
                    command = pair[1]
                elif any(word in phase for word in ("pick", "approach", "grasp", "close")):
                    command = pair[0]
                elif state == "home" and "home" in vocab:
                    command = "home"
                elif state == "release" and "release" in vocab:
                    command = "release"
                else:
                    command = pair[1] if state == "carry" else pair[0]
            labels.append(command)
        segments = _runs(ticks, labels)
    if refine:
        schema = {"type": "object", "required": ["segments"], "properties": {"segments": {"type": "array",
                  "items": {"type": "array", "minItems": 3, "maxItems": 3}}}}
        summary = [{"t": row["t"], "phase": row.get("phase"), "q_target": row["q_target"],
                    "poses": row.get("poses", {})} for row in ticks[::6]]
        text, _ = (ask or gemini.call)(["Refine skill boundaries without changing the task's command vocabulary. "
                                      "Return segments [t0,t1,command] as a contiguous partition of the same time range.\n"
                                      + json.dumps({"task": task, "vocabulary": vocab, "segments": segments,
                                                    "trace_5hz": summary})], role="main", tag="labeling:refine", schema=schema)
        segments = gemini.parse_json(text)["segments"]
    return _validate_segments(segments, ticks, vocab)


def label_trace(source, destination, task=None, *, refine=False, ask=None):
    from r2p.traces import write_trace

    meta, ticks, task = _read(source, task)
    intervals = segment((meta, ticks), task, refine=refine, ask=ask)
    meta = copy.deepcopy(meta)
    meta.update(commands=intervals, task=task, task_name=task["name"])
    result, index = [], 0
    for row in ticks:
        while index + 1 < len(intervals) and row["t"] >= intervals[index][1]:
            index += 1
        result.append({**row, "command": intervals[index][2]})
    write_trace(destination, meta, result)
    return intervals
