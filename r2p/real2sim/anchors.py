"""Grasp evidence from commands, with measured jaw geometry (never q percentiles)."""
from __future__ import annotations

import copy
import numpy as np


def gripper_events(ticks, *, close_tolerance=0.04, open_delta=0.12):
    if not ticks:
        return []
    commanded = np.asarray([row["q_target"][5] for row in ticks], dtype=float)
    if not np.isfinite(commanded).all():
        raise ValueError("Non-finite gripper commands")
    empty_closed = float(commanded.min())
    if np.ptp(commanded) < open_delta:
        return []  # no observed open/close cycle, so no invented grasp
    events = []
    closed = False
    for index, row in enumerate(ticks):
        value = commanded[index]
        if not closed and value <= empty_closed + close_tolerance:
            events.append({"kind": "close", "index": index, "t": row["t"], "empty_closed_rad": empty_closed})
            closed = True
        elif closed and value >= empty_closed + open_delta:
            events.append({"kind": "release", "index": index, "t": row["t"], "empty_closed_rad": empty_closed})
            closed = False
    return events


def extract_anchors(ticks, *, arm=None, min_hold_s=0.4, object_id=None, container_id=None):
    if arm is None:
        from r2p.model import Arm
        arm = Arm()
    events = gripper_events(ticks)
    anchors = []
    for offset, event in enumerate(events):
        if event["kind"] != "close":
            continue
        release = next((item for item in events[offset + 1:] if item["kind"] == "release"), None)
        end = release["index"] if release else len(ticks)
        held = []
        for i in range(event["index"], end):
            if ticks[i]["t"] < event["t"] + min_hold_s or ticks[i]["q"][5] <= event["empty_closed_rad"] + 0.03:
                continue
            window = [row["q"][5] for row in ticks[event["index"]:i+1] if row["t"] >= ticks[i]["t"] - min_hold_s]
            if len(window) >= 3 and np.ptp(window) <= 0.04:
                held.append(i)
        if not held:
            continue  # commanded closed is evidence of intent, not evidence of holding an object
        first = event["index"]
        sample = held[0]
        a, b, _ = arm.tips(ticks[sample]["q"])
        empty = np.asarray(ticks[sample]["q"], dtype=float).copy()
        empty[5] = event["empty_closed_rad"]
        ea, eb, _ = arm.tips(empty)
        width = max(0.0, float(np.linalg.norm(np.asarray(a) - b) - np.linalg.norm(np.asarray(ea) - eb)))
        # Grasp position belongs to the close event, before the lift; width to the settled hold.
        _, _, grasp = arm.tips(ticks[first]["q"])
        release_xyz = None
        if release:
            _, _, release_xyz = arm.tips(ticks[max(first, release["index"] - 1)]["q"])
            release_xyz = np.asarray(release_xyz).tolist()
        anchors.append({"object": object_id, "container": container_id, "grasp_t": event["t"],
                        "grasp_index": first, "hold_index": sample, "grasp_xyz": np.asarray(grasp).tolist(),
                        "width_m": width, "empty_closed_rad": event["empty_closed_rad"],
                        "release_t": release["t"] if release else None, "release_xyz": release_xyz,
                        "source": "gripper_commands_and_measured_jaw_gap"})
    return anchors


anchors_from_commands = extract_anchors


def apply_anchors(scene, anchors, *, mode="auto", pin_pose=True):
    """Associate recorded events, constrain initial poses and measured widths.

    Soft anchors are reported only. Later patches keep their proposed poses so
    the no-regression rule can reject an attempted move away from a grasp.
    """
    if mode not in ("auto", "hard", "soft"):
        raise ValueError("Unknown anchor mode")
    scene = copy.deepcopy(scene)
    objects = scene["objects"]
    pinned = set()
    for anchor in anchors:
        if not anchor.get("object"):
            candidates = [obj for obj in objects if obj.get("role") == "manipulated"]
            if candidates:
                anchor["object"] = min(candidates, key=lambda obj: np.linalg.norm(
                    np.asarray(obj["xyz"])[:2] - np.asarray(anchor["grasp_xyz"])[:2]))["id"]
        if not anchor.get("container") and anchor.get("release_xyz") is not None:
            candidates = [obj for obj in objects if obj.get("role") == "container"]
            if candidates:
                anchor["container"] = min(candidates, key=lambda obj: np.linalg.norm(
                    np.asarray(obj["xyz"])[:2] - np.asarray(anchor["release_xyz"])[:2]))["id"]
        obj = next((obj for obj in objects if obj["id"] == anchor.get("object")), None)
        if obj is None:
            anchor["enforced"] = False
            continue
        discrepancy = float(np.linalg.norm(np.asarray(obj["xyz"])[:2] - np.asarray(anchor["grasp_xyz"])[:2]))
        anchor.setdefault("image_disagreement_m", discrepancy)
        anchor["enforced"] = mode == "hard" or (mode == "auto" and anchor["image_disagreement_m"] <= 0.025)
        if not anchor["enforced"]:
            continue
        if pin_pose and obj["id"] not in pinned:
            obj["xyz"][:2] = anchor["grasp_xyz"][:2]
            pinned.add(obj["id"])
        width = anchor["width_m"]
        if width > 0:
            dimension = 1 if obj["shape"] in ("box", "plate", "stick") else 0
            obj["size_m"][dimension] = float(np.clip(obj["size_m"][dimension], width * .8, width * 1.2))
    return scene
