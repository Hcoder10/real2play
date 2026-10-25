"""A better aggregate score never licenses a visible or physical regression."""
from __future__ import annotations

import math


def keep_candidate(candidate, best, object_bests=None, *, containers=(), anchors_enforced=True,
                   gain=0.003, camera_slack_px=0.0, anchor_slack_m=0.0):
    reasons = []
    object_bests = object_bests or {key: row["iou"] for key, row in best.get("objects", {}).items()}
    if not math.isfinite(candidate.get("score", float("nan"))):
        reasons.append("nonfinite_score")
    elif candidate["score"] - best["score"] <= gain + 1e-12:
        reasons.append("insufficient_gain")
    for key, peak in object_bests.items():
        value = candidate.get("objects", {}).get(key, {}).get("iou")
        if value is None or not math.isfinite(value) or value < peak - 0.10 - 1e-12:
            reasons.append(f"iou_regression:{key}")
        if not anchors_enforced and key in containers:
            previous = best.get("objects", {}).get(key, {}).get("iou", peak)
            if value is None or value < previous - 0.02 - 1e-12:
                reasons.append(f"container_iou_regression:{key}")
    for field, slack, label in (("camera_px", camera_slack_px, "camera_regression"),):
        if best.get(field) is not None:
            value = candidate.get(field)
            if value is None or not math.isfinite(value) or value > best[field] + slack + 1e-12:
                reasons.append(label)
    if anchors_enforced:
        if best.get("anchor_error_m") is not None:
            value = candidate.get("anchor_error_m")
            if value is None or not math.isfinite(value) or value > best["anchor_error_m"] + anchor_slack_m + 1e-12:
                reasons.append("anchor_regression")
        if candidate.get("release_inside") is not True:
            reasons.append("release_outside_container")
    return {"kept": not reasons, "verdict": "KEPT" if not reasons else "REJECTED",
            "reasons": reasons}


keep = keep_candidate
