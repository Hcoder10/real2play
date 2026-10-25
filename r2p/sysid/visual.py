"""Visual joint-map calibration. Collection is injectable; this module has no robot transport.

Offsets are fitted in owner degrees using the shared JointMap/Arm convention.
The camera stays fixed. Reports are candidates, never changes to the active map.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from r2p import gemini
from r2p.model import Arm, JointMap


TIPS_SCHEMA = {"type": "object", "required": ["fixed_tip", "moving_tip", "visible"], "properties": {
    name: {"type": "array", "minItems": 2, "maxItems": 2,
           "items": {"type": "number", "minimum": 0, "maximum": 1000}}
    for name in ("fixed_tip", "moving_tip")}}
TIPS_SCHEMA["properties"]["visible"] = {"type": "boolean"}


def plan(*, xs=(.15, .21, .27), ys=(-.08, 0., .08), zs=(.11, .14)):
    """Propose above-table positions; the mover must check reachability and its safety envelope."""
    points = np.array([[x, y, z] for z in zs for x in xs for y in ys], float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Plan needs finite xyz points")
    if np.any(points[:, 2] < .10) or np.any(np.linalg.norm(points[:, :2], axis=1) > .35):
        raise ValueError("Visual calibration proposals must be >= 10 cm high and within 35 cm reach")
    return {"schema": "r2p_visual_plan/1", "frame": "spec", "execute": False,
            "poses": [{"k": i, "cmd_xyz": p.tolist()} for i, p in enumerate(points)],
            "requires": ["guarded mover", "settled measured owner_deg", "fresh post-settle camera frame"]}


def collect(proposal, *, mover, capture, out_dir, clock=time.time):
    """mover(xyz) -> {ok, owner_deg, settled_at?}; capture(after_t) -> (BGR, captured_at).

    The injected mover owns hardware authorization, guards, settling and cleanup.
    Each successful capture is persisted immediately. No transport is implicit.
    """
    if proposal.get("schema") != "r2p_visual_plan/1" or proposal.get("frame") != "spec":
        raise ValueError("Expected r2p_visual_plan/1 in spec frame")
    rows = []
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "poses.json").exists():
        raise FileExistsError("Use a fresh collection directory")
    for entry in proposal["poses"]:
        xyz = np.asarray(entry["cmd_xyz"], float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or xyz[2] < .10 or np.linalg.norm(xyz[:2]) > .35:
            raise ValueError("Unsafe or invalid proposed calibration point")
        result = mover(xyz.tolist())
        if not result.get("ok"):
            rows.append({**entry, "status": "refused", "reason": result.get("reason", "mover refused")})
        else:
            q = np.asarray(result["owner_deg"], float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError("Mover must return six finite measured owner degrees")
            settled = float(result.get("settled_at", clock()))
            image, captured = capture(settled)
            if not np.isfinite([settled, captured]).all() or captured <= settled:
                raise ValueError("Capture is stale: it must follow settling")
            filename = f"p{int(entry['k']):02d}.jpg"
            gemini.image_part(image)  # validate before writing
            from r2p.real2sim.loop import atomic_image
            atomic_image(destination / filename, image)
            rows.append({**entry, "owner_deg": q.tolist(), "frame": filename,
                         "settled_at": settled, "captured_at": captured, "status": "collected"})
        gemini.atomic_json(destination / "poses.json", rows)
    return rows


def point(source, *, out_path=None, ask=None, camera=None, joint_map=None, force=False):
    """Point once per saved pose with MAIN; persist each reply and reuse completed rows/cache."""
    source = Path(source)
    rows = json.loads(source.read_text(encoding="utf-8"))
    destination = Path(out_path) if out_path else source.with_name("pointed.json")
    if destination.exists() and not force:
        completed = {r["k"]: r for r in json.loads(destination.read_text(encoding="utf-8"))}
        rows = [completed.get(r["k"], r) for r in rows]
    arm = Arm() if camera is not None and joint_map is not None else None
    for row in rows:
        if row.get("status") == "refused" or not force and "tips_px" in row and "visible" in row:
            continue
        image = cv2.imread(str(source.parent / row.get("frame", f"p{row['k']:02d}.jpg")))
        if image is None:
            raise ValueError(f"Missing calibration frame for pose {row['k']}")
        h, w = image.shape[:2]
        prompt = ("Locate the two actual finger ENDPOINTS of this SO-101 gripper, not the hinge, wrist, "
                  "side edges or shadows. Both tips are at the distal end of the black fingers. "
                  "Return fixed_tip and moving_tip as [y,x] normalized 0..1000 in the FIRST full image. "
                  "If either endpoint is occluded or ambiguous set visible=false. Finger identity may be swapped; "
                  "prefer accurate endpoints over guessing identity.")
        parts = [prompt, gemini.image_part(image)]
        if arm is not None:
            cam = camera if camera.image_size == (w, h) else camera.resized((w, h))
            tips = cam.project(np.array(arm.tips(joint_map.to_model(row["owner_deg"]))[:2]))
            if np.isfinite(tips).all():
                lo = np.maximum(np.floor(tips.min(0) - 110), [0, 0]).astype(int)
                hi = np.minimum(np.ceil(tips.max(0) + 110), [w, h]).astype(int)
                if np.all(hi - lo > 20):
                    parts += [f"Second image is a detail crop of full-image pixels x={lo[0]}..{hi[0]}, "
                              f"y={lo[1]}..{hi[1]}. Still return coordinates in the FIRST image.",
                              gemini.image_part(image[lo[1]:hi[1], lo[0]:hi[0]])]
        text, record = (ask or gemini.call)(parts, role="main", tag="visual:point", schema=TIPS_SCHEMA)
        answer = gemini.parse_json(text)
        errors = gemini.schema_errors(answer, TIPS_SCHEMA)
        if errors:
            raise gemini.JSONError("; ".join(errors))
        row.update(visible=answer["visible"], model=record.get("served_by"), record=record,
                   image_size=[w, h], tips_px={name: [answer[name][1] * w / 1000, answer[name][0] * h / 1000]
                                             for name in ("fixed_tip", "moving_tip")})
        gemini.atomic_json(destination, rows)
    gemini.atomic_json(destination, rows)
    return rows


def fit(rows, camera, joint_map, *, arm=None, f_scale=4., max_offset_deg=45., gate_px=5.):
    """Soft-L1 fit with independent leave-one-pose-out refits, including finger swaps.

    All five arm offsets are fitted; signs and gripper endpoints are preserved.
    LOO predictions are generated without using that pose in fitting or selection.
    """
    if isinstance(rows, (str, Path)):
        rows = json.loads(Path(rows).read_text(encoding="utf-8"))
    usable = [r for r in rows if r.get("visible") and "tips_px" in r]
    if len(usable) < 6:
        raise ValueError("At least six visible poses are required for leave-one-out calibration")
    if min(f_scale, max_offset_deg, gate_px) <= 0:
        raise ValueError("Fit scale, bounds and gate must be positive")
    arm = arm or Arm()
    start = np.asarray(joint_map.offset, float)
    owner = np.array([r["owner_deg"] for r in usable], float)
    pixels = np.array([[r["tips_px"][n] for n in ("fixed_tip", "moving_tip")] for r in usable], float)
    if owner.shape != (len(usable), 6) or pixels.shape != (len(usable), 2, 2) or not np.isfinite(owner).all() or not np.isfinite(pixels).all():
        raise ValueError("Calibration poses need six finite owner degrees and two finite pixel points")
    # Rows may have been captured at another resolution; retain the calibrated
    # intrinsics and scale observations into that image's coordinate system.
    for i, row in enumerate(usable):
        size = np.asarray(row.get("image_size", camera.image_size), float)
        if size.shape != (2,) or np.any(size <= 0) or not np.isfinite(size).all():
            raise ValueError("Invalid observation image size")
        pixels[i] *= np.asarray(camera.image_size) / size

    def mapping(offset):
        return JointMap(sign=joint_map.sign.copy(), offset=np.asarray(offset), gc=joint_map.gc, go=joint_map.go)

    def residual(offset, indices):
        jm = mapping(offset)
        result = []
        for i in indices:
            uv = camera.project(np.array(arm.tips(jm.to_model(owner[i]))[:2]))
            direct, swapped = uv - pixels[i], uv - pixels[i, ::-1]
            result.extend((direct if np.sum(direct**2) <= np.sum(swapped**2) else swapped).ravel())
        return np.array(result)

    lower, upper = start - max_offset_deg, start + max_offset_deg
    lower[4], upper[4] = start[4] - 180, start[4] + 180

    def optimize(indices):
        candidates = []
        for roll in (0, 90, -90, 179):
            seed = np.clip(start + [0, 0, 0, 0, roll], lower + 1e-6, upper - 1e-6)
            result = least_squares(lambda x: residual(x, indices), seed, bounds=(lower, upper),
                                   loss="soft_l1", f_scale=f_scale, max_nfev=250, diff_step=1e-4)
            candidates.append(result)
        return min(candidates, key=lambda r: r.cost)

    indices = list(range(len(usable)))
    solution = optimize(indices)
    loo = []
    for i, row in enumerate(usable):
        held = optimize([j for j in indices if j != i])
        error = residual(held.x, [i]).reshape(2, 2)
        before = residual(start, [i]).reshape(2, 2)
        loo.append({"k": row["k"], "tip_rms_px": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                    "midpoint_px": float(np.linalg.norm(error.mean(0))),
                    "before_tip_rms_px": float(np.sqrt(np.mean(np.sum(before**2, axis=1)))),
                    "offset_deg": held.x.tolist(), "converged": bool(held.success)})
    train = np.linalg.norm(residual(solution.x, indices).reshape(-1, 2), axis=1)
    errors = np.array([r["tip_rms_px"] for r in loo])
    baseline = np.array([r["before_tip_rms_px"] for r in loo])
    rank = int(np.linalg.matrix_rank(solution.jac))
    passed = bool(solution.success and all(r["converged"] for r in loo) and rank == 5
                  and np.percentile(errors, 90) <= gate_px and np.mean(errors**2) <= np.mean(baseline**2) + 1e-8)
    from r2p.frames import JOINTS
    candidate = {"joints": {name: {"sign": float(joint_map.sign[i]), "offset_deg": float(solution.x[i]),
                                    "measured": False} for i, name in enumerate(JOINTS[:5])},
                 "gripper": {"closed_rad": joint_map.gc, "open_rad": joint_map.go},
                 "source": "visual soft-L1 candidate; physical verification is still required"}
    return {"schema": "r2p_visual_fit/1", "method": "soft_l1", "n_poses": len(usable),
            "start_offset_deg": start.tolist(), "offset_deg": solution.x.tolist(), "candidate": candidate,
            "train_tip_rms_px": float(np.sqrt(np.mean(train**2))), "loo": loo,
            "loo_tip_rms_px": float(np.sqrt(np.mean(errors**2))), "loo_median_px": float(np.median(errors)),
            "loo_p90_px": float(np.percentile(errors, 90)), "before_tip_rms_px": float(np.sqrt(np.mean(baseline**2))),
            "jacobian_rank": rank, "gate_px": gate_px, "accepted": passed, "verified": False,
            "limitations": "Pixel reprojection only; does not establish millimetre accuracy or hardware safety."}


def report(result, out_dir):
    destination = Path(out_dir)
    gemini.atomic_json(destination / "fit.json", result)
    text = (f"Visual calibration: {result['n_poses']} poses\n\n"
            f"LOO tip RMS: {result['loo_tip_rms_px']:.2f} px; median {result['loo_median_px']:.2f} px; "
            f"p90 {result['loo_p90_px']:.2f} px.\n"
            f"Initial tip RMS: {result['before_tip_rms_px']:.2f} px.\n"
            f"Gate (p90 <= {result['gate_px']:.2f} px): {'PASS' if result['accepted'] else 'FAIL'}.\n\n"
            f"Offsets (degrees): {result['offset_deg']}\n\n{result['limitations']}\n"
            "The candidate has not been activated or physically verified.\n")
    (destination / "report.md").write_text(text, encoding="utf-8")
    return destination / "report.md"
