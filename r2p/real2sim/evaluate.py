"""Calibrated MuJoCo renders, replay and independent held-out measurements."""
from __future__ import annotations

import numpy as np


def centroid(mask):
    y, x = np.where(np.asarray(mask, bool))
    return np.array([x.mean(), y.mean()]) if len(x) else None


def mask_metrics(real, predicted):
    real, predicted = np.asarray(real, bool), np.asarray(predicted, bool)
    union = np.count_nonzero(real | predicted)
    iou = np.count_nonzero(real & predicted) / union if union else 1.0
    a, b = centroid(real), centroid(predicted)
    error = float(np.linalg.norm(a - b)) if a is not None and b is not None else (
        0.0 if a is None and b is None else float(np.hypot(*real.shape)))
    return {"iou": iou, "centroid_err_px": error}


class CalibratedRenderer:
    """CPU render of TaskTwin state through C2's calibrated mesh renderer."""
    def __init__(self, twin, camera):
        from r2p.model import Arm
        from r2p.synthbench.render import CpuRenderer
        self.twin, self.camera = twin, camera
        self.renderer = CpuRenderer(Arm(), camera)

    def render(self):
        import mujoco
        poses = self.twin.poses()
        for pose in poses.values():
            if "quat" in pose:
                matrix = np.empty(9)
                mujoco.mju_quat2Mat(matrix, np.asarray(pose["quat"], float))
                pose["R"] = matrix.reshape(3, 3)
        bgr, masks, _ = self.renderer.render(self.twin.q(), list(self.twin.objects.values()), poses, board=False)
        masks.pop("arm", None)
        return bgr, masks

    def close(self):
        self.renderer = None


def physics_replay(session, scene, *, indices=None, physics="so101_bench", twin_factory=None, render=True):
    """Replay q_target at 30 Hz; align captures to their recorded times, never to row number."""
    if twin_factory is None:
        from r2p.sim import TaskTwin
        twin_factory = TaskTwin
    indices = list(range(len(session.frames))) if indices is None else list(indices)
    q0 = session.ticks[0]["q"] if session.ticks else session.q(0)
    twin = twin_factory(scene, physics=physics, q0=q0)
    renderer = None
    if render and hasattr(twin, "m") and hasattr(twin, "body_ids"):
        renderer = CalibratedRenderer(twin, session.camera)
    snapshots = {}
    try:
        sim_t = session.ticks[0]["t"] if session.ticks else session.times[0]
        command_index = 0
        for index in sorted(indices, key=lambda i: session.times[i]):
            while session.ticks and sim_t + 1 / 30 <= session.times[index] + 1e-7:
                while command_index + 1 < len(session.ticks) and session.ticks[command_index + 1]["t"] <= sim_t + 1e-7:
                    command_index += 1
                twin.step(session.ticks[command_index]["q_target"])
                sim_t += 1 / 30
            item = {"poses": twin.poses(), "q": np.asarray(twin.q()).tolist(), "grip_closed": bool(twin.grip_closed())}
            if render:
                if renderer:
                    item["image"], item["masks"] = renderer.render()
                else:
                    # Injected twins expose a calibrated renderer for lightweight tests.
                    output = twin.render(session.camera)
                    if not isinstance(output, tuple) or len(output) != 2:
                        raise ValueError("An injected twin must render (BGR image, object masks)")
                    item["image"], item["masks"] = output
            snapshots[index] = item
    finally:
        if renderer:
            renderer.close()
        if hasattr(twin, "close"):
            twin.close()
    return snapshots


def anchor_metrics(scene, anchors):
    objects = {obj["id"]: obj for obj in scene["objects"]}
    errors, inside, details = [], [], []
    for anchor in anchors:
        obj = objects.get(anchor.get("object"))
        if obj is None:
            continue
        error = float(np.linalg.norm(np.asarray(obj["xyz"])[:2] - np.asarray(anchor["grasp_xyz"])[:2]))
        errors.append(error)
        row = {"object": obj["id"], "grasp_xy_err_m": error}
        container = objects.get(anchor.get("container"))
        if container and anchor.get("release_xyz") is not None:
            delta = np.asarray(anchor["release_xyz"])[:2] - np.asarray(container["xyz"])[:2]
            size = obj["size_m"]
            radius = max(size[:2]) / 2 if obj["shape"] in ("box", "plate", "stick") else size[0] / 2
            if container["shape"] == "bowl":
                contained = np.linalg.norm(delta) + radius <= container["size_m"][0] / 2
            else:
                yaw = np.deg2rad(container.get("yaw_deg", 0))
                local = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta
                contained = np.all(np.abs(local) + radius <= np.asarray(container["size_m"][:2]) / 2)
            inside.append(bool(contained))
            row.update(container=container["id"], release_inside=bool(contained))
        details.append(row)
    return {"anchor_error_m": float(np.mean(errors)) if errors else None,
            "release_inside": all(inside), "anchors": details}


def evaluate(session, scene, indices, appearances, snapshots, anchors=()):
    rows = {obj["id"]: [] for obj in scene["objects"]}
    trajectory = []
    for index in indices:
        image = session.frame(index)
        for oid in rows:
            real = session.masks.get(index, {}).get(oid)
            if real is None:
                real = appearances[oid].mask(image)
            predicted = snapshots[index]["masks"].get(oid, np.zeros(image.shape[:2], bool))
            rows[oid].append(mask_metrics(real, predicted))
            center = centroid(real)
            pose = snapshots[index]["poses"].get(oid)
            if center is not None and pose is not None:
                projection = np.asarray(session.camera.project(np.asarray(pose["xyz"]))).reshape(2)
                trajectory.append(float(np.linalg.norm(projection - center)))
    objects = {oid: {"iou": float(np.mean([row["iou"] for row in values])),
                     "centroid_err_px": float(np.mean([row["centroid_err_px"] for row in values])),
                     "n_frames": len(values)} for oid, values in rows.items()}
    traj_px = float(np.mean(trajectory)) if trajectory else None
    camera_errors = []
    for point in session.meta.get("camera_keypoints", []):
        if point["frame"] in indices:
            camera_errors.append(float(np.linalg.norm(session.camera.project(point["xyz"]) - point["uv"])))
    score = np.mean([row["iou"] - 0.005 * row["centroid_err_px"] for row in objects.values()])
    if traj_px is not None:
        score -= 0.004 * traj_px
    return {"score": float(score), "objects": objects, "camera_px": float(np.mean(camera_errors)) if camera_errors else None,
            "camera_locked": True, "physics": {"traj_err_px": traj_px, "replayed": bool(session.ticks)},
            **anchor_metrics(scene, anchors)}


def overlay(real, snapshot):
    import cv2

    marked = real.copy()
    for mask in snapshot["masks"].values():
        contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(marked, contours, -1, (0, 255, 255), 1)
    return np.concatenate([cv2.resize(marked, (640, 480)), cv2.resize(snapshot["image"], (640, 480))], axis=1)


def physics_image(session, indices, snapshots, appearances):
    import cv2

    last = indices[-1]
    real = session.frame(last).copy()
    for oid in snapshots[last]["poses"]:
        measured, predicted = [], []
        for index in indices:
            mask = session.masks.get(index, {}).get(oid)
            if mask is None:
                mask = appearances[oid].mask(session.frame(index))
            center = centroid(mask)
            if center is not None:
                measured.append(center)
            pose = snapshots[index]["poses"].get(oid)
            if pose:
                predicted.append(session.camera.project(pose["xyz"]))
        for points, color in ((measured, (0, 255, 0)), (predicted, (0, 0, 255))):
            if len(points) >= 2:
                cv2.polylines(real, [np.asarray(points).reshape(-1, 2).round().astype(np.int32)], False, color, 2)
    return np.concatenate([cv2.resize(real, (640, 480)), cv2.resize(snapshots[last]["image"], (640, 480))], axis=1)
