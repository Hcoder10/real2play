"""Appearance-tunable composite prototype and held-out frame metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RenderParams:
    albedo: dict[str, list[float]] = field(default_factory=dict)  # RGB multiplier by object id / arm
    roughness: float = .65
    light_direction: list[float] = field(default_factory=lambda: [-.4, -.3, 1.])
    light_intensity: float = .3
    light_color: list[float] = field(default_factory=lambda: [1., 1., 1.])
    ambient: float = .7
    shadow_softness_px: float = 3.
    shadow_strength: float = .22
    exposure: float = 0.  # stops
    white_balance: list[float] = field(default_factory=lambda: [1., 1., 1.])  # RGB
    blur_sigma: float = 0.
    noise_std: float = 0.  # 8-bit sensor units
    jpeg_quality: int = 100


class CompositeRenderer:
    def __init__(self, world, background, params=None, rng=None):
        self.world = world
        self.background = cv2.imread(str(background)) if isinstance(background, (str, Path)) else np.asarray(background).copy()
        if self.background is None:
            raise FileNotFoundError(background)
        self.params = RenderParams(**params) if isinstance(params, dict) else (params or RenderParams())
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.masks = {}
        self.shadow = None

    def render(self, q=None, poses=None):
        if q is not None:
            # Replay rendering must not invent contact events.
            q = np.asarray(q, float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError("Expected six finite recorded joints")
            self.world._q = q.copy()
        if poses is not None:
            for oid, pose in poses.items():
                self.world.set_pose(oid, pose["xyz"], pose.get("yaw_deg", 0))
        frame, self.masks, self.shadow = self.world.render_layers(background=self.background, board=False, params=self.params)
        p = self.params
        foreground = np.logical_or.reduce([m.astype(bool) for m in self.masks.values()])
        # A real background already contains its exposure/WB. Tune rendered pixels
        # before adding common sensor effects, retaining the photographed workspace.
        transformed = frame.astype(float) * 2**p.exposure * np.asarray(p.white_balance)[::-1]
        frame[foreground] = np.clip(transformed[foreground], 0, 255).astype(np.uint8)
        if p.blur_sigma > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), p.blur_sigma)
        if p.noise_std > 0:
            frame = np.clip(frame.astype(float)+self.rng.normal(0, p.noise_std, frame.shape), 0, 255).astype(np.uint8)
        if p.jpeg_quality < 100:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(p.jpeg_quality)])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return frame

    def compare(self, real_frame, real_masks, rendered=None):
        rendered = self.render() if rendered is None else rendered
        return compare_frames(rendered, real_frame, self.masks, real_masks)


def compare_frames(rendered, real, rendered_masks, real_masks):
    if rendered.shape != real.shape:
        raise ValueError("Comparison images must have the same calibrated shape")
    lab_a = cv2.cvtColor(rendered.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    lab_b = cv2.cvtColor(real.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    error = np.linalg.norm(lab_a-lab_b, axis=-1)
    per_object = {}
    for oid in sorted(set(rendered_masks) | set(real_masks)):
        a = np.asarray(rendered_masks.get(oid, np.zeros(real.shape[:2])), bool)
        b = np.asarray(real_masks.get(oid, np.zeros(real.shape[:2])), bool)
        union, intersection = a | b, a & b
        iou = float(intersection.sum()/union.sum()) if union.any() else 1.
        lab = float(error[intersection].mean()) if intersection.any() else (0. if not union.any() else 100.)
        per_object[oid] = {"iou": iou, "lab_error": lab}
    mean_iou = float(np.mean([m["iou"] for m in per_object.values()])) if per_object else 1.
    mean_lab = float(np.mean([m["lab_error"] for m in per_object.values()])) if per_object else 0.
    return {"per_object": per_object, "iou": mean_iou, "lab_error": mean_lab,
            "loss": 1-mean_iou+mean_lab/100}
