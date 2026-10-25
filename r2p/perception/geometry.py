"""Known-size object silhouette fitting, all outputs in the footprint frame."""
from __future__ import annotations

import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares


def dimensions(obj):
    size = np.asarray(obj.get("size_m", [.03, .03, .03]), float).ravel()
    shape = obj.get("shape", "box")
    if shape == "stick":
        return np.array([size[0], size[1], size[1]])
    if shape in ("cylinder", "bowl"):
        if obj.get("axis", "z") == "x":
            return np.array([size[1], size[0], size[0]])
        if obj.get("axis", "z") == "y":
            return np.array([size[0], size[1], size[0]])
        return np.array([size[0], size[0], size[1]])
    if shape == "sphere":
        return np.repeat(size[0], 3)
    return np.resize(size, 3)


def rotation(yaw_deg):
    a = np.deg2rad(yaw_deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def is_screwdriver(obj):
    return obj.get("shape") == "stick" and obj.get("appearance", "screwdriver") == "screwdriver"


def silhouette_points(obj, *, appearance=True):
    shape = obj.get("shape", "box")
    size = dimensions(obj)
    if shape in ("bowl", "cylinder") and obj.get("axis", "z") == "z":
        # A bowl's observed rim is a circle at its known top height. Include
        # the outer wall's lower edge, which is also coloured in real images.
        angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        radius = size[0]/2 + (obj.get("wall_m", .005) if shape == "bowl" else 0)
        ring = np.column_stack((radius*np.cos(angles), radius*np.sin(angles), np.zeros(64)))
        return np.vstack((ring + [0, 0, size[2]/2], ring - [0, 0, size[2]/2]))
    if shape == "sphere":
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 32), np.linspace(0, np.pi, 16))
        return size[0]/2 * np.column_stack((np.sin(phi).ravel()*np.cos(theta).ravel(),
                                            np.sin(phi).ravel()*np.sin(theta).ravel(), np.cos(phi).ravel()))
    offset = np.zeros(3)
    if appearance and is_screwdriver(obj):
        offset[0] = -.2 * size[0]
        size = size.copy()
        size[0] *= .6
    return np.array(list(itertools.product([-1, 1], repeat=3))) * size/2 + offset


def fit_pose(cam, obj, mask, bottom=0.0, previous=None):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No object silhouette")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(float)
    if len(hull) < 3:
        raise ValueError("Degenerate object silhouette")
    size = dimensions(obj)
    z = bottom + size[2]/2
    points = silhouette_points(obj)
    scale_bounds = obj.get("_size_scale_range")
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    observed = np.max(hull @ directions.T, axis=0)
    center = cam.deproject(hull.mean(axis=0), h=z)[:2]
    def residual(p):
        scale = p[3] if len(p) > 3 else 1.
        pixels = cam.project(scale*points @ rotation(p[2]).T + [p[0], p[1], bottom+scale*size[2]/2])
        return np.max(pixels @ directions.T, axis=0) - observed
    round_shape = obj.get("shape") in ("bowl", "sphere") or (obj.get("shape") == "cylinder" and obj.get("axis", "z") == "z")
    directed = is_screwdriver(obj)
    yaws = [0.] if round_shape else np.arange(-180. if directed else -90., 180. if directed else 90., 15.)
    hint = None if previous is None else previous.get("directed_yaw_deg", previous.get("yaw_deg"))
    warm = previous is not None and "xyz" in previous and previous.get("fit_px", 100) < 3
    if warm and not round_shape:
        yaws = [hint]
    seeds = []
    for yaw in yaws:
        xy = center.copy()
        if is_screwdriver(obj):
            xy += (rotation(yaw) @ [.2*size[0], 0, 0])[:2]
        p = np.array([*xy, yaw])
        if scale_bounds is not None:
            scale = (previous["size_m"][0]/obj["size_m"][0]) if warm else 1.
            p = np.append(p, np.clip(scale, *scale_bounds))
        penalty = 0.
        if directed and hint is not None:
            angle_error = (yaw-hint+180)%360-180
            penalty = .01*angle_error**2
        seeds.append((float(np.mean(residual(p)**2))+penalty, p))
    candidates = []
    for _, p in sorted(seeds, key=lambda item: item[0])[:1 if warm else 2]:
        if round_shape and scale_bounds is None:
            fit = least_squares(lambda xy: residual([*xy, 0]), p[:2], diff_step=1e-4, max_nfev=18)
            result = np.array([*fit.x, 0])
        else:
            kwargs = {} if scale_bounds is None else {"bounds": ([-np.inf, -np.inf, -np.inf, scale_bounds[0]],
                                                                   [np.inf, np.inf, np.inf, scale_bounds[1]])}
            fit = least_squares(residual, p, diff_step=1e-4, x_scale=[.1, .1, 45]+([1] if scale_bounds else []),
                                max_nfev=22, **kwargs)
            result = fit.x
        penalty = 0. if not directed or hint is None else .0001*((result[2]-hint+180)%360-180)**2
        candidates.append((float(np.mean(residual(result)**2))+penalty, result))
    error, best = min(candidates, key=lambda item: item[0])
    # The handle gives a directed stick; store the specified undirected yaw.
    yaw = float((best[2]+90) % 180-90) if not round_shape else 0.0
    scale = float(best[3]) if len(best) > 3 else 1.
    half_angle = np.deg2rad(best[2])/2
    return {"xyz": [float(best[0]), float(best[1]), float(bottom+size[2]*scale/2)], "yaw_deg": yaw,
            "quat": [float(np.cos(half_angle)), 0., 0., float(np.sin(half_angle))],
            "directed_yaw_deg": float((best[2]+180)%360-180),
            "size_m": (np.asarray(obj["size_m"])*scale).tolist(), "fit_px": float(np.sqrt(error))}
