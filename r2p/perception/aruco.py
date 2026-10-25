"""Printed-board calibration, with measured board placement in the spec frame."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .camera import CameraModel


@dataclass
class BoardConfig:
    grid: tuple[int, int] = (4, 3)
    marker_length_m: float = 0.04
    separation_m: float = 0.01
    dictionary: str = "DICT_4X4_50"
    # Marker 0 top-left: measured from footprint centre, NOT the MJCF origin.
    origin_xyz: tuple[float, float, float] = (0.12, -0.095, 0.0)
    yaw_deg: float = 0.0

    @property
    def dimensions(self):
        return np.asarray(self.grid) * (self.marker_length_m + self.separation_m) - self.separation_m

    def board(self):
        return cv2.aruco.GridBoard(tuple(self.grid), self.marker_length_m, self.separation_m,
                                  cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary)))

    def to_spec(self, points):
        # Printed top row toward robot. Columns run +y, rows run +x.
        p = np.asarray(points, float)
        xyz = p[..., [1, 0, 2]].copy()
        xyz[..., 2] *= -1
        a = np.deg2rad(self.yaw_deg)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        return xyz @ rot.T + self.origin_xyz

    @classmethod
    def from_dict(cls, cfg):
        cfg = cfg.get("board", cfg)
        args = {k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg}
        if "origin_xyz" not in args and "dist_from_base_front_m" in cfg:
            if "base_front_x_m" not in cfg:
                raise ValueError("Board front-face offset needs measured base_front_x_m from footprint centre")
            args["origin_xyz"] = (cfg["base_front_x_m"] + cfg["dist_from_base_front_m"],
                                  cfg.get("left_edge_y_m", -0.095), cfg.get("board_z_m", 0.0))
            right = cfg.get("dist_from_base_front_right_m")
            if right is not None:
                width = cls(**args).dimensions[0]
                args["yaw_deg"] = -np.rad2deg(np.arctan2(right - cfg["dist_from_base_front_m"], width))
        return cls(**args)


def detect(frame, cfg=None):
    cfg = cfg or BoardConfig()
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cfg.board().getDictionary(), parameters)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def correspondences(frame, cfg=None, *, spec=False):
    cfg = cfg or BoardConfig()
    corners, ids = detect(frame, cfg)
    if ids is None:
        raise ValueError("No board markers detected")
    board = cfg.board()
    lookup = dict(zip(board.getIds().ravel(), board.getObjPoints()))
    pairs = [(lookup[int(i)], c.reshape(4, 2)) for c, i in zip(corners, ids.ravel()) if int(i) in lookup]
    if len(pairs) < 2:
        raise ValueError("At least two board markers must be visible")
    xyz, uv = (np.concatenate(v).astype(np.float32) for v in zip(*pairs))
    return (cfg.to_spec(xyz).astype(np.float32) if spec else xyz), uv


def calibrate_intrinsics(frames, cfg=None):
    """Fit intrinsics from 8-12 tilted views (more views are also accepted)."""
    frames = list(frames)
    if len(frames) < 8:
        raise ValueError("Capture at least 8 tilted board views (8-12 recommended)")
    size = frames[0].shape[1::-1]
    xyzs, uvs = [], []
    for frame in frames:
        if frame.shape[1::-1] != size:
            raise ValueError("Calibration views must have identical image size")
        xyz, uv = correspondences(frame, cfg)
        xyzs.append(xyz)
        uvs.append(uv)
    # Fix unsupported high-order radial terms; retain tangential and k1/k2.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        xyzs, uvs, size, None, None, flags=cv2.CALIB_FIX_K3)
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rvecs])
    if np.max(np.linalg.norm(normals - normals.mean(axis=0), axis=1)) < 0.06:
        raise ValueError("Board views have too little tilt diversity; recapture tilted views")
    per_view = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(x, r, t, K, dist)[0].reshape(-1, 2) - u)**2, axis=1))))
                for x, u, r, t in zip(xyzs, uvs, rvecs, tvecs)]
    return CameraModel(K, dist.ravel(), image_size=size), {"reprojection_px": float(rms), "per_view_px": per_view}


def calibrate_extrinsics(frame, intrinsics, cfg=None):
    cfg = cfg or BoardConfig()
    if frame.shape[1::-1] != intrinsics.image_size:
        raise ValueError("Extrinsic image size differs from intrinsic calibration")
    xyz, uv = correspondences(frame, cfg, spec=True)
    ok, rvec, tvec = cv2.solvePnP(xyz, uv, intrinsics.K, intrinsics.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Board pose could not be solved")
    rvec, tvec = cv2.solvePnPRefineLM(xyz, uv, intrinsics.K, intrinsics.dist, rvec, tvec)
    cam = CameraModel(intrinsics.K, intrinsics.dist, cv2.Rodrigues(rvec)[0], tvec, intrinsics.image_size)
    error = float(np.sqrt(np.mean(np.sum((cam.project(xyz) - uv)**2, axis=1))))
    if cam.position[2] <= cfg.origin_xyz[2]:
        raise ValueError("Estimated camera is under the board; check board orientation")
    forward = cam.R[2]
    report = {"reprojection_px": error, "position_m": cam.position.tolist(),
              "height_m": float(cam.position[2]),
              "elevation_deg": float(np.rad2deg(np.arcsin(np.clip(-forward[2], -1, 1))))}
    return cam, report


def footprint_overlay(frame, cam, size_m=(0.10, 0.08)):
    """Footprint outline and +x/+y axes for a human calibration sanity check."""
    out = frame.copy()
    x, y = np.asarray(size_m) / 2
    corners = cam.project([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]])
    cv2.polylines(out, [np.rint(corners).astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    points = np.rint(cam.project([[0, 0, 0], [.08, 0, 0], [0, .08, 0]])).astype(int)
    for end, color, label in ((points[1], (0, 0, 255), "+x"), (points[2], (0, 255, 0), "+y")):
        cv2.arrowedLine(out, tuple(points[0]), tuple(end), color, 2)
        cv2.putText(out, label, tuple(end), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return out


def board_image(cfg=None, pixels_per_m=10000):
    """Black marker extent only; dimensions and marker/gap ratios are exact."""
    cfg = cfg or BoardConfig()
    size = tuple(np.rint(cfg.dimensions * pixels_per_m).astype(int))
    return cfg.board().generateImage(size, marginSize=0, borderBits=1)


def make_board(path, cfg=None, dpi=300):
    """US Letter PNG, physical marker dimensions, 300 dpi, print at 100%."""
    cfg = cfg or BoardConfig()
    ppm = dpi / 0.0254
    width, height = round(8.5 * dpi), round(11 * dpi)
    board = Image.fromarray(board_image(cfg, ppm)).convert("RGB")
    if board.width > width - 2 * round(.25 * dpi) or board.height > height - 2 * round(.25 * dpi):
        raise ValueError("Board does not fit US Letter at 100% scale")
    page = Image.new("RGB", (width, height), "white")
    origin = ((width - board.width) // 2, (height - board.height) // 2)
    page.paste(board, origin)
    draw = ImageDraw.Draw(page)
    draw.text((origin[0], origin[1] - 70), "TOP ROW TOWARD ROBOT", fill="black")
    draw.text((origin[0], origin[1] + board.height + 30),
              f"Print 100% / actual size. Markers: {cfg.marker_length_m * 1000:g} mm. Verify with ruler.", fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path, dpi=(dpi, dpi))
    return page


def render_board(cam, cfg=None, *, board_R=None, board_t=None, background=210):
    """CPU calibration fixture: homography of printed markers, optional tilted pose.

    board_R/t map board coordinates to spec coordinates. Distortion is applied
    after rendering the undistorted view, so calibration tests exercise it too.
    """
    cfg = cfg or BoardConfig()
    w, h = cfg.dimensions
    texture = board_image(cfg)
    src = np.float32([[0, 0], [texture.shape[1], 0], [texture.shape[1], texture.shape[0]], [0, texture.shape[0]]])
    points = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]])
    xyz = cfg.to_spec(points) if board_R is None else points @ np.asarray(board_R).T + board_t
    pinhole = CameraModel(cam.K, np.zeros(5), cam.R, cam.t, cam.image_size)
    matrix = cv2.getPerspectiveTransform(src, pinhole.project(xyz).astype(np.float32))
    result = cv2.warpPerspective(texture, matrix, cam.image_size, flags=cv2.INTER_LINEAR, borderValue=background)
    if np.any(cam.dist):
        u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
        uv = np.stack((u, v), -1).astype(np.float32)
        mapped = cv2.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.dist, P=cam.K).reshape(uv.shape)
        result = cv2.remap(result, mapped[..., 0], mapped[..., 1], cv2.INTER_LINEAR, borderValue=background)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
