"""Camera geometry in metres in the footprint frame, and BGR image sources.

Extrinsics follow OpenCV: camera_xyz = R @ spec_xyz + t. Image size is
(width, height). A failed source never replaces a good frame with a blank one;
age_s therefore also exposes stalled cameras to the runner.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    image_size: tuple[int, int] = (640, 480)

    def __post_init__(self):
        self.K = np.asarray(self.K, float).reshape(3, 3)
        self.dist = np.asarray(self.dist, float).ravel()
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.t = np.asarray(self.t, float).reshape(3)
        self.image_size = tuple(map(int, self.image_size))
        if not all(np.isfinite(v).all() for v in (self.K, self.dist, self.R, self.t)):
            raise ValueError("Camera calibration must be finite")
        if min(self.image_size) <= 0 or self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("Invalid camera dimensions or focal length")
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-5) or np.linalg.det(self.R) < 0:
            raise ValueError("R must be a proper rotation")

    @property
    def position(self):
        return -self.R.T @ self.t

    @property
    def width(self):
        return self.image_size[0]

    @property
    def height(self):
        return self.image_size[1]

    def project(self, xyz):
        p = np.asarray(xyz, float)
        uv, _ = cv2.projectPoints(p.reshape(-1, 3), cv2.Rodrigues(self.R)[0], self.t, self.K, self.dist)
        return uv.reshape(p.shape[:-1] + (2,))

    def deproject(self, uv, h=0.0):
        """Intersect pixel rays with z=h; reject parallel/behind-camera rays."""
        p = np.asarray(uv, float)
        ray = cv2.undistortPoints(p.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        ray = np.column_stack((ray, np.ones(len(ray)))) @ self.R
        if np.any(np.abs(ray[:, 2]) < 1e-10):
            raise ValueError("Ray is parallel to the requested plane")
        distance = (np.asarray(h) - self.position[2]) / ray[:, 2]
        if np.any(distance <= 0):
            raise ValueError("Requested plane is behind the camera")
        xyz = self.position + distance[:, None] * ray
        return xyz.reshape(p.shape[:-1] + (3,))

    def resized(self, size):
        size = tuple(map(int, size))
        scale = np.array(size) / self.image_size
        K = self.K.copy()
        K[0] *= scale[0]
        K[1] *= scale[1]
        return CameraModel(K, self.dist.copy(), self.R.copy(), self.t.copy(), size)

    def to_dict(self):
        return {"schema": "r2p_camera/1", "frame": "spec", "K": self.K.tolist(),
                "dist": self.dist.tolist(), "R": self.R.tolist(), "t": self.t.tolist(),
                "image_size": list(self.image_size)}

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**metadata, **self.to_dict()}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("frame", "spec") != "spec":
            raise ValueError("Only spec-frame camera calibrations are supported")
        size = data.get("image_size", [data.get("width", 640), data.get("height", 480)])
        return cls(data["K"], data.get("dist", np.zeros(5)), data["R"], data["t"], size)

    @classmethod
    def look_at(cls, position=(0.55, -0.35, 0.55), target=(0.20, 0, 0),
                image_size=(640, 480), fovy_deg=48):
        position, target = np.asarray(position, float), np.asarray(target, float)
        z = target - position
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1])
        if np.linalg.norm(x) < 1e-9:
            x = np.array([0., 1., 0.])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.array([x, y, z])
        w, h = image_size
        f = h / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        return cls([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.zeros(5), R, -R @ position, image_size)


def _size(cfg):
    return int(cfg.get("width", 640)), int(cfg.get("height", 480))


class _ThreadCamera:
    def __init__(self, cfg):
        self.size = _size(cfg)
        self._frame = None
        self._captured = 0.0
        self.error = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._period = 1 / max(float(cfg.get("fps", 30)), 0.1)
        self._thread = threading.Thread(target=self._loop, daemon=True, name=type(self).__name__)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                frame, source_age = self._capture()
                if frame is None or frame.size == 0:
                    raise OSError("Camera returned no image")
                if frame.shape[1::-1] != self.size:
                    frame = cv2.resize(frame, self.size)
                with self._lock:
                    self._frame = frame
                    self._captured = time.monotonic() - max(0.0, source_age)
                    self.error = None
                self._ready.set()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(max(0, self._period - (time.monotonic() - start)))

    @property
    def age_s(self):
        with self._lock:
            return time.monotonic() - self._captured if self._frame is not None else float("inf")

    def read(self, timeout=3.0):
        if not self._ready.wait(timeout):
            raise TimeoutError(f"No camera frame: {self.error}")
        with self._lock:
            return self._frame.copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LiveCamera(_ThreadCamera):
    def __init__(self, cfg=None, **kwargs):
        cfg = {**(cfg or {}), **kwargs}
        backend = cfg.get("backend", "AVFOUNDATION" if sys.platform == "darwin" else "ANY")
        if isinstance(backend, str):
            backend = getattr(cv2, "CAP_" + backend.upper().removeprefix("CAP_"))
        self.capture = cv2.VideoCapture(int(cfg.get("index", 0)), backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise OSError(f"Could not open camera {cfg.get('index', 0)}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, _size(cfg)[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, _size(cfg)[1])
        self.capture.set(cv2.CAP_PROP_FPS, cfg.get("fps", 30))
        super().__init__(cfg)

    def _capture(self):
        ok, frame = self.capture.read()
        if not ok:
            raise OSError("Camera read failed")
        return frame, 0.0

    def close(self):
        super().close()
        self.capture.release()


class HttpCamera(_ThreadCamera):
    def __init__(self, cfg=None, **kwargs):
        cfg = {**(cfg or {}), **kwargs}
        self.url = cfg.get("url", "http://127.0.0.1:8792/frame/{index}.jpg").format(index=cfg.get("index", 0))
        self.timeout = float(cfg.get("timeout_s", 2.0))
        super().__init__(cfg)

    def _capture(self):
        start = time.monotonic()
        with urlopen(self.url, timeout=self.timeout) as response:
            frame = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
            age = float(response.headers.get("X-Frame-Age", "0"))
        return frame, age + time.monotonic() - start


class StillCamera:
    def __init__(self, image, size=None):
        self.frame = cv2.imread(str(image)) if isinstance(image, (str, Path)) else np.asarray(image).copy()
        if self.frame is None:
            raise FileNotFoundError(image)
        if size is not None:
            self.frame = cv2.resize(self.frame, tuple(size))
        self.age_s = 0.0

    def read(self, **kwargs):
        return self.frame.copy()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SynthCamera(StillCamera):
    def __init__(self, world):
        self.world = world
        self.age_s = 0.0

    def read(self, **kwargs):
        return self.world.render()

    def close(self):
        self.world.close()


def open_camera(cfg):
    """Open a camera config, or a bench config with a nested camera section."""
    cfg = cfg.get("camera", cfg)
    source = cfg.get("source", "device").lower()
    if source in ("http", "camserver"):
        return HttpCamera(cfg)
    if source in ("device", "live"):
        return LiveCamera(cfg)
    if source in ("still", "image"):
        return StillCamera(cfg.get("image", cfg.get("path")), _size(cfg))
    if source in ("synth", "synthetic"):
        from r2p.synthbench import World
        world = cfg.get("world")
        if world is None:
            model = cfg.get("cam_model") or CameraModel.look_at(image_size=_size(cfg))
            world = World(cfg["task"], np.random.default_rng(cfg.get("seed", 0)), model)
        return SynthCamera(world)
    raise ValueError(f"Unknown camera source: {source}")
