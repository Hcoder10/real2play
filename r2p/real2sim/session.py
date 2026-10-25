"""Recorded attempts and raw videos. Frames are decoded only when requested."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SessionData:
    path: Path
    meta: dict
    ticks: list[dict]
    frames: list[Path | int]
    times: list[float]
    camera: object | None = None
    video: Path | None = None
    fps: float = 30.0
    masks: dict = field(default_factory=dict)

    @property
    def kind(self):
        return self.meta.get("session_kind", "synthetic" if self.meta.get("source") == "synthetic" else "bench")

    def frame(self, index):
        import cv2

        ref = self.frames[index]
        if isinstance(ref, Path):
            image = cv2.imread(str(ref))
        else:
            capture = cv2.VideoCapture(str(self.video))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, ref)
                ok, image = capture.read()
                if not ok:
                    image = None
            finally:
                capture.release()
        if image is None:
            raise ValueError(f"Cannot decode frame {index} in {self.path}")
        return image

    def tick_index(self, index):
        if not self.ticks:
            return None
        times = np.array([row["t"] for row in self.ticks])
        return int(np.argmin(abs(times - self.times[index])))

    def q(self, index):
        if not self.ticks:
            from r2p.frames import HOME_Q
            return HOME_Q.copy()
        return np.asarray(self.ticks[self.tick_index(index)]["q"], dtype=float)

    def joint_table(self, hz=5):
        lines = ["t | measured q (MJCF rad) | commanded q_target (MJCF rad)"]
        next_t = -float("inf")
        for row in self.ticks:
            if row["t"] + 1e-8 >= next_t:
                lines.append(f"{row['t']:.3f} | {np.round(row['q'], 4).tolist()} | "
                             f"{np.round(row['q_target'], 4).tolist()}")
                next_t = row["t"] + 1 / hz
        return "\n".join(lines)

    def video_path(self, destination):
        if self.video and self.video.suffix.lower() in (".mp4", ".mov", ".webm"):
            return self.video
        from r2p.gemini import frames_to_video
        if self.video:
            import cv2
            def decoded():
                capture = cv2.VideoCapture(str(self.video))
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        yield frame
                finally:
                    capture.release()
            return frames_to_video(decoded(), destination, self.fps)
        # Resample by recorded timestamps, holding the last observation across a
        # dropped camera frame. Uniform frame indices would distort event timing.
        def timed():
            cursor, image = 0, self.frame(0)
            count = round((self.times[-1] - self.times[0]) * self.fps) + 1
            for tick in range(count):
                t = self.times[0] + tick / self.fps
                while cursor + 1 < len(self.frames) and self.times[cursor + 1] <= t + 1e-8:
                    cursor += 1
                    image = self.frame(cursor)
                yield image
        return frames_to_video(timed(), destination, self.fps)


def _load_trace(path):
    try:
        from r2p.traces import read_trace
    except ImportError:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or "meta" not in rows[0]:
            raise ValueError("Trace must begin with a meta line")
        meta, ticks = rows[0]["meta"], rows[1:]
    else:
        meta, ticks = read_trace(path)
    if meta.get("schema") != "r2p_trace/1":
        raise ValueError("Expected r2p_trace/1")
    previous = -float("inf")
    for row in ticks:
        if not np.isfinite(row["t"]) or row["t"] <= previous:
            raise ValueError("Trace timestamps must be strictly increasing")
        for key in ("q", "q_target"):
            q = np.asarray(row[key], dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError(f"{key} must contain six finite MJCF radians")
        previous = row["t"]
    return meta, list(ticks)


def load_session(source, *, camera=None):
    if isinstance(source, SessionData):
        return source
    import cv2

    path = Path(source)
    camera_path = Path(camera) if isinstance(camera, (str, Path)) else path / "camera.json"
    if camera is None and camera_path.is_file() or isinstance(camera, (str, Path)):
        from r2p.perception import CameraModel
        camera = CameraModel.load(camera_path)
    if path.is_file() and path.suffix.lower() in (".mp4", ".mov", ".webm", ".avi"):
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if not np.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError("Video has no readable frames or timestamps")
        # Sample for fitting, but the task writer receives the original WHOLE video.
        indices = sorted(set(range(0, count, max(1, round(fps)))) | {count - 1})
        return SessionData(path, {"session_kind": "video"}, [], indices, [i / fps for i in indices], camera, path, fps)
    if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return SessionData(path, {"session_kind": "image"}, [], [path], [0.0], camera)
    trace_paths = [path / name for name in ("trace.jsonl", "joints.jsonl", "attempt.jsonl")]
    trace_path = next((item for item in trace_paths if item.exists()), None)
    if trace_path is None:
        candidates = sorted(path.glob("*.jsonl"))
        if len(candidates) == 1:
            trace_path = candidates[0]
        else:
            raise FileNotFoundError(f"No unambiguous r2p_trace/1 file in {path}")
    meta, ticks = _load_trace(trace_path)
    if (path / "meta.json").exists():
        meta.update(json.loads((path / "meta.json").read_text(encoding="utf-8")))
    frames, times = [], []
    for row in ticks:
        reference = row.get("frame")
        if reference is None:
            continue
        if isinstance(reference, int):
            candidates = [path / "frames" / f"{reference:06d}.jpg", path / "frames" / f"{reference:04d}.jpg"]
        else:
            candidates = [path / str(reference), path / "frames" / str(reference)]
        filename = next((candidate for candidate in candidates if candidate.is_file()), None)
        if filename is None:
            raise FileNotFoundError(f"Referenced session frame is missing: {reference}")
        if filename not in frames:
            frames.append(filename)
            times.append(float(row["t"]))
    if not frames:
        frames = sorted(p for p in (path / "frames").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        frame_times = meta.get("frame_times")
        if frame_times is not None and len(frame_times) == len(frames):
            times = list(map(float, frame_times))
        elif len(frames) == len(ticks):
            times = [float(row["t"]) for row in ticks]
        elif "frame_fps" in meta:
            times = [float(meta.get("frame_start_t", 0)) + i / meta["frame_fps"] for i in range(len(frames))]
        elif frames:
            raise ValueError("Frame/joint counts differ: supply frame references, frame_times or frame_fps")
    if not frames:
        raise ValueError("Session has no frames")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be increasing")
    video = next(iter(sorted(path.glob("*.mp4"))), None)
    return SessionData(path, meta, ticks, frames, times, camera, video, float(meta.get("fps", 30)))


def split_frames(session, count=8):
    if len(session.frames) < 2:
        raise ValueError("Twin fitting requires at least two frames for a held-out evaluation")
    indices = np.unique(np.linspace(0, len(session.frames) - 1, min(count, len(session.frames)), dtype=int)).tolist()
    return indices[::2], indices[1::2]
