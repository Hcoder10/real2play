"""Constant-rate movies from timestamped, disk-backed camera samples."""
from pathlib import Path
import math

import cv2

from r2p.gemini import frames_to_video


def timeline_frames(samples, *, start, end, fps=30):
    """Hold the most recent capture; do not compress camera stalls out of time."""
    samples = sorted(samples, key=lambda item: item[0])
    if not samples:
        raise ValueError("No video samples")
    index, loaded, frame = 0, None, None
    for tick in range(max(1, math.ceil((end-start)*fps)+1)):
        t = start+tick/fps
        while index+1 < len(samples) and samples[index+1][0] <= t+1e-9:
            index += 1
        source = samples[index][1]
        if loaded != index:
            frame = cv2.imread(str(source)) if isinstance(source, (str, Path)) else source
            if frame is None:
                raise ValueError(f"Unreadable video frame: {source}")
            loaded = index
        yield frame


def timeline_video(samples, path, *, start, end, fps=30):
    return frames_to_video(timeline_frames(samples, start=start, end=end, fps=fps), path, fps=fps)
