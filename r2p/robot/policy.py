"""Policy chunk validation and ACT temporal ensembling."""
from collections import deque
from pathlib import Path
import json
import numpy as np


def check_chunk(chunk):
    chunk = np.asarray(chunk, float)
    if chunk.shape != (15, 6) or not np.isfinite(chunk).all():
        raise ValueError("policy must return finite absolute targets with shape (15, 6)")
    return chunk


class Ensembler:
    def __init__(self, m=0.1):
        self.m = float(m)
        self.chunks = deque()

    def add(self, tick, chunk):
        self.chunks.append((tick, check_chunk(chunk).copy()))

    def target(self, tick):
        while self.chunks and self.chunks[0][0] + 15 <= tick:
            self.chunks.popleft()
        values = [c[tick - t] for t, c in self.chunks if 0 <= tick - t < len(c)]
        if not values:
            raise ValueError("no policy prediction covers this tick")
        # Insertion order is oldest first, which intentionally gets rank zero.
        weights = np.exp(-self.m * np.arange(len(values)))
        return np.average(values, axis=0, weights=weights)

    def clear(self):
        self.chunks.clear()


def policy_config(policy, name=None):
    cfg = getattr(policy, "config", None)
    if cfg is None and name is not None:
        path = Path(name)
        if not path.is_dir():
            from .config import ROOT
            path = ROOT / "exports" / name
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "start_q" not in cfg:
        raise ValueError("policy export must declare config.start_q")
    return cfg


def load_policy(name):
    from r2p.policy import load
    return load(name)
