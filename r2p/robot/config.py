"""Bench configuration, with recursive platform overlays."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def overlay(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        result[key] = overlay(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path=None, platform=None):
    path = Path(path) if path else ROOT / "config" / "bench.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    extra = path.with_name(f"{path.stem}.{platform or sys.platform}{path.suffix}")
    if extra.exists():
        cfg = overlay(cfg, json.loads(extra.read_text(encoding="utf-8")))
    return cfg


def config_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_joint_map(cfg=None):
    from r2p.model import JointMap
    cfg = cfg or load_config()
    return JointMap.load(config_path(cfg.get("joint_map", "config/joint_map.json")))


def sag_gain(cfg=None):
    cfg = cfg or load_config()
    path = config_path(f"config/physics/{cfg.get('rig', 'so101_bench')}.json")
    if path.exists():
        physics = json.loads(path.read_text(encoding="utf-8"))
        if "sag_gain" in physics:
            import numpy as np
            gains = np.asarray(physics["sag_gain"], float)
            if gains.shape == (6,) and np.allclose(gains[1:4], gains[1]):
                return float(gains[1])
        for key in ("gravity_comp_k", "gravity_sag_k", "sag_k"):
            if key in physics:
                return float(physics[key])
        if isinstance(physics.get("gravity_sag"), dict):
            return float(physics["gravity_sag"].get("k", cfg["gravity_comp_k"]))
    return float(cfg.get("gravity_comp_k", 0.05))
