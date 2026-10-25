"""Frames, units and conventions shared by every component (docs/SPEC.md section 3). ONE frame everywhere:

spec frame = robot base FOOTPRINT centre on the table surface, metres, x forward (the arm reaches +x at q=0), y left,
z up. The official MJCF (assets/so101/so101_new_calib.xml) has its base origin at BASE_ORIGIN_MJCF in this frame's
coordinates offset: spec = mjcf - BASE_ORIGIN_MJCF.
Roblox: 1 m = STUDS_PER_M studs, spec (x, y, z) -> Roblox (X, Y, Z) = (x, z, -y) * STUDS_PER_M + table origin.
Joint vectors are MJCF radians in JOINTS order (gripper last).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "assets" / "so101" / "so101_new_calib.xml"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BASE_ORIGIN_MJCF = np.array([0.0168, 0.0, -0.0024])
HZ = 30                     # control and logging rate
HORIZON = 15                # action chunk length (absolute joint targets, MJCF rad)
STUDS_PER_M = 10.0

# fingertip points (body, local position) on the official meshes; the jaw midpoint is the grasp point
FIXED_TIP = ("gripper", np.array([-0.0089, -0.0002, -0.1044]))
MOVING_TIP = ("moving_jaw_so101_v1", np.array([-0.0115, -0.0819, 0.0195]))

# where every expert trace starts (fingers ~12 cm above the table); exports declare their own start_q
HOME_Q = np.array([-5e-05, -0.06786, -0.28181, 1.65806, 0.0, 0.9])


def mjcf_to_spec(p):
    return np.asarray(p, float) - BASE_ORIGIN_MJCF


def spec_to_mjcf(p):
    return np.asarray(p, float) + BASE_ORIGIN_MJCF


def spec_to_roblox(p, table_top_y=7.5):
    """spec point (m) -> Roblox world (studs) with the spec origin at (0, table_top_y, 0)."""
    x, y, z = np.asarray(p, float)
    return np.array([x * STUDS_PER_M, table_top_y + z * STUDS_PER_M, -y * STUDS_PER_M])
