"""Offline fits. Returned offsets remain unverified until bench validation."""
import numpy as np
from .io import vector


SAG_POSES = [(0.18, -20, .05), (0.18, 20, .10), (0.24, -20, .05), (0.24, 0, .10),
             (0.24, 20, .15), (0.30, -15, .05), (0.30, 0, .10), (0.30, 15, .15)]


def fit_sag(samples, arm):
    x, y = [], []
    for sample in samples:
        command, measured = vector(sample["commanded"]), vector(sample["measured"])
        x.extend(-np.asarray(arm.gravity_torque(command))[1:4])
        y.extend((measured-command)[1:4])
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 6 or x @ x < 1e-8:
        raise ValueError("insufficient gravity excitation")
    k = float(x @ y / (x @ x))
    if k < 0 or not np.isfinite(k):
        raise ValueError("fit has the wrong sag sign")
    return {"k": k, "n": len(samples), "rms_rad": float(np.sqrt(np.mean((y-k*x)**2)))}


def fit_touch(samples, arm, joint_map):
    from scipy.optimize import least_squares
    from r2p.model import JointMap
    if len(samples) < 6:
        raise ValueError("capture at least six distinct table touches")

    def residual(offset):
        jm = JointMap(joint_map.sign, offset, joint_map.gc, joint_map.go)
        return np.concatenate([arm.tips(jm.to_model(sample["owner_deg"]))[0] - np.asarray(sample["xyz"])
                               for sample in samples])

    fit = least_squares(residual, joint_map.offset, bounds=(-180, 180), loss="soft_l1", f_scale=.003)
    rank = int(np.linalg.matrix_rank(fit.jac))
    return {"offset_deg": fit.x.tolist(), "rms_m": float(np.sqrt(np.mean(residual(fit.x)**2))),
            "rank": rank, "usable": bool(fit.success and rank == 5), "verified": False, "n": len(samples)}


def stable_pose(read, *, hz=30, seconds=1, sleep=None):
    import time
    sleep = sleep or time.sleep
    values = []
    for _ in range(int(hz*seconds)):
        values.append(vector(read()))
        sleep(1/hz)
    values = np.asarray(values)
    if np.any(np.ptp(values[:, :5], axis=0) > .5):
        raise ValueError("arm moved more than 0.5 degrees during capture")
    return np.median(values, axis=0)
