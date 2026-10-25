"""Training-time export metadata. Runtime feature math lives in the NumPy file."""
from __future__ import annotations

import mujoco
import numpy as np

from r2p.frames import BASE_ORIGIN_MJCF, FIXED_TIP, HOME_Q, JOINTS, MOVING_TIP
from r2p.model import Arm
from r2p.tasks import command_vocab, commands, yaw_symmetry
from .runtime import expand_features, observation


def export_config(task, name, *, start_q=HOME_Q, seeds=(0,), width=512, blocks=3):
    arm = Arm()
    m = arm.m
    bodies = []
    for b in range(1, m.nbody):
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, m.body_quat[b])
        j = int(m.body_jntadr[b])
        bodies.append({"parent": int(m.body_parentid[b]), "pos": m.body_pos[b].tolist(),
                       "rotation": matrix.reshape(3, 3).tolist(), "joint": j,
                       "axis": m.jnt_axis[j].tolist() if j >= 0 else [0, 0, 1],
                       "anchor": m.jnt_pos[j].tolist() if j >= 0 else [0, 0, 0]})
    k = task["limits"]["max_objects"]
    c = len(command_vocab(task))
    return {"schema": "r2p_policy/1", "name": str(name), "task": task["name"], "task_spec": task,
            "K": k, "object_ids": [obj["id"] for obj in task["objects"]],
            "obs_frame": "spec_footprint", "obs_layout": [
                {"name": "commanded_q", "start": 0, "size": 6},
                {"name": "slots", "start": 6, "size": 9*k, "shape": [k, 9]},
                {"name": "command_onehot", "start": 6+9*k, "size": c},
                {"name": "jaw_midpoint_finger_axis", "start": 6+9*k+c, "size": 6},
                {"name": "yaw_symmetry", "start": 12+9*k+c, "size": 4*k}],
            "input_dim": 12+13*k+c, "command_vocab": command_vocab(task), "commands": commands(task),
            "start_q": np.asarray(start_q, float).tolist(), "obs_const": [], "horizon": 15,
            "hz": 30, "action": "absolute_q", "joints": list(JOINTS), "joint_limits": arm.range.tolist(),
            "target_space": "servo", "gravity_compensation": "included_in_expert_targets",
            "yaw_sym": [yaw_symmetry(obj) for obj in task["objects"]]+[0]*(k-len(task["objects"])),
            "fk": {"bodies": bodies, "origin": BASE_ORIGIN_MJCF.tolist(),
                   "tips": [{"body": int(m.body(body).id), "local": point.tolist()} for body, point in [FIXED_TIP, MOVING_TIP]]},
            "width": width, "blocks": blocks, "activation": "gelu_tanh", "seeds": list(seeds),
            "ensemble_size": len(seeds), "eval": {"clean": None, "real": None, "realx0.5": None, "n": 0}}


__all__ = ["export_config", "expand_features", "observation"]
