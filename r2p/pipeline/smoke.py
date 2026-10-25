"""Explicit synthetic fixtures for the offline integration run, never real evidence."""
import copy
import json
from pathlib import Path


def twin_inputs(session):
    """Use recorded synthetic scene/masks; run the actual physical replay and metrics."""
    from r2p.model import Arm
    from r2p.real2sim.session import split_frames
    from r2p.synthbench.render import CpuRenderer
    import mujoco
    import numpy as np
    scene = copy.deepcopy(session.meta["scene"])
    renderer = CpuRenderer(Arm(), session.camera)
    critic, heldout = split_frames(session)
    for index in critic + heldout:
        tick = session.ticks[session.tick_index(index)]
        poses = copy.deepcopy(tick["poses"])
        for pose in poses.values():
            if "quat" in pose:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, np.asarray(pose["quat"], float))
                pose["R"] = rotation.reshape(3, 3)
        _, masks, _ = renderer.render(tick["q"], scene["objects"], poses, board=True)
        session.masks[index] = {oid: mask for oid, mask in masks.items() if oid != "arm"}

    def oracle(*args, **kwargs):
        return json.dumps({"objects": [], "done": True, "why": "Offline smoke fixture; no model critique"}), {
            "model": "synthetic-oracle", "served_by": "synthetic-oracle", "cached": False,
            "attempts": 0, "latency": 0}
    return {"scene": scene, "ask": oracle, "anchors": "soft"}


def hold_export(task, directory):
    """A named fixed-hold baseline when the offline run has no incumbent yet."""
    import shutil
    import numpy as np
    from r2p.policy.features import export_config
    from r2p.policy import runtime
    from r2p.policy.eval import evaluate
    from r2p.model import Arm
    from r2p.robot.guard import Guard
    from r2p.robot.gravity import GravityComp
    from .storage import atomic_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = export_config(task, directory.name, width=1, blocks=0)
    arm = Arm()
    limits = Guard(arm).limits
    start = np.clip(config["start_q"], *limits.T)
    config["start_q"] = start.tolist()
    target = np.clip(GravityComp(arm)(start), *limits.T)
    config.update(synthetic=True, purpose="Fixed-hold baseline for offline smoke gate; not a trained policy")
    dim = config["input_dim"]
    norm = {"x_mean": [0.] * dim, "x_std": [1.] * dim,
            "y_mean": target.tolist() * 15, "y_std": [1.] * 90}
    atomic_json(directory / "config.json", config)
    atomic_json(directory / "norm.json", norm)
    np.savez_compressed(directory / "weights.npz", **{
        "member0.input.weight": np.zeros((1, dim)), "member0.input.bias": np.zeros(1),
        "member0.output.weight": np.zeros((90, 1)), "member0.output.bias": np.zeros(90)})
    shutil.copyfile(runtime.__file__, directory / "runtime.py")
    for noise in ("clean", "real", "realx0.5"):
        evaluate(directory, n=1, noise=noise, max_ticks=60)
    return str(directory)
