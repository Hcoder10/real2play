"""Argon command source and camera adapter for the existing guarded Session."""
from __future__ import annotations
import json
import numpy as np
from .contracts import instruction


def argon_command(frame_bgr, target_cell, status, *, ask=None):
    """One budgeted MAIN request at a subtask boundary; injectable/cache-only in tests."""
    from r2p import gemini
    allowed = ["pick(piece)", f"place_at(piece,{target_cell})", "release", "home"]
    schema = {"type": "object", "properties": {"command": {"type": "string", "enum": allowed}}, "required": ["command"]}
    prompt = ("Control one tic-tac-toe cap placement. Issue one command from the allowed vocabulary. "
              "Pick before placing; place_at includes lowering, opening and retreating. Use home after successful placement. "
              "Never invent joint angles. Return JSON only.\n"+json.dumps({"allowed": allowed, "target_cell": target_cell, "status": status}))
    response = (ask or gemini.call)([prompt, gemini.image_part(frame_bgr)], role="main", tag="vla:command", schema=schema)
    if isinstance(response, tuple):
        response, _record = response
    value = response.get("json") if isinstance(response, dict) else None
    if value is None:
        text = response.get("text") if isinstance(response, dict) else response
        value = gemini.parse_json(text)
    command = value["command"]
    if command not in allowed:
        raise ValueError("Argon returned an unsupported VLA command")
    instruction(command, target_cell)  # same translator as the training labels
    return command


def prepare_images(scene_bgr, wrist_bgr, camera, width=256, height=192):
    """Rectify the scene camera and match the calibrated training image transform."""
    import cv2
    if tuple(scene_bgr.shape[1::-1]) != tuple(camera["image_size"]):
        raise ValueError("scene camera resolution differs from calibration")
    K = np.asarray(camera["K"])
    scene = cv2.undistort(scene_bgr, K, np.asarray(camera.get("dist", [0]*5)))
    return {"scene": cv2.resize(scene, (width, height))[..., ::-1].copy(),
            "wrist": cv2.resize(wrist_bgr, (width, height))[..., ::-1].copy()}


class SessionPolicy:
    """Adapter passed to r2p.robot.runner.Session; Argon owns set_command().

    observe() must run once per control tick, with that tick's measured joints and
    synchronized camera frames. set_command() is called only by the high-level
    controller. There is no hidden replacement of an explicit Argon command.
    """
    def __init__(self, policy, target_cell, command="pick(piece)"):
        self.policy, self.config, self.target_cell = policy, policy.config, target_cell
        self.observation = None
        self.set_command(command)

    def set_command(self, command):
        instruction(command, self.target_cell)
        self.command = command

    def observe(self, measured_q, scene_bgr, wrist_bgr, tick):
        self.observation = {"q_measured": np.asarray(measured_q), "tick": int(tick),
                            "images": prepare_images(scene_bgr, wrist_bgr, self.config["camera"])}

    def reset(self):
        self.observation = None
        self.policy.reset()

    def act(self, obs):
        if self.observation is None:
            raise ValueError("VLA requires current scene/wrist frames and measured joints")
        return self.policy.act({**obs, **self.observation, "command": self.command, "target_cell": self.target_cell})
