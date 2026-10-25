"""Torch-free RPC client for Argon and the real runner's SessionPolicy adapter."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np


class RemotePolicy:
    def __init__(self, url="http://127.0.0.1:9106", timeout=10):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.config = self._request("/config")

    def _request(self, route, value=None):
        payload = json.dumps(value, allow_nan=False).encode() if value is not None else None
        request = Request(self.url+route, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def reset(self):
        self._request("/reset", {})

    def act(self, obs):
        import cv2
        payload = {k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
                   for k, v in obs.items() if k in ("q", "q_measured", "tick", "command_age_s", "command", "target_cell")}
        payload["images"] = {}
        for key in ("scene", "wrist"):
            rgb = obs["images"][key]
            ok, image = cv2.imencode(".png", rgb[..., ::-1])
            if not ok:
                raise ValueError("image encode failed")
            payload["images"][key] = base64.b64encode(image).decode()
        reply = self._request("/act", payload)
        from r2p.robot.policy import check_chunk
        if reply["target_space"] != "servo" or reply["action"] != "absolute_q" or reply["hz"] != 30:
            raise ValueError("incompatible inference server")
        return check_chunk(reply["chunk"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:9106")
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--wrist", required=True, type=Path)
    p.add_argument("--state", required=True, type=Path, help="JSON: q, q_measured, tick, optional command_age_s")
    p.add_argument("--cell", required=True)
    p.add_argument("--out", required=True, type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--argon", action="store_true")
    source.add_argument("--command")
    a = p.parse_args()
    import cv2
    from .adapter import argon_command, prepare_images
    policy = RemotePolicy(a.url)
    scene, wrist = cv2.imread(str(a.scene)), cv2.imread(str(a.wrist))
    if scene is None or wrist is None:
        raise ValueError("scene and wrist image files must be readable")
    state = json.loads(a.state.read_text())
    command = argon_command(scene, a.cell, state) if a.argon else a.command
    obs = {**state, "command": command, "target_cell": a.cell,
           "images": prepare_images(scene, wrist, policy.config["camera"])}
    chunk = policy.act(obs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"command": command, "chunk": chunk.tolist(), "hz": 30, "target_space": "servo",
                                "motion_sent": False}, indent=2))
    print(f"Wrote 15 absolute servo targets to {a.out}")


if __name__ == "__main__":
    main()
