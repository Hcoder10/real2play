"""Two calibrated MuJoCo cameras; replay recorded poses without changing physics."""
from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from r2p.frames import JOINTS, ROOT
from r2p.sim.twin import build_xml, load_physics


def numbers(x):
    return " ".join(str(float(v)) for v in x)


class Cameras:
    def __init__(self, scene, physics="so101_bench", camera=None, mounts=None):
        self.camera = camera or json.loads((ROOT/"config/camera.json").read_text())
        self.mounts = mounts or json.loads(Path(__file__).with_name("cameras.json").read_text())
        root = ET.fromstring(build_xml(scene, load_physics(physics)))
        world = root.find("worldbody")
        cfg = self.camera
        R, t, K = np.asarray(cfg["R"]), np.asarray(cfg["t"]), np.asarray(cfg["K"])
        rotation = R.T @ np.diag([1, -1, -1])
        width, height = cfg["image_size"]
        ET.SubElement(world, "camera", name="vla_scene", pos=numbers(-R.T@t),
                      xyaxes=numbers(np.r_[rotation[:, 0], rotation[:, 1]]),
                      resolution=f"{width} {height}", sensorsize="1 1",
                      focal=numbers([K[0, 0]/width, K[1, 1]/height]),
                      principal=numbers([.5-K[0, 2]/width, .5-K[1, 2]/height]))
        mount = self.mounts["wrist"]
        ET.SubElement(world.find(f".//body[@name='{mount['parent_body']}']"), "camera", name="vla_wrist",
                      pos=numbers(mount["pos"]), quat=numbers(mount["quat"]), fovy=str(mount["fovy_deg"]))
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self.renderer = mujoco.Renderer(self.m, self.mounts["height"], self.mounts["width"])
        self.scene = scene

    def render(self, q, poses):
        self.d.qpos[self.qadr] = q
        for obj in self.scene["objects"]:
            if obj.get("fixed") or not obj.get("present", True) or obj["id"] not in poses:
                continue
            pose = poses[obj["id"]]
            adr = self.m.joint("free:"+obj["id"]).qposadr[0]
            self.d.qpos[adr:adr+3] = pose["xyz"]
            if "quat" in pose:
                self.d.qpos[adr+3:adr+7] = pose["quat"]
            else:
                angle = np.deg2rad(pose.get("yaw_deg", 0))/2
                self.d.qpos[adr+3:adr+7] = [np.cos(angle), 0, 0, np.sin(angle)]
        mujoco.mj_forward(self.m, self.d)
        output = []
        for name in ("vla_scene", "vla_wrist"):
            self.renderer.update_scene(self.d, camera=name)
            output.append(self.renderer.render().copy())
        return np.stack(output)

    def close(self):
        self.renderer.close()
