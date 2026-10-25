"""Official SO-101 inertias/kinematics with compliant servos and contact-only grasps.

The world itself is the spec frame. Only the imported robot base is translated.
The two non-convex finger meshes use explicit narrow collision pads: a single
convex hull of a complete jaw fills its open mouth and cannot model a grasp.
There are no welds, attachment heuristics, object teleports or gravity cancellation.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from r2p.frames import BASE_ORIGIN_MJCF, HOME_Q, HZ, JOINTS, MJCF, ROOT


def load_physics(physics="so101_bench"):
    if isinstance(physics, dict):
        return deepcopy(physics)
    path = Path(physics)
    if not path.is_file():
        path = ROOT / "config" / "physics" / f"{physics}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _vec(x):
    return " ".join(str(float(v)) for v in x)


def _rgba(color):
    if isinstance(color, (list, tuple)):
        return _vec(list(color[:3]) + [color[3] if len(color) > 3 else 1])
    colors = {"red": "#d03030", "blue": "#306dc8", "brown": "#a76c3b", "green": "#479859",
              "yellow": "#eac536", "white": "#eeeeee", "black": "#222222", "orange": "#e78b35"}
    s = colors.get(color, color or "#aaaaaa").lstrip("#")
    if len(s) != 6:
        s = "aaaaaa"
    return _vec([int(s[i:i+2], 16) / 255 for i in (0, 2, 4)] + [1])


def extents(obj):
    """Full local XYZ extents, including the task's square-prism stick."""
    s = np.asarray(obj["size_m"], float)
    shape = obj["shape"]
    if shape == "sphere":
        return np.repeat(s[0], 3)
    if shape == "stick":
        return np.array([s[0], s[1], s[1]])
    if shape in ("cylinder", "bowl"):
        result = np.array([s[0], s[0], s[1]])
        if shape == "cylinder" and obj.get("axis") in ("x", "y"):
            result = np.roll(result, 1 if obj["axis"] == "x" else 2)
        return result
    return s


def build_xml(scene, physics):
    root = ET.parse(MJCF).getroot()
    root.find("compiler").set("meshdir", str(MJCF.parent / "assets"))
    ET.SubElement(root, "option", timestep=str(1 / HZ / physics.get("substeps", 16)),
                  gravity=_vec(physics.get("gravity", [0, 0, -9.81])), integrator="implicitfast",
                  cone="elliptic", iterations="60", impratio=str(physics["contact"].get("impratio", 10)), tolerance="1e-9")
    ET.SubElement(root, "size", njmax="4000", nconmax="1000")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    world = root.find("worldbody")
    world.find("body[@name='base']").set("pos", _vec(-BASE_ORIGIN_MJCF))
    contact = physics["contact"]
    common = dict(solref=_vec(contact["solref"]), solimp=_vec(contact["solimp"]), condim="4")
    for geom in world.iter("geom"):
        if geom.get("class") == "collision":
            geom.attrib.update(common)
            geom.set("friction", _vec(contact["friction"]))
            # Arm-object/table collisions, but no adjacent convex hull self contacts.
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
    # Finger pads follow the official mesh dimensions and the actual moving hinge.
    for body_name, mesh in [("gripper", "wrist_roll_follower_so101_v1"),
                            ("moving_jaw_so101_v1", "moving_jaw_so101_v1")]:
        body = world.find(f".//body[@name='{body_name}']")
        for geom in list(body.findall("geom")):
            if geom.get("class") == "collision" and geom.get("mesh") == mesh:
                body.remove(geom)
    fixed = world.find(".//body[@name='gripper']")
    moving = world.find(".//body[@name='moving_jaw_so101_v1']")
    pad = dict(type="box", contype="2", conaffinity="1", friction="1.2 0.008 0.0001",
               group="3", rgba="0.25 0.25 0.25 1", mass="0", **common)
    ET.SubElement(fixed, "geom", name="fixed_finger", pos="-0.0133 -0.0002 -0.0964",
                  size="0.0044 0.0062 0.008", **pad)
    ET.SubElement(moving, "geom", name="moving_finger", pos="-0.0071 -0.0739 0.0195",
                  size="0.0044 0.008 0.0062", **pad)
    # Narrow stems preserve collisions above each fingertip without a convex bridge.
    ET.SubElement(fixed, "geom", name="fixed_stem", pos="-0.0133 -0.0002 -0.069",
                  size="0.0044 0.0062 0.0194", **pad)
    ET.SubElement(moving, "geom", name="moving_stem", pos="-0.0071 -0.043 0.0195",
                  size="0.0044 0.0229 0.0062", **pad)
    for name in JOINTS:
        j = physics["joints"][name]
        node = world.find(f".//joint[@name='{name}']")
        node.attrib.update(damping="0", armature=str(j["armature"]), frictionloss=str(j["frictionloss"]))
        actuator = root.find(f"actuator/position[@name='{name}']")
        actuator.attrib.update(kp=str(j["kp"]), kv=str(j["kv"]),
                               forcerange=_vec([-j["torque_limit"], j["torque_limit"]]))
    table = scene.get("table", {})
    ET.SubElement(world, "geom", name="table", type="plane", pos="0 0 0", size="0.8 0.6 0.02",
                  rgba=_rgba(table.get("color", "#b08a5a")), contype="1", conaffinity="3",
                  friction=_vec([table.get("friction", 0.6), 0.005, 0.0001]), **common)
    ET.SubElement(world, "light", pos="0.1 -0.3 1.2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    for obj in scene.get("objects", []):
        if not obj.get("present", True):
            continue
        name, shape = obj["id"], obj["shape"]
        s = np.asarray(obj["size_m"], float)
        theta = np.deg2rad(obj.get("yaw_deg", 0)) / 2
        body = ET.SubElement(world, "body", name=f"object:{name}", pos=_vec(obj["xyz"]),
                             quat=_vec(obj.get("quat", [np.cos(theta), 0, 0, np.sin(theta)])))
        if not obj.get("fixed", False):
            ET.SubElement(body, "freejoint", name=f"free:{name}")
        friction = obj.get("friction", physics["object"]["friction"])
        if np.isscalar(friction):
            friction = [friction, 0.005, 0.0001]
        kw = dict(rgba=_rgba(obj.get("color")), mass=str(obj.get("mass_kg", physics["object"]["mass_kg"])),
                  contype="1" if obj.get("collide", True) else "0",
                  conaffinity="3" if obj.get("collide", True) else "0", friction=_vec(friction), **common)
        if shape == "bowl":
            wall = obj.get("wall_m", 0.005)
            radius, height = s / 2
            kw["mass"] = str(float(kw["mass"]) / 17)
            ET.SubElement(body, "geom", name=f"{name}:base", type="cylinder", size=_vec([radius+wall, wall/2]),
                          pos=_vec([0, 0, -height+wall/2]), **kw)
            for i in range(16):
                phi = i * 2 * np.pi / 16
                r = radius + wall / 2
                ET.SubElement(body, "geom", name=f"{name}:wall:{i}", type="box",
                              pos=_vec([r*np.cos(phi), r*np.sin(phi), wall/2]),
                              euler=_vec([0, 0, phi]), size=_vec([wall/2, (radius+wall)*np.tan(np.pi/16), height-wall/2]), **kw)
        elif shape == "sphere":
            ET.SubElement(body, "geom", type="sphere", size=str(s[0]/2), **kw)
        elif shape == "cylinder":
            axis = obj.get("axis", "z")
            quat = {"x": [np.sqrt(.5), 0, np.sqrt(.5), 0],
                    "y": [np.sqrt(.5), np.sqrt(.5), 0, 0], "z": [1, 0, 0, 0]}[axis]
            ET.SubElement(body, "geom", type="cylinder", size=_vec(s/2), quat=_vec(quat), **kw)
        elif shape in ("box", "plate", "zone", "stick"):
            ET.SubElement(body, "geom", type="box", size=_vec(extents(obj)/2), **kw)
        else:
            raise ValueError(f"Unsupported scene shape: {shape}")
    return ET.tostring(root, encoding="unicode")


class TaskTwin:
    def __init__(self, scene, physics="so101_bench", q0=HOME_Q):
        self.scene = deepcopy(scene)
        self.physics = load_physics(physics)
        self.m = mujoco.MjModel.from_xml_string(build_xml(scene, self.physics))
        self.d = mujoco.MjData(self.m)
        self.model, self.data = self.m, self.d
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self.dofadr = np.array([self.m.joint(j).dofadr[0] for j in JOINTS])
        self.actadr = np.array([self.m.actuator(j).id for j in JOINTS])
        self.range = np.array([self.m.joint(j).range for j in JOINTS])
        self.objects = {o["id"]: o for o in scene.get("objects", []) if o.get("present", True)}
        self.body_ids = {k: self.m.body(f"object:{k}").id for k in self.objects}
        self.commanded_q = np.clip(np.asarray(q0, float), *self.range.T)
        self.d.qpos[self.qadr] = self.commanded_q
        self.d.ctrl[self.actadr] = self.commanded_q
        self._delay = deque([self.commanded_q.copy() for _ in range(self.physics.get("latency_ticks", 0))])
        self._renderer = None
        self.ticks = 0
        mujoco.mj_forward(self.m, self.d)

    def step(self, q_target):
        q = np.asarray(q_target, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("q_target must be six finite absolute joint angles")
        self.commanded_q = np.clip(q, *self.range.T)
        self._delay.append(self.commanded_q.copy())
        self.d.ctrl[self.actadr] = self._delay.popleft()
        mujoco.mj_step(self.m, self.d, nstep=self.physics.get("substeps", 16))
        mujoco.mj_forward(self.m, self.d)
        self.ticks += 1
        return self.q()

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def poses(self):
        out = {}
        for name, bid in self.body_ids.items():
            R = self.d.xmat[bid].reshape(3, 3)
            out[name] = {"xyz": self.d.xpos[bid].tolist(), "yaw_deg": float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))),
                         "quat": self.d.xquat[bid].tolist(), "size_m": self.objects[name]["size_m"],
                         "v": float(np.linalg.norm(self.d.cvel[bid, 3:]))}
        return out

    def grip_closed(self):
        return bool(self.commanded_q[5] < 0.15)

    def render(self, camera="front"):
        cfgs = self.scene.get("cameras", [])
        if isinstance(cfgs, dict):
            cfgs = [dict(v, id=k) for k, v in cfgs.items()]
        cfg = camera if isinstance(camera, dict) else next((c for c in cfgs if c.get("id") == camera), {})
        width, height = cfg.get("width", 640), cfg.get("height", 480)
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            self.close()
            self._renderer = mujoco.Renderer(self.m, height=height, width=width)
        pos = np.asarray(cfg.get("pos", [0.55, -0.45, 0.45]))
        look = np.asarray(cfg.get("lookat", [0.2, 0, 0.04]))
        delta = pos-look
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look
        cam.distance = np.linalg.norm(delta)
        cam.azimuth = np.rad2deg(np.arctan2(delta[1], delta[0]))
        cam.elevation = -np.rad2deg(np.arctan2(delta[2], np.linalg.norm(delta[:2])))
        self.m.vis.global_.fovy = cfg.get("fovy_deg", 55)
        self._renderer.update_scene(self.d, camera=cam)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
