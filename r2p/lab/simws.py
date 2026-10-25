"""Editable static MJCF workspace and joint-PD replay, with sealed held-out data."""
from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from r2p import gemini
from r2p.agents import safe_path
from r2p.frames import HOME_Q, JOINTS, MJCF, ROOT, BASE_ORIGIN_MJCF
from r2p.model import Arm, JointMap
from r2p.perception.camera import CameraModel
from r2p.sim.twin import build_xml, load_physics
from .motion import Motion, Refused, StallDetector, aim_with_jaw_ik, vector
from .render import Renderer
from .media import timeline_video


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def confined(root, name):
    safe_path(name)
    path = (Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes the sim workspace")
    return path


def _numbers(value):
    return " ".join(str(float(x)) for x in value)


def initial_xml(camera, physics, scene=None):
    root = ET.fromstring(build_xml(scene or {"objects": []}, physics))
    root.find("compiler").attrib.pop("meshdir", None)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", "robot_assets/"+mesh.get("file"))
    # MJCF cameras look along -z with +y up; OpenCV has +z forward and +y down.
    R = camera.R.T @ np.diag([1, -1, -1])
    world = root.find("worldbody")
    ET.SubElement(world, "camera", name="scene", pos=_numbers(camera.position),
                  xyaxes=_numbers(np.r_[R[:, 0], R[:, 1]]),
                  resolution=f"{camera.width} {camera.height}", sensorsize="1 1",
                  focal=_numbers([camera.K[0, 0]/camera.width, camera.K[1, 1]/camera.height]),
                  principal=_numbers([camera.K[0, 2]/camera.width-.5, camera.K[1, 2]/camera.height-.5]))
    ET.SubElement(world.find(".//body[@name='gripper']"), "camera", name="wrist",
                  pos="0.025 0 -0.04", xyaxes="1 0 0 0 1 0", fovy="65")
    return ET.tostring(root, encoding="unicode")


def validate_physics(p):
    if not np.allclose(vector(p.get("gravity", [0, 0, -9.81])), [0, 0, -9.81]):
        raise ValueError("Gravity is fixed; fit sag through servo stiffness")
    if not 1 <= int(p.get("substeps", 32)) <= 256 or not 0 <= int(p.get("latency_ticks", 0)) <= 15:
        raise ValueError("Invalid timestep/latency")
    for name in JOINTS:
        cfg = p["joints"][name]
        for key, limits in {"kp": (.1, 500), "kv": (.001, 30), "armature": (0, 1),
                            "frictionloss": (0, 10), "torque_limit": (.01, 10)}.items():
            v = float(cfg[key])
            if not math.isfinite(v) or not limits[0] <= v <= limits[1]:
                raise ValueError(f"Invalid {name}.{key}")


def expanded_xml(directory):
    directory = Path(directory)
    visited = set()

    def expand(path):
        if path in visited:
            raise ValueError("Recursive/duplicate XML include")
        visited.add(path)
        text = path.read_text(encoding="utf-8")
        if "<!" in text:
            raise ValueError("XML declarations/entities/comments are not accepted in editable assets")
        root = ET.fromstring(text)
        for parent in root.iter():
            for node in list(parent):
                if node.tag == "include":
                    included = expand(confined(directory, node.attrib["file"]))
                    position = list(parent).index(node)
                    parent.remove(node)
                    for child in list(included):
                        parent.insert(position, child)
                        position += 1
        return root

    return expand(confined(directory, "scene.xml"))


def compile_workspace(directory):
    """Validate references and physical rules before loading any user-authored asset."""
    directory = Path(directory)
    p = read_json(directory/"physics.json")
    validate_physics(p)
    root = expanded_xml(directory)
    forbidden = {"equality", "weld", "connect", "exclude", "plugin", "extension", "keyframe", "key",
                 "tendon", "general", "motor", "velocity", "adhesion"}
    for node in root.iter():
        empty_container = node.tag in ("equality", "tendon", "keyframe") and len(node) == 0
        if node.tag in forbidden and not empty_container or node.get("mocap") == "true":
            raise ValueError(f"No sim cheating: {node.tag} / mocap forbidden")
        if node.tag == "flag" and any(v == "disable" for k, v in node.attrib.items() if k in ("contact", "gravity", "actuation")):
            raise ValueError("No sim cheating: disabling physics forbidden")
        if "gravcomp" in node.attrib and float(node.get("gravcomp")) != 0:
            raise ValueError("No sim cheating: gravity compensation on bodies forbidden")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir, texturedir = compiler.get("meshdir", ""), compiler.get("texturedir", "")
    for prefix in (meshdir, texturedir):
        if prefix:
            confined(directory, prefix)
    for node in root.iter():
        if "file" in node.attrib:
            prefix = meshdir if node.tag == "mesh" else texturedir if node.tag == "texture" else ""
            relative = (Path(prefix)/node.get("file")).as_posix()
            asset = confined(directory, relative)
            if not asset.is_file():
                raise ValueError(f"Missing asset: {relative}")
            node.set("file", str(asset))
    compiler.attrib.pop("meshdir", None)
    compiler.attrib.pop("texturedir", None)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    if "gravity" in option.attrib and not np.allclose(np.fromstring(option.get("gravity"), sep=" "), [0, 0, -9.81]):
        raise ValueError("Gravity is fixed")
    option.set("gravity", "0 0 -9.81")
    option.set("timestep", str(1/30/p.get("substeps", 32)))
    for name, friction in p.get("friction_overrides", {}).items():
        values = [float(friction), .005, .0001] if np.isscalar(friction) else vector(friction).tolist()
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 10:
            raise ValueError("Friction overrides must be finite values in 0..10")
        matches = root.findall(f"worldbody/.//geom[@name='{name}']")
        if not matches:
            body = root.find(f"worldbody/.//body[@name='{name}']")
            if body is None:
                body = root.find(f"worldbody/.//body[@name='object:{name}']")
            matches = list(body.iter("geom")) if body is not None else []
        if not matches:
            raise ValueError(f"Unknown friction override geometry/body: {name}")
        for geom in matches:
            geom.set("friction", _numbers(values))
    actuators = root.find("actuator")
    if actuators is None or len(actuators) != 6:
        raise ValueError("Exactly six robot position servos required")
    for node in actuators:
        name = node.get("name")
        if node.tag != "position" or name not in JOINTS or node.get("joint") != name:
            raise ValueError("Only robot joint-space position PD actuators allowed")
        cfg = p["joints"][name]
        node.attrib.update(kp=str(cfg["kp"]), kv=str(cfg["kv"]),
                           forcerange=_numbers([-cfg["torque_limit"], cfg["torque_limit"]]))
        joint = root.find(f"worldbody/.//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"Missing robot joint: {name}")
        joint.attrib.update(armature=str(cfg["armature"]), frictionloss=str(cfg["frictionloss"]), damping="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.neq or model.nmocap:
        raise ValueError("No sim cheating: equality / mocap")
    base = model.body("base").id
    table_id = model.geom("table").id
    def collides(a, b):
        return bool(int(model.geom_contype[a]) & int(model.geom_conaffinity[b]) or
                    int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))
    jaw_ids = [model.geom(name).id for name in ("fixed_finger", "moving_finger")]
    if not np.allclose(model.body_pos[base], -BASE_ORIGIN_MJCF, atol=1e-9):
        raise ValueError("Calibrated footprint origin cannot be shifted")
    baseline = ET.fromstring(build_xml({"objects": []}, load_physics()))
    known_robot_bodies = {n.get("name") for n in baseline.find("worldbody/body[@name='base']").iter("body")}
    for body in root.find("worldbody/body[@name='base']").iter("body"):
        if body.get("name") not in known_robot_bodies:
            raise ValueError("No sim cheating: attaching new object bodies to the robot is forbidden")
    # Preserve the calibrated robot collision surfaces. Appearance and physical
    # material parameters remain editable, but deleting jaws is not a fit.
    for original in baseline.find("worldbody/body[@name='base']").iter("body"):
        name = original.get("name")
        expected = sum(g.get("class") == "collision" or g.get("group") == "3" for g in original.findall("geom"))
        body_id = model.body(name).id
        start = model.body_geomadr[body_id]
        colliders = [g for g in range(start, start+model.body_geomnum[body_id])
                     if model.geom_contype[g] and model.geom_conaffinity[g]]
        if len(colliders) < expected:
            raise ValueError("No sim cheating: calibrated robot collision geometry was removed")
        if any(not collides(g, table_id) for g in colliders):
            raise ValueError("No sim cheating: robot collision masks isolate it from the table")
    for node in root.findall(".//geom[@class='collision']"):
        if node.get("contype") == "0" or node.get("conaffinity") == "0":
            raise ValueError("No sim cheating: robot collisions cannot be disabled")
    for gid in range(model.body_geomadr[0], model.body_geomadr[0]+model.body_geomnum[0]):
        if model.geom_contype[gid] == 0 or model.geom_conaffinity[gid] == 0:
            raise ValueError("No sim cheating: environment collisions cannot be disabled")
    for body in range(1, model.nbody):
        ancestor = body
        while model.body_parentid[ancestor] and ancestor != base:
            ancestor = model.body_parentid[ancestor]
        if ancestor == base:
            continue
        own_geoms = list(range(model.body_geomadr[body], model.body_geomadr[body]+model.body_geomnum[body]))
        colliders = [g for g in own_geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
        if own_geoms and not colliders:
            # A detailed visual shell may have convex collision proxies on
            # another body in the same object subtree. The object cannot be a ghost.
            subtree_geoms = []
            for other in range(1, model.nbody):
                root_body = other
                while model.body_parentid[root_body]:
                    root_body = model.body_parentid[root_body]
                if root_body == ancestor:
                    subtree_geoms.extend(range(model.body_geomadr[other], model.body_geomadr[other]+model.body_geomnum[other]))
            if not any(model.geom_contype[g] or model.geom_conaffinity[g] for g in subtree_geoms):
                raise ValueError("No sim cheating: object collisions cannot be disabled")
        for gid in colliders:
            if not collides(gid, table_id) or not all(collides(gid, jaw) for jaw in jaw_ids):
                raise ValueError("No sim cheating: collision masks isolate an object from the table or jaws")
    if not np.allclose(model.opt.gravity, [0, 0, -9.81]):
        raise ValueError("Gravity changed")
    return model, p


class Simulation:
    """No runtime model edits, object resets, applied forces, or attachments.

    Initial robot encoder state is set exactly once. Thereafter only ctrl and
    mj_step are used; servo compliance generates sag naturally.
    """
    def __init__(self, directory, q0=None, *, camera=None):
        self.m, self.physics = compile_workspace(directory)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self.actadr = np.array([self.m.actuator(j).id for j in JOINTS])
        self.arm = Arm()
        self.jm = JointMap.load(ROOT/"config/joint_map.json")
        if q0 is None:
            # HOME_Q predates the camera-fitted joint map. Seed a reachable pose
            # using the exact proven bench IK and calibrated 20%-open jaw.
            from scripts.lead.jaw_move import plan, OWNER_LO, OWNER_HI
            _, error, q0 = plan(np.array([.20, -.10, .06]), 20, self.jm.to_owner(HOME_Q),
                                self.arm, self.jm, load_physics(), OWNER_LO, OWNER_HI)
            if error > .004:
                raise ValueError("Could not initialize a calibrated safe simulation pose")
        self.target = vector(q0, 6).copy()
        self.rest_q = self.target.copy()
        self.d.qpos[self.qadr] = self.target
        self.d.ctrl[self.actadr] = self.target
        self.delay = deque()
        self._delayed_target = self.target.copy()
        self.scene_camera = camera or CameraModel.load(ROOT/"config/camera.json")
        self.renderer = Renderer(self.m)
        self.motion = None
        self.on_tick = None
        self.on_command = None
        self.speed_limit = 12
        self.stall = StallDetector()
        mujoco.mj_forward(self.m, self.d)

    def state(self):
        q = self.d.qpos[self.qadr].copy()
        owner = self.jm.to_owner(q)
        # An encoder closure lag is evidence of a stall, not an attachment heuristic.
        holding = self.stall.update(q[5], self.target[5], float(self.d.time))
        return {"q": q.tolist(), "q_target": self.target.tolist(), "xyz": self.arm.tips(q)[2].tolist(),
                "gripper": float(np.clip(owner[5], 0, 100)), "holding": holding,
                "source": "hidden_mujoco" if getattr(self, "hidden", False) else "sim", "t": float(self.d.time)}

    def poses(self):
        poses = {}
        base = self.m.body("base").id
        for body in range(1, self.m.nbody):
            parent = body
            while self.m.body_parentid[parent] and parent != base:
                parent = self.m.body_parentid[parent]
            if parent == base:
                continue
            name = self.m.body(body).name
            if name:
                poses[name.removeprefix("object:")] = {"xyz": self.d.xpos[body].tolist(), "quat": self.d.xquat[body].tolist()}
        return poses

    def camera(self, name="scene"):
        if name == "scene":
            return self.scene_camera
        if name != "wrist":
            raise ValueError("camera must be scene or wrist")
        c = self.m.camera("wrist").id
        R = np.diag([1, -1, -1]) @ self.d.cam_xmat[c].reshape(3, 3).T
        f = 240/(2*np.tan(np.deg2rad(self.m.cam_fovy[c])/2))
        return CameraModel([[f, 0, 160], [0, f, 120], [0, 0, 1]], np.zeros(5), R, -R@self.d.cam_xpos[c], (320, 240))

    def frame(self, name="scene"):
        return self.renderer.render(self.d, self.camera(name))

    def advance(self, target, until):
        requested = vector(target, 6)
        if not np.array_equal(requested, self.target):
            due = float(self.d.time)+self.physics.get("latency_ticks", 0)/30
            self.delay.append((due, requested.copy()))
        self.target = requested.copy()
        # Apply controls at their recorded times plus the static servo latency;
        # do not round irregular real events onto a fresh 30 Hz clock.
        while self.d.time + self.m.opt.timestep/2 < until:
            while self.delay and self.delay[0][0] <= self.d.time+self.m.opt.timestep/2:
                _, self._delayed_target = self.delay.popleft()
            self.d.ctrl[self.actadr] = self._delayed_target
            mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)

    def _move_joint(self, target, speed):
        start = np.asarray(self.state()["q_target"])
        target = vector(target, 6)
        limits = np.array([self.m.joint(j).range for j in JOINTS])
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise Refused("Joint target outside limits")
        n = max(1, math.ceil(np.max(np.abs(target-start))/np.deg2rad(speed)*30))
        for q in np.linspace(start, target, n+1)[1:]:
            if self.motion:
                self.motion.check_joints(q, self.arm)
            if self.on_command:
                self.on_command(q.tolist(), float(self.d.time))
            self.advance(q, self.d.time+1/30)
            if self.motion:
                self.motion.check_joints(self.state()["q"], self.arm, measured=True)
            if self.on_tick:
                self.on_tick(self.state())
        return {"status": "ok", "xyz": self.state()["xyz"]}

    def goto(self, xyz, *, speed=12, min_finger=.006):
        from scripts.lead.jaw_move import plan, OWNER_LO, OWNER_HI
        goal, aim = vector(xyz), vector(xyz).copy()
        grip = self.state()["gripper"]
        iterations = 1 if self.motion and self.motion.phase.endswith("until_contact") else 3
        for _ in range(iterations):
            owner, error, _ = plan(aim, grip, self.jm.to_owner(self.state()["q"]), self.arm, self.jm,
                                   load_physics(), OWNER_LO, OWNER_HI)
            if error > .004:
                raise Refused(f"IK error {error*1000:.1f} mm")
            owner[5] = grip
            result = self._move_joint(self.jm.to_model(owner), speed)
            measured = np.asarray(self.state()["xyz"])
            if np.linalg.norm(measured-goal) < .002:
                break
            aim += np.clip(goal-measured, -.015, .015)
        return result

    def gripper(self, value):
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise Refused("Gripper must be 0..100")
        own = self.jm.to_owner(self.target)
        own[5] = value
        return self._move_joint(self.jm.to_model(own), min(self.speed_limit, 8))

    def park(self):
        self.goto([.17, 0, .05], speed=self.speed_limit)
        return self._move_joint(self.rest_q, min(8, self.speed_limit))

    def stop(self):
        # Holding current motor target preserves physical dynamics; never teleport qpos.
        return {"stopped": True}

    def dwell(self, seconds):
        for _ in range(max(1, math.ceil(seconds*30))):
            self.advance(self.target, self.d.time+1/30)
            if self.motion:
                self.motion.check_joints(self.state()["q"], self.arm, measured=True)
            if self.on_tick:
                self.on_tick(self.state())
        return {"held_s": seconds, "holding": self.state()["holding"]}

    def look_at(self, xyz):
        goal = vector(xyz)
        c = self.m.camera("wrist").id
        local_pos, local_quat = self.m.cam_pos[c].copy(), self.m.cam_quat[c].copy()
        local_R = np.empty(9)
        mujoco.mju_quat2Mat(local_R, local_quat)
        def optical_axis(q):
            self.arm.fk(q)
            b = self.arm.m.body("gripper").id
            R = self.arm.d.xmat[b].reshape(3, 3)
            from r2p.frames import BASE_ORIGIN_MJCF
            origin = self.arm.d.xpos[b]-BASE_ORIGIN_MJCF+R@local_pos
            direction = R@local_R.reshape(3, 3)@np.array([0, 0, -1.])
            return origin, direction
        return aim_with_jaw_ik(self, goal, optical_axis)


class SimWorkspace:
    def __init__(self, run, task=None):
        self.run, self.task = Path(run).resolve(), task or {}
        self.directory = self.run/"sim"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.camera = CameraModel.load(ROOT/"config/camera.json")
        if not (self.directory/"scene.xml").exists():
            shutil.copytree(MJCF.parent/"assets", self.directory/"robot_assets", dirs_exist_ok=True)
            gemini.atomic_json(self.directory/"physics.json", load_physics())
            (self.directory/"scene.xml").write_text(initial_xml(self.camera, load_physics()), encoding="utf-8")
        self.holdout_path = self.run/"heldout.json"
        self.holdout = read_json(self.holdout_path) if self.holdout_path.exists() else {"episodes": [], "sealed": False}
        self.latest_replay = None
        self._preview_sim = None
        self.validate()

    def validate(self):
        compile_workspace(self.directory)

    def files(self):
        return [{"path": p.relative_to(self.directory).as_posix(), "bytes": p.stat().st_size}
                for p in sorted(self.directory.rglob("*")) if p.is_file() and "robot_assets" not in p.parts]

    def read(self, path):
        p = confined(self.directory, path)
        if p.suffix.lower() not in (".xml", ".json", ".obj", ".mtl") or p.stat().st_size > 1_000_000:
            raise ValueError("Only small text assets can be read through this tool")
        return {"path": path, "content": p.read_text(encoding="utf-8")}

    def edit_sim(self, path=None, content=None, *, files=None, encoding="utf-8"):
        if self.holdout.get("sealed"):
            raise ValueError("Held-out evaluation is sealed; further editing would leak validation feedback")
        bundle = files or {path: content}
        resolved = [confined(self.directory, name) for name in bundle]
        if len({str(p).casefold() for p in resolved}) != len(resolved):
            raise ValueError("Duplicate asset path")
        old = {}
        try:
            for name, value in bundle.items():
                p = confined(self.directory, name)
                if "robot_assets" in Path(name).parts:
                    raise ValueError("Calibrated robot meshes are read-only; edit materials in scene.xml")
                if p.suffix.lower() not in (".xml", ".json", ".obj", ".stl", ".png", ".jpg", ".jpeg", ".mtl"):
                    raise ValueError("Only static XML/JSON/mesh/texture assets may be written")
                if not isinstance(value, str) or len(value) > 16_000_000:
                    raise ValueError("Invalid asset content")
                raw = base64.b64decode(value, validate=True) if encoding == "base64" else value.encode("utf-8")
                if p.suffix.lower() == ".json":
                    json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                if p.suffix.lower() == ".xml":
                    ET.fromstring(raw)
                if p.suffix.lower() in (".png", ".jpg", ".jpeg") and cv2.imdecode(np.frombuffer(raw, np.uint8), 1) is None:
                    raise ValueError("Invalid texture image")
                old[p] = p.read_bytes() if p.exists() else None
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(raw)
            self.validate()
        except BaseException:
            for p, raw in old.items():
                if raw is None:
                    p.unlink(missing_ok=True)
                else:
                    p.write_bytes(raw)
            raise
        self._preview_sim = None
        return {"edited": list(bundle), "validated": True, "revision": self.revision()}

    def revision(self):
        digest = hashlib.sha256()
        for info in self.files():
            digest.update(info["path"].encode())
            digest.update(confined(self.directory, info["path"]).read_bytes())
        return digest.hexdigest()

    def sim_render(self, camera="scene"):
        sim = self._preview_sim or Simulation(self.directory, camera=self.camera)
        path = self.run/"previews"/(camera+".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sim.frame(camera))
        return {"image": str(path), "state": sim.state(), "renderer": "CPU triangle rasterizer"}

    def sim_plan(self, primitives, start_q=None):
        from .harness import Harness
        sim = Simulation(self.directory, q0=start_q, camera=self.camera)
        destination = self.run/"rehearsal"
        harness = Harness(destination, self.task, sim)
        result = harness.plan(primitives)
        self._preview_sim = sim
        directory = Path(result["directory"])
        return {**result, "sim_only": True, "final_state": sim.state(), "object_poses": sim.poses(),
                "video": str(directory/"scene.mp4") if (directory/"scene.mp4").exists() else None,
                "wrist_video": str(directory/"wrist.mp4") if (directory/"wrist.mp4").exists() else None}

    def episode(self, episode):
        name = str(episode)
        if not name.isdigit():
            raise ValueError("Episode must be its numeric id")
        p = self.run/"episodes"/name
        if not (p/"states.jsonl").exists():
            raise ValueError("Unknown episode")
        return p

    def episode_frames(self, episode, camera="scene", indices=None):
        if str(episode) in self.holdout["episodes"]:
            raise ValueError("Held-out frames cannot be inspected after reservation")
        if camera not in ("scene", "wrist"):
            raise ValueError("Unknown camera")
        source = self.episode(episode)
        rows = [r for r in read_json(source/"frames.json") if r["camera"] == camera]
        selected = [rows[i] for i in (indices if indices is not None else [-1])]
        if len(selected) > 8:
            raise ValueError("At most eight frames per observation")
        return {"episode": str(episode), "frames": selected,
                "images": [str(source/r["path"]) for r in selected]}

    def label_episode(self, episode, object_reads):
        """Freeze independent Argon observations before any candidate comparison."""
        name = str(episode)
        if name in self.holdout["episodes"]:
            raise ValueError("Held-out labels are sealed")
        source = self.episode(name)
        path = source/"object_reads.json"
        if path.exists() and object_reads != read_json(path):
            raise ValueError("Object reads are immutable evidence")
        capture_rows = read_json(source/"frames.json")
        frames = {r["path"]: r for r in capture_rows}
        last = max((r["t"] for r in capture_rows), default=0)
        if not isinstance(object_reads, dict) or not object_reads:
            raise ValueError("Object reads must be nonempty")
        for oid, pose in object_reads.items():
            vector(pose["xyz"])
            if pose.get("source") != "argon_vision" or pose.get("frame") not in frames:
                raise ValueError("Object reads require source=argon_vision and a recorded frame reference")
            if frames[pose["frame"]]["t"] < last-.5:
                raise ValueError("Final object pose must be read from a final frame")
        gemini.atomic_json(path, object_reads)
        return {"episode": name, "labels": list(object_reads), "frozen": True}

    def replay(self, episode, object_reads=None, *, _heldout=False, render=True):
        name = str(episode)
        if name in self.holdout["episodes"] and not _heldout:
            raise ValueError("Held-out episodes cannot be inspected or replayed for editing")
        source = self.episode(name)
        if object_reads is not None:
            self.label_episode(name, object_reads)
        rows = [json.loads(line) for line in (source/"states.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) < 2:
            raise ValueError("Replay requires at least two recorded states")
        if any(b["t"] <= a["t"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Episode timestamps must increase")
        sim = Simulation(self.directory, rows[0]["q"], camera=self.camera)
        captures = read_json(source/"frames.json") if (source/"frames.json").exists() else []
        out = self.run/"replays"/f"{name}-{self.revision()[:10]}"
        out.mkdir(parents=True, exist_ok=True)
        for camera in ("scene", "wrist", "side_by_side"):
            (out/"frames"/camera).mkdir(parents=True, exist_ok=True)
        q_errors, jaw_errors, image_errors = [], [], []
        videos = {"scene": [], "wrist": [], "side_by_side": []}
        capture_index = 0
        t0 = rows[0]["t"]
        # Merge sensor and camera times; integrate each active target until the next actual timestamp.
        events = sorted([(r["t"], "state", i) for i, r in enumerate(rows)] +
                        [(c["t"], "frame", i) for i, c in enumerate(captures) if t0 <= c["t"] <= rows[-1]["t"]])
        commands_path = source/"commands.json"
        commands = read_json(commands_path) if commands_path.exists() else []
        events = sorted(events + [(c["t"], "command", i) for i, c in enumerate(commands)
                                  if t0 <= c["t"] <= rows[-1]["t"]])
        command = rows[0]["q_target"]
        panels = {}
        for t, kind, index in events:
            sim.advance(command, max(0, t-t0))
            if kind == "command":
                command = commands[index]["q_target"]
            elif kind == "state":
                row = rows[index]
                q_errors.append(np.rad2deg(np.asarray(sim.state()["q"])-row["q"]))
                jaw_errors.append(1000*(np.asarray(sim.state()["xyz"])-row["xyz"]))
                if not commands:
                    command = row["q_target"]
            elif render:
                cap = captures[index]
                camera = cap["camera"]
                real = cv2.imread(str(source/cap["path"]))
                if real is None:
                    raise ValueError("Missing recorded frame")
                rendered = sim.frame(camera)
                rendered = cv2.resize(rendered, (real.shape[1], real.shape[0]))
                image_errors.append(float(np.mean(np.abs(real.astype(float)-rendered.astype(float)))/255))
                frame_path = out/"frames"/camera/f"{len(videos[camera]):06d}.jpg"
                cv2.imwrite(str(frame_path), rendered)
                videos[camera].append((t, frame_path))
                panels[camera] = np.concatenate([cv2.resize(real, (480, 270)), cv2.resize(rendered, (480, 270))], 1)
                if "scene" in panels and "wrist" in panels:
                    side_path = out/"frames/side_by_side"/f"{len(videos['side_by_side']):06d}.jpg"
                    panel = np.concatenate([panels["scene"], panels["wrist"]], 0)
                    for label, origin in [("REAL scene", (12, 22)), ("SIM scene", (492, 22)),
                                          ("REAL wrist", (12, 292)), ("SIM wrist", (492, 292))]:
                        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                    cv2.imwrite(str(side_path), panel)
                    videos["side_by_side"].append((t, side_path))
        reads = object_reads
        if reads is None and (source/"object_reads.json").exists():
            reads = read_json(source/"object_reads.json")
        object_errors = []
        for oid, pose in (reads or {}).items():
            predicted = sim.poses().get(oid)
            object_errors.append(float(np.linalg.norm(np.asarray(predicted["xyz"])-pose["xyz"])*1000) if predicted else 1000.)
        residuals = {"joint_rms_deg": float(np.sqrt(np.mean(np.square(q_errors)))),
                     "jaw_path_rms_mm": float(np.sqrt(np.mean(np.sum(np.square(jaw_errors), axis=1)))),
                     "object_final_pose_error_mm": float(np.mean(object_errors)) if object_errors else None,
                     "object_labels": "argon_vision" if object_errors else "unavailable",
                     "image_similarity": 1-float(np.mean(image_errors)) if image_errors else None,
                     "samples": len(rows), "revision": self.revision()}
        for camera, frames in videos.items():
            if frames:
                timeline_video(frames, out/(camera+".mp4"), start=rows[0]["t"], end=rows[-1]["t"])
        gemini.atomic_json(out/"residuals.json", residuals)
        result = {"episode": name, "residuals": residuals, "video": str(out/"side_by_side.mp4") if videos["side_by_side"] else None,
                  "final_state": sim.state(), "object_poses": sim.poses()}
        if not _heldout:
            self.latest_replay = result
        return result

    def reserve_holdout(self, episodes):
        names = list(dict.fromkeys(map(str, episodes)))
        if len(names) < 2 or self.holdout["episodes"]:
            raise ValueError("Reserve >=2 held-out episodes exactly once before fitting")
        # Episodes already exposed through a replay cannot become held out retroactively.
        if any(list((self.run/"replays").glob(f"{name}-*")) for name in names):
            raise ValueError("Already replayed episodes cannot become held out")
        for name in names:
            self.episode(name)
            if read_json(self.episode(name)/"result.json")["status"] != "ok":
                raise ValueError("Held-out episodes must have completed without safety or recording failures")
            expected = {obj["id"] for obj in self.task.get("objects", []) if not obj.get("fixed") and obj.get("role") != "target"}
            if expected:
                path = self.episode(name)/"object_reads.json"
                if not path.exists() or not expected.issubset(read_json(path)):
                    raise ValueError("Label every movable task object independently before reserving held-out episodes")
        self.holdout["episodes"] = names
        baseline = self._score(names, heldout=True)
        self.holdout.update(baseline=baseline, baseline_revision=self.revision(),
                            fingerprints={n: self._episode_hash(n) for n in names})
        gemini.atomic_json(self.holdout_path, self.holdout)
        return {"reserved": len(names), "baseline_recorded": True, "details_withheld": True}

    def _episode_hash(self, name):
        digest = hashlib.sha256()
        for path in sorted(self.episode(name).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(self.episode(name)).as_posix().encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _score(self, names, heldout=False):
        values = [self.replay(n, _heldout=heldout)["residuals"] for n in names]
        metrics = {}
        for key in ("joint_rms_deg", "jaw_path_rms_mm", "object_final_pose_error_mm", "image_similarity"):
            entries = [v[key] for v in values if v[key] is not None]
            metrics[key] = float(np.mean(entries)) if len(entries) == len(values) and entries else None
        metrics["episodes"] = len(values)
        return metrics

    def score(self, split="train"):
        if split not in ("train", "heldout"):
            raise ValueError("split must be train or heldout")
        if split == "train":
            names = [p.name for p in (self.run/"episodes").glob("*") if p.is_dir() and p.name not in self.holdout["episodes"]]
            if not names:
                raise ValueError("No training episodes")
            return self._score(names)
        if len(self.holdout["episodes"]) < 2:
            raise ValueError("At least two untouched held-out episodes are required")
        if self.holdout.get("result"):
            return self.holdout["result"]
        for name, fingerprint in self.holdout["fingerprints"].items():
            if self._episode_hash(name) != fingerprint:
                raise ValueError("Held-out evidence changed")
        # Seal BEFORE running: a failed evaluation cannot reopen editing.
        self.holdout["sealed"] = True
        gemini.atomic_json(self.holdout_path, self.holdout)
        current = self._score(self.holdout["episodes"], heldout=True)
        base = self.holdout["baseline"]
        physics = (current["jaw_path_rms_mm"] < base["jaw_path_rms_mm"]-1e-6 and
                   current["joint_rms_deg"] <= base["joint_rms_deg"]+1e-6)
        visuals = (current["image_similarity"] is not None and base["image_similarity"] is not None and
                   current["image_similarity"] > base["image_similarity"]+1e-6)
        objects = (base["object_final_pose_error_mm"] is None or current["object_final_pose_error_mm"] is not None and
                   current["object_final_pose_error_mm"] <= base["object_final_pose_error_mm"]+1e-6)
        result = {"split": split, "baseline": base, "current": current, "improved": bool(physics and visuals and objects),
                  "sealed": True, "revision": self.revision()}
        self.holdout["result"] = result
        gemini.atomic_json(self.holdout_path, self.holdout)
        return result
