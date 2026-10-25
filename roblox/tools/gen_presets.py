"""Generate Roblox data from the canonical MJCF, task JSON and identified physics.

No old implementation is imported. Mesh IDs are asset data copied separately.
Run from any directory; --check fails if committed generated data is stale.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.ndimage import binary_fill_holes
from r2p.frames import BASE_ORIGIN_MJCF, FIXED_TIP, MOVING_TIP, HOME_Q, JOINTS, MJCF

OUT = ROOT / "roblox/src/ReplicatedStorage/Twin/Generated"


def workshop():
    return json.loads((ROOT / "roblox/presets/game/workshop.json").read_text(encoding="utf-8"))


def workshop_physics(rig=None):
    path = (ROOT / "config/physics" / f"{rig}.json") if rig else ROOT / workshop()["physics_preset"]
    return json.loads(path.read_text(encoding="utf-8"))


def apply_bench(task, bench):
    """Freeze the judge's placement level to the measured bench, in spec metres.

    The arm observations describe a held-cap offset, not a new world frame.
    Keep them in the task/trace provenance; physical contacts determine the
    actual offset during play and SkillOperator measures it before placement.
    """
    task = copy.deepcopy(task)
    task["board"] = copy.deepcopy(bench["board"])
    task["bench_calibration"] = copy.deepcopy(bench["arm"])
    task["bench_source"] = bench["source"]
    task["instruction"] = "Place the cap on the highlighted cell; the robot learns from your play."
    cells = task["board"]["cells"]
    first, last = cells["r0c0"], cells["r2c0"]
    yaw = math.degrees(math.atan2(last[1]-first[1], last[0]-first[0]))
    choices = list(cells.values())
    rnd = task["randomize"]
    rnd.pop("board_pose", None)
    rnd["piece"] = {"xy_choice": copy.deepcopy(bench["supply_xy"])}
    for obj in task["objects"]:
        if obj["id"] == "piece" or obj["id"].startswith("other"):
            obj.update(shape="cylinder", size_m=bench["cap_size_m"].copy(),
                       mass_kg=bench["cap_mass_kg"], friction=bench["cap_friction"], color="white")
        if obj["id"] == "piece":
            obj["xy"] = bench["supply_xy"][0].copy()
            obj.pop("xyz", None)
        if obj["id"] in cells or obj["id"] == "cell":
            obj["size_m"] = [task["board"]["cell_m"]]*2 + [.001]
            obj["yaw_deg"] = yaw
            if obj["id"] in cells:
                obj["xy"] = cells[obj["id"]].copy()
        if obj["id"] == "cell" or obj["id"].startswith("other"):
            rnd[obj["id"]]["xy_choice"] = copy.deepcopy(choices)
    task["bench_note"] = bench["notes"]
    return task


def lua(value):
    if value is None:
        return "nil"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.number)):
        if not np.isfinite(value):
            raise ValueError("non-finite preset value")
        return repr(float(value)) if isinstance(value, (float, np.floating)) else str(value)
    if isinstance(value, str):
        # Luau accepts UTF-8 literals, but not JSON's backslash-u escapes.
        return json.dumps(value, ensure_ascii=False).replace("\\/", "/")
    if isinstance(value, dict):
        return "{" + ",".join(f"[{lua(str(k))}]={lua(v)}" for k, v in value.items() if v is not None) + "}"
    return "{" + ",".join(lua(v) for v in value) + "}"


def matrix(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


def voxel_boxes(triangles, pitch=0.004):
    """Rasterize triangle surfaces, fill closed interiors, merge occupied voxel runs.

    Each resulting box is contained in the voxel occupancy; concave finger gaps
    remain empty. Padding prevents boundary surfaces from breaking flood fill.
    """
    low = np.floor(triangles.min(axis=(0, 1)) / pitch) * pitch - pitch
    high = triangles.max(axis=(0, 1))
    grid = np.zeros(np.ceil((high - low) / pitch).astype(int) + 2, dtype=bool)
    for tri in triangles:
        n = max(1, int(np.ceil(max(np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[0]),
                                      np.linalg.norm(tri[2] - tri[1])) / (pitch * 0.4))))
        u, v = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n)
        mask = u + v <= 1
        pts = tri[0] + u[mask, None] * (tri[1] - tri[0]) + v[mask, None] * (tri[2] - tri[0])
        ijk = np.floor((pts - low) / pitch).astype(int)
        grid[tuple(ijk.T)] = True
    grid = binary_fill_holes(grid)
    boxes = []
    for x, y, z in np.argwhere(grid):
        if not grid[x, y, z]:
            continue
        end = [x + 1, y + 1, z + 1]
        for axis in (2, 1, 0):
            while end[axis] < grid.shape[axis]:
                a, b = [x, y, z], end.copy()
                a[axis] = end[axis]
                b[axis] += 1
                if not grid[a[0]:b[0], a[1]:b[1], a[2]:b[2]].all():
                    break
                end[axis] += 1
        grid[x:end[0], y:end[1], z:end[2]] = False
        boxes.append({"pos": (low + (np.array([x, y, z]) + end) * pitch / 2).tolist(),
                      "size": ((np.array(end) - [x, y, z]) * pitch).tolist()})
    return boxes


def robot_data(with_voxels=True):
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    bodies = []
    for b in range(1, m.nbody):
        j = int(m.body_jntadr[b]) if m.body_jntnum[b] else -1
        ri = matrix(m.body_iquat[b])
        body = {"name": m.body(b).name, "parent": int(m.body_parentid[b]), "pos": m.body_pos[b].tolist(),
                "rot": matrix(m.body_quat[b]).ravel().tolist(), "mass": float(m.body_mass[b]),
                "com": m.body_ipos[b].tolist(), "inertia": (ri @ np.diag(m.body_inertia[b]) @ ri.T).ravel().tolist(),
                "joint": j + 1, "axis": m.jnt_axis[j].tolist() if j >= 0 else [0, 0, 1],
                "joint_pos": m.jnt_pos[j].tolist() if j >= 0 else [0, 0, 0], "visuals": [], "boxes": []}
        groups = {}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_group[g] != 2:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            faces = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            v = v @ matrix(m.geom_quat[g]).T + m.geom_pos[g]
            rgba = m.mat_rgba[m.geom_matid[g]]
            key = "dark" if rgba[0] < 0.3 else "yellow"
            groups.setdefault(key, []).append(v)
            if with_voxels:
                body["boxes"].extend(voxel_boxes(v[faces]))
        # The uploaded asset data groups each link by material, yellow first.
        for index, key in enumerate(sorted(groups, key=lambda k: k != "yellow")):
            v = np.concatenate(groups[key])
            body["visuals"].append({"asset": body["name"] + (f"_{index}" if index else ""),
                                    "center": ((v.min(0) + v.max(0)) / 2).tolist(),
                                    "size": (v.max(0) - v.min(0)).tolist(), "dark": key == "dark"})
        bodies.append(body)
    return {"name": "SO-101", "bodies": bodies, "joints": JOINTS, "range": m.jnt_range.tolist(),
            "home_q": HOME_Q.tolist(), "base_origin": BASE_ORIGIN_MJCF.tolist(),
            "fixed_tip": [next(i + 1 for i, b in enumerate(bodies) if b["name"] == FIXED_TIP[0]), FIXED_TIP[1].tolist()],
            "moving_tip": [next(i + 1 for i, b in enumerate(bodies) if b["name"] == MOVING_TIP[0]), MOVING_TIP[1].tolist()],
            "mjcf_sha256": hashlib.sha256(MJCF.read_text(encoding="utf-8").encode()).hexdigest(), "voxel_m": 0.004}


def write(name, value, check=False):
    path = OUT / (name + ".lua")
    source = "-- Generated by roblox/tools/gen_presets.py; do not edit.\nreturn " + lua(value) + "\n"
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != source:
            raise SystemExit(f"stale generated module: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--robot", action="store_true", help="regenerate expensive collision voxels")
    parser.add_argument("--rig", help="override the workshop physics with config/physics/<rig>.json")
    args = parser.parse_args()
    if args.robot or args.check or not (OUT / "Robot.lua").exists():
        write("Robot", robot_data(), args.check)
    task_paths = sorted((ROOT / "tasks").glob("*/task.json"))
    tasks = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in task_paths}
    game = workshop()
    bench = json.loads((ROOT / game["bench_preset"]).read_text(encoding="utf-8"))
    tasks["place_on_cell"] = apply_bench(tasks["place_on_cell"], bench)
    order = sorted(tasks, key=lambda n: (tasks[n].get("game", {}).get("order", 100), n))
    write("Tasks", {"tasks": tasks, "order": order}, args.check)
    data = workshop_physics(args.rig)
    # Match r2p.sim expert / policy fingerprints, independent of JSON whitespace
    # and Windows checkout line endings.
    write("Physics", {"rig": data["rig"], "sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), "config": data}, args.check)
    write("Game", game, args.check)
    vectors = ROOT / "r2p/tasks/vectors.json"
    if vectors.exists():
        path = ROOT / "tests/roblox/taskcheck_vectors.luau"
        source = "-- Generated from r2p/tasks/vectors.json.\nreturn " + lua(json.loads(vectors.read_text(encoding="utf-8"))) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != source:
                raise SystemExit("stale task-check vectors")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8", newline="\n")
    print(f"Generated {len(tasks)} task presets; physics {data['rig']} (workshop selection)")


if __name__ == "__main__":
    main()
