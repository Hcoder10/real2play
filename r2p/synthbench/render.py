"""CPU triangle rasterizer for the official MuJoCo arm and task primitives.

The official mesh topology is vertex-clustered at 0.7 mm for this prototype.
MuJoCo supplies link transforms; no GL context, CUDA, or GPU is used. Painter
ordering is sufficient for bench views, but is not a production depth renderer.
"""
from __future__ import annotations

import itertools
import cv2
import mujoco
import numpy as np

from r2p.frames import BASE_ORIGIN_MJCF
from r2p.perception.appearance import color_bgr
from r2p.perception.aruco import BoardConfig, render_board
from r2p.perception.geometry import dimensions, is_screwdriver, rotation


def box_mesh(size):
    vertices = np.array(list(itertools.product([-1, 1], repeat=3)), float)*np.asarray(size)/2
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return vertices[faces]


def cylinder_mesh(radius, height, count=64):
    angle = np.linspace(0, 2*np.pi, count, endpoint=False)
    ring = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), np.full(count, -height/2)))
    top = ring + [0, 0, height]
    nxt = np.roll(np.arange(count), -1)
    return np.concatenate((np.stack((ring, ring[nxt], top[nxt]), 1),
                           np.stack((ring, top[nxt], top), 1),
                           np.stack((np.tile([0, 0, height/2], (count, 1)), top, top[nxt]), 1),
                           np.stack((np.tile([0, 0, -height/2], (count, 1)), ring[nxt], ring), 1)))


def object_meshes(obj):
    shape = obj.get("shape", "box")
    size = dimensions(obj)
    color = color_bgr(obj.get("color", "brown"))
    if shape == "bowl":
        radius, wall = size[0]/2, obj.get("wall_m", .005)
        h, floor = size[2], obj.get("floor_m", wall)
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        def ring(r, z):
            return np.column_stack((r*np.cos(theta), r*np.sin(theta), np.full(64, z)))
        rings = [ring(radius+wall, -h/2), ring(radius+wall, h/2),
                 ring(radius, h/2), ring(radius, -h/2+floor)]
        triangles = []
        for a, b in zip(rings[:-1], rings[1:]):
            triangles.extend((np.stack((a, np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)), 1),
                              np.stack((a, np.roll(b, -1, axis=0), b), 1)))
        triangles.append(np.stack((np.tile([0, 0, -h/2+floor], (64, 1)), rings[-1],
                                   np.roll(rings[-1], -1, axis=0)), 1))
        return [(np.concatenate(triangles), color)]
    if shape == "cylinder":
        raw = np.asarray(obj["size_m"])
        triangles = cylinder_mesh(raw[0]/2, raw[1])
        if obj.get("axis") == "x":
            triangles = triangles[..., [2, 1, 0]]
        elif obj.get("axis") == "y":
            triangles = triangles[..., [0, 2, 1]]
        return [(triangles, color)]
    if shape == "sphere":
        from scipy.spatial import ConvexHull
        from r2p.perception.geometry import silhouette_points
        points = silhouette_points(obj)
        return [(points[ConvexHull(points).simplices], color)]
    if is_screwdriver(obj):
        length, width = size[0], size[1]
        return [(box_mesh([.6*length, width, width]) + [-.2*length, 0, 0], color),
                (box_mesh([.4*length, .35*width, .35*width]) + [.3*length, 0, 0], color_bgr("metal"))]
    return [(box_mesh(size), color)]


def _decimate(vertices, faces, cell=.0007):
    _, inverse = np.unique(np.rint(vertices/cell).astype(int), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    clustered = np.column_stack([np.bincount(inverse, weights=vertices[:, i])/count for i in range(3)])
    remapped = inverse[faces]
    keep = (remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2]) & (remapped[:, 0] != remapped[:, 2])
    remapped = np.unique(np.sort(remapped[keep], axis=1), axis=0)
    return clustered[remapped]


class CpuRenderer:
    def __init__(self, arm, cam):
        self.arm, self.cam = arm, cam
        self.meshes = []
        model = arm.m
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[gid] == 3:
                continue
            mesh = model.geom_dataid[gid]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
            faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh]+model.mesh_facenum[mesh]]
            triangles = _decimate(vertices, faces)
            mat = model.geom_matid[gid]
            rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[gid]
            color = (255*np.asarray(rgba[:3])[::-1]).astype(np.uint8)
            self.meshes.append((gid, triangles, color))

    def geometry(self, q, objects, poses):
        arm = self.arm
        arm.fk(q)
        mujoco.mj_forward(arm.m, arm.d)
        geometry = []
        for gid, triangles, color in self.meshes:
            R = arm.d.geom_xmat[gid].reshape(3, 3)
            xyz = triangles @ R.T + arm.d.geom_xpos[gid] - BASE_ORIGIN_MJCF
            geometry.append(("arm", xyz, color))
        for obj in objects:
            pose = poses[obj["id"]]
            R = np.asarray(pose.get("R", rotation(pose.get("yaw_deg", 0))))
            for triangles, color in object_meshes(obj):
                geometry.append((obj["id"], triangles @ R.T + pose["xyz"], color))
        return geometry

    def render(self, q, objects, poses, *, background=None, board=True, params=None):
        cam = self.cam
        if background is None:
            frame = np.full((cam.height, cam.width, 3), (185, 197, 211), np.uint8)
            if board:
                cfg = BoardConfig(origin_xyz=(.07, .13, 0))
                printed = render_board(cam, cfg)
                # Warp support mask independently; board's white cells must remain white.
                w, h = cfg.dimensions
                polygon = cam.project(cfg.to_spec([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]]))
                region = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillConvexPoly(region, np.rint(polygon).astype(np.int32), 1)
                frame[region != 0] = printed[region != 0]
        else:
            frame = np.asarray(background, np.uint8).copy()
            if frame.shape != (cam.height, cam.width, 3):
                raise ValueError("Background must match calibrated image size")
        geometry = self.geometry(q, objects, poses)
        labels = np.zeros(frame.shape[:2], np.int32)
        names = ["arm"] + [obj["id"] for obj in objects]
        ids = {name: i+1 for i, name in enumerate(names)}
        light = np.asarray(getattr(params, "light_direction", [-.4, -.3, 1.]), float)
        light /= max(np.linalg.norm(light), 1e-9)
        if light[2] <= .05:
            raise ValueError("Light must be above the table (direction z > 0)")
        shadow = np.zeros(frame.shape[:2], np.float32)
        all_triangles, all_colors, all_ids = [], [], []
        for name, triangles, color in geometry:
            floor = triangles.reshape(-1, 3).copy()
            floor[:, :2] -= floor[:, 2:3] * light[:2]/light[2]
            floor[:, 2] = 0
            pixels = cam.project(floor)
            hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
            cv2.fillConvexPoly(shadow, hull, 1.)
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            diffuse = np.abs(normals @ light)
            ambient = getattr(params, "ambient", .7)
            intensity = getattr(params, "light_intensity", .3)
            light_color = np.asarray(getattr(params, "light_color", [1, 1, 1]), float)[::-1]
            albedo = getattr(params, "albedo", {}).get(name, [1, 1, 1])
            roughness = getattr(params, "roughness", .65)
            highlight = (1-roughness)*.12 * diffuse**12
            shade = ambient + intensity*diffuse + highlight
            colors = np.clip(color * np.asarray(albedo)[::-1] * shade[:, None] * light_color, 0, 255).astype(np.uint8)
            all_triangles.append(triangles)
            all_colors.append(colors)
            all_ids.append(np.full(len(triangles), ids[name]))
        softness = max(.01, getattr(params, "shadow_softness_px", 3.))
        shadow = cv2.GaussianBlur(shadow, (0, 0), softness)
        strength = getattr(params, "shadow_strength", .22)
        frame = np.clip(frame.astype(float)*(1-strength*shadow[..., None]), 0, 255).astype(np.uint8)
        triangles = np.concatenate(all_triangles)
        colors, geom_ids = np.concatenate(all_colors), np.concatenate(all_ids)
        camera_xyz = triangles @ cam.R.T + cam.t
        pixels = cam.project(triangles.reshape(-1, 3)).reshape(-1, 3, 2)
        # Subpixel faces cannot affect the silhouette at this camera resolution.
        sides = pixels[:, 1:] - pixels[:, :1]
        area = np.abs(sides[:, 0, 0]*sides[:, 1, 1]-sides[:, 0, 1]*sides[:, 1, 0])
        keep = ((camera_xyz[..., 2].min(axis=1) > .01) & (area > .12)
                & (pixels[..., 0].max(axis=1) >= 0) & (pixels[..., 0].min(axis=1) < cam.width)
                & (pixels[..., 1].max(axis=1) >= 0) & (pixels[..., 1].min(axis=1) < cam.height))
        order = np.flatnonzero(keep)
        depth = camera_xyz[:, :, 2].mean(axis=1)
        # A selectable non-colliding target is a decal over the fixed board cells.
        overlays = {ids[obj["id"]] for obj in objects if obj.get("collide") is False and "xy" not in obj}
        depth -= np.isin(geom_ids, list(overlays))*.0001
        order = order[np.argsort(-depth[order], kind="stable")]
        pixel_int = np.rint(pixels).astype(np.int32)
        for i in order:
            cv2.fillConvexPoly(frame, pixel_int[i], tuple(map(int, colors[i])), lineType=cv2.LINE_8)
            cv2.fillConvexPoly(labels, pixel_int[i], int(geom_ids[i]))
        masks = {name: (labels == value).astype(np.uint8) for name, value in ids.items()}
        return frame, masks, shadow
