"""CPU-only MJCF renderer: calibrated optics, editable meshes, colors and textures.

Uses painter-sorted triangles (the existing bench renderer's approximation), not
GPU/GL. It renders the compiled simulation, including arbitrary new object bodies.
"""
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np

from r2p.synthbench.render import _decimate, box_mesh, cylinder_mesh


class Renderer:
    def __init__(self, model):
        self.m = model
        self.meshes = []
        for g in range(model.ngeom):
            if model.geom_group[g] == 3:
                continue  # collision-only robot proxies; visual geometry remains visible
            kind, size = model.geom_type[g], model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                f, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                triangles = _decimate(model.mesh_vert[a:a+n], model.mesh_face[f:f+nf])
            elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
                # Subdivide the tabletop so near-plane clipping cannot discard an
                # entire large face. Plane rendering must stay at exactly z=0.
                xs = np.linspace(-max(size[0], .8), max(size[0], .8), 33)
                ys = np.linspace(-max(size[1], .6), max(size[1], .6), 25)
                triangles = []
                for x0, x1 in zip(xs, xs[1:]):
                    for y0, y1 in zip(ys, ys[1:]):
                        triangles.extend([[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0]],
                                          [[x0, y0, 0], [x1, y1, 0], [x0, y1, 0]]])
                triangles = np.asarray(triangles)
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                triangles = box_mesh(2*np.maximum(size, [.001, .001, .001]))
            elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                triangles = cylinder_mesh(size[0], 2*(size[1]+(size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)), 32)
            elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                from scipy.spatial import ConvexHull
                phi = np.linspace(.01, np.pi-.01, 12)
                theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
                p, t = np.meshgrid(phi, theta)
                v = np.c_[np.sin(p.ravel())*np.cos(t.ravel()), np.sin(p.ravel())*np.sin(t.ravel()), np.cos(p.ravel())]
                v *= size[0] if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size
                triangles = v[ConvexHull(v).simplices]
            else:
                continue
            mat = model.geom_matid[g]
            rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
            if rgba[3] == 0:
                continue
            colors = np.tile(255*np.asarray(rgba[:3])[::-1], (len(triangles), 1))
            # Planar texture sampling at triangle centroids, including cropped camera patches.
            if mat >= 0:
                textures = np.asarray(model.mat_texid[mat]).ravel()
                tex = next((int(t) for t in textures if t >= 0), -1)
                if tex >= 0 and model.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D:
                    w, h, adr = model.tex_width[tex], model.tex_height[tex], model.tex_adr[tex]
                    channels = int(model.tex_nchannel[tex])
                    data = model.tex_data[adr:adr+w*h*channels].reshape(h, w, channels)
                    xy = triangles.mean(1)[:, :2]
                    uv = ((xy / np.maximum(2*size[:2], .001) + .5)*model.mat_texrepeat[mat]) % 1
                    colors = data[(uv[:, 1]*(h-1)).astype(int), (uv[:, 0]*(w-1)).astype(int), :3][:, ::-1]
            self.meshes.append((g, triangles, colors))

    def render(self, data, camera):
        m = self.m
        frame = np.full((camera.height, camera.width, 3), (185, 197, 211), np.uint8)
        triangles, colors = [], []
        light = -data.light_xdir[0] if m.nlight else np.array([-.4, -.3, 1.])
        light = light / max(np.linalg.norm(light), 1e-9)
        diffuse_color = m.light_diffuse[0][::-1] if m.nlight else np.ones(3)*.8
        for gid, local, rgb in self.meshes:
            tri = local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            norm = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-12)
            shade = .65 + .35*np.abs(norm @ light)[:, None]*diffuse_color
            triangles.append(tri)
            colors.append(np.clip(rgb*shade, 0, 255).astype(np.uint8))
        if not triangles:
            return frame
        tri, color = np.concatenate(triangles), np.concatenate(colors)
        camera_xyz = tri @ camera.R.T + camera.t
        pixels = camera.project(tri.reshape(-1, 3)).reshape(-1, 3, 2)
        sides = pixels[:, 1:] - pixels[:, :1]
        area = np.abs(sides[:, 0, 0]*sides[:, 1, 1]-sides[:, 0, 1]*sides[:, 1, 0])
        keep = ((camera_xyz[..., 2].min(1) > .005) & (area > .15) & np.isfinite(pixels).all((1, 2))
                & (pixels[..., 0].max(1) >= 0) & (pixels[..., 0].min(1) < camera.width)
                & (pixels[..., 1].max(1) >= 0) & (pixels[..., 1].min(1) < camera.height))
        order = np.flatnonzero(keep)
        order = order[np.argsort(-camera_xyz[:, :, 2].mean(1)[order], kind="stable")]
        coords = np.clip(np.rint(pixels), -100000, 100000).astype(np.int32)
        for i in order:
            cv2.fillConvexPoly(frame, coords[i], tuple(map(int, color[i])))
        return frame
