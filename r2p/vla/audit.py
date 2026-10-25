"""Projection and recorded-image evidence for the two-camera dataset."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    import cv2
    from .render import Cameras
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    directory = root/manifest["episodes"][0]["directory"]
    images = np.load(directory/"images.npy", mmap_mode="r")
    rows = [np.concatenate(images[i], axis=1) for i in [0, len(images)//3, 2*len(images)//3, len(images)-1]]
    cv2.imwrite(str(out/"dataset_views.jpg"), np.concatenate(rows, axis=0)[..., ::-1])
    points = [[.12, -.12, .02], [.23, -.08, .02], [.23, .12, .02], [.12, .10, .02]]
    scene = {"objects": [{"id": f"marker{i}", "shape": "sphere", "size_m": [.006], "xyz": xyz,
                          "role": "target", "fixed": True, "collide": False, "color": "red"} for i, xyz in enumerate(points)]}
    cam = Cameras(scene, camera=manifest["camera"], mounts=manifest["mounts"])
    try:
        cam.render(manifest["start_q"], {})
        cam.renderer.enable_segmentation_rendering()
        cam.renderer.update_scene(cam.d, camera="vla_scene")
        seg = cam.renderer.render()
        cfg = manifest["camera"]
        camera_xyz = np.asarray(points) @ np.asarray(cfg["R"]).T+cfg["t"]
        pix = camera_xyz @ np.asarray(cfg["K"]).T
        pix = pix[:, :2]/pix[:, 2:]
        pix *= np.array([manifest["mounts"]["width"], manifest["mounts"]["height"]])/cfg["image_size"]
        rows = []
        for i in range(len(points)):
            bid = cam.m.body(f"object:marker{i}").id
            gid = cam.m.body_geomadr[bid]
            y, x = np.where(seg[..., 0] == gid)
            if not len(x):
                continue
            observed = np.array([x.mean()+.5, y.mean()+.5])
            rows.append({"expected": pix[i].tolist(), "observed": observed.tolist(), "error_px": float(np.linalg.norm(observed-pix[i]))})
        report = {"markers": rows, "max_error_px": max(r["error_px"] for r in rows),
                  "visible_markers": len(rows), "scene_distortion": "rectified pinhole; real input must be undistorted"}
        (out/"projection.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if len(rows) < 3 or report["max_error_px"] > 1.5:
            raise RuntimeError("camera projection regression")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
