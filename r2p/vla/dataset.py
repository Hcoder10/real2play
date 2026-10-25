"""Render successful expert episodes to mmap arrays with exact command labels."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from r2p.vla.contracts import action_chunk, commanded_state, instruction, target_cell, Normalizer


def read_trace(path):
    from r2p.traces import read_trace as validated_read
    return validated_read(path)


def render_episode(job):
    from r2p.vla.render import Cameras
    path, out, stride = job
    meta, ticks = read_trace(path)
    if not meta.get("success") or meta.get("target_space") != "servo":
        return None
    directory = Path(out)/Path(path).stem
    directory.mkdir(parents=True, exist_ok=True)
    cell = target_cell(meta["scene"])
    indices = sorted(set(range(0, len(ticks), stride)) | {i for i in range(1, len(ticks)) if ticks[i]["command"] != ticks[i-1]["command"]})
    cam = Cameras(meta["scene"], meta.get("physics_config", "so101_bench"))
    images = np.lib.format.open_memmap(directory/"images.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(indices), 2, cam.mounts["height"], cam.mounts["width"], 3))
    states, actions, texts, sample_ticks = [], [], [], []
    command, start = None, 0
    selected = set(indices)
    try:
        for i, tick in enumerate(ticks):
            if tick["command"] != command:
                command, start = tick["command"], float(tick["t"])
            if i not in selected:
                continue
            previous = ticks[i-1].get("q_exec", ticks[i-1]["q_target"]) if i else meta["start_q"]
            state, _ = commanded_state(previous, tick["q"], float(tick["t"])-start)
            images[len(states)] = cam.render(tick["q"], tick["poses"])
            states.append(state)
            actions.append(action_chunk(ticks, i))
            texts.append(instruction(command, cell))
            sample_ticks.append(i)
        images.flush()
    finally:
        cam.close()
    np.savez(directory/"labels.npz", state=np.asarray(states), action=np.asarray(actions),
             instruction=np.asarray(texts), tick=np.asarray(sample_ticks))
    record = {"trace": str(path), "trace_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "frames": len(states), "ticks": len(ticks), "target_cell": cell, "meta": meta}
    (directory/"episode.json").write_text(json.dumps(record))
    return {"directory": directory.name, "frames": len(states), "seed": meta["seed"], "trace_sha256": record["trace_sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 4 or a.stride < 1 or a.episodes < 1:
        p.error("positive episodes/stride and 1..4 workers required")
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.traces).glob("*.jsonl"))[:a.episodes]
    if not paths:
        raise ValueError("no traces")
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(a.workers) as pool:
        for record in pool.map(render_episode, [(str(p), str(root), a.stride) for p in paths]):
            if record:
                records.append(record)
                print(f"rendered {len(records)}/{len(paths)} {record['frames']} frames", flush=True)
    if not records:
        raise ValueError("no successful servo-target traces")
    labels = [np.load(root/r["directory"]/"labels.npz") for r in records]
    stats = Normalizer.fit(np.concatenate([x["state"] for x in labels]), np.concatenate([x["action"] for x in labels]))
    first = json.loads((root/records[0]["directory"]/"episode.json").read_text())["meta"]
    for r in records:
        other = json.loads((root/r["directory"]/"episode.json").read_text())["meta"]
        for key in ("task", "start_q", "physics_config", "deployment"):
            if first.get(key) != other.get(key):
                raise ValueError(f"mixed trace conventions: {key}")
    from r2p.frames import ROOT
    manifest = {"schema": "r2p_vla_dataset/1", "episodes": records, "frames": sum(r["frames"] for r in records),
                "source_ticks": sum(len(read_trace(p)[1]) for p in paths), "stride": a.stride,
                "hz": 30, "horizon": 15, "seconds": time.monotonic()-started,
                "task_spec": first["task"], "start_q": first["start_q"],
                "physics": first.get("physics_config", "so101_bench"), "deployment": first.get("deployment"),
                "camera": json.loads((ROOT/"config/camera.json").read_text()),
                "mounts": json.loads(Path(__file__).with_name("cameras.json").read_text())}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2))
    (root/"train_stats.json").write_text(json.dumps(stats.stats, indent=2))
    print(json.dumps({k: manifest[k] for k in ("frames", "source_ticks", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
