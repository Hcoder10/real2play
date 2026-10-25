"""Reproducible read-only bench QA. All live Gemini attempts share an 80-request ledger.

Run stages sequentially (the ledger lock prevents competing QA processes).
No serial connection, daemon POST, model fallback, or GPU work is performed.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from r2p import gemini


class QAClient(gemini.Gemini):
    def __init__(self, root, limit=80):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = self.root / "calls.jsonl"
        self.lock = self.root / "qa.lock"
        fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        self.budget_path = self.root / "request_budget.json"
        try:
            logged = sum(r.get("attempts", 0) for r in read_rows(self.journal))
            reserved = json.loads(self.budget_path.read_text())["reserved"] if self.budget_path.exists() else 0
            self.reserved = max(logged, reserved)
            self.limit = limit
            gemini.atomic_json(self.budget_path, {"reserved": self.reserved, "limit": limit})
            super().__init__(max_requests=max(0, limit - self.reserved))
        except BaseException:
            self.close()
            raise

    def _request(self, model, parts, config):
        # Reserve before sending. If this process dies mid-request, a later run
        # still charges the uncertain attempt rather than risking overspend.
        if self.reserved >= self.limit:
            raise gemini.RequestLimit("Persistent QA request budget exhausted")
        self.reserved += 1
        gemini.atomic_json(self.budget_path, {"reserved": self.reserved, "limit": self.limit})
        return super()._request(model, parts, config)

    def _log(self, record):
        super()._log(record)
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(gemini._safe(json.dumps(record)) + "\n")

    def close(self):
        self.lock.unlink(missing_ok=True)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def prepare(root, poses, camera, joint_map):
    from r2p.model import JointMap
    from r2p.traces import write_trace
    rows = json.loads(Path(poses).read_text())
    mapping = JointMap.load(joint_map)
    destination = root / "session"
    (destination / "frames").mkdir(parents=True, exist_ok=True)
    samples = []
    for i, row in enumerate(rows):
        name = f"{i:04d}.jpg"
        shutil.copy2(Path(poses).parent / row.get("frame", f"p{row['k']:02d}.jpg"), destination / "frames" / name)
        samples.append({"k": row["k"], "q": mapping.to_model(row["owner_deg"]).tolist()})
    # Empty command trace: the original calibration log never recorded targets.
    # Measured q is carried as explicitly static samples, not policy labels.
    meta = {"task": "cube_in_bowl", "source": "real", "session_kind": "bench_static_poses",
            "pose_samples": samples, "frame_times": [i * .5 for i in range(len(rows))], "fps": 2,
            "training_eligible": False, "time_basis": "presentation_only",
            "provenance": {"poses": str(poses), "joint_map": str(joint_map), "candidate": mapping.name},
            "note": "Independent calibration photos; no targets, motion timing or success measurement."}
    write_trace(destination / "trace.jsonl", meta, [])
    shutil.copy2(camera, destination / "camera.json")
    return {"session": str(destination), "frames": len(samples), "physics_evidence": False}


def path_summary(rows):
    groups = {}
    for record in rows:
        tag = record["tag"]
        group = tag.split(":")[0]
        bucket = groups.setdefault(group, {"requests": 0, "cache_hits": 0, "errors": 0, "latencies_s": [], "served_by": {}})
        bucket["requests"] += record.get("attempts", 0)
        bucket["cache_hits"] += int(record.get("cached", False))
        bucket["errors"] += int("error" in record)
        for attempt in record.get("attempt_details", []):
            bucket["latencies_s"].append(attempt["latency"])
        if record.get("served_by"):
            model = record["served_by"]
            bucket["served_by"][model] = bucket["served_by"].get(model, 0) + 1
    for bucket in groups.values():
        values = bucket["latencies_s"]
        bucket["latency_p50_s"] = float(np.percentile(values, 50)) if values else None
        bucket["latency_p90_s"] = float(np.percentile(values, 90)) if values else None
    return {"requests": sum(r["requests"] for r in groups.values()), "paths": groups}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "empty", "boards", "twin", "task", "visual", "perception", "label", "summary"))
    parser.add_argument("--root", type=Path, default=Path("runs/agent_g"))
    parser.add_argument("--poses", type=Path, default=Path("runs/bench/vcal/poses.json"))
    parser.add_argument("--camera", type=Path, default=Path("config/camera.json"))
    parser.add_argument("--joint-map", type=Path, default=Path("config/joint_map.json"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-live", type=Path)
    parser.add_argument("--task", help="Known task for the twin (omitting it exercises open scene detection)")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--force-point", action="store_true")
    args = parser.parse_args(argv)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    from r2p.perception import CameraModel
    camera = CameraModel.load(args.camera)
    if args.stage == "prepare":
        result = prepare(root, args.poses, args.camera, args.joint_map)
    elif args.stage == "summary":
        result = path_summary(read_rows(root / "calls.jsonl"))
    else:
        client = QAClient(root)
        try:
            if args.stage in ("empty", "boards"):
                from r2p.ttt import board_images, decide, validate_decision
                if args.stage == "empty":
                    if args.image:
                        path = args.image
                    else:
                        from urllib.request import urlopen
                        path = root / ("live_" + time.strftime("%H%M%S") + ".jpg")
                        with urlopen("http://127.0.0.1:8792/frame/0.jpg", timeout=5) as response:
                            path.write_bytes(response.read())
                    cases = [{"image": str(path), "board": [""] * 9, "history": []}]
                else:
                    if not args.manifest:
                        parser.error("boards requires --manifest")
                    cases = json.loads(args.manifest.read_text())["cases"]
                results = []
                for case in cases:
                    image = cv2.imread(case["image"])
                    start = time.monotonic()
                    row = {"image": case["image"], "expected": case["board"]}
                    try:
                        answer = decide(image, camera, case["history"], ask=client)
                        observed = answer["board"]
                        row.update(answer=answer, exact=observed == case["board"],
                                   correct_cells=sum(a == b for a, b in zip(observed, case["board"])))
                        from r2p.ttt import history_board
                        # Legality against independent truth, not against the model's read.
                        validate_decision({"board": case["board"], "move": answer["move"]}, history_board(case["history"]))
                        row["legal"] = True
                    except (ValueError, gemini.GeminiError) as exc:
                        row.update(error=str(exc), legal=False)
                    row["elapsed_s"] = time.monotonic() - start
                    results.append(row)
                    gemini.atomic_json(root / f"{args.stage}_results.json", results)
                    print(json.dumps({k: row.get(k) for k in ("image", "exact", "legal", "elapsed_s", "error")}), flush=True)
                result = {"n": len(results), "exact_accuracy": sum(r.get("exact", False) for r in results) / len(results),
                          "cell_accuracy": sum(r.get("correct_cells", 0) for r in results) / (9 * len(results)),
                          "legal_rate": sum(r["legal"] for r in results) / len(results),
                          "latency_p50_s": float(np.percentile([r["elapsed_s"] for r in results], 50)),
                          "latency_p90_s": float(np.percentile([r["elapsed_s"] for r in results], 90))}
            elif args.stage == "twin":
                from r2p.real2sim import run_twin
                out = args.out_live or Path("out/live") / ("agent_g_" + time.strftime("%Y%m%d_%H%M%S"))
                result = run_twin(root / "session", task=args.task, ask=client, out_dir=out, iterations=args.iters, max_calls=10)
            elif args.stage == "task":
                from r2p.real2sim import task_from_video
                out = args.out_live or Path("out/live") / ("agent_g_task_" + time.strftime("%Y%m%d_%H%M%S"))
                result = task_from_video(root / "session", ask=client, out_root=root / "tasks",
                    instruction="Author the intended task: put the visible brown cube into the blue bowl. "
                                "These are static calibration views, not evidence of a completed manipulation. "
                                "Use only cube and bowl as task objects; ignore the calibration papers and robot.",
                    twin_options={"out_dir": out, "iterations": args.iters, "max_calls": 8})
            elif args.stage == "visual":
                from r2p.sysid import visual
                from r2p.model import JointMap
                mapping = JointMap.load(args.joint_map)
                rows = visual.point(args.poses, out_path=root / "visual/pointed.json", ask=client,
                                    camera=camera, joint_map=mapping, force=args.force_point)
                result = visual.fit(rows, camera, mapping)
                visual.report(result, root / "visual")
            elif args.stage == "perception":
                from r2p.perception.task_perception import GeminiDetector, TaskPerceiver
                task = json.loads(next((root / "tasks").glob("*/task.json")).read_text())
                image = cv2.imread(str(args.image or root / "session/frames/0000.jpg"))
                detector = GeminiDetector(ask=client)
                with_perceiver = TaskPerceiver(camera, task, detector)
                try:
                    start = time.monotonic()
                    with_perceiver.setup(image)
                    result = {"setup_s": time.monotonic() - start, "poses": with_perceiver.poses,
                              "record": detector.record}
                    with_perceiver.save_setup(root / "perception_setup.json")
                finally:
                    with_perceiver.close()
            elif args.stage == "label":
                from r2p.labeling import segment
                from r2p.traces import read_trace
                # Only real command traces are eligible; static samples have no
                # action labels, so require an explicitly supplied trace here.
                if not args.manifest:
                    parser.error("label requires --manifest pointing to an actual command trace")
                result = {"segments": segment(read_trace(args.manifest), refine=True, ask=client)}
        finally:
            client.close()
    gemini.atomic_json(root / f"{args.stage}_summary.json", result)
    print(json.dumps(result, default=gemini._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
