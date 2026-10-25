"""Fine-tune every pretrained SmolVLA action-expert tensor on two-camera traces."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from .contracts import Normalizer, STATE, CAMERAS


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.episodes = []
        self.index = []
        for item in self.manifest["episodes"]:
            path = self.root/item["directory"]
            with np.load(path/"labels.npz") as f:
                labels = {k: f[k] for k in f.files}
            self.episodes.append((np.load(path/"images.npy", mmap_mode="r"), labels))
            self.index.extend((len(self.episodes)-1, i) for i in range(item["frames"]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ep, row = self.index[index]
        images, labels = self.episodes[ep]
        return labels["state"][row], images[row].copy(), str(labels["instruction"][row]), labels["action"][row]


def collate(rows):
    return np.stack([x[0] for x in rows]), np.stack([x[1] for x in rows]), [x[2] for x in rows], np.stack([x[3] for x in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-fraction", type=float)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--max-seconds", type=float, default=1200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", help="Continue from this fine-tuned export; optimizer restarts explicitly")
    p.add_argument("--image-augmentation", action="store_true")
    p.add_argument("--gpu-duty", type=float, default=.60, help="Maximum GPU duty fraction of the assigned lease")
    a = p.parse_args()
    if not 0 < a.gpu_duty <= 1 or a.steps < 1 or a.batch_size < 1:
        p.error("positive steps/batch and gpu-duty in (0,1] required")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .runtime import configure_device, load_model, make_batch
    device = configure_device(a.device, a.memory_fraction)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    data = Dataset(a.dataset)
    stats = Normalizer(json.loads((Path(a.dataset)/"train_stats.json").read_text()))
    assets, out = Path(a.assets), Path(a.out)
    if out.exists() and (out/"config.json").exists():
        raise ValueError("choose a new export directory")
    out.mkdir(parents=True, exist_ok=True)
    parent = None
    if a.resume:
        parent_path = Path(a.resume)
        parent = json.loads((parent_path/"config.json").read_text())
        if json.loads((parent_path/"train_stats.json").read_text()) != stats.stats:
            raise ValueError("resume normalization differs from this dataset")
        if file_hash(parent_path/"pretrained/model.safetensors") != parent["weights_sha256"]:
            raise ValueError("resume weights hash mismatch")
    model = load_model(Path(a.resume)/"pretrained" if a.resume else assets/"base", assets/"backbone", device, fresh=True)
    tokenizer = AutoTokenizer.from_pretrained(str(assets/"backbone"), local_files_only=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(.9, .95), weight_decay=1e-10)
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate,
                        drop_last=False, persistent_workers=a.workers > 0)
    iterator = iter(loader)
    shutil.copytree(assets/"backbone", out/"backbone", ignore=shutil.ignore_patterns(".cache"))
    shutil.copy2(Path(a.dataset)/"train_stats.json", out/"train_stats.json")
    shutil.copy2(Path(a.dataset)/"manifest.json", out/"dataset_manifest.json")
    cfg = {"schema": "r2p_vla_export/1", "policy_type": "smolvla", "task": "place_on_cell",
           "task_spec": data.manifest["task_spec"], "start_q": data.manifest["start_q"], "physics": data.manifest["physics"],
           "obs_frame": "spec_footprint", "action": "absolute_q", "target_space": "servo", "gravity_compensation": "included_in_expert_targets",
           "horizon": 15, "hz": 30, "K": data.manifest["task_spec"]["limits"]["max_objects"],
           "object_ids": [o["id"] for o in data.manifest["task_spec"]["objects"]],
           "obs_layout": {"state": ["last_commanded_q[6], reset to measured on any joint >4 degrees", "command_age_s"],
                          "images": list(CAMERAS), "image_order": "RGB", "language": "contracts.instruction(command,target_cell)"},
           "obs_const": {"mean": stats.stats[STATE]["mean"], "std": stats.stats[STATE]["std"], "threshold_std": 5, "floor": .001},
           "camera": data.manifest["camera"], "mounts": data.manifest["mounts"], "eval": {},
           "pretrained": json.loads((assets/"provenance.json").read_text()),
           "base_weights_sha256": file_hash(assets/"base/model.safetensors"),
           "deployment": data.manifest.get("deployment"), "deployment_qualified": False,
           "training": {"episodes": len(data.manifest["episodes"]), "frames": len(data), "batch_size": a.batch_size,
                        "seed": a.seed, "lr": a.lr, "normalization": "actual training mean/std; no identity fallback",
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
    cfg["training"].update(image_augmentation=a.image_augmentation,
                           parent_weights_sha256=parent["weights_sha256"] if parent else None,
                           parent_steps=parent["training"].get("total_steps", parent["training"]["steps"]) if parent else 0,
                           optimizer_restarted=bool(parent))
    (out/"runtime.py").write_text("from r2p.vla.runtime import Policy\n")
    started = time.monotonic()
    losses = []

    def save(step):
        model.save_pretrained(out/"pretrained")
        cfg["training"].update(steps=step, seconds=time.monotonic()-started, first_loss=losses[0], last_loss=losses[-1],
                               total_steps=step+cfg["training"]["parent_steps"],
                               peak_vram_gb=torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else 0)
        cfg["weights_sha256"] = file_hash(out/"pretrained/model.safetensors")
        (out/"config.json.tmp").write_text(json.dumps(cfg, indent=2))
        (out/"config.json.tmp").replace(out/"config.json")
        print("saved", step, flush=True)

    model.train()
    log = (out/"training.jsonl").open("w")
    try:
        for step in range(1, a.steps+1):
            step_started = time.monotonic()
            try:
                states, images, texts, actions = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                states, images, texts, actions = next(iterator)
            batch = make_batch(states, images, texts, stats, tokenizer, device)
            if a.image_augmentation:
                for key in CAMERAS:
                    pixels = batch[key]
                    gain = torch.empty((len(pixels), 1, 1, 1), device=device).uniform_(.9, 1.1)
                    batch[key] = (pixels*gain+torch.randn_like(pixels)*(2/255)).clamp_(0, 1)
            batch["action"] = torch.as_tensor(stats.normalize("action", actions), device=device)
            optimizer.zero_grad(set_to_none=True)
            # Warmup short enough for this explicitly time-budgeted adaptation.
            lr = a.lr*min(1., step/50)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, details = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
                time.sleep((time.monotonic()-step_started)*(1/a.gpu_duty-1))
            losses.append(float(loss.detach()))
            if step == 1 or step % 10 == 0:
                row = {"step": step, "loss": losses[-1], "grad_norm": float(norm), "seconds": time.monotonic()-started}
                log.write(json.dumps(row)+"\n")
                log.flush()
                print(json.dumps(row), flush=True)
            if step % a.save_every == 0:
                save(step)
            if time.monotonic()-started >= a.max_seconds:
                break
        save(step)
    finally:
        log.close()


if __name__ == "__main__":
    main()
