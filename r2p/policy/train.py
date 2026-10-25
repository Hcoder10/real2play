"""Train command-conditioned absolute chunks; CPU smoke tests or leased CUDA only.

python -m r2p.policy.train --task T --traces DIR --name NAME --steps S --device cpu
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from r2p.frames import ROOT
from r2p.tasks import load_task
from .data import Augmentation, TraceDataset
from .features import export_config
from .runtime import clamp_constants


def _device(torch, device, memory_fraction):
    if device.startswith("cuda"):
        if not os.environ.get("LANDLORD_LEASE_ID") or "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise RuntimeError("CUDA training requires a landlord lease and its CUDA_VISIBLE_DEVICES environment")
        fraction = memory_fraction or os.environ.get("LANDLORD_MEMORY_FRACTION") or os.environ.get("LANDLORD_TORCH_MEMORY_FRACTION")
        if fraction is None or not 0 < float(fraction) <= 1:
            raise RuntimeError("Pass the lease's torch memory fraction with --memory-fraction (0,1]")
        torch.cuda.set_per_process_memory_fraction(float(fraction), device=torch.device(device))
    elif device != "cpu":
        raise ValueError("device must be cpu or cuda[:index]")
    return torch.device(device)


def train(task, traces, name, steps=1000, device="cpu", *, seeds=(0,), batch_size=128,
          learning_rate=3e-4, out_dir=None, eval_n=5, eval_max_ticks=None,
          memory_fraction=None, augmentation=None, width=512, blocks=3, threads=2):
    import torch
    from .network import StatePolicy

    if steps < 1 or batch_size < 1 or not seeds or eval_n < 1:
        raise ValueError("steps, batch_size, seed count and eval_n must be positive")
    torch.set_num_threads(max(1, threads))
    target_device = _device(torch, device, memory_fraction)
    task = load_task(task) if not isinstance(task, dict) else task
    config = export_config(task, name, seeds=seeds, width=width, blocks=blocks)
    augmentation = augmentation or Augmentation()
    dataset = TraceDataset(task, traces, config, augmentation)
    config["start_q"] = dataset.start_q.tolist()
    config["physics"] = dataset.physics
    config["physics_sha256"] = hashlib.sha256(json.dumps(dataset.physics, sort_keys=True).encode()).hexdigest()
    norm, config["obs_const"] = dataset.normalization()
    tensors = {key: torch.tensor(value, dtype=torch.float32, device=target_device) for key, value in norm.items()}
    weights, losses = {}, []
    for member, seed in enumerate(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = StatePolicy(config["input_dim"], width=width, blocks=blocks).to(target_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        model.train()
        history = []
        for step in range(steps):
            x, y = dataset.sample(batch_size, rng)
            x = clamp_constants(x, config["obs_const"])
            x = (torch.as_tensor(x, dtype=torch.float32, device=target_device)-tensors["x_mean"])/tensors["x_std"]
            y = (torch.as_tensor(y, dtype=torch.float32, device=target_device)-tensors["y_mean"])/tensors["y_std"]
            prediction = model(x)
            loss = torch.nn.functional.smooth_l1_loss(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step == 0 or step == steps-1 or (step+1) % 100 == 0:
                history.append({"step": step+1, "loss": float(loss.detach().cpu())})
        model.eval()
        weights.update({f"member{member}.{key}": value.detach().cpu().numpy() for key, value in model.state_dict().items()})
        losses.append({"seed": seed, "loss": history})
        del model, optimizer
    directory = Path(out_dir) if out_dir is not None else ROOT/"exports"/name
    directory.mkdir(parents=True, exist_ok=True)
    config["training"] = {"steps": steps, "device": device, "batch_size": batch_size,
                          "learning_rate": learning_rate, "augmentation": asdict(augmentation),
                          "traces": dataset.provenance, "samples": len(dataset.raw), "losses": losses,
                          "lease_id": os.environ.get("LANDLORD_LEASE_ID") if device.startswith("cuda") else None}
    np.savez_compressed(directory/"weights.npz", **weights)
    config["weights_sha256"] = hashlib.sha256((directory/"weights.npz").read_bytes()).hexdigest()
    (directory/"norm.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    (directory/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("runtime.py"), directory/"runtime.py")
    # Every completed export contains actual closed-loop evaluations for all noise sets.
    from .eval import evaluate
    for noise in ("clean", "real", "realx0.5"):
        evaluate(directory, n=eval_n, noise=noise, max_ticks=eval_max_ticks, record=True)
    return directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-n", type=int, default=5)
    p.add_argument("--memory-fraction", type=float)
    a = p.parse_args()
    path = train(a.task, a.traces, a.name, a.steps, a.device, seeds=a.seeds, batch_size=a.batch_size,
                 eval_n=a.eval_n, memory_fraction=a.memory_fraction)
    print(json.dumps({"export": str(path), "eval": json.loads((path/"config.json").read_text())["eval"]}, indent=2))


if __name__ == "__main__":
    main()
