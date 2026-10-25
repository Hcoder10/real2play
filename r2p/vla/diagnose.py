"""Teacher-forced errors and image/language sensitivity of a fine-tuned export."""
import argparse
import json
from pathlib import Path
import numpy as np
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-fraction", type=float)
    a = p.parse_args()
    import torch
    from .runtime import Policy
    from .train import Dataset, collate
    policy = Policy.load(a.export, a.device, a.memory_fraction)
    data = Dataset(a.dataset)
    rng = np.random.default_rng(20261025)
    idx = rng.choice(len(data), min(a.samples, len(data)), replace=False)
    errors = []
    for start in range(0, len(idx), 16):
        states, images, texts, labels = collate([data[int(i)] for i in idx[start:start+16]])
        torch.manual_seed(73)
        start_time = time.monotonic()
        prediction = policy.predict(states, images, texts)
        if policy.device.type == "cuda":
            time.sleep((time.monotonic()-start_time)*2/3)
        errors.append(prediction-labels)
    error = np.concatenate(errors)
    states, images, texts, _ = collate([data[int(idx[0])]])
    torch.manual_seed(73)
    prediction = policy.predict(states, images, texts)
    torch.manual_seed(73)
    no_image = policy.predict(states, np.zeros_like(images), texts)
    torch.manual_seed(73)
    no_language = policy.predict(states, images, ["Return the arm to its starting pose."] if texts != ["Return the arm to its starting pose."] else ["Pick up the piece."])
    result = {"samples": len(error), "chunk_mae_rad": float(np.abs(error).mean()),
              "first_action_mae_rad": float(np.abs(error[:, 0]).mean()),
              "per_joint_mae_rad": np.abs(error).mean((0, 1)).tolist(),
              "black_image_prediction_change_rad": float(np.abs(prediction-no_image).mean()),
              "different_language_prediction_change_rad": float(np.abs(prediction-no_language).mean()),
              "note": "Teacher-forced training samples; sensitivity is not closed-loop success.",
              "weights_sha256": policy.config["weights_sha256"]}
    (Path(a.export)/"diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
