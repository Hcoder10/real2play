"""Loadable two-camera SmolVLA adapter, returning (15,6) absolute servo targets."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from .contracts import CAMERAS, STATE, HZ, Normalizer, instruction, commanded_state


def configure_device(device, memory_fraction=None):
    import torch
    if str(device).startswith("cuda"):
        if not os.environ.get("LANDLORD_LEASE_ID"):
            raise RuntimeError("GPU inference/training requires LANDLORD_LEASE_ID")
        if memory_fraction is None or not 0 < memory_fraction <= 1:
            raise ValueError("supply the lease's --memory-fraction")
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    torch.set_num_threads(4)
    return torch.device(device)


def load_model(directory, backbone, device="cpu", *, fresh=False, image_size=None):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_model as load_weights
    config = SmolVLAConfig.from_pretrained(str(directory), local_files_only=True)
    config.vlm_model_name = str(Path(backbone).resolve())
    # Construct the architecture offline, then restore ALL pretrained tensors below.
    # False avoids a redundant backbone download, not pretrained initialization.
    config.load_vlm_weights = False
    config.device = "cpu"
    config.chunk_size = config.n_action_steps = 15
    if image_size is not None:
        config.resize_imgs_with_padding = (image_size, image_size)
    config.input_features = {STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                             **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 192, 256)) for key in CAMERAS}}
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config.empty_cameras = 0
    model = SmolVLAPolicy(config)
    missing, unexpected = load_weights(model, str(Path(directory)/"model.safetensors"), strict=False)
    # Older LeRobot checkpoints included normalizers. We replace only those with
    # actual train_stats; no randomly initialized policy tensor is accepted.
    if missing or any(not k.startswith(("normalize_", "unnormalize_")) for k in unexpected):
        raise ValueError(f"pretrained mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def make_batch(states, images, texts, normalizer, tokenizer, device):
    import torch
    states = normalizer.normalize(STATE, states, clamp_constants=True)
    tokens = tokenizer(list(texts), padding="max_length", truncation=True, max_length=48, return_tensors="pt")
    batch = {STATE: torch.as_tensor(states, device=device),
             "observation.language.tokens": tokens["input_ids"].to(device),
             "observation.language.attention_mask": tokens["attention_mask"].bool().to(device)}
    for i, key in enumerate(CAMERAS):
        pixels = np.asarray(images)[:, i]
        if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("cameras must provide RGB uint8 HWC images")
        batch[key] = torch.as_tensor(pixels.copy(), device=device).permute(0, 3, 1, 2).float()/255
    return batch


class Policy:
    def __init__(self, directory, model, tokenizer, device):
        self.directory, self.model, self.tokenizer, self.device = Path(directory), model, tokenizer, device
        self.config = json.loads((self.directory/"config.json").read_text())
        self.normalizer = Normalizer(json.loads((self.directory/"train_stats.json").read_text()))
        self.reset()

    @classmethod
    def load(cls, directory, device="cpu", memory_fraction=None):
        from transformers import AutoTokenizer
        directory = Path(directory)
        from .train import file_hash
        config = json.loads((directory/"config.json").read_text())
        if file_hash(directory/"pretrained/model.safetensors") != config["weights_sha256"]:
            raise ValueError("export weight hash differs from its recorded training/evaluation")
        device = configure_device(device, memory_fraction)
        model = load_model(directory/"pretrained", directory/"backbone", device)
        tokenizer = AutoTokenizer.from_pretrained(str(directory/"backbone"), local_files_only=True)
        return cls(directory, model, tokenizer, device)

    def reset(self):
        self._command, self._command_start, self._tick = None, 0, 0
        self.model.reset()

    def predict(self, states, images, texts):
        import torch
        batch = make_batch(states, images, texts, self.normalizer, self.tokenizer, self.device)
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            chunk = self.model.predict_action_chunk(batch)
        result = self.normalizer.action(chunk.float().cpu().numpy())
        if result.shape != (len(states), 15, 6) or not np.isfinite(result).all():
            raise ValueError("invalid VLA action chunk")
        return result

    def act(self, obs):
        tick = int(obs.get("tick", self._tick))
        command = instruction(obs["command"], obs.get("target_cell"))
        if command != self._command:
            self._command, self._command_start = command, tick
        age = float(obs.get("command_age_s", (tick-self._command_start)/HZ))
        state, _ = commanded_state(obs["q"], obs.get("q_measured", obs["q"]), age)
        self._tick = tick+1
        images = np.stack([obs["images"][key.rsplit(".", 1)[-1]] for key in CAMERAS])
        return self.predict(state[None], images[None], [command])[0]
