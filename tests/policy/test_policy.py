import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from r2p.frames import HOME_Q
from r2p.model import Arm
from r2p.policy import load
from r2p.policy.data import Augmentation, TraceDataset
from r2p.policy.features import export_config
from r2p.policy.runtime import clamp_constants, command_encoding, fk_features
from r2p.sim import gen_traces
from r2p.tasks import command_onehot, commands, encode_slots, load_task, sample_scene


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    root = tmp_path_factory.mktemp("policy")
    gen_traces("cube_in_bowl", 2, root/"traces", seed=10001, dart=.004)
    return root


def test_numpy_fk_and_command_encoding_match_shared_model():
    arm = Arm()
    task = load_task("place_on_cell")
    config = export_config(task, "test")
    qs = np.random.default_rng(8).uniform(arm.range[:, 0], arm.range[:, 1], (20, 6))
    features = fk_features(qs, config["fk"])
    for q, x in zip(qs, features):
        np.testing.assert_allclose(x[:3], arm.tips(q)[2], atol=1e-10)
        np.testing.assert_allclose(x[3:], arm.jac_tip(q)[2], atol=1e-10)
    for command in commands(task):
        np.testing.assert_array_equal(command_encoding(config, command), command_onehot(task, command))


def test_chunk_labels_augmentations_constants(traces):
    task = load_task("cube_in_bowl")
    config = export_config(task, "test")
    ds = TraceDataset(task, traces/"traces", config)
    plain, target = ds.sample(32, np.random.default_rng(23), augment=False)
    augmented, target_aug = ds.sample(32, np.random.default_rng(23), augment=True)
    assert np.max(np.abs(augmented-plain)) > .01
    offset = augmented[:, :5]-plain[:, :5]
    np.testing.assert_allclose((target_aug-target).reshape(-1, 15, 6)[:, :, :5],
                               np.broadcast_to(offset[:, None, :], (32, 15, 5)), atol=3e-7)
    np.testing.assert_array_equal(target_aug.reshape(-1, 15, 6)[:, :, 5],
                                  target.reshape(-1, 15, 6)[:, :, 5])
    assert target.shape == (32, 90)
    assert ds.labels[:, 5].min() < 0
    norm, constants = ds.normalization()
    assert constants
    x = ds.features[0].copy()
    i = constants[0]["index"]
    x[i] = constants[0]["mean"]+100
    clamped = clamp_constants(x, constants)
    assert clamped[i] == constants[0]["mean"]
    x[i] = constants[0]["mean"]+constants[0]["threshold"]*.5
    assert clamp_constants(x, constants)[i] == x[i]


def test_tiny_cpu_training_standalone_export_parity_and_eval(traces):
    import torch
    from r2p.policy.train import train
    from r2p.policy.network import StatePolicy
    from r2p.policy.eval import evaluate

    directory = train("cube_in_bowl", traces/"traces", "tiny_cpu", steps=2, device="cpu",
                      seeds=(3, 4), batch_size=8, out_dir=traces/"export", eval_n=1, eval_max_ticks=20)
    assert {"config.json", "norm.json", "runtime.py", "weights.npz"} <= {p.name for p in directory.iterdir()}
    policy = load(directory)
    config = policy.config
    assert config["width"] == 512 and config["blocks"] == 3
    assert config["action"] == "absolute_q" and config["obs_frame"] == "spec_footprint"
    np.testing.assert_allclose(config["start_q"], HOME_Q)
    scene = sample_scene(config["task_spec"], np.random.default_rng(12))
    poses = {o["id"]: o for o in scene["objects"]}
    obs = {"q": HOME_Q, "slots": encode_slots(config["task_spec"], poses), "command": config["commands"][0], "tick": 0}
    outputs = policy.act_ensemble(obs)
    assert outputs.shape == (2, 15, 6)
    features = policy.features(obs)
    x = torch.as_tensor(((features-policy.norm["x_mean"])/policy.norm["x_std"]).astype(np.float32))
    for member in range(2):
        model = StatePolicy(config["input_dim"])
        prefix = f"member{member}."
        model.load_state_dict({key[len(prefix):]: torch.from_numpy(value) for key, value in policy.weights.items() if key.startswith(prefix)})
        with torch.no_grad():
            expected = model(x).numpy()*policy.norm["y_std"]+policy.norm["y_mean"]
        limits = np.asarray(config["joint_limits"])
        expected = np.clip(expected.reshape(15, 6), limits[:, 0], limits[:, 1])
        np.testing.assert_allclose(outputs[member], expected, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(policy.act(obs), outputs.mean(axis=0), atol=1e-7)
    spec = importlib.util.spec_from_file_location("standalone_runtime", directory/"runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np.testing.assert_array_equal(module.Policy.load(directory).act(obs), policy.act(obs))
    for noise in ("clean", "real", "realx0.5"):
        report = config["eval"][noise]
        assert report["n"] == 1 and report["max_ticks"] == 20
        assert report["episodes"][0]["ticks"] <= 20
        assert report["weights_sha256"] == config["weights_sha256"]
    repeat = evaluate(directory, n=1, noise="clean", max_ticks=20, record=False)
    assert repeat["episodes"] == config["eval"]["clean"]["episodes"]
    with pytest.raises(ValueError):
        policy.act(dict(obs, q=[np.nan]*6))


def test_cuda_requires_lease_before_allocating(monkeypatch):
    from r2p.policy.train import _device
    monkeypatch.delenv("LANDLORD_LEASE_ID", raising=False)
    class FakeTorch:
        pass
    with pytest.raises(RuntimeError, match="landlord lease"):
        _device(FakeTorch(), "cuda", None)
