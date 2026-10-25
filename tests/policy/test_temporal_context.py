from copy import deepcopy

import numpy as np
import pytest

from r2p.frames import HOME_Q
from r2p.policy.data import TraceDataset
from r2p.policy.features import export_config
from r2p.policy.runtime import Policy, observation
from r2p.tasks import commands, encode_slots, load_task, sample_scene
from r2p.traces import write_trace


def test_command_clock_survives_joint_resets_and_accepts_control_ticks():
    task = load_task("cube_in_bowl")
    config = export_config(task, "clock")
    policy = Policy(config, {}, {})
    # Inspect the actual features passed to inference without requiring training.
    policy.predict_features = lambda features, **kwargs: features
    obs = {"q": HOME_Q.copy(), "slots": np.zeros((config["K"], 9)), "command": commands(task)[0]}
    assert policy.act(obs)[-1] == 0
    obs["q"] = HOME_Q + .2  # A measured-q reset must not restart the skill's clock.
    assert policy.act(obs)[-1] == pytest.approx(1/30)
    assert policy.act(dict(obs, tick=15))[-1] == .5
    assert policy.act(dict(obs, tick=15))[-1] == .5  # Same control tick, same observation.
    assert policy.act(dict(obs, tick=20, command="release"))[-1] == 0
    assert policy.act(dict(obs, tick=25, command="release"))[-1] == pytest.approx(5/30)
    policy.reset()
    assert policy.act(obs)[-1] == 0
    assert policy.act(dict(obs, command_age_s=2.5))[-1] == 2.5
    for invalid in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="command_age_s"):
            observation(dict(obs, command_age_s=invalid), config)
    with pytest.raises(ValueError, match="control tick"):
        policy.act(dict(obs, tick=.5))


def test_trace_clock_matches_runtime_and_restarts_at_command_boundary(tmp_path):
    task = load_task("cube_in_bowl")
    config = export_config(task, "clock")
    scene = sample_scene(task, np.random.default_rng(3))
    poses = {o["id"]: o for o in scene["objects"]}
    rows = [{"t": i/30, "q": HOME_Q, "q_commanded": HOME_Q, "q_target": HOME_Q,
             "poses": poses, "grip_closed": False, "phase": "hold",
             "command": commands(task)[0] if i < 3 else "release"} for i in range(6)]
    path = tmp_path/"trace.jsonl"
    write_trace(path, {"task": task, "source": "expert", "start_q": HOME_Q}, rows)
    dataset = TraceDataset(task, path, config)
    policy = Policy(config, {}, {})
    policy.predict_features = lambda features, **kwargs: features
    for i, row in enumerate(rows):
        actual = policy.act({"q": row["q"], "slots": encode_slots(task, poses),
                             "command": row["command"], "tick": i})
        np.testing.assert_allclose(actual, dataset.features[i], atol=1e-7)
    np.testing.assert_allclose(dataset.features[:, -1], [0, 1/30, 2/30, 0, 1/30, 2/30])

    # Old exports keep their original feature dimension and stateless inference.
    legacy = deepcopy(config)
    legacy.pop("temporal_context")
    old_dataset = TraceDataset(task, path, legacy)
    assert old_dataset.features.shape[1] == dataset.features.shape[1]-1


def test_eval_rate_limit_uses_last_sent_even_when_measured_q_resets(monkeypatch):
    import r2p.policy.eval as evaluation

    task = load_task("cube_in_bowl")
    config = export_config(task, "blocked_gripper")
    scene = sample_scene(task, np.random.default_rng(3))
    poses = {o["id"]: o for o in scene["objects"]}

    class BlockedTwin:
        def __init__(self, scene, physics, q0):
            self.measured = np.array(q0)
            self.commanded_q = self.measured.copy()

        def q(self):
            return self.measured.copy()

        def poses(self):
            return poses

        def step(self, target):
            self.commanded_q = target.copy()

        def grip_closed(self):
            return self.commanded_q[5] < .15

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    policy = Policy(config, {}, {})
    target = HOME_Q.copy()
    target[5] = 0
    policy.act = lambda obs: np.tile(target, (15, 1))
    monkeypatch.setattr(evaluation, "TaskTwin", BlockedTwin)
    rows = []
    result = evaluation.rollout(policy, scene, max_ticks=4, on_step=rows.append)
    assert result["resets"] == 3
    np.testing.assert_allclose([row["target"][5] for row in rows], [.8, .7, .6, .5])
    np.testing.assert_allclose([row["q_input"][5] for row in rows], [.9]*4)
