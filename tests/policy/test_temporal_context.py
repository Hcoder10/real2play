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
    # This test isolates explicit recorded command boundaries. Scheduler parity
    # on physical state transitions is covered separately below.
    config.pop("command_alignment")
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


def test_dataset_recovers_executed_q_before_arm_only_reset(tmp_path):
    task = load_task("cube_in_bowl")
    config = export_config(task, "executed_q")
    scene = sample_scene(task, np.random.default_rng(3))
    poses = {o["id"]: o for o in scene["objects"]}
    closed = HOME_Q.copy()
    closed[5] = -.12
    measured = closed.copy()
    measured[5] = .2  # Contact blocks the jaw; old recorder reset the full input.
    rows = [{"t": i/30, "q": measured.copy(), "q_commanded": measured.copy(),
             "q_target": closed.copy(), "q_exec": closed.copy(), "poses": poses,
             "grip_closed": True, "phase": "close", "command": commands(task)[0]}
            for i in range(3)]
    # Actual arm divergence must still reset all input joints to measurement.
    rows[2]["q"][0] += .2
    path = tmp_path/"contact.jsonl"
    write_trace(path, {"task": task, "source": "expert", "start_q": HOME_Q}, rows)
    dataset = TraceDataset(task, path, config)
    np.testing.assert_allclose(dataset.raw[1, :6], closed, atol=1e-7)
    np.testing.assert_allclose(dataset.raw[2, :6], rows[2]["q"], atol=1e-7)


def test_training_command_clock_and_chunks_follow_inference_tracker(tmp_path):
    from r2p.policy.data import Augmentation, trace_commands
    from r2p.policy.eval import CommandTracker

    task = load_task("cube_in_bowl")
    config = export_config(task, "scheduler_parity")
    scene = sample_scene(task, np.random.default_rng(3))
    initial = {o["id"]: o for o in scene["objects"]}
    obj = task["skeleton"][0]["obj"]
    closed = HOME_Q.copy()
    closed[5] = -.12
    rows = []
    for i in range(10):
        poses = deepcopy(initial)
        if i:
            poses[obj]["xyz"][2] += .05
        target = closed.copy()
        target[0] += i*.001
        rows.append({"t": i/30, "q": closed, "q_commanded": closed,
                     "q_target": target, "q_exec": target, "poses": poses,
                     "grip_closed": True, "phase": "lift", "command": commands(task)[0]})
    meta = {"task": task, "source": "expert", "start_q": HOME_Q}
    expected = []
    tracker = CommandTracker(task, initial)
    for i, row in enumerate(rows):
        if i:
            tracker.update({"objects": row["poses"], "gripper_open": False}, row["q"], HOME_Q)
        expected.append(tracker.command)
    assert trace_commands(task, rows, meta, config) == expected
    assert expected[:4] == [commands(task)[0]]*4
    assert expected[4:] == [commands(task)[1]]*6
    path = tmp_path/"lift.jsonl"
    write_trace(path, meta, rows)
    dataset = TraceDataset(task, path, config, Augmentation(smooth_ticks=1))
    np.testing.assert_allclose(dataset.raw[:, -1], [0, 1/30, 2/30, 3/30, 0, 1/30, 2/30, 3/30, 4/30, 5/30])
    # Boundary padding belongs to the same new command as the observation.
    labels = dataset.labels.reshape(-1, 15, 6)
    np.testing.assert_allclose(labels[3], np.tile(rows[3]["q_target"], (15, 1)), atol=1e-7)
    # The default twin regards exactly .15 rad as open; an open, elevated
    # object must not complete pick during trace alignment either.
    for row in rows:
        row["q_exec"][5] = .15
    assert trace_commands(task, rows, meta, config) == [commands(task)[0]]*len(rows)


@pytest.mark.parametrize('step', [.10, np.deg2rad(4)/30])
def test_eval_rate_limit_uses_last_sent_even_when_measured_q_resets(monkeypatch, step):
    import r2p.policy.eval as evaluation

    task = load_task("cube_in_bowl")
    config = export_config(task, "blocked_gripper")
    if step != .10:
        config['deployment'] = {'joint_step_rad': step}
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
    # A squeezing (blocked) gripper is contact, not divergence: no observation resets (arm joints only), and the
    # gripper target is still rate-limited from the last SENT target.
    assert result["resets"] == 0
    np.testing.assert_allclose([row["target"][5] for row in rows], .9-step*np.arange(1, 5))
