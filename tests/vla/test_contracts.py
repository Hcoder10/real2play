import json
from pathlib import Path
import numpy as np
import pytest
from r2p.vla.contracts import instruction, commanded_state, action_chunk, target_cell, Normalizer, STATE


def test_runtime_and_trace_language_are_identical():
    assert instruction("place_at(piece,cell)", "r1c2") == instruction("place_at(piece,r1c2)")
    assert instruction({"skill": "place_at", "obj": "piece", "xy": "r1c2"}) == "Place the piece on cell r1c2."
    assert instruction("pick(piece)") == "Pick up the piece."
    assert instruction("release") == "Release the piece."
    assert instruction("home") == "Return the arm to its starting pose."
    with pytest.raises(ValueError):
        instruction("place_at(piece,cell)")
    with pytest.raises(ValueError):
        instruction("place_at(piece,r9c9)")


def test_contact_reset_preserves_command_clock():
    commanded = np.zeros(6)
    measured = np.array([0, 0, 0, 0, 0, .2])
    state, reset = commanded_state(commanded, measured, 1.75)
    assert reset
    np.testing.assert_allclose(state[:6], measured)
    assert state[6] == 1.75
    measured[5] = .01
    state, reset = commanded_state(commanded, measured, 2.)
    assert not reset
    np.testing.assert_array_equal(state[:6], commanded)


def test_chunk_never_crosses_commands_or_uses_executed_target():
    ticks = [{"command": "pick(piece)", "q_target": [i]*6, "q_exec": [99]*6} for i in range(3)]
    ticks += [{"command": "release", "q_target": [90]*6}]
    chunk = action_chunk(ticks, 1)
    assert chunk.shape == (15, 6)
    np.testing.assert_array_equal(chunk[0], [1]*6)
    np.testing.assert_array_equal(chunk[1:], np.full((14, 6), 2))


def test_actual_stats_roundtrip_and_constant_clamp():
    state = np.array([[1, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 2]], np.float32)
    actions = np.arange(180, dtype=np.float32).reshape(2, 15, 6)/100
    norm = Normalizer.fit(state, actions)
    np.testing.assert_allclose(norm.action(norm.normalize("action", actions)), actions, atol=1e-6)
    observed = state[0].copy()
    observed[1] = 1
    result = norm.normalize(STATE, observed, clamp_constants=True)
    assert result[1] == 0
    assert result[0] == -1
    assert norm.stats["action"]["count"] == 30


def test_cell_resolution_uses_randomized_scene_not_nominal_board():
    scene = {"objects": [{"id": "cell", "xyz": [.3, .1, 0]},
                         {"id": "r0c0", "xyz": [.2, .1, 0]},
                         {"id": "r1c2", "xyz": [.3, .1, 0]}]}
    assert target_cell(scene) == "r1c2"


def test_no_gpu_without_explicit_lease(monkeypatch):
    pytest.importorskip("torch")
    from r2p.vla.runtime import configure_device
    monkeypatch.delenv("LANDLORD_LEASE_ID", raising=False)
    with pytest.raises(RuntimeError, match="lease|LEASE"):
        configure_device("cuda:0", .167)


def test_argon_uses_main_and_identical_training_sentence():
    from r2p.vla.adapter import argon_command
    calls = []
    def ask(parts, **kwargs):
        calls.append(kwargs)
        return json.dumps({"command": "place_at(piece,r1c2)"}), {"served_by": "gemini-4-argon"}
    command = argon_command(np.zeros((192, 256, 3), np.uint8), "r1c2", {"picked": True}, ask=ask)
    assert calls[0]["role"] == "main"
    assert instruction(command) == instruction("place_at(piece,cell)", "r1c2")


def test_session_adapter_keeps_explicit_argon_command_and_measured_q():
    from r2p.vla.adapter import SessionPolicy
    class Stub:
        config = {"camera": {"image_size": [256, 192], "K": [[200, 0, 128], [0, 200, 96], [0, 0, 1]], "dist": [0]*5}}
        def act(self, obs):
            self.seen = obs
            return np.ones((15, 6))
    policy = Stub()
    adapter = SessionPolicy(policy, "r1c2", "place_at(piece,r1c2)")
    frame = np.zeros((192, 256, 3), np.uint8)
    frame[..., 0] = 255
    adapter.observe(np.zeros(6), frame, frame, 15)
    result = adapter.act({"q": np.ones(6), "command": "pick(piece)"})
    assert result.shape == (15, 6)
    assert policy.seen["command"] == "place_at(piece,r1c2)"
    assert policy.seen["tick"] == 15
    assert policy.seen["images"]["scene"][50, 50, 2] == 255
    np.testing.assert_array_equal(policy.seen["q_measured"], np.zeros(6))


def test_policy_clock_advances_by_control_ticks_and_survives_gripper_reset(tmp_path):
    from r2p.vla.runtime import Policy
    (tmp_path/"config.json").write_text("{}")
    stats = Normalizer.fit(np.zeros((2, 7)), np.zeros((2, 15, 6))).stats
    (tmp_path/"train_stats.json").write_text(json.dumps(stats))
    class Model:
        def reset(self):
            pass
    policy = Policy(tmp_path, Model(), None, None)
    seen = []
    def predict(states, images, texts):
        seen.append(states.copy())
        return np.ones((1, 15, 6))*.123
    policy.predict = predict
    images = {name: np.zeros((192, 256, 3), np.uint8) for name in ("scene", "wrist")}
    obs = {"q": np.ones(6), "q_measured": np.zeros(6), "command": "pick(piece)", "images": images}
    chunk = policy.act(dict(obs, tick=10))
    policy.act(dict(obs, tick=25))
    assert seen[-1][0, 6] == .5
    np.testing.assert_array_equal(seen[-1][0, :6], np.zeros(6))
    np.testing.assert_allclose(chunk, .123)  # no addition of observed q
    policy.act(dict(obs, command="place_at(piece,r1c2)", tick=26))
    assert seen[-1][0, 6] == 0
