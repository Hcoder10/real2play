import json
from pathlib import Path
import numpy as np
from r2p.frames import HOME_Q, ROOT
from r2p.vla.evaluate import Episode


def test_closed_loop_uses_servo_targets_latency_and_actual_checker(monkeypatch):
    class FakeCameras:
        def __init__(self, *args):
            pass
        def render(self, q, poses):
            return np.zeros((2, 192, 256, 3), np.uint8)
        def close(self):
            pass
    monkeypatch.setattr("r2p.vla.evaluate.Cameras", FakeCameras)
    task = json.loads((ROOT/"tests/policy/fixtures/place_on_cell.json").read_text())
    config = {"task_spec": task, "start_q": HOME_Q.tolist(), "physics": "so101_bench", "camera": {}, "mounts": {}}
    ep = Episode(config, 19100, "real", 15)
    try:
        state, images, text = ep.observe(0)
        assert state.shape == (7,) and images.shape == (2, 192, 256, 3)
        assert text == "Pick up the piece."
        ep.chunks.append((0, np.repeat((HOME_Q+.3)[None], 15, axis=0)))
        ep.step(0, 2)
        # real mode delays the first actuator command by one tick; no delta
        # action is added to measured sag, and no second gravity feedforward.
        np.testing.assert_allclose(ep.twin.commanded_q, np.clip(HOME_Q-ep.offset, *ep.twin.range.T))
        np.testing.assert_allclose(ep.last_sent, np.clip(HOME_Q+.1, *ep.twin.range.T))
        ep.step(1, 2)
        assert ep.done and not ep.result["success"]
        assert ep.result["reason"] == "timeout"
        assert "check" in ep.result and ep.result["ticks"] == 2
    finally:
        ep.close()
