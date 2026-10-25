import numpy as np
import pytest

from r2p.frames import HOME_Q
from r2p.perception import CameraModel, TaskPerceiver
from r2p.synthbench import World, OracleDetector


@pytest.mark.parametrize("name", ["cube_in_bowl", "pick_place_stick", "stack_two_cubes"])
@pytest.mark.parametrize("seed", [2, 8])
def test_synthbench_pose_accuracy(name, seed):
    cam = CameraModel.look_at((.50, -.32, .65), (.20, 0, .01), image_size=(960, 720))
    with World(name, np.random.default_rng(seed), cam) as world:
        parked = world.q()
        parked[0] = -1.6
        world.step(parked)  # accuracy protocol uses an unobstructed setup view
        frame = world.render()
        detector = OracleDetector(world)
        with TaskPerceiver(cam, world.task, detector) as perceiver:
            perceiver.setup(frame)
            result = perceiver.estimate(frame, world.q())
            assert detector.calls == ["small"]
            for oid, expected in world.poses().items():
                assert result["poses"][oid]["seen"], (name, seed, oid, result["poses"])
                error = np.linalg.norm(np.asarray(result["poses"][oid]["xyz"])-expected["xyz"])
                assert error < .003, (name, seed, oid, error, result["poses"][oid], expected)
                if world.task["objects"][0]["shape"] == "stick":
                    assert -90 <= result["poses"][oid]["yaw_deg"] < 90


def test_saved_setup_zero_detector_calls_and_stale_result(tmp_path):
    with World("cube_in_bowl", np.random.default_rng(2)) as world:
        image = world.render()
        clock = [0.]
        perceiver = TaskPerceiver(world.cam, world.task, OracleDetector(world), clock=lambda: clock[0])
        perceiver.setup(image)
        path = tmp_path / "setup.json"
        perceiver.save_setup(path)
        def forbidden(*args, **kwargs):
            raise AssertionError("Saved replay must never call a detector")
        replay = TaskPerceiver(world.cam, world.task, forbidden, clock=lambda: clock[0]).load_setup(path)
        assert replay.estimate(image, HOME_Q)["poses"]
        clock[0] = 20.
        result = replay.estimate(np.zeros_like(image), HOME_Q)
        assert result["lost"]
        assert result["redetect_count"] == 0
        assert not replay.check()["ok"]


def test_async_main_redetect_budget_and_cooldown(monkeypatch):
    monkeypatch.delenv("R2P_CACHE_ONLY")
    with World("cube_in_bowl", np.random.default_rng(2)) as world:
        frame = world.render()
        oracle = OracleDetector(world)
        now = [0.]
        class Fake:
            calls = []
            def detect(self, frame, task, *, role):
                self.calls.append(role)
                return oracle.detect(frame, task, role=role) if role == "small" else {}
        fake = Fake()
        with TaskPerceiver(world.cam, world.task, fake, clock=lambda: now[0]) as p:
            p.setup(frame)
            blank = np.zeros_like(frame)
            now[0] = .99
            p.estimate(blank, HOME_Q)
            assert fake.calls == ["small"]
            now[0] = 1.01
            p.estimate(blank, HOME_Q)
            p._worker.join(2)
            assert fake.calls == ["small", "main"]
            now[0] = 2
            p.estimate(blank, HOME_Q)
            assert p.redetect_count == 1
            for i in range(8):
                now[0] += 8.1
                p.estimate(blank, HOME_Q)
                p._worker.join(2)
            assert fake.calls.count("main") == 6
            assert p.redetect_count == 6
