import numpy as np
import pytest

from r2p.frames import HOME_Q
from r2p.perception import CameraModel, TaskPerceiver
from r2p.synthbench import World, OracleDetector


@pytest.mark.parametrize("name", ["cube_in_bowl", "pick_place_stick", "stack_two_cubes", "two_cubes_in_bowl", "push_to_target", "place_on_cell"])
@pytest.mark.parametrize("seed", [2, 4, 8])
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


def test_moving_objects_keep_nearest_component_and_stack_height():
    with World("stack_two_cubes", np.random.default_rng(2)) as world:
        q = world.q()
        q[0] = -1.6
        world.step(q)
        world.set_pose("top", [.24, -.06, .019], -20)
        world.set_pose("base", [.29, .04, .019], 15)
        frame = world.render()
        with TaskPerceiver(world.cam, world.task, OracleDetector(world)) as p:
            p.setup(frame)
            for step in range(1, 5):
                expected = [.24+.003*step, -.06+.002*step, .019]
                world.set_pose("top", expected, -20+step)
                result = p.estimate(world.render(), q)
                assert np.linalg.norm(np.asarray(result["poses"]["top"]["xyz"])-expected) < .003
            # Lifted top silhouette is re-fitted at the support's top plane.
            world.set_pose("top", [.29, .04, .057], -20)
            p.centers["top"] = world.cam.project([.29, .04, .057]).tolist()
            p.poses["top"]["xyz"] = [.29, .04, .04]
            result = p.estimate(world.render(), q)
            assert np.linalg.norm(np.asarray(result["poses"]["top"]["xyz"])-[.29, .04, .057]) < .003


def test_fixed_objects_never_colour_tracked_and_cache_only_blocks_recovery():
    with World("pick_place_stick", np.random.default_rng(2)) as world:
        q = world.q()
        q[0] = -1.6
        world.step(q)
        detector = OracleDetector(world)
        now = [0.]
        with TaskPerceiver(world.cam, world.task, detector, clock=lambda: now[0]) as p:
            p.setup(world.render())
            assert "zone" not in p.models
            original = p.poses["zone"]["xyz"].copy()
            now[0] = 10.
            result = p.estimate(np.zeros((world.cam.height, world.cam.width, 3), np.uint8), q)
            assert result["poses"]["zone"]["xyz"] == original
            assert result["poses"]["zone"]["source"] == "known"
            assert detector.calls == ["small"]
            assert p.redetect_count == 0


def test_gemini_default_detector_makes_one_small_call(monkeypatch):
    import json
    from r2p import gemini
    from r2p.perception.task_perception import GeminiDetector
    calls = []
    def fake_call(parts, **kwargs):
        calls.append(kwargs)
        return json.dumps({"objects": [{"id": "cube", "box_2d": [100, 200, 300, 400]}]}), {"cached": True}
    monkeypatch.setattr(gemini, "call", fake_call)
    detector = GeminiDetector()
    boxes = detector.detect(np.zeros((100, 200, 3), np.uint8), {"objects": [{"id": "cube"}]})
    assert boxes == {"cube": [40, 10, 80, 30]}
    assert len(calls) == 1 and calls[0]["role"] == "small"


def test_known_supply_rigid_fk_and_release_does_not_reset_to_supply():
    import copy
    from types import SimpleNamespace
    from r2p.perception.geometry import rotation
    with World("cube_in_bowl", np.random.default_rng(2)) as world:
        task = copy.deepcopy(world.task)
        supply = np.asarray(world.poses()["cube"]["xyz"])
        task["objects"][0]["xy"] = supply[:2].tolist()
        task["objects"][0]["pose_verified"] = True
        class FakeArm:
            mid = supply + [.003, 0, 0]
            R = np.eye(3)
            m = SimpleNamespace(body=lambda name: SimpleNamespace(id=0))
            def tips(self, q):
                return self.mid-[0, .02, 0], self.mid+[0, .02, 0], self.mid
            def fk(self, q):
                return SimpleNamespace(xmat=np.array([self.R.ravel()]))
        arm = FakeArm()
        with TaskPerceiver(world.cam, task, OracleDetector(world), arm=arm) as p:
            frame = world.render()
            p.setup(frame)
            assert "cube" not in p.models
            q = HOME_Q.copy()
            q[5] = 0
            result = p.estimate(frame, q)
            assert result["poses"]["cube"]["source"] == "fk"
            arm.mid = supply + [.04, .02, .10]
            arm.R = rotation(70)
            result = p.estimate(np.zeros_like(frame), q)
            expected = arm.mid + arm.R @ [-.003, 0, 0]
            assert np.allclose(result["poses"]["cube"]["xyz"], expected)
            assert result["holding"] == ["cube"]
            q[5] = .9
            result = p.estimate(frame, q)
            assert np.allclose(result["poses"]["cube"]["xyz"][:2], expected[:2])
            assert np.isclose(result["poses"]["cube"]["xyz"][2], .019)
            assert not result["holding"]


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
