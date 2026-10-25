import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from r2p.gemini import FakeGemini, QuotaError
from r2p.real2sim import SessionData, extract_anchors, keep_candidate, load_session, run_twin, split_frames, task_from_video
from r2p.real2sim.evaluate import mask_metrics, physics_replay
from r2p.real2sim.patch import apply_patch


class JawArm:
    def tips(self, q):
        a = np.asarray(q[:3], float)
        b = a + [0, 0.01 + q[5] * 0.1, 0]
        return a, b, (a + b) / 2


def ticks(n=60):
    rows = []
    for i in range(n):
        closed = 10 <= i < 40
        q = [0.2, 0, 0.025, 0, 0, 0.3 if closed else 1.0]
        target = q[:]
        target[5] = 0.0 if closed else 1.0
        rows.append({"t": i / 30, "q": q, "q_target": target, "poses": {}, "grip_closed": closed,
                     "phase": "test", "command": "pick(cube)"})
    return rows


def session_fixture(tmp_path, n=8):
    from r2p.perception import CameraModel

    camera = CameraModel.look_at(image_size=(64, 48))
    frames = []
    for i in range(n):
        image = np.full((48, 64, 3), 30 + i * 10, np.uint8)
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), image)
        frames.append(path)
    return SessionData(tmp_path, {"session_kind": "synthetic"}, [], frames, [i / 30 for i in range(n)], camera)


def metrics(score=0.5, cube=0.8, bowl=0.8, camera=1, anchor=0.01, inside=True):
    return {"score": score, "objects": {"cube": {"iou": cube}, "bowl": {"iou": bowl}},
            "camera_px": camera, "anchor_error_m": anchor, "release_inside": inside}


@pytest.mark.parametrize("change, enforced, reason", [
    ({"score": .503}, True, "insufficient_gain"),
    ({"cube": .69}, True, "iou_regression:cube"),
    ({"bowl": .779}, False, "container_iou_regression:bowl"),
    ({"camera": 1.001}, True, "camera_regression"),
    ({"anchor": .0101}, True, "anchor_regression"),
    ({"inside": False}, True, "release_outside_container"),
])
def test_keep_rule_independent_gates(change, enforced, reason):
    candidate = metrics(**{"score": 0.6, **change})
    verdict = keep_candidate(candidate, metrics(), containers=["bowl"], anchors_enforced=enforced)
    assert not verdict["kept"] and reason in verdict["reasons"]


def test_keep_per_object_peak_and_boundary():
    assert keep_candidate(metrics(.51, cube=.7), metrics())["kept"]
    verdict = keep_candidate(metrics(.9, cube=.7), metrics(), {"cube": .95, "bowl": .8})
    assert not verdict["kept"]
    assert keep_candidate(metrics(.6, bowl=.78), metrics(), containers=["bowl"], anchors_enforced=False)["kept"]
    candidate = metrics(.7)
    candidate["camera_px"] = None
    assert not keep_candidate(candidate, metrics())["kept"]


def test_anchors_use_most_closed_command_not_measured_percentile():
    rows = ticks()
    facts = extract_anchors(rows, arm=JawArm())
    assert len(facts) == 1
    assert facts[0]["empty_closed_rad"] == 0
    assert facts[0]["grasp_index"] == 10
    assert facts[0]["release_t"] == 40 / 30
    assert facts[0]["width_m"] == pytest.approx(0.03)
    assert facts[0]["source"] == "gripper_commands_and_measured_jaw_gap"
    # Measured stall alone must not create an anchor when no close was commanded.
    for row in rows:
        row["q_target"][5] = 1
    assert extract_anchors(rows, arm=JawArm()) == []


def test_empty_gripper_close_is_not_a_grasp():
    rows = ticks()
    for row in rows:
        row["q"][5] = row["q_target"][5]
    assert extract_anchors(rows, arm=JawArm()) == []


def test_soft_and_inconsistent_anchors_never_move_the_scene():
    from r2p.real2sim.anchors import apply_anchors
    scene = {"objects": [{"id": "cube", "role": "manipulated", "shape": "box", "xyz": [.3, 0, .015],
                           "size_m": [.03]*3}]}
    facts = [{"grasp_xyz": [.2, 0, .015], "width_m": .02, "release_xyz": None}]
    assert apply_anchors(scene, copy.deepcopy(facts), mode="soft") == scene
    assert apply_anchors(scene, copy.deepcopy(facts), mode="auto") == scene
    hard = apply_anchors(scene, copy.deepcopy(facts), mode="hard")
    assert hard["objects"][0]["xyz"][0] == .2
    assert hard["objects"][0]["size_m"][1] == pytest.approx(.024)


def test_closing_in_transit_without_a_hold_does_not_anchor():
    rows = ticks()
    for i in range(10, 40):
        rows[i]["q"][5] = .9 - .025 * (i - 10)
    assert extract_anchors(rows, arm=JawArm()) == []


def test_session_trace_camera_frames_and_video(tmp_path):
    from r2p.tasks import load_task
    from r2p.traces import write_trace
    from r2p.perception import CameraModel

    task = load_task("cube_in_bowl")
    rows = ticks(5)
    for i, row in enumerate(rows):
        filename = f"frames/{i:04}.jpg"
        (tmp_path / "frames").mkdir(exist_ok=True)
        cv2.imwrite(str(tmp_path / filename), np.zeros((48, 64, 3), np.uint8))
        row["frame"] = filename
    meta = {"schema": "r2p_trace/1", "task": task, "task_name": task["name"], "source": "real", "player": None,
            "level": None, "seed": 1, "scene": {"objects": []}, "success": False, "fail_reason": None,
            "commands": [], "rig": "test"}
    write_trace(tmp_path / "trace.jsonl", meta, rows)
    CameraModel.look_at(image_size=(64, 48)).save(tmp_path / "camera.json")
    session = load_session(tmp_path)
    assert len(session.frames) == 5 and session.frame(0).shape == (48, 64, 3)
    assert session.camera is not None and session.ticks[2]["q_target"] == rows[2]["q_target"]
    assert session.video_path(tmp_path / "generated.mp4").exists()
    assert session.joint_table().count("|") == 4  # header plus first 5 Hz sample
    critic, heldout = split_frames(session)
    assert not set(critic) & set(heldout)


def test_patch_rejects_camera_nan_and_unknown_objects():
    scene = {"objects": [{"id": "cube", "size_m": [.03] * 3, "xyz": [.2, 0, .015]}]}
    patch = {"objects": [{"id": "cube", "xyz": [.21, 0, .015]}], "why": "move", "done": False}
    updated = apply_patch(scene, patch)
    assert updated["objects"][0]["xyz"][0] == .21 and scene["objects"][0]["xyz"][0] == .2
    for bad in ({**patch, "camera": {}}, {**patch, "objects": [{"id": "ghost", "mass_kg": .2}]},
                {**patch, "objects": [{"id": "cube", "friction": float("nan")}]},
                {**patch, "objects": [{"id": "cube", "size_m": [30, 30, 30]}]}):
        with pytest.raises(ValueError):
            apply_patch(scene, bad)


class Mask:
    def mask(self, frame):
        value = np.zeros(frame.shape[:2], bool)
        value[10:20, 10:20] = True
        return value


def replay_stub(session, scene, *, indices, **kwargs):
    return {i: {"image": session.frame(i), "masks": {obj["id"]: Mask().mask(session.frame(i)) for obj in scene["objects"]},
                "poses": {obj["id"]: {"xyz": obj["xyz"]} for obj in scene["objects"]}} for i in indices}


def test_twin_loop_keeps_best_plateau_and_atomic_contract(tmp_path):
    session = session_fixture(tmp_path)
    scene = {"objects": [{"id": "cube", "role": "manipulated", "shape": "box", "xyz": [.2, 0, .015], "size_m": [.03]*3},
                          {"id": "bowl", "role": "container", "shape": "bowl", "xyz": [.3, 0, .02], "size_m": [.1, .04]}]}
    replies = [{"objects": [{"id": "cube", "xyz": [x, 0, .015]}], "why": "fit", "done": False}
               for x in (.21, .22, .23)]
    fake = FakeGemini(replies)
    calls = []

    def evaluator(session, candidate, indices, *args):
        calls.append(indices)
        x = candidate["objects"][0]["xyz"][0]
        return metrics(score=.5 if x == .2 else .6 if x == .21 else .7, cube=.8 if x <= .21 else .6)
    result = run_twin(session, scene=scene, appearances={"cube": Mask(), "bowl": Mask()}, ask=fake,
                      out_dir=tmp_path / "live", replay=replay_stub, evaluator=evaluator, iterations=6)
    assert result["best_iter"] == 1 and result["stop_reason"] == "plateau"
    assert result["scene"]["objects"][0]["xyz"][0] == .21
    run = json.loads((tmp_path / "live/run.json").read_text())
    assert run["schema"] == "r2s_live/1" and run["status"] == "done"
    assert [entry["kept"] for entry in run["iters"]] == [True, True, False, False]
    assert run["calls"]["by_model"] == {"gemini-4-argon": 3}
    assert all(indices == run["eval_frames"] for indices in calls)
    assert not set(run["eval_frames"]) & set(run["critic_frames"])
    for request in fake.requests:
        # prompt, four critic overlays, one critic-only trajectory image
        assert len(request["parts"]) == len(run["critic_frames"]) + 2
    assert cv2.imread(str(tmp_path / "live/iter_1_overlay_0.jpg")).shape == (480, 1280, 3)
    assert (tmp_path / "live/iter_3_physics.jpg").exists()
    assert not list((tmp_path / "live").glob("*.tmp"))


def test_quota_retains_initial_scene(tmp_path):
    session = session_fixture(tmp_path)
    scene = {"objects": [{"id": "cube", "role": "manipulated", "shape": "box", "xyz": [.2, 0, .015], "size_m": [.03]*3}]}
    def unavailable(*args, **kwargs):
        raise QuotaError("Daily quota")
    result = run_twin(session, scene=scene, appearances={"cube": Mask()}, ask=unavailable,
                      out_dir=tmp_path / "live", replay=replay_stub, evaluator=lambda *args: metrics())
    assert result["best_iter"] == 0 and result["stop_reason"] == "quota"
    assert (tmp_path / "live/scene_best.json").exists()


def test_task_video_validation_reasks_and_reuses_whole_video(tmp_path):
    from r2p.tasks import load_task, validate
    session = session_fixture(tmp_path)
    invalid = load_task("cube_in_bowl")
    invalid["goal"][0]["pred"] = "teleport"
    valid = load_task("cube_in_bowl")
    valid.pop("randomize")
    fake = FakeGemini([{"task": invalid}, {"task": valid}])
    fitted = []
    def twin(source, **kwargs):
        fitted.append(kwargs)
        return {"scene": {"version": 0, "objects": [{"id": valid["objects"][0]["id"], "xyz": [.24, .03, .02]}]}}
    result = task_from_video(session, out_root=tmp_path / "tasks", ask=fake, twin_runner=twin)
    assert validate(result["task"]) == [] and result["scene_status"] == "fitted"
    assert len(fake.requests) == 2 and len(fitted) == 1
    assert "goal[0].pred" in fake.requests[1]["parts"][-1]
    assert fake.requests[0]["parts"][1] == fake.requests[1]["parts"][1]
    assert fake.requests[0]["parts"][1]["inline_data"]["mime_type"] == "video/mp4"
    assert Path(result["task_path"]).exists() and Path(result["scene_path"]).exists()


def test_task_video_stops_after_two_reasks(tmp_path):
    session = session_fixture(tmp_path)
    fake = FakeGemini([{"task": {"name": "bad"}}] * 3)
    with pytest.raises(ValueError, match="still invalid"):
        task_from_video(session, out_root=tmp_path / "tasks", ask=fake)
    # Different correction prompts can reuse a cached reply; request count never exceeds three.
    assert len(fake.records) == 3 and len(fake.requests) <= 3
    assert not (tmp_path / "tasks/bad/task.json").exists()


def test_bare_video_does_not_claim_metric_scene_without_calibration(tmp_path):
    from r2p.tasks import load_task
    session = session_fixture(tmp_path)
    session.camera = None
    result = task_from_video(session, out_root=tmp_path / "tasks", ask=FakeGemini([{"task": load_task("cube_in_bowl")}]))
    assert result["scene_status"] == "requires_calibrated_camera" and result["scene_path"] is None


def test_mask_metrics_penalize_missing_objects():
    real = np.zeros((20, 20), bool)
    real[4:8, 5:9] = True
    assert mask_metrics(real, real) == {"iou": 1.0, "centroid_err_px": 0.0}
    lost = mask_metrics(real, np.zeros_like(real))
    assert lost["iou"] == 0 and lost["centroid_err_px"] > 20


def test_physics_replay_uses_commanded_q_and_timestamps(tmp_path):
    session = session_fixture(tmp_path, n=4)
    session.ticks = ticks(4)
    commands = []
    class Twin:
        def __init__(self, scene, physics, q0):
            self._q = q0
        def step(self, q):
            commands.append(q)
            self._q = q
        def q(self):
            return self._q
        def poses(self):
            return {}
        def grip_closed(self):
            return False
    physics_replay(session, {"objects": []}, twin_factory=Twin, render=False)
    assert commands == [row["q_target"] for row in session.ticks[:3]]
