from copy import deepcopy
from dataclasses import replace
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from r2p import gemini
from r2p.tasks import load_task, sample_scene
from r2p.real2sim.active import UnsafeExperiment, start_q
from r2p.real2sim.experiments import (TaskExperiment, Validator, choose_experiment,
    excited_parameters, experiment_menu, json_experiment, resolve)
from r2p.real2sim.general_active import materialize, parameter_space, fit_parameters, replay_scene
from r2p.real2sim.experiment_recording import DualCameraRecorder, SurveyedMover
from r2p.sim.twin import load_physics


@pytest.fixture(scope="module")
def bowl():
    task = load_task("cube_in_bowl")
    return task, sample_scene(task, np.random.default_rng(4))


@pytest.mark.parametrize("name,contact", [("cube_in_bowl", "pick_lift_place"), ("place_on_cell", "pick_lift_place"),
    ("push_to_target", "push"), ("stack_two_cubes", "pick_lift_place"),
    ("pick_place_stick", "pick_lift_place"), ("two_cubes_in_bowl", "pick_lift_place")])
def test_menu_is_derived_from_all_shipped_tasks(name, contact):
    task = load_task(name); scene = sample_scene(task, np.random.default_rng(4))
    menu = experiment_menu(task, scene)
    assert {e.kind for e in menu} == {"hold_pose", "joint_sweep", contact}
    assert all(resolve(task, scene, e) for e in menu)
    if name == "cube_in_bowl":
        assert any(np.linalg.norm(e.offset_xy) > .01 for e in menu)
        assert any("object:bowl:friction" in excited_parameters(task, e) for e in menu)
    if name == "push_to_target":
        assert all(e.location == "target" for e in menu if e.kind == "push")
    if name == "two_cubes_in_bowl":
        assert {task["skeleton"][e.step]["obj"] for e in menu} == {"cube_a", "cube_b"}


def test_scene_change_moves_menu_geometry(bowl):
    task, scene = bowl
    changed = deepcopy(scene); changed["objects"][0]["xyz"][1] += .015
    e = experiment_menu(task, changed)[0]
    assert resolve(task, changed, e)[3][1] == pytest.approx(scene["objects"][0]["xyz"][1]+.015)


def test_success_predicates_change_the_experiment_dwell():
    task = load_task("stack_two_cubes")
    scene = sample_scene(task, np.random.default_rng(4))
    contacts = [e for e in experiment_menu(task, scene) if e.kind == "pick_lift_place"]
    assert all(e.duration_s == 2 for e in contacts)


def test_contact_evidence_cannot_be_replaced_by_static_bowl_evidence(bowl):
    from r2p.real2sim.general_active import require_contact_evidence
    from r2p.real2sim.session import SessionData
    from pathlib import Path
    task, scene = bowl
    session = SessionData(Path("unused"), {"task": task, "scene": scene},
        [{"t": 0., "poses": {"bowl": {"xyz": [0,0,0]}}}], [0], [0.])
    observation = SimpleNamespace(experiment=TaskExperiment("pick_lift_place", 1), session=session)
    with pytest.raises(ValueError, match="missing the manipulated object"):
        require_contact_evidence(observation)


@pytest.mark.parametrize("change", [{"step": True}, {"speed": float("nan")}, {"speed": 100},
    {"kind": "push"}, {"offset_xy": [.2, 0]}, {"step": 999}, {"kind": "joint_sweep", "joint": 5}])
def test_invalid_or_unrelated_proposal_refused(bowl, change):
    task, scene = bowl
    value = {"kind": "pick_lift_place", "step": 1, **change}
    with pytest.raises((UnsafeExperiment, ValueError)):
        e = TaskExperiment.parse(value)
        resolve(task, scene, e)


def test_complete_path_speed_verticality_and_target_protection(bowl):
    task, scene = bowl; validator = Validator(task); q0 = start_q()
    e = TaskExperiment("pick_lift_place", 1)
    plan = validator.plan(e, scene, q0)
    speed = np.max(abs(np.diff(np.vstack((q0, plan.targets)), axis=0)))*30
    assert speed <= np.deg2rad(e.speed)+1e-8
    assert validator.arm.tips(plan.targets[-1])[2][2] >= validator.safety.safe_z-.003
    lift_roll = plan.targets[np.asarray(plan.phases) == "lift", 4]
    assert np.ptp(lift_roll) < np.deg2rad(3), "Lifting must not spin the grasp"
    low = [(p, validator.arm.tips(q)[2]) for p, q in zip(plan.phases, plan.targets) if p in ("lower", "lift", "place", "retreat")]
    for (phase_a, a), (phase_b, b) in zip(low, low[1:]):
        if phase_a == phase_b and min(a[2], b[2]) < .035:
            assert np.linalg.norm(a[:2]-b[:2]) < .001
    blocked = deepcopy(scene)
    blocked["objects"].append({"id": "protected", "role": "target", "shape": "box", "size_m": [.03]*3,
                                "xyz": scene["objects"][0]["xyz"], "fixed": True})
    assert not validator.verdict(e, blocked, q0)["safe"]
    near = deepcopy(scene); near["objects"][0]["xyz"] = [.11, 0, .019]
    assert not validator.verdict(e, near, q0)["safe"]


def test_argon_receives_refusal_and_can_design_new_valid_variant(bowl):
    task, scene = bowl
    class FastValidator:
        safety = Validator(task).safety
        def verdict(self, e, scene, q0): return {"safe": True}
        def plan(self, e, scene, q0): return resolve(task, scene, e)
    requests = []
    class Client:
        def call(self, parts, **kwargs):
            requests.append((json.loads(parts[0]), kwargs))
            answer = {"kind": "push", "step": 1} if len(requests) == 1 else {
                "kind": "hold_pose", "step": 1, "height_m": .103}
            return json.dumps(answer), {}
    e, info = choose_experiment(Client(), task, scene, {}, {}, [{"training_only": 1}], set(), FastValidator(), start_q())
    assert e.height_m == .103 and info["source"] == "main"
    assert requests[1][0]["refusals"][-1]["reason"] == "Push not required by this task skill"
    assert all(k["role"] == "main" and k["thinking"] == "high" for _, k in requests)
    assert requests[0][0]["task"] == task and requests[0][0]["scene"] == scene
    class Quota:
        def call(self, *a, **kw): raise gemini.QuotaError("daily quota")
    with pytest.raises(gemini.QuotaError):
        choose_experiment(Quota(), task, scene, {}, {}, [], set(), FastValidator(), start_q())


def test_parameter_bounds_and_materials_are_per_object(bowl):
    task, scene = bowl; physics = load_physics()
    values, bounds = parameter_space(task, scene, physics)
    assert bounds["object:bowl:mass_kg"][1] > .25
    p, s = materialize(physics, scene, {**values, "object:cube:mass_kg": .04})
    assert s["objects"][0]["mass_kg"] == .04 and s["objects"][1]["mass_kg"] == .25
    with pytest.raises(ValueError, match="training experiments only"):
        fit_parameters([SimpleNamespace(split="heldout")], task, physics, scene, values)
    surveyed = deepcopy(scene); surveyed["objects"][0]["xyz"][0] += .02
    observation = SimpleNamespace(session=SimpleNamespace(meta={"scene": surveyed}))
    assert replay_scene(observation, s)["objects"][0]["xyz"] == surveyed["objects"][0]["xyz"]


def test_real_wrist_limits_are_required_and_applied(bowl):
    from r2p.real2sim.experiments import apply_owner_limits
    from r2p.model import JointMap
    task, _ = bowl; validator = Validator(task); joint_map = JointMap()
    with pytest.raises(UnsafeExperiment, match="joint map differ"):
        apply_owner_limits(validator, joint_map, {})
    state = {"joint_map": joint_map.name, "lo": [-90, -90, -90, -45, -90, 0],
             "hi": [90, 90, 90, 45, 90, 100]}
    apply_owner_limits(validator, joint_map, state)
    assert validator.arm.range[3, 1] == pytest.approx(np.pi/4)
    assert validator.safety.owner_hi == state["hi"]


def test_dual_recording_persists_frames_and_failure(tmp_path):
    def capture():
        t = time.monotonic()
        return np.zeros((32, 32, 3), np.uint8), t, {"sequence": t}
    recorder = DualCameraRecorder(tmp_path/"cameras", capture, capture, hz=50)
    recorder.start(); time.sleep(.06); recorder.check(); recorder.close()
    result = json.loads((recorder.path/"cameras.json").read_text())
    assert all(len(rows) >= 2 for rows in result["frames"].values())
    assert all((recorder.path/rows[0]["file"]).exists() for rows in result["frames"].values())
    recorder.errors.append("wrist disconnected")
    with pytest.raises(UnsafeExperiment, match="wrist disconnected"):
        recorder.check()


def test_sheet_rotation_and_missing_survey_fail_before_motion(bowl):
    task, scene = bowl
    base = SimpleNamespace(scene=deepcopy(scene), capture=lambda: (None, 0, {}), url="test",
        request=lambda url: {}, measured_q=lambda state: start_q())
    points = {"a": [0., 0.], "b": [.1, 0.], "c": [0., .1]}
    wrapper = SurveyedMover(base, survey=lambda *a: {}, sheet_geometry=lambda frame: points)
    wrapper._guard(None)
    points["b"][1] += .005
    with pytest.raises(UnsafeExperiment, match="Sheet moved"):
        wrapper.prepare()
    points["b"][1] = 0
    with pytest.raises(UnsafeExperiment, match="Fresh survey missing"):
        wrapper.prepare()


def test_incomplete_video_campaign_is_explicit(tmp_path):
    from r2p.real2sim.twin_videos import build_videos
    result = build_videos(tmp_path)
    assert result["status"] == "waiting_for_campaign"
    assert result["missing"] and result["recorded_sessions"] == 0


def test_main_cap_sensor_uses_one_budgeted_read_and_consumes_reset_cache(bowl):
    from r2p.real2sim.experiment_recording import ArgonSurveyPerceiver
    from r2p.perception import CameraModel
    task, scene = bowl
    scene = deepcopy(scene)
    cap = scene["objects"][0]
    cap.update(shape="cylinder", size_m=[.034, .017], xyz=[.20, -.05, .0085])
    camera = CameraModel.look_at(image_size=(320, 240))
    uv = camera.project([.20, -.05, .017])
    centre = (uv[::-1]/[240, 320]*1000).tolist()
    answer = {"objects": [{"id": "cube", "support": "table", "top_center_yx": centre,
                            "outline_yx": [[centre[0]+a, centre[1]+b] for a, b in ((-15,-15),(-15,15),(15,15),(15,-15))]}]}
    class Client:
        calls = 0
        def call(self, parts, **kw):
            self.calls += 1
            assert kw["role"] == "main" and kw["thinking"] == "high"
            return json.dumps(answer), {"cached": True}
    client = Client(); sensor = ArgonSurveyPerceiver(task, scene, camera, client)
    frame = np.zeros((240, 320, 3), np.uint8)
    result = sensor.read(frame)
    np.testing.assert_allclose(result["poses"]["cube"]["xyz"], cap["xyz"], atol=1e-8)
    assert sensor.estimate(frame)["poses"]
    assert not sensor.estimate(frame)["poses"]
    assert client.calls == 1


def test_real_recording_checks_measured_jaw_before_write_and_keeps_abort_evidence(bowl, tmp_path):
    from r2p.real2sim.active import _solve
    task, scene = bowl; validator = Validator(task)
    low = _solve(validator.arm, np.array([.2, -.05, .004]), start_q(), .5)
    writes = []
    def request(url, payload=None):
        if payload:
            writes.append(payload)
        return {"q": low.tolist()}
    def capture():
        now = time.monotonic()
        return np.zeros((32, 32, 3), np.uint8), now, {"sequence": now}
    class Mover:
        def execute(self, e, destination, **kwargs):
            self.request(self.url+"/state")
            self.request(self.url+"/cmd", {"joints": start_q().tolist()})
    base = Mover()
    base.scene, base.validator, base.url = deepcopy(scene), validator, "fake"
    base.capture, base.request, base.measured_q = capture, request, lambda state: np.array(state["q"])
    def survey(*args):
        return {o["id"]: {"xyz": o["xyz"], "source": "vision", "seen": True} for o in scene["objects"]}
    wrapper = SurveyedMover(base, survey=survey, wrist_capture=capture,
        sheet_geometry=lambda frame: {"a": [0, 0], "b": [.1, 0], "c": [0, .1]})
    with pytest.raises(UnsafeExperiment, match="measured jaw"):
        wrapper.execute(TaskExperiment("hold_pose", 1), tmp_path/"abort")
    assert writes == []
    assert (tmp_path/"abort_cameras/scene/000000.jpg").exists()
    assert (tmp_path/"abort_cameras/wrist/000000.jpg").exists()
    assert (tmp_path/"abort_cameras/sheet_guard.json").exists()


def test_video_replay_writes_decodable_aligned_mp4s(tmp_path, bowl):
    import cv2
    from r2p.perception import CameraModel
    from r2p.real2sim.session import SessionData
    from r2p.real2sim.twin_videos import render_experiment
    task, scene = bowl
    frame = tmp_path/"frame.png"
    cv2.imwrite(str(frame), np.zeros((48, 64, 3), np.uint8))
    q = start_q().tolist()
    ticks = [{"t": i/30, "q": q, "q_target": q, "poses": {}} for i in range(7)]
    session = SessionData(tmp_path, {"source": "real"}, ticks, [frame, frame], [0., .2],
                          CameraModel.look_at(image_size=(64,48)))
    result = render_experiment(session, load_physics(), load_physics(), scene, scene, tmp_path/"videos", fps=5, width=64)
    assert result["frames"] == 2
    for key, path in result["files"].items():
        cap = cv2.VideoCapture(path)
        try:
            assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == (192 if key == "comparison" else 64)
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 2
        finally:
            cap.release()
