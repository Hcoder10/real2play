"""Active identification acceptance: physical truth, held-out gate, no live IO."""
from copy import deepcopy
from dataclasses import replace
import json
import time

import numpy as np
import pytest

from r2p import gemini
from r2p.model import JointMap
from r2p.perception import CameraModel
from r2p.real2sim.active import (
    ArmdMover, Experiment, Safety, SimMover, UnsafeExperiment, Validator,
    bench_scene, choose_experiment, experiment_menu, fit_parameters,
    initial_parameters, keep_heldout, materialize, offline_client,
    run_campaign, start_q,
)
from r2p.real2sim.session import load_session
from r2p.sim.twin import load_physics


@pytest.fixture(scope="module")
def setup():
    return bench_scene(), Validator(), start_q()


@pytest.mark.parametrize("experiment", [
    {"kind": "dance"}, {"kind": "hold_pose", "xyz": [.5, 0, .12]},
    {"kind": "hold_pose", "xyz": [.18, -.1, -.01]},
    {"kind": "hold_pose", "xyz": [.1653, .0154, .13]},
    {"kind": "joint_sweep", "joint": 5},
    {"kind": "joint_sweep", "joint": 0, "amplitude_deg": 100},
    {"kind": "joint_sweep", "joint": 1, "speed": 21},
    {"kind": "hold_pose", "xyz": [.18, -.1, float("nan")]},
    {"kind": "push", "obj": "piece", "distance_m": .3},
    {"kind": "push", "obj": "r0c0"},
    {"kind": "push", "obj": "piece", "speed": True},
    {"kind": "hold_pose", "xyz": [.18, -.1, .12], "cmd": "release"},
])
def test_unsafe_experiments_refused(setup, experiment):
    scene, validator, q0 = setup
    assert not validator.verdict(experiment, scene, q0)["safe"]


def test_whole_path_bounds_and_safe_finish(setup):
    scene, validator, q0 = setup
    for experiment in experiment_menu(scene):
        verdict = validator.verdict(experiment, scene, q0)
        if not verdict["safe"]:
            assert experiment.kind == "pick_lift_place"
            continue
        plan = validator.plan(experiment, scene, q0)
        speed = np.abs(np.diff(np.vstack((q0, plan.targets)), axis=0))*30
        assert np.max(speed) <= np.deg2rad(experiment.speed)+1e-9
        assert validator.arm.tips(plan.targets[-1])[2][2] >= validator.safety.safe_z-.001
    blocked = deepcopy(scene)
    blocked["objects"].append({"id": "unexpected_cap", "role": "obstacle", "xyz": [.17, -.092, .0085],
                               "size_m": [.034, .017], "shape": "cylinder", "fixed": True})
    assert not validator.verdict(experiment_menu(scene)[2], blocked, q0)["safe"]


def test_legacy_armd_is_read_only_and_frame_is_not_shifted(setup, tmp_path):
    scene, validator, q0 = setup
    jm = JointMap()
    state = {"q": jm.to_owner(q0).tolist(), "joint_map": "identity", "lo": [-180]*5+[0],
             "hi": [180]*5+[100], "min_z": {"finger": .1}, "tip": validator.arm.tips(q0)[2].tolist()}
    requests = []
    def read_only(url, payload=None):
        requests.append((url, payload))
        assert payload is None
        return state
    mover = ArmdMover(scene, CameraModel.look_at(), joint_map=jm, request=read_only)
    np.testing.assert_allclose(mover.measured_q(state), q0)
    with pytest.raises(UnsafeExperiment, match="blocks /state"):
        mover.execute(experiment_menu(scene)[0], tmp_path/"forbidden")
    assert requests and all(payload is None for _, payload in requests)
    assert not (tmp_path/"forbidden").exists()


def test_invalid_argon_choice_falls_back_and_quota_does_not(setup):
    scene, validator, q0 = setup
    parameters = initial_parameters(scene)
    uncertainty = {k: {"identified": False} for k in parameters}
    client = gemini.FakeGemini(responses=[{"kind": "hold_pose", "xyz": [.9, 0, .1]}])
    choice, record = choose_experiment(client, experiment_menu(scene), set(), parameters, uncertainty,
                                       [], validator, scene, q0)
    assert record["source"] == "informative_untried_fallback"
    assert validator.verdict(choice, scene, q0)["safe"]
    assert all(r["model"] == gemini.model_for("main") for r in client.requests)
    class QuotaClient:
        def call(self, *args, **kwargs):
            raise gemini.QuotaError("daily limit")
    with pytest.raises(gemini.QuotaError):
        choose_experiment(QuotaClient(), experiment_menu(scene), set(), parameters, uncertainty, [], validator, scene, q0)


def test_holdout_gate_forbids_visual_and_physical_regressions():
    baseline = {"error": 3., "joint_deg": 1., "jaw_mm": 2., "object_mm": 3., "iou": .8,
                "objects": {"piece": {"iou": .8}}}
    for field, value in (("joint_deg", 1.1), ("jaw_mm", 2.2), ("object_mm", 3.3)):
        proposed = {**baseline, "error": 1., field: value}
        assert not keep_heldout(proposed, baseline)["kept"]
    assert not keep_heldout({**baseline, "error": 1., "objects": {"piece": {"iou": .78}}}, baseline)["kept"]
    assert not keep_heldout({**baseline, "error": 1., "objects": {"piece": {"iou": float("nan")}}}, baseline)["kept"]
    assert keep_heldout({**baseline, "error": 2.}, baseline)["kept"]


def test_final_camera_pose_uses_its_own_timestamp(monkeypatch):
    from r2p.real2sim import active
    camera_tick = {"q": [0]*6, "poses": {"piece": {"xyz": [.2, -.1, .01]}}}
    final_sensor_tick = {"q": [0]*6, "poses": {}}
    camera_prediction = {"q": [0]*6, "poses": {"piece": {"xyz": [.201, -.1, .01]}}}
    later_prediction = {"q": [0]*6, "poses": {"piece": {"xyz": [.3, -.1, .01]}}}
    monkeypatch.setattr(active, "_replay", lambda *args: [(camera_tick, camera_prediction), (final_sensor_tick, later_prediction)])
    result = active._vector(None, {}, {}, joints=False)
    np.testing.assert_allclose(result, [.2, 0])


def test_sim_campaign_recovers_hidden_truth_without_fit_leak(tmp_path, monkeypatch):
    scene, physics = bench_scene(), load_physics()
    truth = {**initial_parameters(scene), "stiffness_scale": 1.35, "cap_friction": .45}
    truth_physics, truth_scene = materialize(physics, scene, truth)
    mover = SimMover(scene, truth_physics=truth_physics, truth_scene=truth_scene)
    def no_network(*args, **kwargs):
        raise AssertionError("Tests cannot call network")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    result = run_campaign(mover, budget=6, out_dir=tmp_path/"campaign", candidate_dir=tmp_path/"physics",
                          client=offline_client(), max_nfev=7)
    assert result["status"] == "complete"
    assert result["experiments_used"] == 6
    assert result["heldout"]["error"] < result["before"]["error"]*.5
    assert result["parameters"]["stiffness_scale"] == pytest.approx(1.35, abs=.02)
    assert result["parameters"]["cap_friction"] == pytest.approx(.45, abs=.04)
    assert result["parameters"]["cap_mass_kg"] == pytest.approx(.004, abs=.0005)
    assert not result["uncertainty"]["table_friction"]["identified"]
    assert not result["candidate_activated"]
    assert (tmp_path/"campaign/report.md").exists()
    session = load_session(tmp_path/"campaign/train_01")
    assert session.meta["schema"] == "r2p_trace/1" and len(session.frames) == 5
    assert session.ticks[-1]["poses"]["piece"]["source"] == "rendered_mask_fit"
    assert session.meta["split"] == "train"
    saved = json.loads((tmp_path/"physics/so101_bench.campaign.json").read_text())
    assert saved["active_twin"]["activated"] is False
    observation = mover.execute(experiment_menu(scene)[0], tmp_path/"leak", split="heldout")
    with pytest.raises(ValueError, match="training experiments only"):
        fit_parameters([observation], physics, scene, initial_parameters(scene))


def test_invalid_budget_never_moves(tmp_path):
    class NeverMover:
        def execute(self, *args, **kwargs):
            pytest.fail("must validate budget before motion")
    for budget in (0, 3, 31, True):
        with pytest.raises(ValueError, match="budget"):
            run_campaign(NeverMover(), budget=budget, out_dir=tmp_path/"invalid")


def test_streaming_mover_records_sensor_and_capture_times(tmp_path, setup):
    scene, validator, q0 = setup
    q = q0.copy()
    commands = []
    def request(url, payload=None):
        nonlocal q
        if payload is not None:
            commands.append(payload)
            if "joints" in payload:
                q = np.asarray(payload["joints"])
        return {"q": q.tolist(), "owner_deg": JointMap().to_owner(q).tolist(), "q_target": q.tolist()}
    camera = CameraModel.look_at(image_size=(64, 48))
    def capture():
        now = time.monotonic()
        return np.zeros((48, 64, 3), np.uint8), now, {"sequence": now, "timestamp_source": "injected_capture"}
    class Perceiver:
        def estimate(self, frame, joints):
            cap = scene["objects"][0]
            return {"poses": {cap["id"]: {"xyz": cap["xyz"], "yaw_deg": 0, "source": "vision", "seen": True}},
                    "masks": {cap["id"]: np.ones((48, 64), np.uint8)}}
    mover = ArmdMover(scene, camera, request=request, capture=capture, perceiver=Perceiver())
    e = Experiment("hold_pose", xyz=tuple(validator.arm.tips(q0)[2]), duration_s=.3)
    observation = mover.execute(e, tmp_path/"stream")
    session = load_session(observation.session.path)
    assert session.meta["resampled"] and session.meta["sensor_hz"] >= 10
    assert session.meta["capture_timestamps"]
    assert len(session.frames) > 2 and all(c.get("stream") for c in commands)
    assert (tmp_path/"stream_raw.json").exists()
    np.testing.assert_allclose(np.diff([t["t"] for t in session.ticks]), 1/30)


def test_streaming_refusal_holds_without_blind_retreat(tmp_path, setup):
    scene, validator, q0 = setup
    commands = []
    def request(url, payload=None):
        if payload is not None:
            commands.append(payload)
            return {"ok": False, "refused": "load"} if "joints" in payload else {"ok": True}
        return {"q": q0.tolist(), "owner_deg": q0.tolist(), "q_target": q0.tolist()}
    class Perceiver:
        def estimate(self, *args):
            return {"poses": {"piece": {"xyz": scene["objects"][0]["xyz"], "source": "vision", "seen": True}}}
    mover = ArmdMover(scene, CameraModel.look_at(image_size=(64, 48)), request=request, perceiver=Perceiver(),
                      capture=lambda: (np.zeros((48, 64, 3), np.uint8), time.monotonic(), {}))
    with pytest.raises(UnsafeExperiment, match="refused/aborted"):
        mover.execute(experiment_menu(scene)[0], tmp_path/"refused")
    assert len(commands) == 2 and commands[-1] == {"hold": True}
    assert not (tmp_path/"refused").exists()
