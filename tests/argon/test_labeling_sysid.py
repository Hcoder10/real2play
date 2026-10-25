import copy

import numpy as np
import pytest

from r2p.gemini import FakeGemini
from r2p.labeling import segment
from r2p.sysid import Parameter, fit_parameters, plan_experiments, validation_gate


def task():
    from r2p.tasks import load_task
    return load_task("cube_in_bowl")


def synthetic_trace():
    rows = []
    for i in range(60):
        q = [0.0] * 6
        q[5] = 0 if 15 <= i < 40 else 1
        q[0] = max(0, i - 42) * 0.01
        rows.append({"t": i / 30, "q": q.copy(), "q_target": q, "poses": {}, "phase": "", "command": ""})
    return rows


def test_gripper_and_motion_segmentation():
    result = segment(synthetic_trace(), task())
    assert [row[2].split("(")[0] for row in result] == ["pick", "place_in", "release", "home"]
    assert result[0][0] == 0 and result[-1][1] == 2.0
    assert result[0][1] == .5 and result[1][1] == 40 / 30
    assert all(a[1] == b[0] for a, b in zip(result, result[1:]))


def test_existing_human_command_labels_are_preserved():
    from r2p.tasks import commands
    rows = synthetic_trace()
    vocab = commands(task())
    for i, row in enumerate(rows):
        row["command"] = vocab[min(3, i // 15)]
    result = segment({"meta": {"source": "human", "task": task()}, "ticks": rows})
    assert [s[2] for s in result] == vocab
    assert [s[:2] for s in result] == [[0, .5], [.5, 1], [1, 1.5], [1.5, 2]]


def test_refinement_is_main_only_and_must_preserve_partition():
    rows = synthetic_trace()
    baseline = segment(rows, task())
    reply = [{"start_tick": round(a * 30), "end_tick": round(b * 30), "command": cmd} for a, b, cmd in baseline]
    fake = FakeGemini([{"segments": reply}])
    assert segment(rows, task(), refine=True, ask=fake) == baseline
    assert fake.requests[0]["model"] == "gemini-4-argon"
    bad = FakeGemini([{"segments": [{"start_tick": 2, "end_tick": 60, "command": "home"}]}] * 2)
    with pytest.raises(ValueError, match="partition"):
        segment(rows, task(), refine=True, ask=bad)


def test_experiment_plan_targets_largest_residual_and_stays_in_joint_range():
    residuals = {"jaw": {"value": 8, "tolerance": 2, "joint": 1},
                 "object": {"value": 6, "tolerance": 5, "object": "cube"},
                 "grasp": {"value": .3, "tolerance": .05, "object": "cube"}}
    q0 = [0] * 6
    plans = plan_experiments(residuals, q0=q0, joint_ranges=[[-.1, .1]] * 6)
    assert plans[0]["kind"] == "grasp" and plans[0]["priority"] == pytest.approx(6)
    excitation = next(p for p in plans if p["kind"] == "excitation")
    assert np.max(np.abs(excitation["q_target"])) <= .1
    assert not any(p["execute"] for p in plans)
    assert np.allclose(excitation["q_target"][0], q0) and np.allclose(excitation["q_target"][-1], q0)


def nonlinear_servo(attempt, physics):
    """Independent synthetic measurement fixture: damped servo, gravity and a sliding object."""
    kp = physics["joints"]["shoulder_pan"]["kp"]
    kv = physics["joints"]["shoulder_pan"]["kv"]
    sag = physics["sag_gain"][0]
    latency = physics["latency_ticks"]
    mu = physics["object"]["friction"]
    commands = np.array([row["q_target"] for row in attempt["ticks"]])
    theta, velocity = 0.0, 0.0
    samples = []
    for i in range(len(commands)):
        samples.append([theta, 0, 0, 0, 0, 0])
        target = commands[max(0, i-latency), 0]
        # Ten substeps independently integrate torque, rather than a fitted closed-form response.
        for _ in range(10):
            acceleration = kp * (target - theta - sag * np.cos(theta)) - kv * velocity
            velocity += acceleration / 300
            theta += velocity / 300
    q = np.asarray(samples)
    end_x = .2 + attempt["push"] * np.exp(-mu * 2)
    return {"q": q, "jaw_xyz": q[:, :3] * .12 + [.2, 0, .08],
            "end_poses": {"cube": {"xyz": [end_x, 0, .015]}},
            "grasp_outcome": end_x > .22, "latency_ticks": latency}


def measured_attempt(truth, frequency, seed):
    t = np.arange(100) / 30
    commands = .25 * np.sin(2 * np.pi * frequency * t) + .1 * np.sin(2 * np.pi * 1.3 * t + seed)
    attempt = {"id": str(seed), "ticks": [{"t": float(time), "q": [0]*6, "q_target": [float(cmd), 0, 0, 0, 0, 0]}
                                           for time, cmd in zip(t, commands)], "push": .08 + seed * .002}
    recorded = nonlinear_servo(attempt, truth)
    for row, q in zip(attempt["ticks"], recorded["q"]):
        row["q"] = q.tolist()
    attempt.update(jaw_xyz=recorded["jaw_xyz"], end_poses=recorded["end_poses"],
                   grasp_outcome=recorded["grasp_outcome"], latency_ticks=truth["latency_ticks"])
    return attempt


def test_sysid_recovers_kp_kv_sag_latency_and_friction_on_heldout(tmp_path):
    truth = {"joints": {"shoulder_pan": {"kp": 7.5, "kv": 1.3}}, "sag_gain": [.04] * 6,
             "latency_ticks": 2, "object": {"friction": .45}}
    initial = {"joints": {"shoulder_pan": {"kp": 5.0, "kv": .7}}, "sag_gain": [.08] * 6,
               "latency_ticks": 0, "object": {"friction": .7}}
    parameters = [Parameter("joints.shoulder_pan.kp", 1, 15), Parameter("joints.shoulder_pan.kv", .1, 4),
                  Parameter("sag_gain.0", 0, .2), Parameter("object.friction", .01, 1.5)]
    train = [measured_attempt(truth, .4, 0), measured_attempt(truth, .8, 1)]
    heldout = [measured_attempt(truth, .6, 2)]
    result = fit_parameters(train, initial, heldout=heldout, parameters=parameters, replay=nonlinear_servo,
                             output=tmp_path / "identified.json")
    assert result["gate"]["passed"] and result["identifiable"]
    fitted = result["physics"]
    assert fitted["latency_ticks"] == 2
    assert fitted["joints"]["shoulder_pan"]["kp"] == pytest.approx(7.5, abs=.01)
    assert fitted["joints"]["shoulder_pan"]["kv"] == pytest.approx(1.3, abs=.01)
    assert fitted["sag_gain"][0] == pytest.approx(.04, abs=.001)
    assert fitted["object"]["friction"] == pytest.approx(.45, abs=.001)
    assert result["gate"]["metrics"]["jaw_rms_mm"] < .01
    assert (tmp_path / "identified.json").exists()
    assert initial["object"]["friction"] == .7


def test_validation_fails_missing_evidence_and_large_errors():
    physics = {"joints": {"shoulder_pan": {"kp": 7.5, "kv": 1.3}}, "sag_gain": [.04] * 6,
               "latency_ticks": 2, "object": {"friction": .45}}
    attempt = measured_attempt(physics, .6, 2)
    bad = copy.deepcopy(physics)
    bad["joints"]["shoulder_pan"]["kp"] = 2
    assert not validation_gate([attempt], bad, replay=nonlinear_servo)["passed"]
    attempt.pop("grasp_outcome")
    gate = validation_gate([attempt], physics, replay=nonlinear_servo)
    assert not gate["passed"] and "missing:grasp_outcome" in gate["failures"]


def test_fit_rejects_train_eval_leakage():
    sample = {"ticks": [{"t": 0, "q": [0]*6, "q_target": [0]*6}]}
    with pytest.raises(ValueError, match="disjoint"):
        fit_parameters([sample], {}, heldout=[copy.deepcopy(sample)])
