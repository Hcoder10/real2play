"""Acceptance: .venv/Scripts/python.exe -m pytest tests/lab -q

All Gemini transports are fake and all robot backends are in-process MuJoCo or
small protocol fakes. No hardware, live requests, GL contexts, CUDA or GPU work.
"""
from copy import deepcopy
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from r2p import gemini
from r2p.agents import FakeAgent
from r2p.lab.agent import LabAgent, StatefulGemini, hidden_truth
from r2p.lab.harness import Harness, RealBackend, annotate, make_server
from r2p.lab.motion import Motion, Refused
from r2p.lab.simws import SimWorkspace, Simulation, compile_workspace, read_json


@pytest.fixture
def workspace(tmp_path):
    return SimWorkspace(tmp_path/"run")


class TinyBackend:
    def __init__(self, xyz=(.2, -.1, .08)):
        self.xyz = list(xyz)
        self.commands, self.stops = [], 0
        self.motion = None
        self.speed_limit = 12
        self.q, self.target = np.zeros(6), np.zeros(6)

    def state(self):
        return {"xyz": self.xyz, "q": self.q.tolist(), "q_target": self.target.tolist(),
                "gripper": 20, "holding": False}

    def goto(self, xyz, **kwargs):
        self.commands.append(("goto", np.asarray(xyz).tolist(), kwargs))
        self.xyz = list(xyz)
        return {"status": "ok"}

    def gripper(self, value):
        self.commands.append(("gripper", value))
        return {"ok": True}

    def stop(self):
        self.stops += 1


@pytest.mark.parametrize("primitive", [
    {"name": "goto", "args": {"x": .5, "y": 0, "z": .1}},
    {"name": "goto", "args": {"x": .2, "y": 0, "z": float("nan")}},
    {"name": "goto", "args": {"x": .2, "y": 0, "z": .1, "speed": 21}},
    {"name": "goto", "args": {"x": .2, "y": 0, "z": .004}},
    {"name": "grasp", "args": {"x": .8, "y": 0, "z": .01}},
])
def test_unsafe_inputs_never_move(primitive):
    backend = TinyBackend()
    with pytest.raises(Refused):
        Motion(backend).execute(primitive)
    assert backend.commands == []


def test_vertical_grasp_exception_does_not_allow_grid_sweeps():
    task = {"board": {"cell_m": .04, "cells": {"r1c1": [.22, 0]}}}
    backend = TinyBackend((.2, -.1, .06))
    motion = Motion(backend, task)
    with pytest.raises(Refused, match="Protected"):
        motion.check_point([.22, 0, .02])
    motion.execute({"name": "grasp", "args": {"x": .22, "y": 0, "z": .01}})
    low = [c for c in backend.commands if c[0] == "goto" and c[1][2] < .035]
    assert low and all(np.allclose(c[1][:2], [.22, 0]) for c in low)
    assert all(c[2]["speed"] <= 8 for c in low)
    assert motion.contact_xy is None
    backend.xyz = [.22, 0, .02]
    with pytest.raises(Refused):
        motion.execute({"name": "move_by", "args": {"dx": .01, "dy": 0, "dz": 0}})


def test_contact_steps_relieve_and_stop_on_lag():
    backend = TinyBackend((.2, -.1, .08))
    original = backend.goto
    def goto(xyz, **kw):
        result = original(xyz, **kw)
        if xyz[2] <= .044:
            backend.q[1] = .10
        return result
    backend.goto = goto
    result = Motion(backend).execute({"name": "move_until_contact", "args": {"x": .2, "y": -.1, "z": .035}})
    assert result["contact"]
    assert result["relieved_m"] == .004
    assert min(c[1][2] for c in backend.commands if c[0] == "goto") >= .042-1e-9
    assert backend.xyz[2] >= .08-1e-9


def test_minimum_measured_height_is_enforced():
    motion = Motion(TinyBackend())
    with pytest.raises(Refused, match="Live abort"):
        motion.check_point([.2, -.1, .006], measured=True)
    motion.contact_xy = np.array([.2, -.1])
    motion.check_point([.2, -.1, .006], measured=True)
    with pytest.raises(Refused):
        motion.check_point([.21, -.1, .006], measured=True)


@pytest.mark.parametrize("name", ["../escape.json", "/escape.json", "C:/escape.json", "sub/../../escape.xml", "x.py"])
def test_edit_path_and_executable_rejection(workspace, name):
    with pytest.raises(ValueError):
        workspace.edit_sim(name, "{}")


@pytest.mark.parametrize("injection", [
    '<equality><weld body1="base"/></equality>',
    '<worldbody><body name="cheat" mocap="true"/></worldbody>',
    '<option><flag contact="disable"/></option>',
    '<worldbody><body name="cheat" gravcomp="1"/></worldbody>',
    '<contact><exclude body1="base" body2="shoulder"/></contact>',
    '<include file="../outside.xml"/>',
])
def test_no_cheating_and_transactional_rollback(workspace, injection):
    original = workspace.read("scene.xml")["content"]
    before = workspace.revision()
    candidate = original.replace("</mujoco>", injection+"</mujoco>")
    with pytest.raises((ValueError, OSError)):
        workspace.edit_sim(files={"new.json": "{}", "scene.xml": candidate})
    assert workspace.read("scene.xml")["content"] == original
    assert workspace.revision() == before
    assert not (workspace.directory/"new.json").exists()


def test_collision_defaults_cannot_disable_environment(workspace):
    root = ET.fromstring(workspace.read("scene.xml")["content"])
    root.find("worldbody/geom[@name='table']").set("conaffinity", "0")
    with pytest.raises(ValueError, match="collisions"):
        workspace.edit_sim("scene.xml", ET.tostring(root, encoding="unicode"))


def test_static_geometry_and_include_are_rendered(workspace):
    root = ET.fromstring(workspace.read("scene.xml")["content"])
    ET.SubElement(root, "include", file="object.xml")
    fragment = '<mujoco><worldbody><body name="object:test" pos=".2 -.1 .02"><freejoint/>' \
               '<geom type="box" size=".02 .02 .02" mass=".03" rgba="1 0 0 1" contype="1" conaffinity="3"/>' \
               '</body></worldbody></mujoco>'
    result = workspace.edit_sim(files={"object.xml": fragment, "scene.xml": ET.tostring(root, encoding="unicode")})
    assert result["validated"]
    sim = Simulation(workspace.directory)
    assert "test" in sim.poses()
    assert sim.frame().shape == (540, 960, 3)
    assert sim.frame("wrist").shape == (240, 320, 3)


def test_same_physics_replay_matches_timestamped_motion(workspace):
    sim = Simulation(workspace.directory)
    harness = Harness(workspace.run, {}, sim)
    result = harness.plan([{"name": "goto", "args": {"x": .21, "y": -.1, "z": .06, "speed": 20}}])
    assert result["status"] == "ok", result
    assert result["recording"]["state_hz"] == pytest.approx(30)
    episode = Path(result["directory"])
    for name in ("scene.mp4", "wrist.mp4", "states.jsonl", "frames.json", "primitives.json", "commands.json"):
        assert (episode/name).stat().st_size > 0
    replay = workspace.replay(result["episode"])
    assert replay["residuals"]["joint_rms_deg"] < 1e-8
    assert replay["residuals"]["jaw_path_rms_mm"] < 1e-8
    assert Path(replay["video"]).stat().st_size > 0
    assert replay["residuals"]["object_final_pose_error_mm"] is None
    assert not np.any(sim.d.xfrc_applied)
    assert not np.any(sim.d.qfrc_applied)


def test_hidden_truth_is_independent_and_not_exposed(workspace):
    truth = hidden_truth(workspace, {})
    try:
        assert not Path(truth._temporary.name).is_relative_to(workspace.run)
        assert truth.physics["joints"]["shoulder_lift"]["kp"] != read_json(workspace.directory/"physics.json")["joints"]["shoulder_lift"]["kp"]
        obs = Harness(workspace.run, {}, truth).obs()
        assert "physics" not in obs and "object_poses" not in obs
        assert Path(obs["scene_cam_path"]).exists()
    finally:
        truth._temporary.cleanup()


def test_heldout_seals_edits_and_rejects_tampering(workspace):
    sim = Simulation(workspace.directory)
    harness = Harness(workspace.run, {}, sim)
    for _ in range(2):
        assert harness.plan([{"name": "done"}])["status"] == "ok"
    workspace.reserve_holdout(["1", "2"])
    with pytest.raises(ValueError, match="Held-out"):
        workspace.replay("1")
    result = workspace.score("heldout")
    assert not result["improved"]
    with pytest.raises(ValueError, match="sealed"):
        workspace.edit_sim("new.json", "{}")


def test_done_requires_evidence_not_an_assertion(tmp_path):
    response = {"plan": "# Plan: Test the completion gate.", "thought": "# Thought: Claiming success needs evidence.",
                "calls": [], "done": True, "why": "I claim the model is exact."}
    client = gemini.FakeGemini([response], cache_dir=tmp_path/"cache")
    agent = LabAgent(tmp_path/"run", {}, client=client, real_mock=True, approve="auto-sim")
    try:
        result = agent.run_loop(1)
        assert not result["complete"] and result["status"] == "incomplete"
        ledger = json.loads((tmp_path/"run/ledger.jsonl").read_text().strip())
        assert ledger["results"][-1]["tool"] == "completion_gate"
        assert client.records[-1]["config"]["generation_config"]["thinking_level"] == "high"
        assert client.records[-1]["served_by"] == "gemini-4-argon"
    finally:
        agent.close()


def test_owner_approval_per_call_and_auto_sim_no_motion(tmp_path):
    # Construct only the approval part to avoid creating a real backend.
    agent = object.__new__(LabAgent)
    agent.real_mock, agent.approval = False, "ask"
    answers = iter(["y", "n"])
    agent.input_fn = lambda _: next(answers)
    primitives = [{"name": "home"}]
    assert agent._approve(primitives)
    assert not agent._approve(primitives)
    agent.approval = "auto-sim"
    assert not agent._approve(primitives)
    backend = TinyBackend()
    harness = Harness(tmp_path/"denied", {}, backend, approve=agent._approve)
    result = harness.plan(primitives)
    assert result["status"] == "refused"
    assert not backend.commands and backend.stops == 0


def test_legacy_daemon_is_rejected_without_commands():
    requests = []
    def request(url, body=None):
        requests.append((url, body))
        return {"q": [0]*6}
    with pytest.raises(Refused, match="streaming armd"):
        RealBackend(request=request)
    assert len(requests) == 1 and requests[0][1] is None


def test_http_observation_plan_and_browser_origin_gate(workspace):
    harness = Harness(workspace.run, {}, Simulation(workspace.directory))
    server = make_server(harness, ("127.0.0.1", 0))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = "http://127.0.0.1:"+str(server.server_address[1])
    try:
        with urlopen(base+"/obs") as response:
            obs = json.load(response)
        assert obs["workspace"]["frame"] == "spec"
        body = json.dumps({"primitives": [{"name": "done"}]}).encode()
        with urlopen(Request(base+"/plan", data=body, headers={"Content-Type": "application/json"})) as response:
            assert json.load(response)["status"] == "ok"
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base+"/plan", data=body, headers={"Origin": "https://example.com"}))
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        worker.join(3)


def test_managed_bundle_still_uses_static_validation(tmp_path):
    def factory(**kwargs):
        return FakeAgent({"files": {"evil.py": "print('not allowed')"}, "tests": [{"name": "fake", "passed": True}],
                          "report": "Generated files"}, cache_dir=tmp_path/"managed", **kwargs)
    agent = LabAgent(tmp_path/"run", {}, client=gemini.FakeGemini([], cache_dir=tmp_path/"cache"),
                     real_mock=True, managed_factory=factory)
    try:
        with pytest.raises(ValueError, match="static"):
            agent.managed_build("Build a mesh")
        assert not (agent.ws.directory/"evil.py").exists()
    finally:
        agent.close()


def test_full_fake_argon_experiment_fit_and_heldout_gate(tmp_path):
    """Exercise the actual loop through collection -> edit -> PD replay -> final gate.

    The canned fit is a test oracle for the known synthetic fixture, not a claim
    that a live model inferred these parameters. The agent never sees truth files.
    """
    calls = 0
    agent = None
    def response(**request):
        nonlocal calls
        calls += 1
        if calls <= 3:
            actions = [{"tool": "real_plan", "args": {"primitives": [
                {"name": "goto", "args": {"x": [.21, .20, .22][calls-1], "y": -.1, "z": .06, "speed": 20}}]}}]
            if calls == 3:
                actions.append({"tool": "reserve_holdout", "args": {"episodes": ["2", "3"]}})
        elif calls == 4:
            physics = read_json(agent.ws.directory/"physics.json")
            for joint in physics["joints"].values():
                joint["kp"] *= .72
                joint["kv"] *= 1.12
            root = ET.fromstring(agent.ws.read("scene.xml")["content"])
            root.find("worldbody/geom[@name='table']").set("rgba", "0.5294117647 0.4039215686 0.3176470588 1")
            actions = [{"tool": "edit_sim", "args": {"files": {"physics.json": json.dumps(physics),
                         "scene.xml": ET.tostring(root, encoding="unicode")}}},
                       {"tool": "replay", "args": {"episode": "1"}}]
        else:
            actions = [{"tool": "score", "args": {"split": "heldout"}}]
        return {"plan": "# Plan: Collect independent motions, reserve two, fit training data and validate once.",
                "thought": f"# Thought: Offline integration turn {calls}; inspect tool evidence before declaring completion.",
                "calls": actions, "done": calls == 5,
                "why": "The recorded held-out gate must confirm both physical and visual improvement."}
    client = gemini.FakeGemini(responder=response, cache_dir=tmp_path/"cache")
    agent = LabAgent(tmp_path/"run", {}, client=client, real_mock=True, approve="auto-sim")
    try:
        status = agent.run_loop(5)
        rows = [json.loads(s) for s in (agent.run/"ledger.jsonl").read_text().splitlines()]
        errors = [r for row in rows for r in row["results"] if "error" in r]
        assert not errors, errors
        assert status["complete"], status
        assert status["heldout"]["current"]["jaw_path_rms_mm"] < status["heldout"]["baseline"]["jaw_path_rms_mm"]
        assert status["heldout"]["current"]["image_similarity"] > status["heldout"]["baseline"]["image_similarity"]
        assert len(rows) == 5 and all(row["record"]["served_by"] == "gemini-4-argon" for row in rows)
        assert agent.ws.holdout["sealed"]
    finally:
        agent.close()


def test_private_heldout_override_is_not_an_agent_tool(tmp_path):
    agent = LabAgent(tmp_path/"run", {}, client=gemini.FakeGemini([], cache_dir=tmp_path/"cache"), real_mock=True)
    try:
        with pytest.raises(ValueError, match="internal"):
            agent.call_tool("replay", {"episode": "1", "_heldout": True})
    finally:
        agent.close()


def test_final_object_labels_are_required_and_immutable(workspace):
    workspace.task = {"objects": [{"id": "cube", "role": "manipulated"}]}
    harness = Harness(workspace.run, {}, Simulation(workspace.directory))
    for _ in range(2):
        harness.plan([{"name": "done"}])
    with pytest.raises(ValueError, match="Label every"):
        workspace.reserve_holdout(["1", "2"])
    for episode in ("1", "2"):
        final = read_json(workspace.episode(episode)/"result.json")["recording"]["final_frames"]["scene"]
        labels = {"cube": {"xyz": [.2, -.1, .02], "source": "argon_vision", "frame": final}}
        workspace.label_episode(episode, labels)
        labels["cube"]["xyz"][0] = .25
        with pytest.raises(ValueError, match="immutable"):
            workspace.label_episode(episode, labels)
    workspace.reserve_holdout(["1", "2"])
    with pytest.raises(ValueError, match="Held-out"):
        workspace.episode_frames("1")
    with (workspace.episode("1")/"states.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="evidence changed"):
        workspace.score("heldout")


def test_sheet_measurement_only_away_from_board_and_movement_latches(tmp_path, monkeypatch):
    from scripts.lead import measure_grid
    from r2p.perception.camera import CameraModel
    backend = TinyBackend((.22, 0, .06))
    backend.frame = lambda name: np.zeros((540, 960, 3), np.uint8)
    backend.camera = lambda name: CameraModel.look_at()
    task = {"board": {"cells": {"r1c1": [.22, 0]}, "center_xy": [.22, 0], "cell_m": .04}}
    harness = Harness(tmp_path, task, backend)
    measurements = []
    def measure(*_):
        measurements.append(1)
        return {"r1c1": [.23, 0]}, .04, None
    monkeypatch.setattr(measure_grid, "measure", measure)
    assert not harness.sheet_guard()["checked"]
    assert not measurements
    backend.xyz = [.1, -.15, .08]
    with pytest.raises(Refused, match="Sheet moved"):
        harness.sheet_guard()
    assert len(measurements) == 3
    with pytest.raises(Refused, match="fresh survey"):
        harness.sheet_guard()


def test_server_side_interaction_state_is_forwarded(tmp_path):
    seen = []
    def transport(**kwargs):
        seen.append(kwargs["config"])
        return {"id": f"turn-{len(seen)}", "model": "gemini-4-argon", "output_text": "{}"}
    client = StatefulGemini(transport=transport, cache_dir=tmp_path)
    client.call(["first"], tag="lab:test", thinking="high")
    client.call(["second"], tag="lab:test", thinking="high")
    assert seen[0]["store"] is True
    assert seen[1]["previous_interaction_id"] == "turn-1"


def test_stall_requires_time_and_stationary_encoder():
    from r2p.lab.motion import StallDetector
    detector = StallDetector()
    assert not detector.update(.3, -.1, 0)
    for i in range(1, 9):
        assert not detector.update(.3-i*.025, -.1, i/30)
    detector = StallDetector()
    values = [detector.update(.2, -.1, i/30) for i in range(10)]
    assert not values[0] and values[-1]
    assert not detector.update(.2, .4, 1)


def test_video_timeline_preserves_capture_gaps():
    from r2p.lab.media import timeline_frames
    first = np.zeros((2, 2, 3), np.uint8)
    last = np.full((2, 2, 3), 255, np.uint8)
    frames = list(timeline_frames([(0, first), (1, last)], start=0, end=1, fps=4))
    assert len(frames) == 5
    assert all(not frame.any() for frame in frames[:-1])
    assert frames[-1].min() == 255


def test_wrist_aim_uses_executed_ik(workspace):
    sim = Simulation(workspace.directory)
    motion = Motion(sim)
    point = [.22, -.1, .02]
    result = motion.execute({"name": "look_at", "args": dict(zip(("x", "y", "z"), point))})
    cam = sim.camera("wrist")
    pixel = cam.project(point)
    assert np.linalg.norm(pixel-np.array([cam.width/2, cam.height/2])) < 35


def test_footprint_shift_is_rejected(workspace):
    root = ET.fromstring(workspace.read("scene.xml")["content"])
    root.find("worldbody/body[@name='base']").set("pos", "0 0 0")
    with pytest.raises(ValueError, match="footprint"):
        workspace.edit_sim("scene.xml", ET.tostring(root, encoding="unicode"))


def test_rehearsal_returns_movies_and_final_state(workspace):
    result = workspace.sim_plan([{"name": "wait", "args": {"seconds": .1}}])
    assert result["status"] == "ok"
    assert Path(result["video"]).exists() and Path(result["wrist_video"]).exists()
    assert len(result["final_state"]["q"]) == 6
    assert workspace.sim_render()["state"]["t"] == result["final_state"]["t"]


def test_irregular_timestamp_replay_not_row_indexed(workspace):
    sim = Simulation(workspace.directory)
    directory = workspace.run/"episodes/1"
    directory.mkdir(parents=True)
    initial = np.asarray(sim.state()["q"])
    commands = []
    rows = []
    for t, delta in [(0., 0.), (.017, .02), (.051, -.01), (.118, .03), (.201, 0.)]:
        sim.advance(sim.target, t)
        command = initial.copy()
        command[0] += delta
        commands.append({"t": t, "q_target": command.tolist()})
        sim.advance(command, t)
        rows.append({"t": t, "q": sim.state()["q"], "q_target": command.tolist(), "xyz": sim.state()["xyz"]})
    (directory/"states.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    gemini.atomic_json(directory/"commands.json", commands)
    residuals = workspace.replay("1", render=False)["residuals"]
    assert residuals["joint_rms_deg"] < 1e-8
    assert residuals["jaw_path_rms_mm"] < 1e-8


def test_real_adapter_uses_proven_motion_with_streaming_speed_cap(workspace):
    reference = Simulation(workspace.directory)
    sent = []
    def request(url, body=None):
        if body is not None:
            assert body.get("stream") is True
            sent.append(body)
            reference.advance(body["joints"], reference.d.time+1/30)
        state = reference.state()
        return {**state, "owner_deg": reference.jm.to_owner(state["q"]).tolist(),
                "rest_q": reference.rest_q.tolist()}
    backend = RealBackend(request=request)
    result = Motion(backend).execute({"name": "goto", "args": {"x": .21, "y": -.1, "z": .06, "speed": 12}})
    assert sent and result["miss_mm"] < 4
    targets = np.array([reference.rest_q, *[row["joints"] for row in sent]])
    assert np.max(np.rad2deg(np.abs(np.diff(targets, axis=0)))) <= 12/30+1e-6


def test_failed_argon_response_still_has_ledger_and_status(tmp_path):
    agent = LabAgent(tmp_path/"run", {}, client=gemini.FakeGemini(["not JSON"], cache_dir=tmp_path/"cache"), real_mock=True)
    try:
        with pytest.raises(gemini.JSONError):
            agent.run_loop(1)
        ledger = read_json(agent.run/"status.json")
        assert ledger["status"] == "error"
        row = json.loads((agent.run/"ledger.jsonl").read_text())
        assert row["raw_response"] == "not JSON" and row["turn"] == 1
    finally:
        agent.close()


def test_physics_friction_override_reaches_compiled_model(workspace):
    physics = read_json(workspace.directory/"physics.json")
    physics["friction_overrides"] = {"table": [.92, .004, .0002]}
    workspace.edit_sim("physics.json", json.dumps(physics))
    model, _ = compile_workspace(workspace.directory)
    np.testing.assert_allclose(model.geom_friction[model.geom("table").id], [.92, .004, .0002])


def test_nonzero_collision_masks_cannot_make_a_ghost_table(workspace):
    root = ET.fromstring(workspace.read("scene.xml")["content"])
    root.find("worldbody/geom[@name='table']").attrib.update(contype="16", conaffinity="16")
    with pytest.raises(ValueError, match="collision masks"):
        workspace.edit_sim("scene.xml", ET.tostring(root, encoding="unicode"))


def test_visual_shell_requires_real_collision_proxy(workspace):
    root = ET.fromstring(workspace.read("scene.xml")["content"])
    body = ET.SubElement(root.find("worldbody"), "body", name="object:mesh", pos=".2 .1 .03")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", type="box", size=".02 .02 .02", mass=".03", contype="0", conaffinity="0")
    with pytest.raises(ValueError, match="collisions"):
        workspace.edit_sim("scene.xml", ET.tostring(root, encoding="unicode"))
    ET.SubElement(body, "geom", type="box", size=".02 .02 .02", mass="0", contype="1", conaffinity="3", group="3")
    assert workspace.edit_sim("scene.xml", ET.tostring(root, encoding="unicode"))["validated"]
