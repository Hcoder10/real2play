import copy

import pytest

from r2p.pipeline.evaluation import evaluate_pair
from r2p.pipeline.gate import GateError, compare
from r2p.pipeline.storage import digest


@pytest.fixture
def plan():
    return {"noise": ["clean", "real"], "real_noise": {"pose_std_m": .003},
            "episodes": [{"id": f"s{i}", "scene": {"objects": [{"id": "cube", "xyz": [i,0,0]}]},
                          "noise_seed": i+19} for i in range(4)]}


def report(plan, name, clean, real):
    return {"policy": name, "plan_hash": digest(plan), "results": {
        noise: [{"scene_id": episode["id"], "noise_seed": episode["noise_seed"], "success": i < score}
                for i, episode in enumerate(plan["episodes"])] for noise, score in [("clean",clean),("real",real)]}}


def test_gate_pass_names_incumbent_and_scenes(plan):
    result = compare(report(plan,"new",4,3), report(plan,"old",3,2), plan)
    assert result["accepted"] and result["selected"] == "new"
    assert "incumbent old" in result["message"] and "IDENTICAL scenes" in result["message"]


@pytest.mark.parametrize("clean,real", [(3,2),(4,1),(2,4),(1,1)])
def test_gate_holds_ties_and_condition_regressions(plan, clean, real):
    result = compare(report(plan,"new",clean,real), report(plan,"old",3,2), plan)
    assert not result["accepted"] and result["selected"] == "old"


@pytest.mark.parametrize("mutation", ["scene", "seed", "hash", "missing", "numeric_success", "reorder"])
def test_gate_rejects_unpaired_or_invalid_evidence(plan, mutation):
    candidate = report(plan,"new",4,4)
    row = candidate["results"]["real"][0]
    if mutation == "scene": row["scene_id"] = "different"
    if mutation == "seed": row["noise_seed"] += 1
    if mutation == "hash": candidate["plan_hash"] = "different"
    if mutation == "missing": candidate["results"].pop("real")
    if mutation == "numeric_success": row["success"] = 1
    if mutation == "reorder": candidate["results"]["real"].reverse()
    with pytest.raises(GateError):
        compare(candidate, report(plan,"old",3,2), plan)


def test_no_clean_only_gate(plan):
    plan["noise"] = ["clean"]
    with pytest.raises(GateError, match="real-noise"):
        compare(report(plan,"new",4,4),report(plan,"old",2,2),plan)


def test_pair_uses_identical_scenes_noise_and_independent_policy_state(plan):
    calls, loaded = [], []
    def loader(name):
        loaded.append(name)
        return name
    def evaluate(policy, task, scene, noise, seed, noise_spec):
        calls.append((policy, copy.deepcopy(scene), noise, seed))
        scene["objects"][0]["xyz"][0] = 999  # must not contaminate the incumbent or saved plan
        return policy == "new"
    before = copy.deepcopy(plan)
    candidate, incumbent = evaluate_pair("new", "old", {}, plan, loader=loader, evaluator=evaluate)
    assert [c[1:] for c in calls[:8]] == [c[1:] for c in calls[8:]]
    assert len(loaded) == 16 and plan == before
    assert compare(candidate, incumbent, plan)["accepted"]
