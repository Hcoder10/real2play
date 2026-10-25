import copy
import json
from pathlib import Path
import pytest

from r2p.tasks import check, check_window
from r2p.tasks.make_vectors import build_vectors, fixture, signature

VECTORS = json.loads((Path(__file__).resolve().parents[2]/"r2p/tasks/vectors.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case",VECTORS["cases"],ids=lambda case:case["name"])
def test_shared_check_vectors(case):
    assert signature(check(case["task"],case["state"])) == case["expected"]


@pytest.mark.parametrize("case",VECTORS["window_cases"],ids=lambda case:case["name"])
def test_shared_window_vectors(case):
    assert signature(check_window(case["task"],case["states"],case["hz"])) == case["expected"]


def test_vector_regeneration_and_coverage():
    assert VECTORS == build_vectors()
    assert len(VECTORS["cases"]) >= 300 and len(VECTORS["window_cases"]) >= 5
    assert {g["pred"] for c in VECTORS["cases"] for g in c["task"]["goal"]} == {
        "inside","on_top","at","near","lifted","grasped","upright","yaw","not_moved","stacked","gripper_open"}


def test_checker_does_not_mutate_inputs():
    task,state = fixture({"pred":"inside","a":"a","b":"b"})
    saved = copy.deepcopy((task,state))
    check(task,state)
    check_window(task,[state]*15)
    assert (task,state) == saved


@pytest.mark.parametrize("pose", [{"xyz":[float("nan"),0,0]}, {"xyz":[0,0]},
                                    {"xyz":[0,0,0],"quat":[0,0,0,0]}, {"xyz":[0,0,0],"size_m":[-1,1,1]}])
def test_invalid_predicate_evidence_fails_closed(pose):
    task,state = fixture({"pred":"inside","a":"a","b":"b"})
    state["objects"]["a"] = pose
    assert not check(task,state)["ok"]


@pytest.mark.parametrize("hz", [0,-1,float("nan"),float("inf")])
def test_invalid_window_rate(hz):
    task,state = fixture({"pred":"inside","a":"a","b":"b"})
    with pytest.raises(ValueError):
        check_window(task,[state]*15,hz)
