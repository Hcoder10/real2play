"""Build portable checker vectors from independently specified expected outcomes.

Run python -m r2p.tasks.make_vectors [--check]. The checker is used only to
verify the independently assigned expectations, never to produce them.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from . import check, check_window, validate

OUTPUT = Path(__file__).with_name("vectors.json")


def fixture(goal, *, a_shape="box", a_size=None, b_shape="bowl", b_size=None):
    a_size = a_size or [0.04, 0.02, 0.02]
    b_size = b_size or [0.11, 0.03]
    task = {
        "schema": "argon_task/1", "name": "checker_fixture",
        "objects": [
            {"id": "a", "role": "manipulated", "shape": a_shape, "size_m": a_size, "xy": [0.24, 0]},
            {"id": "b", "role": "container", "shape": b_shape, "size_m": b_size, "xy": [0.24, 0]},
            {"id": "c", "role": "target", "shape": "box", "size_m": [0.04, 0.04, 0.02], "xy": [0.24, 0]},
        ],
        "goal": copy.deepcopy(goal if isinstance(goal, list) else [goal]),
        "skeleton": [{"skill": "home"}], "limits": {"episode_s": 10, "max_objects": 3},
    }
    state = {"objects": {"a": {"xyz": [0.24, 0, 0.02], "yaw_deg": 0},
                         "b": {"xyz": [0.24, 0, 0.015], "yaw_deg": 0},
                         "c": {"xyz": [0.24, 0, 0.01], "yaw_deg": 0}}, "gripper_open": True}
    return task, state


def expected(task, goal_ok, *, gripper=True, settled=True, window=None):
    bits = goal_ok if isinstance(goal_ok, list) else [goal_ok]
    assert len(bits) == len(task["goal"])
    entries = [{"pred": goal["pred"], "ok": bool(bit)} for goal, bit in zip(task["goal"], bits)]
    if gripper is not None and not any(g["pred"] in ("grasped", "gripper_open") for g in task["goal"]):
        entries.append({"pred": "gripper_open", "ok": gripper})
    entries.append({"pred": "settled", "ok": settled})
    if window is False:
        entries.append({"pred": "window", "ok": False})
    return {"ok": all(entry["ok"] for entry in entries), "per_predicate": entries}


def signature(result):
    return {"ok": result["ok"], "per_predicate": [{"pred": p["pred"], "ok": p["ok"]} for p in result["per_predicate"]]}


def build_vectors():
    cases, windows = [], []

    def add(name, task, state, goal_ok, **kwargs):
        cases.append({"name": name, "task": copy.deepcopy(task), "state": copy.deepcopy(state),
                      "expected": expected(task, goal_ok, **kwargs)})

    # Aperture radius .055 - .004 = .051. These are centre tests, as specified.
    for shape, dims in (("bowl", [.11, .03]), ("cylinder", [.11, .03]), ("sphere", [.11])):
        for angle in (0, 30, 90, 175, -179):
            for delta, want in ((-.0005, True), (.0005, False)):
                task, state = fixture({"pred": "inside", "a": "a", "b": "b"}, b_shape=shape, b_size=dims)
                radius = .051 + delta
                state["objects"]["a"]["xyz"][:2] = [.24 + radius*math.cos(math.radians(angle)), radius*math.sin(math.radians(angle))]
                add(f"inside_{shape}_radial_{angle}_{delta}", task, state, want)
    for delta, want in ((-.0005, True), (.0005, False)):
        task, state = fixture({"pred": "inside", "a": "a", "b": "b"})
        state["objects"]["a"]["xyz"][2] = .04 + delta
        add(f"inside_rim_{delta}", task, state, want)
    for pred, margin in (("inside", .004), ("on_top", 0)):
        for yaw in (0, 30, 90, -135, 179):
            for axis, halfwidth in ((0, .06), (1, .03)):
                for delta, want in ((-.0005, True), (.0005, False)):
                    task, state = fixture({"pred": pred, "a": "a", "b": "b"}, b_shape="box", b_size=[.12,.06,.02])
                    state["objects"]["b"].update(xyz=[.24,0,.01], yaw_deg=yaw)
                    local = [0, 0]
                    local[axis] = halfwidth - margin + delta
                    angle = math.radians(yaw)
                    state["objects"]["a"]["xyz"] = [.24+local[0]*math.cos(angle)-local[1]*math.sin(angle),
                                                       local[0]*math.sin(angle)+local[1]*math.cos(angle), .02 if pred == "inside" else .03]
                    add(f"{pred}_rectangle_{yaw}_axis{axis}_{delta}", task, state, want)
    for sign in (-1, 1):
        for delta, want in ((-.0005, True), (.0005, False)):
            task, state = fixture({"pred": "on_top", "a": "a", "b": "b"})
            state["objects"]["a"]["xyz"][2] = .04 + sign*(.012 + delta)
            add(f"on_top_vertical_{sign}_{delta}", task, state, want)
    for pred in ("at", "near", "not_moved"):
        for tolerance in (.005, .012, .03, .04, .1):
            for angle in (0, 45, 90, 179, -179):
                for delta, want in ((-.0005, True), (.0005, False)):
                    goal = {"pred": pred, "a": "a"}
                    goal.update({"xy": "b", "tol_m": tolerance} if pred == "at" else
                                {"b": "b", "d_m": tolerance} if pred == "near" else {"tol_m": tolerance})
                    task, state = fixture(goal)
                    radius = tolerance + delta
                    state["objects"]["a"]["xyz"] = [.24+radius*math.cos(math.radians(angle)),radius*math.sin(math.radians(angle)),.5]
                    state["initial"] = {"a": {"xyz": [.24, 0, .01]}}
                    add(f"{pred}_xy_{tolerance}_{angle}_{delta}", task, state, want)
    for delta, want in ((-.0005, True), (.0005, False)):
        task, state = fixture({"pred": "at", "a": "a", "xy": [.24,0], "tol_m": .012})
        state["objects"]["a"]["xyz"][0] += .012 + delta
        add(f"at_literal_{delta}", task, state, want)
    for height in (0, .005, .03, .1):
        for delta, want in ((-.0005, False), (.0005, True)):
            task, state = fixture({"pred": "lifted", "a": "a", "h_m": height})
            state["objects"]["a"]["xyz"][2] = .01 + height + delta
            add(f"lifted_{height}_{delta}", task, state, want)
    for holding, want in ((["a"], True), ([], False), (["b"], False)):
        task, state = fixture({"pred": "grasped", "a": "a"})
        state.update(gripper_open=False, holding=holding)
        add(f"grasped_holding_{holding}", task, state, want)
    for delta, want in ((-.0005, True), (.0005, False)):
        task, state = fixture({"pred": "grasped", "a": "a"})
        state.update(gripper_open=False, site_xyz=[.24+.04+delta,0,.02])
        add(f"grasped_distance_{delta}", task, state, want)
    for delta, want in ((-.0005, False), (.0005, True)):
        task, state = fixture({"pred": "grasped", "a": "a"})
        state["objects"]["a"]["xyz"][2] = .015 + delta
        state.update(gripper_open=False, site_xyz=state["objects"]["a"]["xyz"])
        add(f"grasped_lift_{delta}", task, state, want)
    task, state = fixture({"pred": "grasped", "a": "a"})
    state["site_xyz"] = [.24,0,.02]
    add("grasped_open_near_jaws_is_not_grasp", task, state, False)
    state["gripper_open"] = False
    state["objects"]["a"]["xyz"][2] = .01
    add("grasped_closed_on_table_is_not_grasp", task, state, False)
    for tol in (1, 5, 15, 45):
        for delta, want in ((-.5, True), (.5, False)):
            task, state = fixture({"pred": "upright", "a": "a", "tol_deg": tol})
            half = math.radians(tol + delta)/2
            state["objects"]["a"]["quat"] = [math.cos(half), math.sin(half),0,0]
            add(f"upright_tilt_{tol}_{delta}", task, state, want)
    task, state = fixture({"pred": "upright", "a": "a", "tol_deg": 1})
    add("upright_yaw_only", task, state, True)
    for shape, size, period in (("box",[.04,.04,.02],90), ("box",[.04,.02,.02],180),
                                ("plate",[.04,.04,.002],90), ("plate",[.04,.02,.002],180),
                                ("stick",[.12,.02],180), ("cylinder",[.02,.12],180)):
        for target in (-180, -179, 0, 179, 180):
            for sign in (-1,1):
                for delta, want in ((-.5, True), (.5, False)):
                    task, state = fixture({"pred": "yaw", "a": "a", "deg": target, "tol_deg": 5}, a_shape=shape, a_size=size)
                    if shape == "cylinder":
                        task["objects"][0]["axis"] = "x"
                    state["objects"]["a"]["yaw_deg"] = (target + period + sign*(5+delta)+180)%360-180
                    add(f"yaw_{shape}_{size[1]}_{target}_{sign}_{delta}", task, state, want)
    for shape, size in (("sphere",[.04]), ("bowl",[.11,.03]), ("cylinder",[.04,.02])):
        for yaw in (-180,-90,0,90,180):
            task, state = fixture({"pred": "yaw", "a": "a", "deg": 22, "tol_deg": 1}, a_shape=shape, a_size=size)
            state["objects"]["a"]["yaw_deg"] = yaw
            add(f"yaw_round_{shape}_{yaw}", task, state, True)
    for failed_link in (None, 0, 1):
        task, state = fixture({"pred": "stacked", "objs": ["c","a","b"]}, b_shape="box", b_size=[.04,.04,.02])
        state["objects"]["a"]["xyz"][2] = .03
        state["objects"]["b"]["xyz"][2] = .05
        if failed_link is not None:
            state["objects"]["a" if failed_link == 0 else "b"]["xyz"][0] += .1
        add(f"stacked_failed_link_{failed_link}", task, state, failed_link is None)
    for delta, want in ((-.0005, True), (.0005, False)):
        task, state = fixture({"pred": "stacked", "objs": ["c","a","b"]}, b_shape="box", b_size=[.04,.04,.02])
        state["objects"]["a"]["xyz"][2] = .03
        state["objects"]["b"]["xyz"][2] = .05 + .012 + delta
        add(f"stacked_vertical_{delta}", task, state, want)

    # Missing/optional objects, including an object-id target and a stack member.
    goals = [{"pred":"inside","a":"a","b":"b"}, {"pred":"on_top","a":"a","b":"b"},
             {"pred":"at","a":"a","xy":"b","tol_m":.01}, {"pred":"near","a":"a","b":"b","d_m":.01},
             {"pred":"lifted","a":"a","h_m":.005}, {"pred":"grasped","a":"a"},
             {"pred":"upright","a":"a","tol_deg":5}, {"pred":"yaw","a":"a","deg":0,"tol_deg":5},
             {"pred":"not_moved","a":"a","tol_m":.01}, {"pred":"stacked","objs":["c","a","b"]}]
    for goal in goals:
        for optional in (False, True):
            task, state = fixture(dict(goal, optional=optional))
            state["objects"].pop("a")
            add(f"missing_a_{goal['pred']}_optional_{optional}", task, state, optional)
    for optional in (False, True):
        task, state = fixture({"pred":"at","a":"a","xy":"b","tol_m":.01,"optional":optional})
        state["objects"].pop("b")
        add(f"missing_target_optional_{optional}", task, state, optional)
    task, state = fixture({"pred":"not_moved","a":"a","tol_m":.01})
    add("not_moved_missing_initial", task, state, False)
    for delta, want in ((-.0005,True),(.0005,False)):
        task, state = fixture({"pred":"not_moved","a":"a","tol_m":.01,"optional":True})
        state["initial"] = {"a":{"xyz":[.24,0,.02]}}
        state["objects"]["a"]["xyz"][0] += .01 + delta
        add(f"optional_present_still_checked_{delta}",task,state,want)
        task, state = fixture({"pred":"on_top","a":"a","b":"b"})
        state["objects"]["a"]["xyz"] = [.24+.055+delta,0,.04]
        add(f"on_top_round_radius_{delta}",task,state,want)
    for yaw, want in ((-179,True),(-175,False),(179,True),(175,False)):
        task, state = fixture({"pred":"yaw","a":"a","deg":180,"tol_deg":3})
        state["objects"]["a"].pop("yaw_deg")
        half = math.radians(yaw)/2
        state["objects"]["a"]["quat"] = [2*math.cos(half),0,0,2*math.sin(half)]
        add(f"yaw_quaternion_wrap_{yaw}",task,state,want)
    task, state = fixture([{"pred":"inside","a":"a","b":"b"},
                           {"pred":"inside","a":"c","b":"b"},
                           {"pred":"at","a":"a","xy":"b","tol_m":.01}])
    state["objects"]["c"]["xyz"][0] += .1
    add("conjunction_keeps_repeated_predicate_positions",task,state,[True,False,True])
    for open_ in (False, True):
        task, state = fixture({"pred":"inside","a":"a","b":"b"})
        state["gripper_open"] = open_
        add(f"implicit_gripper_flat_{open_}", task, state, True, gripper=open_)
        state.pop("gripper_open")
        state["gripper"] = {"open": open_}
        add(f"implicit_gripper_nested_{open_}", task, state, True, gripper=open_)
        task["goal"] = [{"pred":"gripper_open"}]
        add(f"explicit_gripper_{open_}", task, state, open_)
    task, state = fixture({"pred":"inside","a":"a","b":"b"})
    state.pop("gripper_open")
    add("missing_gripper_implicit_skipped", task, state, True, gripper=None)
    task["goal"] = [{"pred":"gripper_open"}]
    add("missing_gripper_explicit_fails", task, state, False, gripper=None)
    for speed, want in ((.4995,True), (.5005,False), (.5,False)):
        task, state = fixture({"pred":"inside","a":"a","b":"b"})
        state["objects"]["a"]["v"] = speed
        add(f"settled_scalar_{speed}", task, state, True, settled=want)
        state["objects"]["a"]["v"] = [0,speed,0]
        add(f"settled_vector_{speed}", task, state, True, settled=want)
    task, state = fixture({"pred":"inside","a":"a","b":"b"})
    state["objects"]["c"]["v"] = 5
    add("unreferenced_velocity_ignored", task, state, True)
    state["objects"]["a"]["size_m"] = [.04,.02,.04]
    state["objects"]["a"]["xyz"][2] = .045
    add("size_override_changes_rim", task, state, True)

    task, good = fixture({"pred":"inside","a":"a","b":"b"})
    bad = copy.deepcopy(good)
    bad["objects"]["a"]["xyz"][0] += .2

    def window(name, task, states, want, hz=30):
        windows.append({"name":name,"task":copy.deepcopy(task),"states":copy.deepcopy(states),"hz":hz,"expected":want})

    window("empty",task,[],{"ok":False,"per_predicate":[{"pred":"window","ok":False}]})
    window("short_14",task,[good]*14,expected(task,True,window=False))
    window("held_15",task,[good]*15,expected(task,True))
    window("bad_before_final_window",task,[bad]+[good]*15,expected(task,True))
    window("bad_last_tick",task,[good]*14+[bad],expected(task,False))
    window("bad_middle_tick",task,[good]*7+[bad]+[good]*7,expected(task,False))
    window("held_at_10hz",task,[good]*5,expected(task,True),hz=10)
    closed = dict(good,gripper_open=False)
    window("closed_mid_window",task,[good]*7+[closed]+[good]*7,expected(task,True,gripper=False))
    moving = copy.deepcopy(good)
    moving["objects"]["a"]["v"] = .5005
    window("moving_mid_window",task,[good]*7+[moving]+[good]*7,expected(task,True,settled=False))
    task, state = fixture({"pred":"not_moved","a":"a","tol_m":.01})
    moved = copy.deepcopy(state)
    moved["objects"]["a"]["xyz"][0] += .02
    window("initial_is_whole_episode_start",task,[state]+[moved]*15,expected(task,False))
    task, state = fixture({"pred":"not_moved","a":"a","tol_m":.01,"optional":True})
    absent = copy.deepcopy(state)
    absent["objects"].pop("a")
    window("optional_disappears",task,[state]*8+[absent]*7,expected(task,True))
    return {"schema":"r2p_check_vectors/1","cases":cases,"window_cases":windows}


def verify(vectors):
    names = set()
    for case in vectors["cases"] + vectors["window_cases"]:
        assert case["name"] not in names, case["name"]
        names.add(case["name"])
        assert not validate(case["task"]), (case["name"], validate(case["task"]))
        result = check(case["task"], case["state"]) if "state" in case else check_window(case["task"], case["states"], case["hz"])
        assert signature(result) == case["expected"], (case["name"], result, case["expected"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the checked-in vectors are stale")
    args = parser.parse_args(argv)
    vectors = build_vectors()
    verify(vectors)
    # One case per line keeps the large parity artifact readable and diffs localized.
    output = '{\n  "schema": "r2p_check_vectors/1",\n  "cases": [\n'
    output += ",\n".join("    " + json.dumps(case, separators=(",",":"), allow_nan=False) for case in vectors["cases"])
    output += '\n  ],\n  "window_cases": [\n'
    output += ",\n".join("    " + json.dumps(case, separators=(",",":"), allow_nan=False) for case in vectors["window_cases"])
    output += "\n  ]\n}\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != output:
            parser.exit(1, "vectors.json is stale; run python -m r2p.tasks.make_vectors\n")
    else:
        OUTPUT.write_text(output, encoding="utf-8")
    print(f"ok {len(vectors['cases'])} check cases, {len(vectors['window_cases'])} window cases")


if __name__ == "__main__":
    main()
