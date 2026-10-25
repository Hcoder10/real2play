import pytest

from r2p.tasks import load_task, validate


def test_load_by_name_directory_file_and_dict():
    task = load_task("cube_in_bowl")
    assert task == load_task("tasks/cube_in_bowl") == load_task("tasks/cube_in_bowl/task.json")
    clone = load_task(task)
    clone["objects"][0]["size_m"][0] = 100
    assert task["objects"][0]["size_m"][0] == .038


@pytest.mark.parametrize("value", [None, [], 4, "task", {}, {"objects":[None]}])
def test_malformed_tasks_report_errors(value):
    assert validate(value)


@pytest.mark.parametrize("section,key,value", [
    ("object", "shape", "capsule"), ("object", "shape", []),
    ("object", "role", "toy"), ("object", "grasp", "magic"),
    ("object", "size_m", [0, .03, .03]), ("object", "size_m", [.03, .03]),
    ("object", "size_m", [True, .03, .03]), ("object", "size_m", [float("nan"), .03, .03]),
    ("object", "axis", "w"), ("object", "fixed", 1), ("object", "mass_kg", -1),
    ("object", "yaw_deg", float("inf")), ("object", "xyz", [.2,0]),
    ("goal", "pred", "touches"), ("goal", "a", "unknown"), ("goal", "optional", "yes"),
    ("goal", "b", []), ("skill", "skill", "teleport"), ("skill", "obj", "unknown"),
    ("skill", "approach", "below"), ("limits", "max_objects", 1),
    ("limits", "max_objects", True), ("limits", "episode_s", 0),
    ("random", "r", [.3,.1]), ("random", "az_deg", [0,float("nan")]),
    ("random", "r", [-1,1]), ("random", "p_present", 1.1), ("random", "p_present", True),
    ("random", "xy_choice", []), ("random", "xy_choice", [[.2]]),
    ("random", "size_scale", [-1,1]), ("random", "size_m", [.1,.1]),
    ("random", "size_m", [[.2,.1],.1,.1]), ("random", "mass_kg", -1),
    ("random", "friction", [0,-1]), ("random", "invented_distribution", [0,1]),
])
def test_closed_vocab_and_numeric_validation(section, key, value):
    task = load_task("cube_in_bowl")
    targets = {"object":task["objects"][0], "goal":task["goal"][0], "skill":task["skeleton"][0],
               "limits":task["limits"], "random":task["randomize"]["cube"]}
    targets[section][key] = value
    assert validate(task)
    with pytest.raises(ValueError, match="Invalid task"):
        load_task(task)


@pytest.mark.parametrize("goal", [
    {"pred":"at","a":"cube","xy":"bowl","tol_m":.01},
    {"pred":"at","a":"cube","xy":[.2,0],"tol_m":.01},
    {"pred":"stacked","objs":["bowl","cube"]},
    {"pred":"gripper_open"},
])
def test_valid_goal_variants(goal):
    task = load_task("cube_in_bowl")
    task["goal"] = [goal]
    assert not validate(task)


@pytest.mark.parametrize("goal", [
    {"pred":"at","a":"cube","xy":"absent","tol_m":.01},
    {"pred":"at","a":"cube","xy":[.2,0,1],"tol_m":.01},
    {"pred":"at","a":"cube","xy":[.2,0],"tol_m":0},
    {"pred":"at","a":"cube","xy":[.2,0]},
    {"pred":"stacked","objs":[]}, {"pred":"stacked","objs":["cube"]},
    {"pred":"stacked","objs":["cube","cube"]},
    {"pred":"stacked","objs":["cube","ghost"]},
    {"pred":"yaw","a":"cube","deg":float("nan"),"tol_deg":1},
])
def test_invalid_goal_arguments(goal):
    task = load_task("cube_in_bowl")
    task["goal"] = [goal]
    assert validate(task)


def test_duplicate_ids_unknown_randomizer_and_support_cycle():
    task = load_task("cube_in_bowl")
    task["objects"][1]["id"] = "cube"
    assert any("duplicate" in error for error in validate(task))
    task = load_task("cube_in_bowl")
    task["randomize"]["ghost"] = {"r":[.2,.3], "az_deg":[0,20]}
    assert validate(task)
    task = load_task("cube_in_bowl")
    task["objects"][0]["on"] = "bowl"
    task["objects"][1]["on"] = "cube"
    assert any("cycle" in error for error in validate(task))


def test_position_required_and_conflicting_randomizers():
    task = load_task("cube_in_bowl")
    task.pop("randomize")
    assert validate(task)
    task["objects"][0]["xy"] = [.2,.1]
    task["objects"][1]["xyz"] = [.2,-.1,.015]
    assert not validate(task)
    task["randomize"] = {"cube":{"r":[.2,.3]}}
    assert validate(task)
    task["randomize"]["cube"].update(az_deg=[0,10], x=[.2,.3], y=[0,.1])
    assert validate(task)


def test_permanently_absent_objects_need_no_position_distribution():
    task = load_task("place_on_cell")
    for oid in ("other1","other2","other3","other4"):
        task["randomize"][oid] = {"p_present":0}
    assert not validate(task)
