import copy
import math
import numpy as np
import pytest

from r2p.tasks import (command_onehot, command_vocab, commands, encode_obs, encode_slots,
                       load_task, pose_of, poses_from_scene, sample_scene, slot_size, yaw_symmetry)

NAMES = ("cube_in_bowl", "pick_place_stick", "place_on_cell", "push_to_target", "stack_two_cubes", "two_cubes_in_bowl")


@pytest.mark.parametrize("name", NAMES)
def test_sampled_scenes_are_reachable_deterministic_and_do_not_mutate_task(name):
    task = load_task(name)
    original = copy.deepcopy(task)
    for seed in range(30):
        scene = sample_scene(task, np.random.default_rng(seed))
        assert scene == sample_scene(task, np.random.default_rng(seed))
        for obj in scene["objects"]:
            assert .10 <= math.hypot(*obj["xyz"][:2]) <= .32
            dims = slot_size(obj["shape"], obj["size_m"], obj.get("axis", "z"))
            assert obj["xyz"][2] == pytest.approx(dims[2]/2)
            assert all(field in obj for field in ("id","role","shape","size_m","xyz","yaw_deg","color","mass_kg","friction","fixed"))
    assert task == original


def test_board_layout_optional_pieces_and_slot_identity():
    task = load_task("place_on_cell")
    grid = {obj["id"]:obj for obj in task["objects"] if obj["id"].startswith("r")}
    for i, x in enumerate((.115,.160,.205)):
        for j, y in enumerate((-.045,0,.045)):
            obj = grid[f"r{i}c{j}"]
            assert obj["xy"] == [x,y]
            assert task["board"]["cells"][f"r{i}c{j}"] == [x,y]
            assert obj["fixed"] and not obj["collide"]
            assert obj["size_m"][:2] == [.045,.045]
    by_id = {o["id"]:o for o in task["objects"]}
    assert by_id["piece"]["size_m"] == [.03,.03,.014]
    assert by_id["piece"]["xy"] == [.13,.13]
    targets, counts = set(), set()
    for seed in range(100):
        scene = sample_scene(task, np.random.default_rng(seed))
        poses = poses_from_scene(scene)
        choices = [tuple(p["xyz"][:2]) for oid,p in poses.items() if oid == "cell" or oid.startswith("other")]
        assert len(set(choices)) == len(choices)
        targets.add(tuple(poses["cell"]["xyz"][:2]))
        counts.add(len(choices)-1)
        slots = encode_slots(task, poses)
        for index,obj in enumerate(task["objects"]):
            assert slots[index,8] == (obj["id"] in poses)
    assert len(targets) == 9
    assert counts == {0,1,2,3,4}


@pytest.mark.parametrize("shape,size,axis,want", [
    ("box",[.1,.2,.3],"z",[.1,.2,.3]), ("plate",[.1,.2,.003],"z",[.1,.2,.003]),
    ("sphere",[.1],"z",[.1,.1,.1]), ("bowl",[.11,.03],"z",[.11,.11,.03]),
    ("stick",[.12,.022],"z",[.12,.022,.022]),
    ("cylinder",[.02,.1],"x",[.1,.02,.02]), ("cylinder",[.02,.1],"y",[.02,.1,.02]),
    ("cylinder",[.02,.1],"z",[.02,.02,.1]),
])
def test_shape_extents_and_encoding(shape,size,axis,want):
    assert slot_size(shape,size,axis) == want
    task = load_task("pick_place_stick")
    task["objects"][0].update(shape=shape,size_m=size,axis=axis)
    slots = encode_slots(task,{"stick":{"xyz":[.2,.1,.05],"yaw_deg":30}})
    np.testing.assert_allclose(slots[0], [.2,.1,.05,.5,math.sqrt(3)/2,*want,1],atol=1e-7)
    assert not slots[1:].any()


@pytest.mark.parametrize("yaw,want", [(-270,-90),(-180,0),(-90,-90),(90,-90),(180,0),(270,-90),(200,20)])
def test_stick_yaw_canonical_interval(yaw,want):
    task = load_task("pick_place_stick")
    slot = encode_slots(task,{"stick":{"xyz":[.2,0,.011],"yaw_deg":yaw}})[0]
    np.testing.assert_allclose(slot[3:5],[math.sin(math.radians(want)),math.cos(math.radians(want))],atol=1e-7)


@pytest.mark.parametrize("shape,size,axis,want", [
    ("box",[.04,.04,.02],"z",4), ("box",[.04,.02,.02],"z",2),
    ("plate",[.04,.04,.002],"z",4), ("plate",[.04,.02,.002],"z",2),
    ("stick",[.12,.02],"z",2), ("sphere",[.04],"z",0), ("bowl",[.11,.03],"z",0),
    ("cylinder",[.02,.1],"z",0), ("cylinder",[.02,.1],"x",2), ("cylinder",[.02,.1],"y",2),
])
def test_yaw_symmetries(shape,size,axis,want):
    assert yaw_symmetry(dict(shape=shape,size_m=size,axis=axis)) == want


def test_pose_quaternion_sizes_and_observation_layout():
    xyz, yaw = pose_of({"xyz":[.2,0,.01], "quat":[2*math.cos(math.pi/4),0,0,2*math.sin(math.pi/4)]})
    assert yaw == pytest.approx(90)
    xyz2,yaw2 = pose_of([.2,0,.01,math.cos(math.pi/4),0,0,math.sin(math.pi/4)])
    np.testing.assert_equal(xyz,xyz2)
    assert yaw2 == yaw
    task = load_task("cube_in_bowl")
    poses = {"cube":{"xyz":[.2,0,.01],"yaw_deg":0,"size_m":[.02,.02,.02]}}
    q = np.arange(6)/10
    obs = encode_obs(task,q,poses,"pick(cube)")
    np.testing.assert_array_equal(obs[:6],q.astype(np.float32))
    np.testing.assert_array_equal(obs[6:42],encode_slots(task,poses).ravel())
    np.testing.assert_array_equal(obs[42:],command_onehot(task,"pick(cube)"))
    np.testing.assert_allclose(obs[11:14],[.02]*3)
    with pytest.raises(ValueError):
        encode_obs(task,[0]*5,poses)


def test_commands_are_fixed_skill_by_slot_not_skeleton_position():
    task = load_task("cube_in_bowl")
    assert commands(task) == ["pick(cube)","place_in(cube,bowl)","release","home"]
    original = command_onehot(task,"pick(cube)")
    task["skeleton"].reverse()
    np.testing.assert_array_equal(original,command_onehot(task,"pick(cube)"))
    assert len(command_vocab(task)) == len(original) == 8*(4+1)
    assert command_vocab(task)[0] == "pick(slot:0)"
    task["objects"][0]["id"] = "renamed"
    np.testing.assert_array_equal(original,command_onehot(task,"pick(renamed)"))
    for command in ("wait(0.5)","place_at(renamed,[0.2,0.1])","push_to(renamed,bowl)"):
        assert command_onehot(task,command).sum() == 1
    assert command_onehot(task,{"skill":"release"}).sum() == 1


@pytest.mark.parametrize("command", ["pick(ghost)","teleport(cube)","pick","pick(cube,bowl)",
    "place_at(cube,[0.2])","place_at(cube,ghost)","wait(-1)","wait(nan)","home(cube)"])
def test_bad_commands_fail(command):
    with pytest.raises(ValueError):
        command_onehot(load_task("cube_in_bowl"),command)


def test_randomized_physics_presence_and_cartesian_coordinates():
    task = load_task("cube_in_bowl")
    task["randomize"] = {"cube":{"x":[.2,.21],"y":[.05,.06],"yaw_rel_deg":[10,10],
                                "size_m":[[.02,.03],.03,.04],"size_scale":[2,2],"mass_kg":[.1,.1],"friction":[.7,.7]},
                         "bowl":{"p_present":0}}
    task["objects"][1]["xy"] = [.2,-.1]
    scene = sample_scene(task,np.random.default_rng(9))
    assert len(scene["objects"]) == 1
    obj = scene["objects"][0]
    assert .2 <= obj["xyz"][0] <= .21 and .05 <= obj["xyz"][1] <= .06
    assert .04 <= obj["size_m"][0] <= .06
    assert obj["size_m"][1:] == [.06,.08]
    assert obj["xyz"][2] == .04
    assert obj["yaw_deg"] == pytest.approx(math.degrees(math.atan2(obj["xyz"][1],obj["xyz"][0]))+10)
    assert obj["mass_kg"] == .1 and obj["friction"] == .7


def test_minimum_separation_and_impossible_scene():
    task = load_task("stack_two_cubes")
    for seed in range(30):
        a,b = sample_scene(task,np.random.default_rng(seed))["objects"]
        radius = math.hypot(.038,.038)/2
        assert math.dist(a["xyz"][:2],b["xyz"][:2]) >= 2*radius+.04
    task["randomize"] = {"min_sep_m":.04}
    for obj in task["objects"]:
        obj["xy"] = [.2,0]
    with pytest.raises(ValueError,match="Cannot sample"):
        sample_scene(task,np.random.default_rng(0),tries=3)


def test_stack_support_dependency_is_independent_of_object_order():
    task = load_task("stack_two_cubes")
    task["randomize"] = {}
    task["objects"][0]["on"] = "base"
    task["objects"][1]["xy"] = [.2,0]
    scene = sample_scene(task,np.random.default_rng(0))
    a,b = scene["objects"]
    assert a["id"] == "top" and b["id"] == "base"
    assert a["xyz"] == pytest.approx([.2,0,.057])
    assert b["xyz"] == pytest.approx([.2,0,.019])


def test_three_object_stack_can_share_footprint():
    task = load_task("stack_two_cubes")
    task["randomize"] = {}
    task["objects"][0]["on"] = "base"
    task["objects"][1]["xy"] = [.2,0]
    task["objects"].insert(0,dict(task["objects"][0],id="third",on="top"))
    scene = sample_scene(task,np.random.default_rng(0))
    assert scene["objects"][0]["xyz"] == pytest.approx([.2,0,.095])
