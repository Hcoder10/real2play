import pytest

from r2p.sim.expert import evaluate


@pytest.mark.parametrize("task", ["cube_in_bowl", "place_on_cell", "pick_place_stick",
                                 "stack_two_cubes", "push_to_target", "two_cubes_in_bowl"])
def test_ten_sampled_scenes_per_task(task):
    report = evaluate(task, n=10, seed=10000)
    # place_on_cell moves the whole printed board (+-25/15 mm, +-8 deg) and uses 30-37 mm caps: 224/256 on the rig.
    assert report["success"] >= (.8 if task == "place_on_cell" else .9), report
    if task == "place_on_cell":
        assert report["median_placement_mm"] <= 6, report
