import pytest

from r2p.sim.expert import evaluate


@pytest.mark.parametrize("task", ["cube_in_bowl", "place_on_cell", "pick_place_stick",
                                 "stack_two_cubes", "push_to_target", "two_cubes_in_bowl"])
def test_ten_sampled_scenes_per_task(task):
    report = evaluate(task, n=10, seed=10000)
    assert report["success"] >= .9, report
    if task == "place_on_cell":
        assert report["median_placement_mm"] <= 6, report
