import numpy as np

from r2p.synthbench import World, CompositeRenderer, RenderParams, compare_frames


def test_composite_render_and_metric():
    with World("cube_in_bowl", np.random.default_rng(2)) as world:
        bg = np.full((world.cam.height, world.cam.width, 3), 180, np.uint8)
        renderer = CompositeRenderer(world, bg)
        image = renderer.render()
        assert image.shape == bg.shape
        assert renderer.shadow.max() > 0
        assert renderer.masks["arm"].sum() > 100
        metric = compare_frames(image, image, renderer.masks, renderer.masks)
        assert metric["loss"] == 0
        renderer.params = RenderParams(exposure=1, white_balance=[1, .8, .7], noise_std=1, jpeg_quality=95)
        changed = renderer.render()
        assert renderer.compare(image, renderer.masks, changed)["lab_error"] > 0


def test_rigid_attach_rotation_release_and_jaw_floor():
    with World("cube_in_bowl", np.random.default_rng(2)) as world:
        q = world.q()
        q[5] = -.17
        mid, R, _ = world._gripper(q)
        world.set_pose("cube", mid)
        world.step(q)
        assert world.held is not None
        assert world.q()[5] > q[5]  # object width prevents empty-closed jaws
        oid, offset, local_R, width = world.held
        for index in (0, 4, 1):
            q[index] += .12
            world.step(q)
            mid, R, gap = world._gripper(world.q())
            assert gap >= width-1e-6
            assert np.allclose(world.poses()[oid]["xyz"], mid+R@offset)
        q[5] = .9
        world.step(q)
        assert world.held is None
        assert np.isclose(world.poses()[oid]["xyz"][2], .019)
        assert world.truth_check()["ok"] is False


def test_release_container_floor_support_top_and_finger_push():
    with World("cube_in_bowl", np.random.default_rng(2)) as world:
        bowl = world.poses()["bowl"]["xyz"]
        world.set_pose("cube", [*bowl[:2], .10])
        world._release("cube")
        assert np.isclose(world.poses()["cube"]["xyz"][2], .005+.019)
        assert world.truth_check()["ok"]
        world.set_pose("cube", [.25, -.08, .019])
        world._push(np.array([.21, -.08, .019]), np.array([.25, -.08, .019]))
        before = world.poses()["cube"]["xyz"][0]
        world._push(np.array([.24, -.08, .019]), np.array([.27, -.08, .019]))
        assert world.poses()["cube"]["xyz"][0] > before
    with World("stack_two_cubes", np.random.default_rng(2)) as world:
        world.set_pose("base", [.25, .03, .019])
        world.set_pose("top", [.25, .03, .10])
        world._release("top")
        assert np.isclose(world.poses()["top"]["xyz"][2], .057)
        assert world.truth_check()["ok"]


def test_not_moved_truth_has_initial_evidence():
    with World("place_on_cell", np.random.default_rng(4)) as world:
        xy = world.poses()["cell"]["xyz"][:2]
        world.set_pose("piece", [*xy, .007])
        assert world.truth_check()["ok"]
        oid = next(oid for oid in world.poses() if oid.startswith("other"))
        original = world.poses()[oid]["xyz"]
        world.set_pose(oid, [original[0]+.02, *original[1:]])
        assert not world.truth_check()["ok"]
