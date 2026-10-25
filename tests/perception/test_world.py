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
