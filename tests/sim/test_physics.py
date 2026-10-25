import numpy as np
import pytest

from r2p.frames import BASE_ORIGIN_MJCF
from r2p.sim import TaskTwin
from r2p.sim.expert import Expert
from r2p.sim.ik import JawIK


@pytest.mark.parametrize("size", [[.038, .038, .038], [.030, .030, .014]])
def test_physical_grasp_lifts_and_holds_two_seconds(size):
    obj = dict(id="piece", shape="box", size_m=size, xyz=[.23, 0, size[2]/2],
               yaw_deg=0, mass_kg=.03, friction=.6)
    twin = TaskTwin({"objects": [obj]})
    expert = Expert(dict(name="grasp", objects=[obj], skeleton=[], goal=[], limits={"max_objects": 1}))
    expert._bind(twin)
    assert expert.pick("piece")
    assert twin.m.neq == 0  # No weld or other attachment constraint.
    expert._hold(60, -.12, "hold")
    heights = [state["objects"]["piece"]["xyz"][2] for state in expert.states[-60:]]
    assert min(heights) - size[2]/2 > .05
    assert max(heights)-min(heights) < .003
    assert twin.q()[5] > .05  # The physical object stops the hinge from closing.
    assert twin.d.ncon > 0


@pytest.mark.parametrize("reach,band", [(.18, (2, 5)), (.30, (9, 12))])
def test_static_sag_emerges_from_finite_servo_stiffness(reach, band):
    ik = JawIK()
    q = ik.solve([reach, 0, .08], grip=.3, iterations=160)
    twin = TaskTwin({"objects": []}, q0=q)
    for _ in range(120):
        twin.step(q)
    sag_mm = (ik.arm.tips(q)[2][2]-ik.arm.tips(twin.q())[2][2])*1000
    assert band[0] < sag_mm < band[1]
    np.testing.assert_array_equal(twin.d.qfrc_applied, 0)
    np.testing.assert_allclose(twin.m.body("base").pos, -BASE_ORIGIN_MJCF)


def test_shapes_frame_latency_and_invalid_command():
    objects = [dict(id="bowl", shape="bowl", size_m=[.11,.03], xyz=[.2,0,.015], fixed=True),
               dict(id="zone", shape="plate", size_m=[.05,.05,.002], xyz=[.3,0,.001], fixed=True, collide=False),
               dict(id="stick", shape="stick", size_m=[.12,.022], xyz=[.2,.12,.011]),
               dict(id="ball", shape="sphere", size_m=[.025], xyz=[.3,.15,.0125]),
               dict(id="cylinder", shape="cylinder", size_m=[.025,.06], axis="x", xyz=[.25,-.12,.0125])]
    twin = TaskTwin({"objects": objects})
    np.testing.assert_allclose(twin.poses()["stick"]["xyz"], objects[2]["xyz"])
    bowl = twin.body_ids["bowl"]
    assert np.sum(twin.m.geom_bodyid == bowl) == 17
    zone = twin.m.geom_bodyid == twin.body_ids["zone"]
    assert not twin.m.geom_contype[zone].any()
    assert not twin.m.geom_conaffinity[zone].any()
    old = twin.commanded_q.copy()
    new = old.copy(); new[0] += .1
    twin.step(new)
    np.testing.assert_allclose(twin.d.ctrl[twin.actadr], old)
    twin.step(new)
    np.testing.assert_allclose(twin.d.ctrl[twin.actadr], new)
    with pytest.raises(ValueError):
        twin.step([np.nan]*6)


def test_render_rgb():
    with TaskTwin({"objects": []}) as twin:
        rgb = twin.render({"width": 160, "height": 120})
        assert rgb.shape == (120, 160, 3)
        assert rgb.dtype == np.uint8
        assert rgb.std() > 5
