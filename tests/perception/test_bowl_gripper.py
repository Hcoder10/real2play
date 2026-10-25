"""The bowl's wider contact hold must remain closed in saved-setup vision."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from r2p.perception import CameraModel, TaskPerceiver
from r2p.tasks import load_task


@pytest.mark.parametrize('threshold,held', [(.35, False), (.450214, True)])
def test_cube_contact_hold_uses_declared_threshold(threshold, held):
    task = deepcopy(load_task('cube_in_bowl'))
    obj = task['objects'][0]
    obj.update(xy=[.2, 0], pose_verified=True)
    supply = np.array([.2, 0, .019])
    arm = SimpleNamespace(
        tips=lambda q: (supply-[0, .019, 0], supply+[0, .019, 0], supply),
        m=SimpleNamespace(body=lambda name: SimpleNamespace(id=0)),
        fk=lambda q: SimpleNamespace(xmat=np.array([np.eye(3).ravel()])))
    cam = CameraModel.look_at((.5, -.3, .6), (.2, 0, .02), image_size=(64, 48))
    def forbidden(*args, **kwargs):
        raise AssertionError('No detector requests in threshold regression')
    with TaskPerceiver(cam, task, forbidden, arm=arm,
                       gripper_open_threshold_rad=threshold) as p:
        p._setup = True
        p.poses[obj['id']] = {'xyz': supply.tolist(), 'yaw_deg': 0, 'seen': True, 'source': 'vision'}
        result = p.estimate(np.zeros((48, 64, 3), np.uint8), np.array([0, 0, 0, 0, 0, .42]))
        assert bool(result['holding']) is held
        assert result['gripper_open'] is (not held)
