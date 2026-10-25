from types import SimpleNamespace
import numpy as np
import pytest

from r2p.frames import JOINTS


class FlatArm:
    range = np.array([[-3, 3]]*5 + [[-.17, 1.745]])

    def gravity_torque(self, q):
        return np.array([0., .5, -.3, .1, 0, 0])

    def geometry(self, q):
        return {"link": .2, "gripper": .15, "finger": .10+q[1], "finger_r": .3, "finger_zmax": .2}

    def tips(self, q):
        mid = np.array([.25, 0, .12+q[1]])
        return mid-[.02, 0, 0], mid+[.02, 0, 0], mid


@pytest.fixture
def flat_arm():
    return FlatArm()


class FakeBus:
    def __init__(self):
        self.events = []
        self.is_calibrated = True
        self.motors = dict.fromkeys(JOINTS, None)
        self.present = dict(zip(JOINTS, [10, 20, 30, 40, 50, 60]))

    def connect(self):
        self.events.append(("connect",))

    def disconnect(self, disable_torque=True):
        self.events.append(("disconnect", disable_torque))

    def disable_torque(self):
        self.events.append(("torque_off",))

    def sync_read(self, name, normalize=True):
        self.events.append(("read", name, normalize))
        return self.present.copy() if name == "Present_Position" else dict.fromkeys(JOINTS, -32)

    def sync_write(self, name, values, normalize=True):
        self.events.append(("write", name, dict(values), normalize))


@pytest.fixture
def follower():
    bus = FakeBus()
    return SimpleNamespace(bus=bus, config=SimpleNamespace(disable_torque_on_disconnect=True),
                           configure=lambda: bus.events.append(("configure_torque_on",)))


@pytest.fixture
def cube_task():
    from r2p.tasks import load_task
    return load_task("cube_in_bowl")
