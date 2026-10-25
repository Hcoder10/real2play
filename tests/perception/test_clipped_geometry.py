import cv2
import numpy as np

from r2p.perception import CameraModel, TaskPerceiver
from r2p.perception.geometry import fit_pose, silhouette_points, rotation
from r2p.perception.task_perception import _known_pose


def test_fit_visible_portion_of_sensor_clipped_cube():
    camera = CameraModel.look_at(image_size=(640, 480))
    obj = {"shape": "box", "size_m": [.04] * 3}
    center = camera.deproject([320, 475], h=.02)
    polygon = camera.project(silhouette_points(obj) @ rotation(15).T + center)
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(polygon.round().astype(np.int32)), 1)
    fitted = fit_pose(camera, obj, mask)
    assert fitted["clipped"]
    assert fitted["fit_px"] < 1
    assert np.linalg.norm(np.array(fitted["xyz"]) - center) < .002


def test_generated_fixed_pose_is_not_a_surveyed_pose():
    obj = {"shape": "bowl", "size_m": [.1, .03], "xyz": [.2, .1, .015], "fixed": True,
           "pose_source": "calibrated_twin"}
    assert _known_pose(obj) is None
    obj["pose_verified"] = True
    assert _known_pose(obj)["source"] == "known"


def test_randomized_supply_requires_vision_but_frozen_supply_is_known():
    obj = {"id": "piece", "shape": "cylinder", "size_m": [.036, .017], "xy": [.15, -.09]}
    task = {"objects": [obj], "randomize": {"piece": {"x": [.1, .2], "y": [-.09, -.09]}}}
    with TaskPerceiver(CameraModel.look_at(), task) as p:
        assert _known_pose(p.objects["piece"]) is None
    task["randomize"]["piece"]["x"] = [.15, .15]
    with TaskPerceiver(CameraModel.look_at(), task) as p:
        assert _known_pose(p.objects["piece"])["xyz"] == [.15, -.09, .0085]
