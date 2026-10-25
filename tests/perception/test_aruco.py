import cv2
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from r2p.perception.aruco import (BoardConfig, calibrate_extrinsics, calibrate_intrinsics,
                                 footprint_overlay, make_board, render_board)
from r2p.perception.camera import CameraModel


def test_rendered_board_calibration_under_one_cm(tmp_path):
    cam = CameraModel.look_at((.48, -.28, .55), (.19, 0, 0), image_size=(1280, 960))
    cfg = BoardConfig()
    rng = np.random.default_rng(19)
    mapping = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.]])
    views = []
    for _ in range(12):
        R = Rotation.from_euler("xyz", rng.uniform(-22, 22, 3), degrees=True).as_matrix() @ mapping
        t = np.array(cfg.origin_xyz)+rng.uniform([-.02, -.02, .01], [.02, .02, .09])
        views.append(render_board(cam, cfg, board_R=R, board_t=t))
    intrinsic, report = calibrate_intrinsics(views, cfg)
    flat = render_board(cam, cfg)
    recovered, extrinsic = calibrate_extrinsics(flat, intrinsic, cfg)
    assert np.linalg.norm(cam.position-recovered.position) < .01, (cam.position, recovered.position)
    assert report["reprojection_px"] < 1
    assert extrinsic["reprojection_px"] < 1
    assert extrinsic["elevation_deg"] > 40
    assert not np.array_equal(flat, footprint_overlay(flat, recovered))


def test_print_scale_and_measured_placement(tmp_path):
    path = tmp_path / "letter.png"
    make_board(path)
    with Image.open(path) as image:
        assert image.size == (2550, 3300)
        assert abs(image.info["dpi"][0]-300) < .01
    with pytest.raises(ValueError, match="base_front"):
        BoardConfig.from_dict({"dist_from_base_front_m": .08})
    cfg = BoardConfig.from_dict({"base_front_x_m": .04, "dist_from_base_front_m": .08})
    assert np.allclose(cfg.to_spec([0, 0, 0]), [.12, -.095, 0])
    assert np.allclose(cfg.to_spec([.1, .1, 0]), [.22, .005, 0])
    with pytest.raises(ValueError, match="8"):
        calibrate_intrinsics([np.zeros((30, 30, 3), np.uint8)])
