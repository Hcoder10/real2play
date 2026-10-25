import json

import cv2
import numpy as np
import pytest

from r2p.gemini import FakeGemini
from r2p.model import Arm, JointMap
from r2p.perception import CameraModel
from r2p.sysid import visual


def test_plan_and_injected_collection_freshness(tmp_path):
    proposal = visual.plan(xs=(.2,), ys=(0,), zs=(.12,))
    moves = []
    def mover(xyz):
        moves.append(xyz)
        return {"ok": True, "owner_deg": [0] * 6, "settled_at": 5}
    capture = lambda after: (np.zeros((30, 40, 3), np.uint8), after + 1)
    rows = visual.collect(proposal, mover=mover, capture=capture, out_dir=tmp_path)
    assert len(moves) == 1 and rows[0]["captured_at"] == 6
    assert (tmp_path / "p00.jpg").exists() and (tmp_path / "poses.json").exists()
    with pytest.raises(ValueError, match="stale"):
        visual.collect(proposal, mover=mover, capture=lambda t: (capture(t)[0], t - 1), out_dir=tmp_path / "stale")
    with pytest.raises(ValueError, match="10 cm"):
        visual.plan(zs=(.05,))


def test_point_resumes_and_preserves_full_image_coordinates(tmp_path):
    cv2.imwrite(str(tmp_path / "p00.jpg"), np.zeros((100, 200, 3), np.uint8))
    source = tmp_path / "poses.json"
    source.write_text(json.dumps([{"k": 0, "owner_deg": [0] * 6}]))
    client = FakeGemini([{"fixed_tip": [500, 250], "moving_tip": [500, 750], "visible": True}])
    rows = visual.point(source, ask=client)
    assert rows[0]["tips_px"]["fixed_tip"] == [50, 50]
    assert rows[0]["tips_px"]["moving_tip"] == [150, 50]
    assert visual.point(source, ask=client) == rows
    assert len(client.requests) == 1


def synthetic_rows(outlier=False):
    camera = CameraModel.look_at(image_size=(640, 480))
    arm = Arm()
    base = JointMap(sign=np.array([-1, 1, 1, 1, -1.]))
    truth = JointMap(sign=base.sign, offset=np.array([2., -3., 4., -5., 8.]))
    rng = np.random.default_rng(123)
    rows = []
    for i in range(10):
        owner = np.r_[rng.uniform([-30, -40, 20, 20, -60], [30, 20, 70, 70, 60]), rng.uniform(20, 70)]
        uv = camera.project(np.array(arm.tips(truth.to_model(owner))[:2]))
        if i % 2:
            uv = uv[::-1]
        if outlier and i == 2:
            uv += [50, -45]
        rows.append({"k": i, "owner_deg": owner.tolist(), "visible": True,
                     "tips_px": dict(zip(("fixed_tip", "moving_tip"), uv.tolist()))})
    return rows, camera, base, truth


def test_soft_l1_recovers_offsets_and_loo_with_swapped_fingers(tmp_path):
    rows, camera, base, truth = synthetic_rows()
    result = visual.fit(rows, camera, base)
    assert np.allclose(result["offset_deg"], truth.offset, atol=.02)
    assert result["accepted"] and result["loo_p90_px"] < .02
    assert len(result["loo"]) == 10 and result["jacobian_rank"] == 5
    assert result["candidate"]["joints"]["shoulder_pan"]["sign"] == -1
    assert not result["verified"]
    assert visual.report(result, tmp_path).exists()


def test_outlier_does_not_move_fit_and_fails_heldout_gate():
    rows, camera, base, truth = synthetic_rows(outlier=True)
    result = visual.fit(rows, camera, base)
    assert np.max(np.abs(np.array(result["offset_deg"]) - truth.offset)) < 2
    assert not result["accepted"]
    assert result["loo"][2]["tip_rms_px"] > 50
    with pytest.raises(ValueError, match="six"):
        visual.fit(rows[:4], camera, base)
