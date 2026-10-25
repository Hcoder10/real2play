import cv2
import numpy as np

from r2p.perception.appearance import Appearance, lab_indices, seed_mask


def test_lab_fit_and_whole_blob_background_rollback():
    frame = np.full((100, 160, 3), (180, 190, 195), np.uint8)
    frame[20:60, 20:60] = (20, 180, 30)
    frame[20:60, 100:140] = (20, 180, 30)
    target = np.zeros(frame.shape[:2], np.uint8)
    target[20:60, 20:60] = 1
    other = np.zeros_like(target)
    other[20:60, 100:140] = 1
    indices = lab_indices(frame)
    model = Appearance.fit(frame, target)
    assert model.foreground.shape == (8, 32, 32)
    assert np.count_nonzero(model.mask(frame) & target) == target.sum()
    # Force a high-confidence threshold to exercise the rollback, not just
    # whether adaptation returns. An identically coloured blob is inseparable.
    model.threshold = .8
    prior = model.background.copy()
    assert not model.learn_background(indices, other, target, weight=1.)
    assert np.array_equal(model.background, prior)
    assert np.count_nonzero(model.mask(frame) & target) >= .9*target.sum()
    different = np.zeros_like(target)
    different[70:90, 100:140] = 1
    assert model.learn_background(indices, different, target)


def test_grabcut_task_colour_rejects_wrong_object():
    frame = np.full((150, 180, 3), 210, np.uint8)
    frame[40:100, 30:75] = (20, 20, 220)
    frame[40:100, 100:145] = (220, 20, 20)
    mask = seed_mask(frame, [20, 30, 155, 110], "blue")
    assert mask[50:90, 110:140].mean() > .95
    assert mask[50:90, 40:65].sum() == 0
