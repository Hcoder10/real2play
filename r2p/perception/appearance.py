"""One-frame Lab likelihood models and conservative distractor adaptation."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageColor
from scipy.ndimage import gaussian_filter

BINS = (8, 32, 32)


def color_bgr(color):
    if isinstance(color, (list, tuple, np.ndarray)):
        rgb = np.asarray(color, float)[:3]
        if rgb.max() <= 1:
            rgb *= 255
        return rgb[::-1].astype(np.uint8)
    names = {"wood": "#996633", "brown": "#965523", "metal": "#a0a0a0",
             "yellow": "#e2bd24", "blue": "#2455cc"}
    try:
        rgb = ImageColor.getrgb(names.get(str(color).lower(), str(color)))
    except ValueError:
        rgb = (150, 150, 150)
    return np.array(rgb[::-1], np.uint8)


def lab_indices(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int32)
    # Fine a*/b* bins in the useful range separate pale handles and pink cloth.
    return (np.minimum(lab[..., 0] // 32, 7) * 1024
            + np.clip((lab[..., 1] - 64) // 4, 0, 31) * 32
            + np.clip((lab[..., 2] - 64) // 4, 0, 31))


def seed_mask(frame, box, color=None):
    """GrabCut within xyxy pixels, then select task-colour foreground components."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = np.rint(box).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise ValueError("Detector box is empty or too small")
    labels = np.zeros((h, w), np.uint8)
    labels[y0:y1, x0:x1] = cv2.GC_PR_FGD
    # Work on a crop to keep one-shot setup practical at 720p.
    pad = 8
    left, top, right, bottom = max(0, x0-pad), max(0, y0-pad), min(w, x1+pad), min(h, y1+pad)
    local = labels[top:bottom, left:right].copy()
    try:
        cv2.grabCut(frame[top:bottom, left:right], local, None, np.zeros((1, 65)), np.zeros((1, 65)),
                    4, cv2.GC_INIT_WITH_MASK)
        labels[top:bottom, left:right] = local
    except cv2.error:
        # A full-image or uniform box may not have enough background for GMMs.
        pass
    mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    if color is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        expected = cv2.cvtColor(color_bgr(color).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        if expected[1] > 45:
            delta = np.abs(hsv[..., 0].astype(float) - float(expected[0]))
            gate = (np.minimum(delta, 180-delta) < 25) & (hsv[..., 1] > 30)
        else:
            gate = hsv[..., 1] < 90
        candidate = mask & gate.astype(np.uint8)
        if candidate.sum() >= max(8, .08 * mask.sum()):
            mask = candidate
    if mask.sum() < 8:
        raise ValueError("No usable foreground pixels inside detector box")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


class Appearance:
    def __init__(self, foreground, background, threshold=0.5):
        self.foreground = np.asarray(foreground, float).reshape(BINS)
        self.background = np.asarray(background, float).reshape(BINS)
        self.threshold = float(threshold)
        self._update()

    @staticmethod
    def _hist(indices):
        return gaussian_filter(np.bincount(indices.ravel(), minlength=np.prod(BINS)).reshape(BINS).astype(float),
                               sigma=(.65, .65, .65))

    def _update(self):
        fg = self.foreground / max(self.foreground.sum(), 1)
        bg = self.background / max(self.background.sum(), 1)
        self.lut = (fg / (fg + bg + 1e-10)).ravel().astype(np.float32)

    @classmethod
    def fit(cls, frame, mask, indices=None):
        indices = lab_indices(frame) if indices is None else indices
        mask = np.asarray(mask, bool)
        if mask.sum() < 8 or (~mask).sum() < 8:
            raise ValueError("Appearance fit needs foreground and background pixels")
        model = cls(cls._hist(indices[mask]), cls._hist(indices[~mask]))
        scores = model.lut[indices]
        best = (-1, .5)
        for threshold in np.linspace(.03, .97, 48):
            predicted = scores >= threshold
            iou = np.count_nonzero(predicted & mask) / max(1, np.count_nonzero(predicted | mask))
            if iou > best[0]:
                best = (iou, threshold)
        model.threshold = float(best[1])
        return model

    def backproject(self, frame=None, indices=None):
        if indices is None:
            indices = lab_indices(frame)
        return self.lut[indices]

    def mask(self, frame=None, indices=None):
        return (self.backproject(frame, indices) >= self.threshold).astype(np.uint8)

    def learn_background(self, indices, distractor_mask, object_mask, weight=.25):
        """Learn complete distractor blobs; rollback if >10% tracked pixels vanish."""
        distractor_mask, object_mask = np.asarray(distractor_mask, bool), np.asarray(object_mask, bool)
        if not distractor_mask.any() or not object_mask.any():
            return False
        old = self.background.copy()
        extra = self._hist(indices[distractor_mask])
        extra *= max(self.background.sum(), 1) / max(extra.sum(), 1)
        self.background = (1-weight) * self.background + weight * extra
        self._update()
        if np.count_nonzero(self.lut[indices[object_mask]] >= self.threshold) < .9 * object_mask.sum():
            self.background = old
            self._update()
            return False
        return True

    def to_dict(self):
        return {"bins": list(BINS), "foreground": self.foreground.ravel().tolist(),
                "background": self.background.ravel().tolist(), "threshold": self.threshold}

    @classmethod
    def from_dict(cls, data):
        if tuple(data["bins"]) != BINS:
            raise ValueError("Unsupported appearance histogram layout")
        return cls(data["foreground"], data["background"], data["threshold"])
