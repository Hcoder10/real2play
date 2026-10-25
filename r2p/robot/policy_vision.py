"""Saved cap appearance with surveyed targets and contact-gated FK tracking."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from r2p.tasks import sample_scene, slot_size


def load_placement_setup(calibration, task, path, *, gripper_open_threshold_rad=.35):
    from r2p.perception import TaskPerceiver
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    saved = payload["task"]
    def shapes(spec):
        return [(o["id"], o["role"], o["shape"], o["size_m"], o.get("color")) for o in spec["objects"]]
    if saved["name"] != task["name"] or shapes(saved) != shapes(task):
        raise ValueError("saved appearance setup has different objects/sizes/colors; capture a new setup")
    # C2 deliberately validates exact task+camera equality on load. Load its
    # original task, then explicitly replace only surveyed placement geometry.
    perceiver = TaskPerceiver(calibration, saved, detector=None,
                             gripper_open_threshold_rad=gripper_open_threshold_rad)
    perceiver.load_setup(path)
    perceiver.task = deepcopy(task)
    perceiver.objects = {o["id"]: deepcopy(o) for o in task["objects"]}
    scene = sample_scene(task, np.random.default_rng(0))
    active_ids = {o["id"] for o in scene["objects"]}
    for mapping in (perceiver.poses, perceiver.models, perceiver.centers, perceiver.areas):
        for oid in list(mapping):
            if oid not in active_ids:
                del mapping[oid]
    fixed = {o["id"]: {**o, "seen": True, "source": "measured_task"}
             for o in scene["objects"] if o.get("fixed")}
    perceiver.poses.update(fixed)
    perceiver.initial = deepcopy(perceiver.poses)
    return ContactPerceiver(perceiver, task, fixed)


class ContactPerceiver:
    """Allow C2's rigid FK attachment only after measured closing-pad contact.

    This is a geometric observation heuristic, not a measured grasp force.
    Release returns to vision as soon as the pads separate from the cap.
    """
    def __init__(self, perceiver, task, fixed):
        from r2p.model import Arm
        self.perceiver, self.task, self.fixed = perceiver, task, fixed
        self.arm = Arm()
        self.previous_gap = None

    def estimate(self, frame, q):
        a, b, _ = self.arm.tips(q)
        gap = float(np.linalg.norm(b-a))
        closing = self.previous_gap is not None and gap < self.previous_gap-1e-7
        self.previous_gap = gap
        held = self.perceiver._held
        if held is not None:
            obj = self.perceiver.objects[held[0]]
            width = min(slot_size(obj["shape"], obj["size_m"])[:2])
            if gap > width+.003:
                self.perceiver._held = None
        contact = False
        if closing and gap > 1e-8:
            axis = (b-a)/gap
            for obj in self.task["objects"]:
                if obj["role"] != "manipulated":
                    continue
                pose = self.perceiver.poses.get(obj["id"], {})
                if not pose.get("seen") or pose.get("source") != "vision":
                    continue
                size = slot_size(obj["shape"], obj["size_m"])
                width = min(size[:2])
                p = np.asarray(pose["xyz"])
                along = float(np.dot(p-a, axis))
                lateral = p-(a+along*axis)
                contact |= (width-.006 <= gap <= width+.003 and 0 <= along <= gap
                            and np.linalg.norm(lateral[:2]) < width/2 and abs(lateral[2]) < size[2]/2)
        if self.perceiver._held is None:
            self.perceiver._was_closed = not contact
        result = self.perceiver.estimate(frame, q)
        result["poses"].update(deepcopy(self.fixed))
        return result

    def close(self):
        self.perceiver.close()
