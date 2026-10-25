"""Z1 free-roll bowl expert, frozen from P3 contact-controller work.

Run: python -m r2p.sim.expert --task place_on_cell --n 100
Only absolute servo targets are produced. DART changes the executed target, never
the recorded training label. Object motion always comes from physical contacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import hashlib
import time
from pathlib import Path

import numpy as np

from r2p.frames import HOME_Q, HZ
from r2p.tasks import check_window, commands, load_task, sample_scene
from .ik import JawIK
from .twin import TaskTwin, extents, load_physics


class Expert:
    def __init__(self, task, *, dart=0.0, seed=0, deployment=None):
        self.task = load_task(task) if not isinstance(task, dict) else deepcopy(task)
        self.ik = JawIK()
        self.arm = self.ik.arm
        self.deployment = None
        self.close_grip, self.release_grip = -.12, .5
        if deployment is not None:
            from .bench import contact_profile
            self.deployment = contact_profile(deployment)
            self.arm.range = np.asarray(self.deployment["model_limits"], float)
            self.arm.range[4] = self.deployment["roll_window_rad"]
            self.ik = JawIK(self.arm)
            self.close_grip = self.deployment["gripper_close_rad"]
            self.release_grip = self.deployment["gripper_open_rad"]
        self.dart = float(dart)
        self.rng = np.random.default_rng(seed)
        self.ticks, self.states = [], []
        self.segments = []
        self.command, self.phase = "home", "home"
        self.held = None
        self.fail_reason = None

    def _bind(self, twin):
        self.twin = twin
        self.qplan = twin.q()
        self.nominal_command = twin.commanded_q.copy()
        self.start_q = self.qplan.copy()
        if self.deployment and not np.allclose(self.start_q, self.deployment["start_q"], atol=1e-5):
            raise ValueError("bench twin must start at the profile start_q")
        self.initial = twin.poses()
        self.goal_xyz = self.arm.tips(twin.q())[2]
        self.yaw = None

    def _gripper_open(self):
        if self.deployment:
            threshold = self.deployment["gripper_open_threshold_rad"]
            return bool(self.twin.q()[5] > threshold and self.twin.commanded_q[5] > threshold)
        return not self.twin.grip_closed()

    def _tick(self, label):
        label = np.clip(np.asarray(label, float), *self.arm.range.T)
        if self.deployment:
            # The profile's close angle is a travel endpoint, not a contact
            # position. Driving a stalled cap all the way there saturates the
            # 0.8 Nm actuator and rolls it out of these asymmetric finger pads.
            # Bound the position-servo effort while retaining physical contact.
            if label[5] < self.release_grip - .01:
                kp = self.twin.physics["joints"]["gripper"]["kp"]
                effort = self.deployment["expert_control"]["gripper_hold_torque_nm"]
                label[5] = max(label[5], self.twin.q()[5] - effort / kp)
            step = self.deployment["joint_step_rad"]
            previous = self.twin.commanded_q[5]
            label[5] = previous + np.clip(label[5] - previous, -step, step)
            label[5] = np.clip(label[5], *self.arm.range[5])
        self.nominal_command = label.copy()
        q_before = self.twin.q()
        poses_before = self.twin.poses()
        closed_before = not self._gripper_open()
        q_input = self.twin.commanded_q.copy()
        if np.max(np.abs(q_input-q_before)) > np.deg2rad(4):
            q_input = q_before.copy()
        executed = label.copy()
        if self.dart:
            executed[:5] += self.rng.normal(0, self.dart, 5)
        executed = np.clip(executed, *self.arm.range.T)
        if self.deployment:
            # DART is an executed-target disturbance. The bench still applies
            # its rate guard; otherwise .01 rad noise bypasses the .0023 rad
            # step and the resulting contacts do not represent this profile.
            previous = self.twin.commanded_q
            step = self.deployment["joint_step_rad"]
            executed = previous + np.clip(executed-previous, -step, step)
        self.twin.step(executed)
        self.ticks.append({"t": len(self.ticks)/HZ, "q": q_before.tolist(), "q_commanded": q_input.tolist(),
                           "q_target": label.tolist(), "q_exec": executed.tolist(), "poses": poses_before,
                           "grip_closed": closed_before, "command": self.command, "phase": self.phase})
        self.states.append({"objects": self.twin.poses(), "gripper_open": self._gripper_open(),
                            "site_xyz": self.arm.tips(self.twin.q())[2].tolist(), "initial": self.initial})

    def _servo(self, xyz, grip, yaw=None):
        measured = self.twin.q()
        # Predict a fraction of the hinge motion, while respecting a contact-held jaw.
        g = measured[5] + np.clip(grip-measured[5], -.015, .015)
        self.qplan[5] = g
        self.qplan = self.ik.solve(xyz, self.qplan, grip=g, yaw=yaw, iterations=8)
        label = self.ik.compensate(self.qplan, self.twin.physics)
        label[5] = grip
        # Local feedback corrects payload and friction residuals without accumulating
        # measured joint sag into an ever-descending commanded pose.
        actual = self.arm.tips(measured)[2]
        predicted = self.arm.tips(self.qplan)[2]
        J = self.arm.jac_tip(measured)[0]
        correction = J.T @ np.linalg.solve(J@J.T + 0.0001*np.eye(3), np.clip(predicted-actual, -.005, .005))
        label[:5] += .35 * correction
        # The planned path must advance independently of DART. Anchoring each
        # small nominal step to the disturbed execution attenuates its speed
        # toward zero when noise is larger than the bench's rate allowance.
        previous = self.nominal_command if self.deployment else self.twin.commanded_q
        step = self.deployment["joint_step_rad"] if self.deployment else .075
        label[:5] = previous[:5] + np.clip(label[:5]-previous[:5], -step, step)
        self._tick(label)

    def move(self, xyz, grip=-.12, yaw=None, *, speed=.075, settle=12, phase="move"):
        self.phase = phase
        xyz = np.asarray(xyz, float)
        start = self.arm.tips(self.twin.q())[2].copy()
        n = max(12, int(np.linalg.norm(xyz-start)/speed*HZ))
        if self.deployment:
            end = self.ik.solve(xyz, self.qplan, grip=grip, yaw=yaw)
            # Preserve the joint-rate distribution of the half-speed live guard.
            n = max(n, int(np.ceil(1.5*np.max(np.abs(end[:5]-self.qplan[:5]))/self.deployment["joint_step_rad"])))
        for t in range(1, n+1):
            u = t/n
            u = u*u*(3-2*u)
            self._servo(start+(xyz-start)*u, grip, yaw)
        for _ in range(settle):
            self._servo(xyz, grip, yaw)
        if self.deployment:
            for _ in range(90):
                if np.linalg.norm(self.arm.tips(self.twin.q())[2]-xyz) < .003:
                    break
                self._servo(xyz, grip, yaw)
        self.goal_xyz, self.yaw = xyz.copy(), yaw

    def _hold(self, ticks, grip, phase):
        self.phase = phase
        if self.deployment and phase in ("close", "recovery_close", "release"):
            # There is one fixed jaw. Holding the fingertip midpoint while the
            # hinge moves sweeps that jaw through the cap, especially on release.
            # Hold the arm targets instead, then rebase FK on the new opening.
            label = self.nominal_command.copy()
            travel = abs(grip - label[5]) / self.deployment["joint_step_rad"]
            label[5] = grip
            for _ in range(max(ticks, int(np.ceil(travel)) + 30)):
                self._tick(label)
            self.qplan = self.twin.q()
            self.goal_xyz = self.arm.tips(self.qplan)[2].copy()
            return
        for _ in range(ticks):
            self._servo(self.goal_xyz, grip, self.yaw)

    def _carry_height(self, xy, preferred):
        """Highest reachable bench carry height with the measured grasp opening.

        The calibrated wrist-flex stop makes 50 mm unreachable at near-row cells.
        Do not spend a whole carry commanding that unreachable Cartesian pose.
        The final placement still uses the observed cap-to-jaw offset below.
        """
        grip = self.twin.q()[5]
        for height in np.arange(preferred, .011, -.002):
            target = np.r_[xy, height]
            q = self.ik.solve(target, self.qplan, grip=grip, iterations=120)
            if np.linalg.norm(self.arm.tips(q)[2] - target) < .002:
                return float(height)
        return preferred

    def _grasp_yaw(self, obj):
        pose = self.twin.poses()[obj]
        dims = extents(self.twin.objects[obj])
        angle = np.deg2rad(pose["yaw_deg"]) + (np.pi/2 if dims[0] > dims[1]+1e-6 else 0)
        # A square has four equivalent grasps; choose the reachable wrist roll.
        period = np.pi/2 if abs(dims[0]-dims[1]) < 1e-6 else np.pi
        bearing = np.arctan2(pose["xyz"][1], pose["xyz"][0])
        base = bearing+(angle-bearing+period/2) % period-period/2
        candidates = [base+i*period for i in range(-4,5)
                      if self.arm.range[4,0]+.02 < base+i*period+self.qplan[0]+.06 < self.arm.range[4,1]-.02]
        return min(candidates, key=lambda y: abs(y+self.qplan[0]+.06-self.qplan[4])) if candidates else base

    def pick(self, obj):
        definition = self.twin.objects[obj]
        dims = extents(definition)
        width = min(dims[:2])
        yaw = self._grasp_yaw(obj)
        open_grip = float(np.clip((width+.018-.017)/.081, .35, .95))
        if self.deployment:
            open_grip = self.release_grip
        for attempt in range(2):
            pos = np.asarray(self.twin.poses()[obj]["xyz"])
            yaw = self._grasp_yaw(obj)
            low = max(.0035, pos[2]-dims[2]/2+.0035)
            safe = max(.09, pos[2]+.06)
            if self.deployment:
                safe = self.deployment["travel_height_m"]
                # FK reports the bottom tips, whereas contact pads extend 16 mm
                # upward. A 9 mm tip height pinches a 14-20 mm cap above its
                # centre and lets it tip out during the slow bench carry.
                tip_height = self.deployment["expert_control"]["grasp_tip_height_m"]
                low = max(tip_height, pos[2]-dims[2]/2+tip_height)
            current = self.arm.tips(self.twin.q())[2]
            if current[2] < safe-.01:
                self.move([*current[:2], safe], open_grip, yaw, phase="approach_up")
            self.move([*pos[:2], safe], open_grip, yaw, phase="approach")
            self.move([*pos[:2], low], open_grip, yaw, speed=.045, settle=20, phase="descend")
            self._hold(35, self.close_grip, "close" if not attempt else "recovery_close")
            self.move([*pos[:2], safe if self.deployment else low+.065], self.close_grip, yaw, speed=.045, settle=20, phase="lift")
            actual = np.asarray(self.twin.poses()[obj]["xyz"])
            if actual[2]-pos[2] > (.03 if self.deployment else .04):
                self.held = obj
                return True
            self.phase = "recovery"
        self.fail_reason = f"grasp_miss:{obj}"
        return False

    def _target(self, step):
        target = step.get("target", step.get("xy"))
        if isinstance(target, str):
            return np.asarray(self.twin.poses()[target]["xyz"]), target
        return np.r_[np.asarray(target, float), 0.0], None

    def place(self, step):
        obj, skill = step["obj"], step["skill"]
        dest, target_id = self._target(step)
        dims = extents(self.twin.objects[obj])
        current = np.asarray(self.twin.poses()[obj]["xyz"])
        mid = self.arm.tips(self.twin.q())[2]
        offset = current-mid
        yaw = self.yaw
        if target_id:
            target = self.twin.objects[target_id]
            target_yaw = np.deg2rad(self.twin.poses()[target_id]["yaw_deg"])
            if skill in ("place_on", "place_at", "place_in") and not self.deployment:
                period = np.pi/2 if abs(dims[0]-dims[1]) < 1e-6 else np.pi
                obj_yaw = np.deg2rad(self.twin.poses()[obj]["yaw_deg"])
                yaw += (target_yaw-obj_yaw+period/2) % period-period/2
            if skill == "place_on":
                surface = dest[2]+extents(target)[2]/2
            elif skill == "place_in":
                # Release above the rim: putting the whole closed hand inside a
                # shallow bowl can pinch its wall or the previously placed cube.
                surface = dest[2]+extents(target)[2]/2+.004
                # Spread successive pieces inside a shared container.
                peers = [s["obj"] for s in self.task["skeleton"]
                         if s["skill"] == "place_in" and s.get("target") == target_id]
                if len(peers) > 1:
                    dest[1] += (peers.index(obj)-(len(peers)-1)/2)*(dims[1]+.006)
            else:
                surface = dest[2]+extents(target)[2]/2 if target.get("collide", True) else 0
        else:
            surface = 0
        safe = max(.09, surface+.065)
        if self.deployment:
            safe = self.deployment["travel_height_m"]
        carry_xy = dest[:2]-offset[:2]
        carry_height = self._carry_height(carry_xy, safe) if self.deployment else safe
        self.move([*carry_xy, carry_height], self.close_grip, yaw, speed=.065, settle=25, phase="carry")
        # Measure the grasp after rotating, so the held object, not the jaw, is centred.
        current = np.asarray(self.twin.poses()[obj]["xyz"])
        mid = self.arm.tips(self.twin.q())[2]
        if self.deployment and np.linalg.norm(current-mid) > max(dims)+.025:
            self.held = None
            self.fail_reason = f"dropped_during_carry:{obj}"
            return False
        offset = current-mid
        target_mid = np.r_[dest[:2], surface+dims[2]/2+.002]-offset
        target_mid[2] = max(.003, target_mid[2])
        if self.deployment:
            target_mid[2] = max(.012, target_mid[2])
        self.move(target_mid, self.close_grip, yaw, speed=.035, settle=20, phase="lower")
        for _ in range(12):
            error = dest[:2]-np.asarray(self.twin.poses()[obj]["xyz"])[:2]
            self.goal_xyz[:2] += np.clip(error, -.0008, .0008)
            self._servo(self.goal_xyz, self.close_grip, yaw)
        self._hold(6 if skill == "place_in" else 22, self.release_grip, "release")
        self.held = None
        self.move([*self.goal_xyz[:2], safe], self.release_grip, yaw, speed=.045, settle=10, phase="retreat")
        return True

    def push(self, step):
        obj = step["obj"]
        target, _ = self._target(step)
        dims = extents(self.twin.objects[obj])
        for attempt in range(3):
            pos = np.asarray(self.twin.poses()[obj]["xyz"])
            delta = target[:2]-pos[:2]
            distance = np.linalg.norm(delta)
            if distance < .009:
                break
            direction = delta/max(distance, 1e-8)
            yaw = np.arctan2(direction[1], direction[0])+np.pi/2
            radius = np.linalg.norm(dims[:2])/2+.022
            behind = pos[:2]-direction*radius
            self.move([*behind, .08], .02, yaw, phase="push_approach")
            self.move([*behind, .012], .02, yaw, speed=.04, phase="push_descend")
            self.phase = "push" if attempt == 0 else "recovery_push"
            for _ in range(220):
                pos = np.asarray(self.twin.poses()[obj]["xyz"])
                delta = target[:2]-pos[:2]
                distance = np.linalg.norm(delta)
                if distance < .007:
                    break
                if np.dot(delta, direction) < 0:
                    break
                # Observe the object's progress instead of assuming the fingertip
                # sits at a particular offset from a rotating cube face.
                self.goal_xyz[:2] += direction*min(.0008, distance/4)
                self._servo(self.goal_xyz, .02, yaw)
            actual = self.arm.tips(self.twin.q())[2]
            # Back off with the fist closed. Opening beside the cube swats it.
            self.move([*(actual[:2]-direction*.012), actual[2]], .02, yaw, speed=.035,
                      settle=3, phase="push_backoff")
            self.move([*self.goal_xyz[:2], .09], .02, yaw, phase="push_retreat")
            self._hold(10, .5, "push_open")

    def home(self):
        current = self.arm.tips(self.twin.q())[2]
        self.move([*current[:2], max(.05 if self.deployment else .1, current[2])], self.release_grip, self.yaw, phase="home_up")
        start = (self.nominal_command if self.deployment else self.twin.commanded_q).copy()
        self.phase = "home"
        target = self.ik.compensate(self.start_q if self.deployment else HOME_Q, self.twin.physics)
        ticks = max(60, int(np.ceil(np.max(np.abs(target-start))/self.deployment["joint_step_rad"]))) if self.deployment else 60
        for i in range(ticks):
            u = (i+1)/ticks
            self._tick(start+(target-start)*u)
        for _ in range(20):
            self._tick(target)

    def run(self, twin=None, *, seed=0, scene=None):
        self.rng = np.random.default_rng(seed)
        own_twin = twin is None
        if own_twin:
            scene = scene or sample_scene(self.task, np.random.default_rng(seed))
            twin = TaskTwin(scene, q0=self.deployment["start_q"] if self.deployment else HOME_Q)
        self.ticks, self.states, self.segments = [], [], []
        self.fail_reason, self.held = None, None
        self._bind(twin)
        try:
            for step, command in zip(self.task["skeleton"], commands(self.task)):
                self.command = command
                start = len(self.ticks)/HZ
                skill = step["skill"]
                if skill == "pick":
                    ok = self.pick(step["obj"])
                elif skill in ("place_in", "place_on", "place_at"):
                    ok = self.place(step)
                elif skill == "push_to":
                    self.push(step)
                    ok = True
                elif skill == "home":
                    self.home()
                    ok = True
                elif skill == "release":
                    self._hold(15, self.release_grip, "release")
                    ok = True
                elif skill == "wait":
                    self._hold(max(1, round(step["s"]*HZ)), twin.commanded_q[5], "wait")
                    ok = True
                else:
                    raise ValueError(f"Unsupported skill {skill}")
                self.segments.append([start, len(self.ticks)/HZ, command])
                if not ok:
                    break
            result = check_window(self.task, self.states, hz=HZ)
            success = bool(result["ok"])
            meta = {"schema": "r2p_trace/1", "task": self.task, "task_name": self.task["name"], "source": "expert",
                    "player": "reference_expert", "level": self.task["name"], "seed": seed,
                    "scene": twin.scene, "success": success, "fail_reason": None if success else self.fail_reason or "goal_not_met",
                    "commands": self.segments, "rig": "so101_bench", "start_q": self.start_q.tolist(), "dart_std_rad": self.dart}
            meta["physics_config"] = deepcopy(twin.physics)
            meta["physics_sha256"] = hashlib.sha256(json.dumps(twin.physics, sort_keys=True).encode()).hexdigest()
            meta["target_space"] = "servo"
            if self.deployment:
                meta["deployment"] = deepcopy(self.deployment)
            return {"meta": meta, "ticks": self.ticks, "check": result, "final": twin.poses()}
        finally:
            if own_twin:
                twin.close()

    rollout = run


def evaluate(task, n=100, seed=10000, dart=0.0, progress=False):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    task = load_task(task) if not isinstance(task, dict) else task
    expert = Expert(task, dart=dart, seed=seed)
    failures, errors, outcomes = Counter(), [], []
    start = time.perf_counter()
    for i in range(n):
        result = expert.run(seed=seed+i)
        success = result["meta"]["success"]
        outcomes.append(success)
        if not success:
            failures[result["meta"]["fail_reason"]] += 1
        for goal in task["goal"]:
            if goal["pred"] == "at":
                target = result["final"][goal["xy"]]["xyz"][:2] if isinstance(goal["xy"], str) else goal["xy"]
                errors.append(float(np.linalg.norm(np.asarray(result["final"][goal["a"]]["xyz"][:2])-target)*1000))
        if progress and ((i+1) % 10 == 0 or i == 0):
            print(f"{task['name']} {i+1}/{n}: {sum(outcomes)}/{i+1} successes", flush=True)
    return {"task": task["name"], "n": n, "seed": seed, "success": float(np.mean(outcomes)),
            "successes": sum(outcomes), "failures": dict(failures),
            "median_placement_mm": float(np.median(errors)) if errors else None,
            "seconds": time.perf_counter()-start, "outcomes": outcomes,
            "physics_sha256": hashlib.sha256(json.dumps(load_physics(), sort_keys=True).encode()).hexdigest(),
            "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
            "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ("expert.py", "ik.py", "twin.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="cube_in_bowl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--dart", type=float, default=0)
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = evaluate(args.task, args.n, args.seed, args.dart, progress=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
