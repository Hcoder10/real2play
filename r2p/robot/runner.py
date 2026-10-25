"""30 Hz session runner; hardware, policy, perception and clocks are injectable."""
from copy import deepcopy
import math
import time
import numpy as np

from .config import load_config, overlay, sag_gain
from .gravity import GravityComp
from .guard import Guard
from .io import vector
from .operator import Operator
from .policy import Ensembler, policy_config
from .record import TraceWriter


def observation_q(commanded, measured, threshold_deg=4):
    """One divergent joint resets the entire observation to the measured pose."""
    commanded, measured = vector(commanded), vector(measured)
    reset = bool(np.any(np.abs(commanded - measured) > np.deg2rad(threshold_deg)))
    return (measured if reset else commanded).copy(), reset


class Session:
    def __init__(self, robot, policy, task, *, arm=None, joint_map=None, camera=None,
                 perceiver=None, world=None, cfg=None, operator=None, gravity_comp=True,
                 speed=1.0, realtime=True, record=None, policy_cfg=None, publisher=None):
        from r2p.model import Arm, JointMap
        from r2p.tasks import load_task
        self.cfg = overlay(load_config(), cfg or {})
        self.robot, self.policy = robot, policy
        self.task = load_task(task) if not isinstance(task, dict) else task
        self.arm = arm or Arm()
        self.joint_map = joint_map or getattr(robot, "joint_map", None) or JointMap()
        self.policy_cfg = policy_cfg or policy_config(policy)
        self.start_q = vector(self.policy_cfg["start_q"])
        self.hz = self.cfg["control_hz"]
        if self.hz != 30:
            raise ValueError("policy and trace contract requires 30 Hz")
        self.dt = 1 / self.hz
        self.guard = Guard(self.arm, self.cfg["guard"], hz=self.hz, speed=speed, start_q=self.start_q)
        self.comp = GravityComp(self.arm, sag_gain(self.cfg) if gravity_comp else 0)
        self.operator = operator or Operator(self.cfg["control_file"], keyboard=False)
        self.camera, self.perceiver, self.world = camera, perceiver, world
        self.realtime, self.record = realtime, record
        self.publisher = publisher
        self.last_commanded = self.joint_map.to_model(self.robot.read())
        self.last_sent = self.last_commanded.copy()
        self.last_nominal = self.last_sent.copy()

    def _step(self):
        if hasattr(self.robot, "step"):
            self.robot.step(self.dt)
        if self.world is not None:
            self.world.update(self.joint_map.to_model(self.robot.read()), self.robot)

    def _wait(self, started):
        if self.realtime:
            time.sleep(max(0, self.dt - (time.perf_counter() - started)))

    def _hold(self):
        # Guard refusals use last target; explicit operator stops use measured hold.
        self.robot.hold()
        self.last_sent = self.joint_map.to_model(self.robot.read())
        self.last_commanded = self.last_sent.copy()
        self.last_nominal = self.last_sent.copy()

    def _send(self):
        accepted_owner = self.robot.write(self.joint_map.to_owner(self.last_sent))
        if accepted_owner is not None:
            self.last_sent = self.joint_map.to_model(accepted_owner)
        self.last_commanded = self.last_sent.copy()

    def _hold_after_error(self, exc):
        message = f"{type(exc).__name__}: {exc}"
        try:
            self._hold()
        except Exception as hold_error:
            message += f"; hold also failed: {type(hold_error).__name__}: {hold_error}"
        return message

    def _overloaded(self):
        loads = self.robot.loads()
        return loads is not None and any(abs(float(v)) > self.cfg["load_limit"] for v in loads.values())

    def move_start(self, *, retreat=False):
        """Guarded, interpolated approach to the export's start pose."""
        goal = self.comp(self.start_q)
        goal = np.clip(goal, self.guard.limits[:, 0], self.guard.limits[:, 1])
        refused = 0
        for _ in range(30 * 45):
            started = time.perf_counter()
            events = self.operator.poll()
            if "torque_off" in events:
                self.robot.torque_off()
                return False
            if any(x in events for x in ("stop", "quit")) or self._overloaded():
                self._hold()
                return False
            if "pause" in events:
                self._hold()
            if self.operator.paused:
                self._step()
                self._wait(started)
                continue
            measured = self.joint_map.to_model(self.robot.read())
            if np.max(np.abs(self.last_sent - goal)) < np.deg2rad(0.25) and np.max(np.abs(measured - self.start_q)) < np.deg2rad(4):
                return True
            result = self.guard.project(self.last_sent, goal, measured=measured, retreat=retreat)
            if result.accepted:
                refused = 0
                self.last_sent = result.target
            else:
                refused += 1
            self._send()
            self._step()
            self._wait(started)
            if refused >= 60:
                return False
        return False

    def _observe(self, q):
        frame = self.camera.read() if self.camera is not None else None
        if self.camera is not None and getattr(self.camera, "age_s", 0) > self.cfg["perception"]["max_lost_s"]:
            raise RuntimeError("stale camera")
        if self.perceiver is not None:
            observation = self.perceiver.estimate(frame, q)
            poses = observation["poses"]
        elif self.world is not None:
            poses = self.world.poses()
        else:
            raise RuntimeError("Session requires a perceiver or synthetic truth world")
        return frame, poses

    def run(self, *, timeout_s=None, prepare=True):
        from r2p.tasks import check, encode_slots, commands
        timeout_s = float(timeout_s or self.cfg["timeout_s"])
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("timeout must be positive and finite")
        timings, refusal_count, clipped, resets, consecutive = [], 0, 0, 0, 0
        goal_ticks = 0
        perceived = {"ok": False, "per_predicate": []}
        truth = None
        writer = None
        tick = 0
        outcome = "timeout"
        error = None
        initial_poses = None
        command_segments = []
        ensemble = Ensembler(self.cfg["ensemble_m"])
        replan = int(self.cfg["replan_every"])
        if not 1 <= replan <= 15:
            raise ValueError("replan_every must be between 1 and 15")
        self.operator.score = None
        if self.record:
            writer = TraceWriter(self.record, {"task": self.task, "task_name": self.task["name"],
                "source": "bot" if self.world is not None else "real", "player": "policy",
                "level": self.task["name"], "seed": getattr(self.world, "seed", None),
                "scene": getattr(self.world, "scene", None), "success": False,
                "fail_reason": None, "commands": [], "rig": self.cfg["rig"]})
        started_run = time.perf_counter()
        try:
            if prepare and not self.move_start():
                outcome = "start_refused"
            else:
                loop_start = time.monotonic()
                for tick in range(int(math.ceil(timeout_s * self.hz))):
                    started = time.perf_counter()
                    if self.realtime and time.monotonic() - loop_start >= timeout_s:
                        break
                    events = self.operator.poll()
                    if "torque_off" in events:
                        self.robot.torque_off()
                        outcome = "operator"
                        break
                    if any(x in events for x in ("stop", "quit")):
                        self._hold()
                        self.move_start(retreat=True)
                        outcome = "operator"
                        break
                    if "pause" in events:
                        self._hold()
                        ensemble.clear()
                    if self.operator.paused:
                        goal_ticks = 0
                        self._step()
                        self._wait(started)
                        continue
                    if self._overloaded():
                        self._hold()
                        outcome = "load_stop"
                        break
                    measured = self.joint_map.to_model(self.robot.read())
                    q_obs, reset = observation_q(self.last_commanded, measured, self.cfg["divergence_deg"])
                    if reset:
                        self.last_commanded = q_obs.copy()
                        ensemble.clear()
                        resets += 1
                    frame, poses = self._observe(measured)
                    required = [o["id"] for o in self.task["objects"] if not o.get("optional", False)
                                and self.task.get("randomize", {}).get(o["id"], {}).get("p_present", 1) == 1]
                    if any(oid not in poses or poses[oid].get("seen", True) is False for oid in required):
                        self.robot.write(self.joint_map.to_owner(self.last_sent))
                        goal_ticks = 0
                        ensemble.clear()
                        self._step()
                        self._wait(started)
                        continue
                    if initial_poses is None:
                        initial_poses = deepcopy(poses)
                    vocabulary = commands(self.task)
                    command = getattr(self.policy, "command", None) or (vocabulary[0] if vocabulary else "")
                    if not command_segments or command_segments[-1][2] != command:
                        command_segments.append([tick*self.dt, (tick+1)*self.dt, command])
                    else:
                        command_segments[-1][1] = (tick+1)*self.dt
                    obs = {"q": q_obs, "slots": encode_slots(self.task, poses), "command": command}
                    if tick % replan == 0 or not ensemble.chunks:
                        ensemble.add(tick, self.policy.act(obs))
                    nominal = ensemble.target(tick)
                    # Expert exports already contain servo-space compensation.
                    correction = (np.zeros(6) if self.policy_cfg.get("gravity_compensation") == "included_in_expert_targets"
                                  else self.comp.offset(nominal))
                    guarded = self.guard.project(self.last_sent, nominal + correction, measured=measured)
                    if guarded.accepted:
                        self.last_sent = guarded.target
                        self.last_commanded = guarded.target.copy()
                        consecutive = 0
                        clipped += int(guarded.clipped)
                    else:
                        refusal_count += 1
                        consecutive += 1
                    self._send()
                    if guarded.accepted:
                        self.last_nominal = self.last_sent - correction
                    mid = self.arm.tips(measured)[2]
                    gripper_open = bool(measured[5] > 0.5 and self.last_sent[5] > 0.5)
                    state = {"objects": poses, "gripper_open": gripper_open, "site_xyz": mid.tolist(), "initial": initial_poses}
                    perceived = check(self.task, state)
                    eligible = perceived["ok"] and gripper_open and mid[2] >= 0.07
                    goal_ticks = goal_ticks + 1 if eligible else 0
                    if self.world is not None:
                        truth = self.world.truth_check()
                    if self.publisher and frame is not None and tick % 6 == 0:
                        self.publisher.update({"source": "bot" if self.world is not None else "real",
                                               "status": "running", "task": self.task["name"],
                                               "command": command, "phase": getattr(self.policy, "phase", "policy"),
                                               "synthetic": self.world is not None}, frame)
                    if writer:
                        writer.submit({"t": tick * self.dt, "q": measured,
                            "q_commanded": q_obs,
                            "q_target": self.last_nominal,
                            "q_exec": self.last_sent, "poses": poses, "grip_closed": not gripper_open,
                            "command": command, "phase": getattr(self.policy, "phase", "policy"),
                            "guard": guarded.reason}, frame)
                    timings.append((time.perf_counter() - started) * 1000)
                    self._step()
                    if goal_ticks >= self.hz:
                        outcome = "goal_reached"
                        break
                    if consecutive >= self.cfg["max_consecutive_refusals"]:
                        outcome = "held_after_refusals"
                        break
                    if getattr(self.policy, "done", False):
                        outcome = "policy_done"
                        break
                    self._wait(started)
        except KeyboardInterrupt:
            error = self._hold_after_error(KeyboardInterrupt())
            outcome = "operator"
        except Exception as exc:
            error = self._hold_after_error(exc)
            outcome = "error"
        finally:
            # No disconnect or torque change here. The daemon retains the hold.
            success_auto = bool(truth["ok"] if truth is not None else perceived["ok"])
            override = self.operator.score
            summary = {"outcome": outcome, "success": success_auto if override is None else override,
                "perception_check": deepcopy(perceived), "truth_check": truth, "operator_override": override,
                "timing_ms": {"p50": float(np.percentile(timings, 50)) if timings else 0,
                              "p90": float(np.percentile(timings, 90)) if timings else 0},
                "refusals": refusal_count, "clipped_ticks": clipped, "commanded_q_resets": resets,
                "clipped_fraction": clipped / len(timings) if timings else 0,
                "ticks": len(timings), "duration_s": time.perf_counter() - started_run,
                "error": error, "record_dropped": writer.dropped if writer else 0}
            summary["observation_source"] = "perception" if self.perceiver is not None else "synthetic_truth"
            if self.publisher:
                self.publisher.update({"source": "bot" if self.world is not None else "real",
                                       "synthetic": self.world is not None, "status": "idle", **summary})
            if writer:
                writer.meta["commands"] = command_segments
                writer.close(summary)
        return summary
