"""Headless Python port of Roblox SkillOperator's pick/place_at/release/home.

The player uses the game's phase machine and Kinematics.ik equations. Contacts
run in MuJoCo, not Roblox: these are explicitly simulated bot players. There is
no Expert delegation, attachment, pose assignment, or success shortcut.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from r2p.frames import HOME_Q, HZ, JOINTS, ROOT
from r2p.model import Arm
from r2p.sim import TaskTwin
from r2p.sim.twin import extents, load_physics
from r2p.tasks import check_window, commands, load_task, sample_scene
from r2p.traces import write_trace
from r2p.labeling import segment


def physics_hash(physics):
    # Same encoding as Roblox presets and policy exports.
    return hashlib.sha256(json.dumps(physics, sort_keys=True).encode()).hexdigest()


class GamePlayer:
    def __init__(self, task, physics, q=HOME_Q):
        if any(s['skill'] not in {'pick', 'place_at', 'release', 'home', 'wait'} for s in task['skeleton']):
            raise ValueError('Headless player supports pick/place_at/release/home/wait')
        self.task, self.physics = task, physics
        self.arm = Arm()
        self.q = np.array(q, float).copy()
        self.index, self.stage, self.age = 0, 1, 0.
        self.retry = self.carry_retries = 0
        self.pick_yaw = self.yaw = self.place_offset = self.goal = None
        self.release_hold = self.done = False
        self.commands = commands(task)

    def advance(self):
        self.stage += 1
        self.age = 0.

    def next(self):
        self.index += 1
        self.stage, self.age, self.retry = 1, 0., 0
        self.pick_yaw = self.place_offset = None
        self.done = self.index >= len(self.task['skeleton'])

    def compensate(self, q):
        kp = np.array([self.physics['joints'][j]['kp'] for j in JOINTS])
        gravity = -self.physics.get('gravity', [0, 0, -9.81])[2] / 9.81
        return np.clip(q + self.arm.gravity_torque(q)*gravity/kp, *self.arm.range.T)

    def ik(self, target, q, roll=None):
        """Literal Kinematics.lua solver using the shared official-model Jacobian."""
        out = q.copy()
        count = 5 if roll is None else 4
        if roll is not None:
            out[4] = np.clip(roll, *self.arm.range[4])
        unit = lambda v: v / max(np.linalg.norm(v), 1e-12)
        for _ in range(12):
            a, b, mid = self.arm.tips(out)
            linear, angular, down = self.arm.jac_tip(out)
            linear, angular = linear[:, :count], angular[:, :count]
            across = unit(b-a)
            orientation = np.cross(across, unit(np.r_[across[:2], 0.]))
            orientation += np.cross(unit(down-across*np.dot(down, across)), [0, 0, -1])
            gram = linear@linear.T + .000015*np.eye(3)
            position = linear.T@np.linalg.solve(gram, np.clip(target-mid, -.025, .025))
            posture = angular.T@np.linalg.solve(angular@angular.T+.01*np.eye(3), orientation)
            projection = linear.T@np.linalg.solve(gram, linear@posture)
            out[:count] = np.clip(out[:count]+np.clip(position+.6*(posture-projection), -.15, .15),
                                  self.arm.range[:count, 0]+1e-5, self.arm.range[:count, 1]-1e-5)
        return out

    def step(self, state, dt=1/HZ):
        if self.done:
            return self.compensate(self.q), 'home', 'done'
        self.age += dt
        step = self.task['skeleton'][self.index]
        skill, command = step['skill'], self.commands[self.index]
        phase, grip, target = f'{skill}:{self.stage}', self.q[5], None
        obj = next((o for o in self.task['objects'] if o['id'] == step.get('obj')), None)
        pose = state['objects'].get(step.get('obj'))
        site = np.asarray(state['site_xyz'])
        if skill in {'pick', 'place_at'} and pose is None:
            self.next()
            return self.q.copy(), command, 'absent'
        size = extents({**obj, 'size_m': pose.get('size_m', obj['size_m'])}) if obj and pose else None
        thin = obj is not None and obj['shape'] == 'box' and size[2] < .02
        if skill == 'pick':
            xyz = np.asarray(pose['xyz'])
            if self.pick_yaw is None:
                bearing = np.arctan2(xyz[1], xyz[0])
                yaw = np.deg2rad(pose.get('yaw_deg', 0)) + (np.pi/2 if size[0] > size[1]+1e-6 else 0)
                period = np.pi/2 if abs(size[0]-size[1]) < 1e-6 else np.pi
                self.pick_yaw = bearing+(yaw-bearing+period/2) % period-period/2
                self.yaw = self.pick_yaw
            clearance = .0015 if thin else .0035
            z = max(clearance, xyz[2]-size[2]/2+clearance)
            if thin:
                self.yaw = None
            height = (.09 if thin else .10) if self.stage == 1 else (.07 if thin else .10) if self.stage == 4 else z
            target = np.r_[xyz[:2], height]
            grip = np.clip((min(size[:2])+.018-.017)/.081, .35, .95) if self.stage < 3 else -.16
            if thin and self.stage < 3:
                grip = max(grip, .55)
            if self.stage == 1 and (np.linalg.norm(site-target) < .008 or self.age > 6):
                self.advance()
            elif self.stage == 2 and (np.linalg.norm(site-target) < .005 or self.age > 4):
                self.advance()
            elif self.stage == 3 and self.age > (1.15 if thin else .85):
                self.advance()
            elif self.stage == 4 and self.age > 1.2 and xyz[2]-size[2]/2 > .04:
                self.next()
            elif self.stage == 4 and self.age > 3.5:
                if self.retry < 2:
                    self.retry += 1
                    self.stage, self.age, self.pick_yaw = 1, 0., None
                else:
                    self.next()
        elif skill == 'place_at':
            self.release_hold = True
            if (self.index and self.task['skeleton'][self.index-1]['skill'] == 'pick'
                    and pose['xyz'][2]-size[2]/2 < .012 and site[2] > .04
                    and self.age > 1.5 and self.carry_retries < 2):
                self.carry_retries += 1
                self.index -= 1
                self.stage, self.age = 1, 0.
                self.pick_yaw = self.place_offset = None
                retry = self.compensate(self.q)
                retry[5] = .55
                return retry, self.commands[self.index], 'recover_lost_piece'
            dst = step.get('target', step.get('xy'))
            dest = state['objects'].get(dst) if isinstance(dst, str) else {'xyz': [*dst, 0]}
            if dest is None:
                self.next()
                return self.q.copy(), command, 'absent_target'
            held = np.asarray(pose['xyz'])-site
            if np.linalg.norm(held) < .06 and (self.place_offset is None or self.stage == 1):
                self.place_offset = held.copy()
            centre = np.zeros(3) if self.place_offset is None else self.place_offset
            height = .022 if self.place_offset is None else max(.003, size[2]/2+.003-centre[2])
            target = np.r_[np.asarray(dest['xyz'])[:2]-centre[:2], .09 if self.stage == 1 else height]
            grip = -.16
            if self.stage == 1 and (np.linalg.norm(site-target) < .008 or self.age > 6):
                self.advance()
            elif self.stage == 2 and (np.linalg.norm(site-target) < .006 or self.age > 4):
                self.next()
        elif skill == 'release':
            if self.release_hold:
                grip = .55
                self.q[5] = grip
            else:
                target, grip = self.goal, .8
            if self.age > .8:
                self.next()
                self.release_hold = False
        elif skill == 'home':
            self.q = HOME_Q.copy()
            if self.age > 1.5:
                self.next()
        elif skill == 'wait' and self.age >= step['s']:
            self.next()
        if target is not None:
            if self.goal is None:
                self.goal = site.copy()
            delta = target-self.goal
            speed = .045 if thin else .075
            self.goal += delta*min(1., speed*dt/max(np.linalg.norm(delta), 1e-12))
            actual_grip = state.get('q', self.q)[5]
            self.q[5] = actual_grip+np.clip(grip-actual_grip, -.015, .015)
            roll = None
            if self.yaw is not None:
                roll = self.yaw+self.q[0]+.06
                while roll > self.arm.range[4, 1]:
                    roll -= np.pi
                while roll < self.arm.range[4, 0]:
                    roll += np.pi
            previous = self.q.copy()
            self.q = self.ik(self.goal, self.q, roll)
            if thin:
                self.q[:5] = previous[:5]+np.clip(self.q[:5]-previous[:5], -np.deg2rad(3), np.deg2rad(3))
        target_q = self.compensate(self.q)
        target_q[5] = grip
        return target_q, command, phase


def collect_players(task, directory, n=8, seed=500, physics='so101_bench'):
    task, physics = load_task(task), load_physics(physics)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for episode in range(n):
        episode_seed = seed+episode
        scene = sample_scene(task, np.random.default_rng(episode_seed))
        player = GamePlayer(task, physics)
        ticks, states = [], deque(maxlen=15)
        with TaskTwin(scene, physics=physics, q0=HOME_Q) as twin:
            initial = twin.poses()
            for i in range(round(task['limits']['episode_s']*HZ)):
                state = {'objects': twin.poses(), 'site_xyz': player.arm.tips(twin.q())[2].tolist(),
                         'q': twin.q().tolist(), 'gripper_open': not twin.grip_closed(), 'initial': initial}
                target, command, phase = player.step(state)
                ticks.append({'t': i/HZ, 'q': twin.q().tolist(), 'q_target': target.tolist(),
                              'poses': state['objects'], 'grip_closed': bool(target[5] < .15),
                              'command': command, 'phase': phase})
                # Match Station.tick: observations precede the action's physics tick.
                states.append({**state, 'gripper_open': bool(target[5] >= .15)})
                twin.step(target)
                if check_window(task, list(states), hz=HZ)['ok']:
                    break
            success = check_window(task, list(states), hz=HZ)['ok']
        meta = {'task': task, 'source': 'bot', 'player': 'headless_skill_operator_python',
                'level': task['name'], 'seed': episode_seed, 'scene': scene, 'success': bool(success),
                'fail_reason': None if success else 'timeout', 'rig': physics.get('rig', 'so101_bench'),
                'start_q': HOME_Q.tolist(), 'headless': True, 'synthetic': True, 'target_space': 'servo',
                'physics_config': physics, 'physics_sha256': physics_hash(physics),
                'provenance': 'Python port of Roblox SkillOperator; MuJoCo contacts; no human or Roblox engine',
                'player_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'game_code_sha256': {name: hashlib.sha256((ROOT/'roblox/src/ReplicatedStorage/Twin'/name).read_bytes()).hexdigest()
                                    for name in ('SkillOperator.lua', 'Kinematics.lua')}}
        meta['commands'] = segment((meta, ticks), task)
        path = directory/f'headless_{episode_seed:08d}.jsonl'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite player evidence: {path}')
        write_trace(path, meta, ticks)
        paths.append(path)
        print(f'HEADLESS {episode+1}/{n}: success={success}, ticks={len(ticks)}', flush=True)
    return paths
