"""Game -> current contact twin -> command labels -> CPU learning -> paired gate.

Raw files are immutable evidence. Only accepted replay observations enter the
training set. A HOLD is a completed experiment, never an invented improvement.
"""
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from r2p.frames import HOME_Q, HZ
from r2p.labeling import segment
from r2p.model import Arm
from r2p.sim import TaskTwin
from r2p.sim.twin import load_physics
from r2p.tasks import check_window, load_task, sample_scene
from r2p.traces import read_trace, write_trace
from .game_player import collect_players, physics_hash
from .storage import atomic_json, digest, file_digest, read_json, run_lock, utcnow


@dataclass(frozen=True)
class ReplayLimits:
    joint_rms_deg: float = 1.
    joint_max_deg: float = 4.
    jaw_rms_mm: float = 2.
    object_max_mm: float = 5.
    object_yaw_deg: float = 5.
    object_tilt_deg: float = 5.


def replay_trace(source, destination, task, physics, *, limits=None):
    """Replay all targets from reset, never reset objects to recorded poses.

    Station traces observe q/poses BEFORE q_target is applied. Their grip_closed
    is the current target flag. Relabeled training observations use the previous
    commanded q (or measured q after >4 degree divergence).
    """
    limits = limits or ReplayLimits()
    meta, ticks = read_trace(source)
    if meta['source'] not in {'human', 'bot'} or meta.get('test_only') or meta.get('attract'):
        raise ValueError('Only non-fixture human/bot gameplay is eligible')
    if meta.get('headless') and meta['source'] != 'bot':
        raise ValueError('Headless players must be labeled source=bot')
    if not ticks:
        raise ValueError('Empty game trace')
    recorded_task = meta['task']
    if isinstance(recorded_task, str):
        if recorded_task != task['name']:
            raise ValueError('Trace task name differs from selected task')
        recorded_task = task
    if digest(recorded_task) != digest(task):
        raise ValueError('Task revision differs from the selected task; recollect or use the recorded task JSON')
    if len(ticks) > round(task['limits']['episode_s']*HZ)+HZ:
        raise ValueError('Trace exceeds the task episode budget')
    scene = meta['scene']
    objects = scene.get('objects', [])
    expected = {o['id']: o for o in task['objects']}
    ids = [o['id'] for o in objects]
    if len(set(ids)) != len(ids) or not set(ids) <= set(expected) or not objects:
        raise ValueError('Invalid scene object identities')
    for obj in objects:
        definition = expected[obj['id']]
        if obj['shape'] != definition['shape'] or bool(obj.get('fixed')) != bool(definition.get('fixed')):
            raise ValueError('Scene shape/fixed flags disagree with the task')
    arm = Arm()
    targets = np.asarray([row['q_target'] for row in ticks])
    if np.any(targets < arm.range[:, 0]-1e-6) or np.any(targets > arm.range[:, 1]+1e-6):
        raise ValueError('Absolute target exceeds robot joint limits')
    start = np.asarray(meta.get('start_q', ticks[0]['q']), float)
    if start.shape != (6,) or not np.isfinite(start).all() or np.any(start < arm.range[:, 0]) or np.any(start > arm.range[:, 1]):
        raise ValueError('Invalid start_q')
    intervals = segment((meta, ticks), task, refine=False)
    rows, states = [], deque(maxlen=15)
    q_errors, jaw_errors, object_errors, yaw_errors, tilt_errors = [], [], [], [], []
    index, resets = 0, 0
    commanded = start.copy()
    with TaskTwin(deepcopy(scene), q0=start, physics=physics) as twin:
        initial = twin.poses()
        for i, row in enumerate(ticks):
            measured, poses = twin.q(), twin.poses()
            if set(row['poses']) != set(poses):
                raise ValueError(f'Missing/unknown recorded objects at tick {i}')
            q_errors.append(measured-np.asarray(row['q']))
            jaw_errors.append(np.linalg.norm(arm.tips(measured)[2]-arm.tips(row['q'])[2]))
            for oid, pose in poses.items():
                recorded = row['poses'][oid]
                if 'size_m' in recorded and not np.allclose(recorded['size_m'], pose['size_m'], atol=1e-6):
                    raise ValueError(f'Object dimensions changed at tick {i}')
                object_errors.append(np.linalg.norm(np.asarray(pose['xyz'])-recorded['xyz']))
                # Cylinders/spheres have no meaningful yaw; use task symmetry for the rest.
                from r2p.tasks import yaw_symmetry
                symmetry = yaw_symmetry(expected[oid])
                if symmetry:
                    period = 360/symmetry
                    yaw_errors.append(abs((pose['yaw_deg']-recorded['yaw_deg']+period/2) % period-period/2))
                if 'quat' in recorded and expected[oid]['shape'] != 'sphere':
                    import mujoco
                    quaternion = np.asarray(recorded['quat'], float)
                    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
                        raise ValueError(f'Invalid object quaternion at tick {i}')
                    reference, actual = np.zeros(9), np.zeros(9)
                    mujoco.mju_quat2Mat(reference, np.asarray(pose['quat']))
                    mujoco.mju_quat2Mat(actual, quaternion/np.linalg.norm(quaternion))
                    axis = {'x': 0, 'y': 1, 'z': 2}[expected[oid].get('axis', 'z')]
                    dot = np.dot(reference.reshape(3, 3)[:, axis], actual.reshape(3, 3)[:, axis])
                    tilt_errors.append(float(np.rad2deg(np.arccos(np.clip(dot, -1, 1)))))
            q_input = commanded.copy()
            if np.max(abs(q_input-measured)) > np.deg2rad(4):
                q_input = measured.copy()
                resets += 1
            while index+1 < len(intervals) and row['t'] >= intervals[index][1]:
                index += 1
            target = targets[i]
            rows.append({'t': i/HZ, 'q': measured.tolist(), 'q_commanded': q_input.tolist(),
                         'q_target': target.tolist(), 'poses': poses, 'grip_closed': twin.grip_closed(),
                         'command': intervals[index][2], 'phase': row['phase']})
            states.append({'objects': poses, 'initial': initial, 'gripper_open': bool(target[5] >= .15),
                           'site_xyz': arm.tips(measured)[2].tolist()})
            twin.step(target)
            commanded = target.copy()
        success = bool(check_window(task, list(states), hz=HZ)['ok'])
    metrics = {'joint_rms_deg': float(np.rad2deg(np.sqrt(np.mean(np.square(q_errors))))),
               'joint_max_deg': float(np.rad2deg(np.max(np.abs(q_errors)))),
               'jaw_rms_mm': float(1000*np.sqrt(np.mean(np.square(jaw_errors)))),
               'object_max_mm': float(1000*max(object_errors, default=0)),
               'object_yaw_deg': float(max(yaw_errors, default=0)),
               'object_tilt_deg': float(max(tilt_errors, default=0))}
    failures = [key for key, threshold in asdict(limits).items() if metrics[key] > threshold]
    report = {'source': str(source), 'source_sha256': file_digest(source), 'metrics': metrics,
              'limits': asdict(limits), 'accepted': not failures, 'failures': failures,
              'recorded_success': meta['success'], 'replay_success': success,
              'commanded_q_resets': resets, 'ticks': len(rows), 'commands': intervals,
              'source_physics_sha256': meta.get('physics_sha256'), 'physics_sha256': physics_hash(physics)}
    if failures:
        return report
    # Shift interval times if an uploaded trace starts at a nonzero timestamp.
    intervals = [[a-ticks[0]['t'], b-ticks[0]['t'], c] for a, b, c in intervals]
    relabeled = {**deepcopy(meta), 'task': task, 'task_name': task['name'], 'commands': intervals,
                 'physics_config': physics, 'physics_sha256': physics_hash(physics), 'start_q': start.tolist(),
                 'success': success, 'fail_reason': None if success else 'replay_goal_not_met',
                 'target_space': 'servo', 'replay': report,
                 'labeling': {'provider': 'r2p.labeling.segment', 'mode': 'deterministic_game_phase_events',
                              'model_calls': 0, 'argon_refined': False}}
    write_trace(destination, relabeled, rows)
    report['relabeled'] = str(destination)
    return report


def ingest(directory, output, task, physics):
    output = Path(output)
    reports, seen = [], set()
    files = sorted(Path(directory).rglob('*.jsonl'))
    for path in files:
        sha = file_digest(path)
        row = {'source': str(path), 'source_sha256': sha, 'accepted': False}
        try:
            if sha in seen:
                raise ValueError('Duplicate trace content')
            seen.add(sha)
            row = replay_trace(path, output/f'{sha}.jsonl', task, physics)
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            row['reason'] = str(exc)
        reports.append(row)
    accepted = [r for r in reports if r['accepted']]
    counts = {'ingested': len(files), 'rejected': len(files)-len(accepted), 'relabeled': len(accepted),
              'replay_successes': sum(r['replay_success'] for r in accepted)}
    return {'counts': counts, 'traces': reports, 'physics_sha256': physics_hash(physics)}


def run_loop(from_game, task='place_on_cell', *, root=Path.cwd(), output=None, physics='so101_bench',
             collect_headless=0, experts=4, steps=1000, eval_n=12, seed=2026, max_ticks=None, validation_game=None):
    from r2p.policy.train import train
    from r2p.policy.data import Augmentation
    from .collection import collect
    from .evaluation import evaluate_pair, make_plan, rollout
    from .game_commands import GameCommands
    from .gate import compare

    if min(experts, steps, eval_n) < 1 or collect_headless < 0 or (max_ticks is not None and max_ticks < 15):
        raise ValueError('Positive expert/step/eval counts and >=15 evaluation ticks required')
    root = Path(root).resolve()
    task, physics = load_task(task), load_physics(physics)
    source = Path(from_game).resolve()
    run = Path(output).resolve() if output else root/'runs'/f'{task["name"]}-game-{uuid.uuid4().hex[:10]}'
    if run.exists() and any(run.iterdir()):
        raise ValueError(f'Use a fresh output directory; evidence already exists: {run}')
    if run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError('Input traces and output run must not overlap')
    run.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'r2p_game_loop/1', 'status': 'RUNNING', 'task': task['name'], 'run': str(run),
                'started': utcnow(), 'evidence': 'simulation', 'model_calls': 0,
                'task_sha256': digest(task), 'physics_sha256': physics_hash(physics),
                'options': {'experts': experts, 'steps': steps, 'eval_n': eval_n, 'seed': seed, 'max_ticks': max_ticks}}
    from r2p.frames import ROOT
    code_paths = ['r2p/pipeline/game_loop.py', 'r2p/pipeline/game_player.py', 'r2p/pipeline/game_commands.py', 'r2p/pipeline/evaluation.py',
                  'r2p/policy/data.py', 'r2p/policy/train.py', 'r2p/policy/eval.py', 'r2p/policy/runtime.py',
                  'r2p/sim/twin.py', 'r2p/sim/expert.py', 'r2p/sim/ik.py', 'r2p/labeling.py']
    manifest['code_sha256'] = {name: file_digest(ROOT/name) for name in code_paths}
    def save(stage):
        manifest['stage'] = stage
        atomic_json(run/'game_loop.json', manifest)
        print(f'GAME LOOP {stage}', flush=True)
    with run_lock(root/'runs'/'game-loop.lock'):
        try:
            atomic_json(run/'task.json', task)
            atomic_json(run/'physics.json', physics)
            save('ingest')
            if collect_headless:
                collect_players(task, source, n=collect_headless, seed=seed+1000, physics=physics)
            if not source.is_dir():
                raise ValueError(f'Game trace directory does not exist: {source}')
            replay = ingest(source, run/'replayed', task, physics)
            atomic_json(run/'replay.json', replay)
            manifest['counts'] = replay['counts']
            save('replay_label')
            accepted = [r for r in replay['traces'] if r['accepted']]
            # Keep the snapshot of every ingested input, including blocked runs.
            for row in replay['traces']:
                target = run/'raw'/f'{row["source_sha256"]}.jsonl'
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(row['source'], target)
            if not accepted:
                raise ValueError('No physically consistent gameplay survived replay; see replay.json')
            source_counts = Counter()
            for row in accepted:
                meta, _ = read_trace(row['relabeled'])
                source_counts['headless' if meta.get('headless') else meta['source']] += 1
            manifest['sources'] = dict(source_counts)
            manifest['labeling'] = 'Deterministic game phase events via r2p.labeling; no live Argon calls'
            save('experts')
            collect(task, experts, run/'expert', workers=1, physics=physics, seed=seed)
            dataset = run/'mixed'
            dataset.mkdir()
            for path in (run/'expert').glob('*.jsonl'):
                shutil.copyfile(path, dataset/path.name)
            save('train_before')
            # Equal architecture, updates, seeds and augmentation: only data differs.
            kwargs = dict(steps=steps, device='cpu', seeds=(seed,), width=128, blocks=2, batch_size=128,
                          eval_n=1, eval_max_ticks=max_ticks, threads=2,
                          augmentation=Augmentation(smooth_ticks=1, joint_offset_std=np.deg2rad(.25),
                                                    position_std=.0005, stale_probability=.1,
                                                    aim_radial_m=0., aim_lateral_m=0.))
            before = train(task, dataset, 'game_before', out_dir=run/'before', **kwargs)
            for path in (run/'replayed').glob('*.jsonl'):
                shutil.copyfile(path, dataset/path.name)
            save('train_after')
            after = train(task, dataset, 'game_after', out_dir=run/'after', **kwargs)
            save('evaluate')
            plan = make_plan(task, sample_scene(task, np.random.default_rng(seed+100000)), eval_n, seed+200000)
            plan['physics'] = physics
            plan['command_protocol'] = 'game_state_skeleton_v1_no_model_requests'
            plan['training_scene_hashes'] = sorted({digest(read_trace(p)[0]['scene']) for p in dataset.glob('*.jsonl')})
            if any(digest(e['scene']) in plan['training_scene_hashes'] for e in plan['episodes']):
                raise ValueError('Evaluation scene overlaps training; choose a different seed')
            if max_ticks is not None:
                plan['max_ticks'] = max_ticks
            atomic_json(run/'eval_plan.json', plan)
            def game_rollout(*args):
                return rollout(*args, tracker_factory=GameCommands)
            challenger, incumbent = evaluate_pair(after, before, task, plan, evaluator=game_rollout)
            atomic_json(run/'after_eval.json', challenger)
            atomic_json(run/'before_eval.json', incumbent)
            for directory, report in ((after, challenger), (before, incumbent)):
                config = read_json(directory/'config.json')
                config['game_loop_eval'] = {**report, 'n': eval_n, 'command_protocol': plan['command_protocol']}
                atomic_json(directory/'config.json', config)
            if validation_game is not None:
                from .game_diagnostics import imitation_error
                validation = ingest(Path(validation_game), run/'validation_replayed', task, physics)
                atomic_json(run/'validation_replay.json', validation)
                if not validation['counts']['relabeled']:
                    raise ValueError('No held-out validation gameplay survived replay')
                diagnostics = {name: imitation_error(run/name, run/'validation_replayed') for name in ('before', 'after')}
                atomic_json(run/'imitation_eval.json', diagnostics)
                manifest['imitation_eval'] = diagnostics
            gate = compare(challenger, incumbent, plan)
            atomic_json(run/'gate.json', gate)
            manifest['gate'] = gate
            manifest['evaluation'] = {'n': eval_n, 'noise': plan['noise'], 'evidence': 'simulation',
                                      'plan_hash': digest(plan), 'truncated': max_ticks is not None}
            points = []
            for label, count, report in [('before', experts, incumbent), ('after', experts+len(accepted), challenger)]:
                scores = {noise: sum(r['success'] for r in rows)/len(rows) for noise, rows in report['results'].items()}
                points.append({'label': label, 'episodes': count, **scores, 'eval_n': eval_n,
                               'game_episodes': 0 if label == 'before' else len(accepted), 'expert_episodes': experts})
            manifest['curve'] = points
            manifest['code_sha256_at_finish'] = {name: file_digest(ROOT/name) for name in code_paths}
            manifest['code_changed_during_run'] = [name for name in code_paths
                                                  if manifest['code_sha256'][name] != manifest['code_sha256_at_finish'][name]]
            manifest['artifacts'] = {str(p.relative_to(run)): file_digest(p) for p in run.rglob('*')
                                     if p.is_file() and p.name != 'game_loop.json'}
            manifest['status'] = 'DONE'
            manifest['finished'] = utcnow()
            save('gate')
            # One atomic pointer prevents partial runs becoming dashboard evidence.
            publication = {**manifest, 'manifest': str(run/'game_loop.json'), 'input_dir': str(source),
                           'selected_eval': str(run/('after_eval.json' if gate['accepted'] else 'before_eval.json'))}
            atomic_json(root/'runs'/'game_loop_latest.json', publication)
            print(json.dumps({'run': str(run), 'counts': manifest['counts'], 'sources': manifest['sources'],
                              'curve': points, 'gate': gate['decision']}, indent=2), flush=True)
            return manifest
        except Exception as exc:
            manifest.update(status='BLOCKED', error=str(exc), finished=utcnow())
            atomic_json(run/'game_loop.json', manifest)
            raise
