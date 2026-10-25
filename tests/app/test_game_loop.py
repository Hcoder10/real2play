"""Physical replay and evidence gates; no hardware, accelerator or model calls."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from r2p.frames import HOME_Q
from r2p.pipeline.game_loop import ingest, replay_trace, run_loop
from r2p.pipeline.game_player import GamePlayer, physics_hash
from r2p.sim import TaskTwin
from r2p.sim.twin import load_physics
from r2p.tasks import commands, load_task, sample_scene
from r2p.traces import read_trace, write_trace


@pytest.fixture
def trace(tmp_path):
    task = load_task('place_on_cell')
    physics = load_physics()
    scene = sample_scene(task, np.random.default_rng(101))
    rows = []
    vocab = commands(task)
    with TaskTwin(scene, physics=physics, q0=HOME_Q) as twin:
        for i in range(30):
            target = HOME_Q.copy()
            target[0] += .15*min(1, i/8)
            rows.append({'t': i/30, 'q': twin.q().tolist(), 'q_target': target.tolist(),
                         'poses': twin.poses(), 'grip_closed': False,
                         'command': vocab[0] if i < 17 else 'release', 'phase': 'test'})
            twin.step(target)
    meta = {'task': task, 'source': 'bot', 'player': 'test', 'scene': scene, 'success': False,
            'headless': True, 'physics_sha256': physics_hash(physics), 'start_q': HOME_Q.tolist()}
    path = tmp_path/'input'/'trace.jsonl'
    write_trace(path, meta, rows)
    return path, task, physics


def test_replay_uses_current_physics_targets_commanded_q_and_segments(trace, tmp_path, monkeypatch):
    import r2p.gemini
    monkeypatch.setattr(r2p.gemini, 'call', lambda *a, **k: pytest.fail('Model request forbidden'))
    source, task, physics = trace
    destination = tmp_path/'replay'/'trace.jsonl'
    report = replay_trace(source, destination, task, physics)
    assert report['accepted'] and report['metrics']['object_max_mm'] == 0
    assert report['physics_sha256'] == physics_hash(physics)
    meta, rows = read_trace(destination)
    _, original = read_trace(source)
    previous = HOME_Q.copy()
    for row, raw in zip(rows, original):
        expected = np.array(row['q']) if np.max(abs(previous-row['q'])) > np.deg2rad(4) else previous
        np.testing.assert_allclose(row['q_commanded'], expected)
        np.testing.assert_array_equal(row['q_target'], raw['q_target'])
        previous = np.array(row['q_target'])
    assert meta['commands'] == [[0, 17/30, commands(task)[0]], [17/30, 1., 'release']]
    assert meta['labeling']['model_calls'] == 0
    assert meta['replay']['source_sha256']


@pytest.mark.parametrize('tamper', ['teleport', 'joint', 'missing', 'out_of_bounds', 'fixture', 'wrong_task', 'size', 'tilt'])
def test_rejects_impossible_or_ineligible_gameplay(trace, tmp_path, tamper):
    source, task, physics = trace
    meta, rows = read_trace(source)
    if tamper == 'teleport':
        rows[12]['poses']['piece']['xyz'][0] += .2
    elif tamper == 'joint':
        rows[12]['q'][0] += .3
    elif tamper == 'missing':
        del rows[12]['poses']['piece']
    elif tamper == 'out_of_bounds':
        rows[12]['q_target'][0] = 99
    elif tamper == 'fixture':
        meta['test_only'] = True
    elif tamper == 'wrong_task':
        meta['task']['instruction'] += ' changed revision'
    elif tamper == 'size':
        rows[12]['poses']['piece']['size_m'][0] *= 2
    else:
        rows[12]['poses']['piece']['quat'] = [.70710678, .70710678, 0., 0.]
    write_trace(source, meta, rows)
    result = ingest(source.parent, tmp_path/'replay', task, physics)
    assert result['counts'] == {'ingested': 1, 'rejected': 1, 'relabeled': 0, 'replay_successes': 0}
    assert not list((tmp_path/'replay').glob('*.jsonl'))


def test_ingest_isolates_bad_files_and_deduplicates(trace, tmp_path):
    source, task, physics = trace
    (source.parent/'bad.jsonl').write_text('{invalid', encoding='utf-8')
    (source.parent/'copy.jsonl').write_bytes(source.read_bytes())
    result = ingest(source.parent, tmp_path/'replay', task, physics)
    assert result['counts']['ingested'] == 3
    assert result['counts']['rejected'] == 2
    assert result['counts']['relabeled'] == 1


def test_chunks_are_absolute_and_stop_at_command_boundary(trace, tmp_path):
    from r2p.policy.data import Augmentation, TraceDataset
    from r2p.policy.features import export_config
    source, task, physics = trace
    destination = tmp_path/'replay'/'trace.jsonl'
    replay_trace(source, destination, task, physics)
    dataset = TraceDataset(task, destination.parent, export_config(task, 'test'), Augmentation(smooth_ticks=1))
    _, rows = read_trace(destination)
    chunk = dataset.labels[16].reshape(15, 6)
    np.testing.assert_allclose(chunk, np.tile(rows[16]['q_target'], (15, 1)), atol=1e-6)
    np.testing.assert_allclose(dataset.raw[16, :6], rows[16]['q_commanded'], atol=1e-6)


def test_nonzero_timestamps_rebased(trace, tmp_path):
    source, task, physics = trace
    meta, rows = read_trace(source)
    for row in rows:
        row['t'] += 10
    write_trace(source, meta, rows)
    destination = tmp_path/'replay'/'trace.jsonl'
    replay_trace(source, destination, task, physics)
    meta, rows = read_trace(destination)
    assert meta['commands'][0][0] == 0 and rows[0]['t'] == 0
    assert meta['commands'][-1][1] == pytest.approx(1)


def test_changed_physics_is_actually_used(trace, tmp_path):
    source, task, physics = trace
    changed = deepcopy(physics)
    for row in changed['joints'].values():
        row['kp'] *= .01
    result = replay_trace(source, tmp_path/'replay.jsonl', task, changed)
    assert not result['accepted']
    assert result['source_physics_sha256'] != result['physics_sha256']
    assert result['metrics']['joint_max_deg'] > 4


def test_player_port_matches_actual_luau(tmp_path):
    from r2p.pipeline.game_parity import verify
    from roblox.tools.test import runner
    try:
        runner()
    except SystemExit:
        pytest.skip('Pure Luau executable unavailable')
    task, physics = load_task('place_on_cell'), load_physics()
    scene = sample_scene(task, np.random.default_rng(500))
    player, rows = GamePlayer(task, physics), []
    with TaskTwin(scene, physics=physics) as twin:
        for i in range(30):
            state = {'q': twin.q().tolist(), 'objects': twin.poses(), 'site_xyz': player.arm.tips(twin.q())[2].tolist()}
            target, command, phase = player.step(state)
            rows.append({'t': i/30, 'q': state['q'], 'poses': state['objects'], 'q_target': target.tolist(),
                         'command': command, 'phase': phase, 'grip_closed': bool(target[5] < .15)})
            twin.step(target)
    path = tmp_path/'player.jsonl'
    write_trace(path, {'task': task, 'source': 'bot', 'scene': scene, 'headless': True,
                       'physics_config': physics, 'start_q': HOME_Q.tolist(), 'success': False}, rows)
    assert verify(path)['pass']


def test_blocked_run_never_publishes(trace, tmp_path):
    source, task, _ = trace
    meta, rows = read_trace(source)
    meta['test_only'] = True
    write_trace(source, meta, rows)
    with pytest.raises(ValueError, match='No physically consistent'):
        run_loop(source.parent, task, root=tmp_path, output=tmp_path/'run', experts=1, steps=1, eval_n=1)
    assert not (tmp_path/'runs/game_loop_latest.json').exists()
    manifest = json.loads((tmp_path/'run/game_loop.json').read_text())
    assert manifest['status'] == 'BLOCKED'


def test_cli_loop_options():
    from r2p.cli import parser
    args = parser().parse_args(['loop', '--from-game', 'traces', '--task', 'place_on_cell'])
    assert args.eval_n == 12 and args.steps == 1000 and args.collect_headless == 0


def test_diagnostic_refuses_training_scene_reuse(trace, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from r2p.pipeline import game_diagnostics
    source, task, physics = trace
    destination = tmp_path/'replay'/'trace.jsonl'
    replay_trace(source, destination, task, physics)
    monkeypatch.setattr(game_diagnostics, 'load', lambda _: SimpleNamespace(config={
        'training': {'traces': [{'path': str(source)}]}, 'physics_sha256': physics_hash(physics)}))
    with pytest.raises(ValueError, match='held-out scenes'):
        game_diagnostics.imitation_error('unused', destination.parent)


def test_release_command_requires_lowered_placement_but_not_already_open_jaws():
    from r2p.pipeline.game_commands import GameCommands
    task = load_task('place_on_cell')
    poses = {o['id']: deepcopy(o) for o in sample_scene(task, np.random.default_rng(500))['objects']}
    tracker = GameCommands(task, deepcopy(poses))
    tracker.index = 1
    poses['piece']['xyz'][:2] = poses['cell']['xyz'][:2]
    poses['piece']['xyz'][2] = .1
    state = {'objects': poses, 'gripper_open': False}
    for _ in range(6):
        tracker.update(state, HOME_Q, HOME_Q)
    assert tracker.command == 'place_at(piece,cell)'
    poses['piece']['xyz'][2] = .015
    for _ in range(4):
        tracker.update(state, HOME_Q, HOME_Q)
    assert tracker.command == 'release'
    for _ in range(4):
        tracker.update(state, HOME_Q, HOME_Q)
    assert tracker.command == 'release'


def test_dashboard_publishes_replay_curve_and_honest_sources(trace, tmp_path):
    from dashboard.adapters import DashboardData
    from r2p.pipeline.storage import atomic_json
    source, _, _ = trace
    report = tmp_path/'eval.json'
    atomic_json(report, {'policy': 'before', 'results': {'clean': [{'success': False}], 'real': [{'success': False}]}})
    counts = {'ingested': 3, 'rejected': 2, 'relabeled': 1}
    atomic_json(tmp_path/'runs/game_loop_latest.json', {'status': 'DONE', 'input_dir': str(source.parent),
        'selected_eval': str(report), 'sources': {'headless': 1}, 'counts': counts, 'gate': {'decision': 'HOLD'},
        'curve': [{'episodes': 4, 'clean': 0., 'real': 0.}, {'episodes': 5, 'clean': 0., 'real': 0.}]})
    adapter = DashboardData(tmp_path, {'traces': []})
    assert adapter.game()['groups']['human']['rounds'] == 0
    assert adapter.game()['headless']['rounds'] == 1
    assert adapter.game()['replay'] == counts
    policy = adapter.policy()
    assert policy['n'] == 1 and len(policy['scaling']) == 2
    assert 'headless simulated' in policy['message'] and 'HOLD' in policy['message']
