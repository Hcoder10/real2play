"""Held-out replay action error, distinct from closed-loop task success."""
from pathlib import Path

import numpy as np

from r2p.policy import load
from r2p.policy.data import Augmentation, TraceDataset
from r2p.traces import read_trace
from .storage import digest


def imitation_error(export, traces):
    policy = load(export)
    config = policy.config
    training = config['training']['traces']
    training_scenes = {digest(read_trace(row['path'])[0]['scene']) for row in training}
    reports = []
    for path in sorted(Path(traces).glob('*.jsonl')):
        meta, rows = read_trace(path)
        if digest(meta['scene']) in training_scenes:
            raise ValueError('Imitation diagnostic requires held-out scenes')
        if not meta.get('replay', {}).get('accepted'):
            raise ValueError('Diagnostic observations must have passed current twin replay')
        if meta['physics_sha256'] != config['physics_sha256']:
            raise ValueError('Diagnostic and training physics differ')
    dataset = TraceDataset(config['task_spec'], traces, config, Augmentation(smooth_ticks=1))
    predictions = np.concatenate([policy.predict_features(dataset.features[i:i+256])
                                  for i in range(0, len(dataset.features), 256)])
    errors = np.abs(predictions-dataset.labels.reshape(-1, 15, 6))
    offset = 0
    for source in dataset.provenance:
        end = offset+source['ticks']
        reports.append({'path': source['path'], 'seed': source['seed'], 'ticks': source['ticks'],
                        'chunk_mae_rad': float(errors[offset:end].mean()),
                        'first_action_mae_rad': float(errors[offset:end, 0].mean())})
        offset = end
    return {'schema': 'r2p_game_imitation_eval/1', 'policy': str(export), 'n_episodes': len(reports),
            'ticks': len(errors), 'chunk_mae_rad': float(errors.mean()),
            'first_action_mae_rad': float(errors[:, 0].mean()), 'episodes': reports,
            'meaning': 'Teacher-forced held-out twin replay predictions; NOT a task-success rate'}
