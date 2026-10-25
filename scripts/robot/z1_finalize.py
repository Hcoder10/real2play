"""Attach actual evaluation evidence and rehearsal qualification to a bowl export."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', required=True)
    p.add_argument('--rehearsal', required=True)
    a = p.parse_args()
    directory = Path(a.export)
    config = json.loads((directory/'config.json').read_text())
    digest = hashlib.sha256((directory/'weights.npz').read_bytes()).hexdigest()
    for noise in ('clean', 'real', 'realx0.5'):
        path = directory/f'eval_{noise}.json'
        if path.exists():
            report = json.loads(path.read_text())
            if report['weights_sha256'] != digest:
                raise ValueError(f'Stale evaluation: {path}')
            config['eval'][noise] = report
    config['eval']['n'] = min((config['eval'].get(n) or {}).get('n', 0)
                              for n in ('clean', 'real', 'realx0.5'))
    rehearsal = json.loads(Path(a.rehearsal).read_text())
    accepted = (rehearsal.get('success') is True and rehearsal.get('outcome') == 'goal_reached'
                and rehearsal.get('sent_owner_limit_hits') == 0)
    evaluated = config['eval']['n'] >= 30
    config['qualification'] = {
        'ready_for_real': bool(accepted and evaluated),
        'reason': ('Guarded rehearsal passed; review recorded noise rates before operation.' if accepted and evaluated
                   else f"Not qualified: rehearsal={rehearsal.get('outcome')}, success={rehearsal.get('success')}, full_noise_eval={evaluated}"),
        'report': a.rehearsal, 'weights_sha256': digest,
        'start_accepted': rehearsal.get('start', {}).get('reason') is None,
        'refusals': rehearsal.get('refusals'),
        'sent_owner_limit_hits': rehearsal.get('sent_owner_limit_hits')}
    (directory/'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config['qualification'], indent=2))


if __name__ == '__main__':
    main()
