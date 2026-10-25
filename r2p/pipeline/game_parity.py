"""Compare headless player targets to the actual pure-Luau SkillOperator."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from r2p.frames import ROOT
from r2p.model import Arm
from r2p.traces import read_trace
from .storage import atomic_json, file_digest


def verify(trace, out=None):
    from roblox.tools.gen_presets import lua
    from roblox.tools.test import runner

    meta, ticks = read_trace(trace)
    if not meta.get('headless'):
        raise ValueError('This parity check expects headless SkillOperator traces')
    arm = Arm()
    states = [{'objects': t['poses'], 'q': t['q'], 'site_xyz': arm.tips(t['q'])[2].tolist()} for t in ticks]
    folder = ROOT/'roblox'/'out'
    folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='l_player_parity_', dir=folder) as temporary:
        path = Path(temporary)/'check.luau'
        path.write_text(
            'local Physics=require("../../src/ReplicatedStorage/Twin/Generated/Physics")\n'
            f'Physics.config={lua(meta["physics_config"])}\n'
            'local S=require("../../src/ReplicatedStorage/Twin/SkillOperator")\n'
            f'local player=S.new({lua(meta["task"])},{lua(meta["start_q"])})\n'
            f'local states={lua(states)}\n'
            'for _,state in ipairs(states) do\n'
            ' local q,c,p=player:step(state,1/30)\n'
            ' local values={} for _,v in ipairs(q) do table.insert(values,string.format("%.17g",v)) end\n'
            ' print(table.concat(values,",").."|"..c.."|"..p)\nend\n', encoding='utf-8')
        completed = subprocess.run([runner(), str(path)], check=True, capture_output=True, text=True, timeout=120)
    lines = [line.split('|') for line in completed.stdout.splitlines()]
    if len(lines) != len(ticks):
        raise ValueError('Luau did not return one action per tick')
    q = np.asarray([[float(v) for v in row[0].split(',')] for row in lines])
    error = float(np.max(abs(q-np.asarray([t['q_target'] for t in ticks]))))
    mismatches = sum((r[1], r[2]) != (t['command'], t['phase']) for r, t in zip(lines, ticks))
    report = {'trace': str(trace), 'trace_sha256': file_digest(trace), 'ticks': len(ticks),
              'max_target_error_rad': error, 'phase_command_mismatches': mismatches,
              'pass': error < 1e-6 and mismatches == 0,
              'meaning': 'Same recorded observations through Luau SkillOperator and Python port; not engine parity'}
    if out:
        atomic_json(out, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = verify(args.trace, args.out)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['pass'] else 1)
