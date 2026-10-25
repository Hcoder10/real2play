"""Deterministic expert collection, bounded to four CPU workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from r2p.tasks import load_task
from r2p.traces import write_trace
from .expert import Expert


def _episode(args):
    task, seed, out_dir, dart = args
    result = Expert(task, dart=dart, seed=seed).run(seed=seed)
    path = Path(out_dir) / f"expert_{task['name']}_{seed:08d}.jsonl"
    write_trace(path, result["meta"], result["ticks"])
    return str(path)


def gen_traces(task, n, out_dir, workers=1, *, seed=0, dart=0.01):
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer between 1 and 4")
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    task = load_task(task) if not isinstance(task, dict) else task
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(task, seed+i, str(out_dir), dart) for i in range(n)]
    if workers == 1:
        return [Path(_episode(job)) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [Path(p) for p in pool.map(_episode, jobs)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dart", type=float, default=.01)
    a = p.parse_args()
    paths = gen_traces(a.task, a.n, a.out_dir, a.workers, seed=a.seed, dart=a.dart)
    print(f"Wrote {len(paths)} traces to {a.out_dir}")


if __name__ == "__main__":
    main()
