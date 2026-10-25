"""Collect B's expert traces with an explicitly identified physics version."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _episode(job):
    import numpy as np
    from r2p.sim import TaskTwin
    from r2p.sim.expert import Expert
    from r2p.tasks import sample_scene
    from r2p.traces import write_trace

    task, physics, seed, out = job
    scene = sample_scene(task, np.random.default_rng(seed))
    with TaskTwin(scene, physics=physics) as twin:
        result = Expert(task, seed=seed, dart=0.01).run(twin, seed=seed)
    result["meta"]["physics"] = physics
    path = Path(out) / f"expert_{seed:08d}.jsonl"
    write_trace(path, result["meta"], result["ticks"])
    return path


def collect(task, n, out_dir, workers, physics, seed=0):
    if not 1 <= workers <= 4:
        raise ValueError("At most four expert workers are permitted")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(task, physics, seed+i, str(out_dir)) for i in range(n)]
    if workers == 1:
        return [_episode(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_episode, jobs))
