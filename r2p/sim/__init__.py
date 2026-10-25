"""Physical reference twin and expert trace collection (all positions in the spec frame)."""
from .twin import TaskTwin


def gen_traces(task, n, out_dir, workers=1, **kwargs):
    from .traces import gen_traces as generate
    return generate(task, n, out_dir, workers=workers, **kwargs)


__all__ = ["TaskTwin", "gen_traces"]
