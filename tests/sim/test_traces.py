import numpy as np
import pytest

from r2p.sim import gen_traces
from r2p.traces import read_trace, validate_trace


def test_trace_format_commands_dart_and_determinism(tmp_path):
    paths = gen_traces("place_on_cell", 1, tmp_path/"a", workers=1, seed=10001, dart=.004)
    meta, ticks = read_trace(paths[0])
    assert not validate_trace(meta, ticks)
    assert meta["success"]
    assert [segment[2] for segment in meta["commands"]] == ["pick(piece)", "place_at(piece,cell)", "release", "home"]
    assert any(not np.array_equal(t["q_target"], t["q_exec"]) for t in ticks)
    for t in ticks:
        assert len(t["q_commanded"]) == 6
        assert t["q_target"][5] == t["q_exec"][5]
    again = gen_traces("place_on_cell", 1, tmp_path/"b", workers=1, seed=10001, dart=.004)
    assert paths[0].read_bytes() == again[0].read_bytes()
    with pytest.raises(ValueError, match="workers"):
        gen_traces("place_on_cell", 1, tmp_path, workers=5)


def test_two_worker_collection(tmp_path):
    paths = gen_traces("cube_in_bowl", 2, tmp_path, workers=2, seed=10001, dart=0)
    assert len(paths) == 2
    assert [read_trace(path)[0]["seed"] for path in paths] == [10001, 10002]
