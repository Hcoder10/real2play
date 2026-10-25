import importlib.util
from pathlib import Path
import sys

TOOLS=Path(__file__).resolve().parents[2]/"roblox/tools"
sys.path.insert(0,str(TOOLS))
from parity import compare

def test_parity_fails_for_mismatch_missing_objects_and_truncated_runs():
    ref={"q":[[0]*6],"jaw":[[.2,0,.1]],"poses":{"cube":{"xyz":[.2,0,.01]}}}
    assert compare(ref,ref)["pass"]
    bad={**ref,"q":[[.2]*6]}
    assert not compare(ref,bad)["pass"]
    assert not compare(ref,{**ref,"jaw":[[.21,0,.1]]})["pass"]
    assert not compare(ref,{**ref,"poses":{}})["pass"]
    assert not compare(ref,{**ref,"q":[]})["pass"]


def test_parity_checks_orientation_and_quaternion_sign():
    ref={"q":[[0]*6],"jaw":[[.2,0,.1]],"poses":{"cube":{"xyz":[.2,0,.01],"quat":[1,0,0,0]}}}
    equivalent={**ref,"poses":{"cube":{"xyz":[.2,0,.01],"quat":[-1,0,0,0]}}}
    rotated={**ref,"poses":{"cube":{"xyz":[.2,0,.01],"quat":[0,0,0,1]}}}
    assert compare(ref,equivalent)["pass"]
    assert not compare(ref,rotated)["pass"]
    assert not compare(ref,{**ref,"poses":{"cube":{"xyz":[.2,0,.01]}}})["pass"]
