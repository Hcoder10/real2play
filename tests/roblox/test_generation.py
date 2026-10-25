import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"roblox/tools"))
from gen_presets import lua, apply_bench, workshop, workshop_physics
import json

def test_lua_preserves_unicode_and_json_control_characters():
    assert "机器人" in lua({"name":"机器人 🦾"})
    assert "\\u" not in lua("机器人")
    assert lua([True,False,None])=="{true,false,nil}"
    assert lua("line\nnext")=='"line\\nnext"'


def test_measured_bench_task_and_physics():
    game=workshop()
    bench=json.loads((ROOT/game["bench_preset"]).read_text())
    canonical=json.loads((ROOT/"tasks/place_on_cell/task.json").read_text())
    original=json.loads(json.dumps(canonical))
    task=apply_bench(canonical,bench)
    assert task["board"]==bench["board"]
    assert task["objects"][0]["size_m"]==[.033,.017]
    assert task["randomize"]["piece"]["xy_choice"]==bench["supply_xy"]
    assert "board_pose" not in task["randomize"]
    assert task["bench_calibration"]["xy_offset_mm"]==[10,11]
    assert task["bench_calibration"]["z_offset_mm"]==-1.5
    assert canonical==original
    physics=workshop_physics()
    assert physics["joints"]["shoulder_lift"]["kp"]==20
    assert physics["object"]=={"mass_kg":.003,"friction":.6}
