import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.run([sys.executable,"-m","r2p.tasks",*args],cwd=ROOT,text=True,capture_output=True)


def test_cli_validates_windows_literal_glob():
    result = run("validate","tasks/*/task.json")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("ok ") == 6


def test_cli_sample_and_check(tmp_path):
    scene_path=tmp_path/"scene.json"
    result=run("sample","cube_in_bowl","--seed","3","--out",str(scene_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(scene_path.read_text())["task"] == "cube_in_bowl"
    state={"objects":{"cube":{"xyz":[.23,0,.024]},"bowl":{"xyz":[.23,0,.015]}},"gripper_open":True}
    state_path=tmp_path/"state.json"
    state_path.write_text(json.dumps(state))
    result=run("check","cube_in_bowl",str(state_path))
    assert result.returncode == 0 and json.loads(result.stdout)["ok"]
    state["gripper_open"]=False
    state_path.write_text(json.dumps(state))
    result=run("check","cube_in_bowl",str(state_path))
    assert result.returncode == 1 and not json.loads(result.stdout)["ok"]


def test_cli_check_window_and_bad_task(tmp_path):
    state={"objects":{"cube":{"xyz":[.23,0,.024]},"bowl":{"xyz":[.23,0,.015]}},"gripper_open":True}
    path=tmp_path/"states.json"
    path.write_text(json.dumps([state]*15))
    result=run("check","cube_in_bowl",str(path))
    assert result.returncode == 0 and json.loads(result.stdout)["required_ticks"] == 15
    result=run("validate","no_such_task")
    assert result.returncode == 1 and "ERROR" in result.stdout
