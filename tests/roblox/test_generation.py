import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"roblox/tools"))
from gen_presets import lua

def test_lua_preserves_unicode_and_json_control_characters():
    assert "机器人" in lua({"name":"机器人 🦾"})
    assert "\\u" not in lua("机器人")
    assert lua([True,False,None])=="{true,false,nil}"
    assert lua("line\nnext")=='"line\\nnext"'
