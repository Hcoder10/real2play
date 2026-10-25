"""Run a real Studio Play test and verify its authenticated traces on disk."""
from __future__ import annotations
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from trace_server import make_server
from r2p.traces import read_trace

def main():
    out=ROOT/"roblox/out/smoke"; out.mkdir(parents=True,exist_ok=True)
    key=secrets.token_urlsafe(32)
    args=out/"args.json"; result=out/"result.json"
    with make_server("127.0.0.1",0,out/"traces",key) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        args.write_text(json.dumps({"trace_url":f"http://localhost:{server.server_port}","trace_key":key}),encoding="utf-8")
        try:
            env=os.environ.copy();env["PLAY"]="1"
            proc=subprocess.run(["node",str(ROOT/"roblox/tools/runmode.mjs"),str(ROOT/"tests/roblox/play_smoke.luau"),"@"+str(args),"180",str(result)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=210)
            if proc.returncode:
                print(proc.stderr or proc.stdout);return 1
            data=json.loads(result.read_text(encoding="utf-8"))
            ids=data["value"]["trace_ids"]
            for trace_id in ids:
                meta,ticks=read_trace(out/"traces"/(trace_id+".jsonl"))
                assert meta["test_only"] and meta["source"]=="bot" and meta["success"] and len(ticks)>=15
            data["value"]["traces_received"]=len(ids)
            result.write_text(json.dumps(data,indent=2),encoding="utf-8")
            print(json.dumps(data,indent=2));return 0
        finally:
            args.unlink(missing_ok=True);server.shutdown();thread.join()

if __name__=="__main__":raise SystemExit(main())
