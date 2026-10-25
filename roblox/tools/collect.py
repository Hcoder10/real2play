"""Play task skeletons headlessly in Studio Run mode and collect canonical bot traces."""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import threading
from trace_server import make_server
from r2p.traces import read_trace

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks",nargs="+")
    parser.add_argument("--rounds",type=int,default=1)
    parser.add_argument("--seed",type=int,default=100)
    parser.add_argument("--limit",type=int,default=50)
    parser.add_argument("--out",type=Path,default=ROOT/"roblox/out/bots")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    key=secrets.token_urlsafe(32); path=args.out.resolve()/"args.json"
    with make_server("127.0.0.1",0,args.out/"traces",key) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        values={"rounds":args.rounds,"seed":args.seed,"limit":args.limit,"trace_url":f"http://localhost:{server.server_port}","trace_key":key}
        if args.tasks:values["tasks"]=args.tasks
        path.write_text(json.dumps(values),encoding="utf-8")
        timeout=90*(len(args.tasks or [0]*6))*args.rounds+90
        try:
            result=subprocess.run(["node",str(ROOT/"roblox/tools/runmode.mjs"),str(ROOT/"roblox/tools/collect.luau"),"@"+str(path),str(timeout),str(args.out.resolve()/"result.json")],cwd=ROOT,capture_output=True,text=True,timeout=timeout+30)
            if result.returncode:print(result.stderr or result.stdout);return 1
            data=json.loads((args.out/"result.json").read_text(encoding="utf-8"))
            complete=True
            for item in data["value"]["results"]:
                item["trace_received"]=(args.out/"traces"/(item["trace_id"]+".jsonl")).exists()
                complete=complete and item["trace_received"]
                if item["trace_received"]:
                    meta,ticks=read_trace(args.out/"traces"/(item["trace_id"]+".jsonl"))
                    assert meta["source"]=="bot" and len(ticks)==item["ticks"] and meta["success"]==item["success"]
                    item["physics_sha256"]=meta["physics_sha256"]
                print(json.dumps({k:v for k,v in item.items() if k not in ("phases","final")}))
            (args.out/"result.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
            return 0 if complete else 1
        finally:path.unlink(missing_ok=True);server.shutdown();thread.join()

if __name__=="__main__":raise SystemExit(main())
