"""Measure the same absolute commands in r2p.sim and Luau/Roblox Run mode.

Default: pure JointSim numerical gate. --studio includes actual HingeConstraint
angles, jaw positions and object end poses; a failed gate exits nonzero.
No real robot, Gemini requests or GPU are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from r2p.frames import HOME_Q
from r2p.model import Arm
from r2p.sim import TaskTwin
from gen_presets import lua, workshop_physics
from test import runner


def compare(reference, candidate, *, joint_deg=1.0, jaw_mm=2.0, object_mm=5.0, object_deg=5.0):
    if len(reference["q"]) != len(candidate["q"]) or not reference["q"]:
        return {"pass":False,"error":"tick count mismatch or empty replay"}
    q=np.asarray(reference["q"]); actual=np.asarray(candidate["q"])
    jaw=np.asarray(reference["jaw"]); actual_jaw=np.asarray(candidate["jaw"])
    if q.shape!=actual.shape or jaw.shape!=actual_jaw.shape or not np.isfinite(actual).all() or not np.isfinite(actual_jaw).all():
        return {"pass":False,"error":"invalid replay shape or non-finite values"}
    joints=float(np.rad2deg(np.sqrt(np.mean((q-actual)**2))))
    ee=float(np.sqrt(np.mean(np.sum((jaw-actual_jaw)**2,axis=1)))*1000)
    errors={}; rotations={}
    for name,pose in reference.get("poses",{}).items():
        if name not in candidate.get("poses",{}):
            return {"pass":False,"error":f"missing object {name}"}
        other=candidate["poses"][name]
        position=np.asarray(other.get("xyz",[]),dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all():
            return {"pass":False,"error":f"invalid object position {name}"}
        errors[name]=float(np.linalg.norm(np.asarray(pose["xyz"])-position)*1000)
        if "quat" in pose:
            a=np.asarray(pose["quat"],dtype=float); b=np.asarray(other.get("quat",[]),dtype=float)
            if b.shape!=(4,) or not np.isfinite(b).all() or np.linalg.norm(b)<1e-9:
                return {"pass":False,"error":f"invalid object orientation {name}"}
            dot=abs(float(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b))))
            rotations[name]=float(np.rad2deg(2*np.arccos(np.clip(dot,0,1))))
        elif "yaw_deg" in pose:
            yaw=other.get("yaw_deg",float("nan"))
            if not np.isfinite(yaw):return {"pass":False,"error":f"missing object yaw {name}"}
            rotations[name]=abs(float((pose["yaw_deg"]-yaw+180)%360-180))
    objects=max(errors.values(),default=0)
    return {"pass":joints<=joint_deg and ee<=jaw_mm and objects<=object_mm and max(rotations.values(),default=0)<=object_deg,
            "joint_rms_deg":joints,"jaw_rms_mm":ee,"object_end_mm":errors,"object_end_rotation_deg":rotations,
            "limits":{"joint_rms_deg":joint_deg,"jaw_rms_mm":jaw_mm,"object_end_mm":object_mm,"object_end_rotation_deg":object_deg},"ticks":len(q)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio",action="store_true")
    parser.add_argument("--input",type=Path,help="JSON {scene,q0,commands}; default is a smooth held-out excitation")
    parser.add_argument("--out",type=Path,default=ROOT/"roblox/out/parity.json")
    parser.add_argument("--ticks",type=int,default=180)
    args=parser.parse_args()
    if args.input:
        payload=json.loads(args.input.read_text(encoding="utf-8-sig"))
    else:
        commands=[]
        for i in range(args.ticks):
            t=i/30; q=HOME_Q.copy()
            ramp=min(1,max(0,(t-1)/2))
            q[:5]+=ramp*np.array([.12,.08,.08,-.10,.12])*np.sin(.7*t+np.arange(5)*.4)
            commands.append(q.tolist())
        objects=[dict(id="cube",role="manipulated",shape="box",size_m=[.03]*3,xyz=[.15,.20,.04],yaw_deg=15,mass_kg=.03)] if args.studio else []
        payload={"scene":{"objects":objects,"table":{"size_m":[.8,.6],"center_xy":[.2,0]}},"q0":HOME_Q.tolist(),"commands":commands}
    arm=Arm(); reference={"q":[],"jaw":[]}
    with TaskTwin(payload["scene"],q0=payload["q0"],physics=workshop_physics()) as twin:
        for command in payload["commands"]:
            q=twin.step(command); reference["q"].append(q.tolist()); reference["jaw"].append(arm.tips(q)[2].tolist())
        reference["poses"]=twin.poses()
        payload["physics"]=twin.physics
    out=args.out.resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    args_path=out.with_name(out.stem+"_input.json"); args_path.write_text(json.dumps(payload),encoding="utf-8")
    if args.studio:
        result_path=out.with_name(out.stem+"_studio.json")
        proc=subprocess.run(["node",str(ROOT/"roblox/tools/runmode.mjs"),str(ROOT/"roblox/tools/parity_studio.luau"),"@"+str(args_path),"180",str(result_path)],cwd=ROOT,capture_output=True,text=True)
        if proc.returncode:
            raise SystemExit(proc.stderr or proc.stdout)
        result=json.loads(result_path.read_text())
        if not result.get("ok"):raise SystemExit(str(result))
        candidate=result["value"]
    else:
        inputs=ROOT/"roblox/out/parity_inputs.luau"; inputs.parent.mkdir(exist_ok=True)
        inputs.write_text("return "+lua(payload),encoding="utf-8")
        proc=subprocess.run([runner(),str(ROOT/"roblox/tools/parity_runner.luau")],cwd=ROOT,capture_output=True,text=True,check=True)
        candidate=json.loads(proc.stdout)
    report=compare(reference,candidate)
    report["engine"]="Roblox Run mode" if args.studio else "pure Luau JointSim"
    report["physics"]=payload["physics"]; report["input"]=str(args_path)
    report["physics_sha256"]=hashlib.sha256(json.dumps(payload["physics"],sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    out.with_name(out.stem+"_reference.json").write_text(json.dumps(reference),encoding="utf-8")
    out.with_name(out.stem+"_candidate.json").write_text(json.dumps(candidate),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="physics"},indent=2))
    return 0 if report["pass"] else 1

if __name__=="__main__":raise SystemExit(main())
