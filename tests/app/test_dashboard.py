from contextlib import contextmanager
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

from dashboard.adapters import DashboardData
from dashboard.server import make_server, qr_png
from r2p.pipeline.storage import atomic_json


def config(**overrides):
    return {"real2sim_live":"out/live", "real2sim_run":"pinned", "headline_eval":"exports/pinned/config.json",
            "scaling":"runs/scaling.json", "traces":["roblox/out/traces"], "runs":"runs", "deploy_runs":"runs",
            "ttt_status":"runs/ttt/status.json", "camera_status_url":"http://127.0.0.1:8791/status.json",
            "camera_url":"http://127.0.0.1:8791/frame.jpg", "game_url":None, **overrides}


def trace(root, name, *, source="human", success=True, task="cube_in_bowl", player="Ada", level=None, times=(0,1,2)):
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    meta={"schema":"r2p_trace/1", "task":{"name":task}, "task_name":task, "source":source,
          "success":success, "player":player, "level":level or task}
    path.write_text("\n".join([json.dumps({"meta":meta}),*[json.dumps({"t":t}) for t in times]])+"\n",encoding="utf-8")
    return path


def data(root, status=None, **overrides):
    return DashboardData(root,config(),fetch_json=lambda _:status or {},**overrides)


def make_twin(root):
    base=root/"out/live/pinned"
    base.mkdir(parents=True)
    from PIL import Image
    Image.new("RGB",(1280,480),(20,30,40)).save(base/"overlay.jpg")
    Image.new("RGB",(640,480),(20,30,40)).save(base/"motion.jpg")
    iterations=[]
    for i, ious, kept, motion in [(0,[.4,.6],True,70),(1,[.8,.9],True,12),(2,[.2,.4],False,90)]:
        detail={"metrics":{"score":-3+i,"objects":{str(j):{"iou":v} for j,v in enumerate(ious)},
                           "physics":{"traj_err_px":motion}}, "images":{"overlay_0":"overlay.jpg","physics":"motion.jpg"}}
        atomic_json(base/f"iter_{i}.json",detail)
        iterations.append({"iter":i,"file":f"iter_{i}.json","kept":kept,"score":-3+i,"best_so_far":-1})
    atomic_json(base/"run.json",{"schema":"r2s_live/1","session_kind":"synthetic","status":"done",
                                 "best_iter":1,"iters":iterations})
    return base


def test_twin_gap_uses_object_iou_not_unbounded_score_and_pins_best(tmp_path):
    make_twin(tmp_path)
    result=data(tmp_path).twin()
    assert result["gap"] == pytest.approx(.15)
    assert result["motion_px"] == 12 and result["shown_iter"] == 1
    assert [r["best_gap"] for r in result["history"]] == pytest.approx([.5,.15,.15])
    assert result["synthetic"]
    assert [r["label"] for r in result["images"]] == ["SIM","REAL","MOTION"]
    assert [r["crop"] for r in result["images"]] == ["right","left","full"]


def test_twin_does_not_select_latest_unpinned_run(tmp_path):
    base=make_twin(tmp_path)
    atomic_json(base.parent/"newer"/"run.json",{"schema":"r2s_live/1","iters":[]})
    assert data(tmp_path).twin()["run"] == "pinned"
    adapter=DashboardData(tmp_path,config(real2sim_run=None))
    assert adapter.twin()["available"] is False


def test_twin_missing_metrics_are_unknown(tmp_path):
    base=make_twin(tmp_path)
    atomic_json(base/"iter_1.json",{"metrics":{"score":.99},"images":{}})
    result=data(tmp_path).twin()
    assert result["gap"] is None and result["motion_px"] is None


def test_image_files_confined_to_explicit_images_under_root(tmp_path):
    adapter=data(tmp_path)
    assert adapter.asset("../../outside.jpg",tmp_path) is None
    secret=tmp_path/"secret.json"
    secret.write_text("{}")
    assert adapter.asset("secret.json",tmp_path) is None


def test_game_counts_metadata_not_filename_separates_bots_and_duration(tmp_path):
    trace(tmp_path,"roblox/out/traces/human_misleading.jsonl",source="bot",level="place_on_cell")
    trace(tmp_path,"roblox/out/traces/bot_misleading.jsonl",source="human",success=True,level="place_on_cell",times=(8,9,10))
    trace(tmp_path,"roblox/out/traces/other.jsonl",source="human",success=False,player="Bea",times=(0,4))
    result=data(tmp_path).game()
    assert result["groups"]["human"]["rounds"]==2
    assert result["groups"]["human"]["successes"]==1
    assert result["groups"]["bot"]["rounds"]==1
    assert result["groups"]["human"]["hours"]==pytest.approx((6+2/30)/3600)
    assert result["leaderboard"][0]["player"]=="Ada"
    ttt=next(row for row in result["levels"] if row["name"]=="place_on_cell")
    assert ttt["human"]==1 and ttt["bot"]==1


def test_in_progress_invalid_trace_excluded_then_recovered(tmp_path):
    path=tmp_path/"roblox/out/traces/partial.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"meta":')
    adapter=data(tmp_path)
    assert adapter.game()["invalid_traces"]==1
    trace(tmp_path,"roblox/out/traces/partial.jsonl")
    assert adapter.game()["groups"]["human"]["rounds"]==1


def test_real_hides_synthetic_status_and_attempts_by_default(tmp_path):
    trace(tmp_path,"runs/probe/attempt_01/trace.jsonl",source="bot")
    trace(tmp_path,"runs/probe/attempt_02/trace.jsonl",source="real")
    status={"source":"bot","argon_says":"mock reasoning"}
    result=data(tmp_path,status).real()
    assert result["frame"] is None and not result["available"] and result["argon_says"] is None
    assert len(result["attempts"])==1 and not result["attempts"][0]["synthetic"]
    shown=data(tmp_path,status,show_dry_runs=True).real()
    assert shown["label"]=="SIM DRY RUN" and len(shown["attempts"])==2


def test_real_status_has_argon_says_and_explicit_provenance(tmp_path):
    result=data(tmp_path,{"source":"real","argon_says":"Take the corner","status":"moving"}).real()
    assert result["available"] and result["argon_says"]=="Take the corner" and result["label"]=="REAL"
    assert not data(tmp_path,{"status":"moving"}).real()["available"]


def test_offline_camera_is_unknown_not_synthetic(tmp_path):
    def offline(_): raise OSError("no publisher")
    adapter=DashboardData(tmp_path,config(),fetch_json=offline)
    result=adapter.real()
    assert not result["available"] and "offline" in result["message"]


def test_pinned_eval_and_scaling(tmp_path):
    atomic_json(tmp_path/"exports/pinned/config.json",{"name":"fixed-policy","eval":{"clean":.8,"real":.6,"n":100}})
    atomic_json(tmp_path/"exports/newer/config.json",{"name":"newer","eval":{"clean":1,"real":1,"n":1}})
    atomic_json(tmp_path/"runs/scaling.json",{"points":[{"episodes":400,"clean":.8},{"episodes":50,"clean":.5}]})
    result=data(tmp_path).policy()
    assert result["name"]=="fixed-policy" and result["real"]==.6
    assert result["evidence"]=="TWIN EVALUATION"
    assert [p["episodes"] for p in result["scaling"]]==[50,400]


def test_paired_eval_report_adapter(tmp_path):
    atomic_json(tmp_path/"exports/pinned/config.json",{"policy":"paired","results":{
        "clean":[{"success":True},{"success":True}],"real":[{"success":True},{"success":False}]}})
    result=data(tmp_path).policy()
    assert result["clean"]==1 and result["real"]==.5 and result["n"]==2


def test_ttt_model_fallback_and_board_are_reported(tmp_path):
    atomic_json(tmp_path/"runs/ttt/status.json",{"synthetic":True,"board":["X","",None,"O","","","","",""],
                                               "why":"Prevent a fork","model":"local-minimax","source":"minimax"})
    result=data(tmp_path).ttt()
    assert result["board"][0]=="X" and result["board"][2]==""
    assert result["why"]=="Prevent a fork" and result["model"]=="local-minimax" and result["synthetic"]


def test_run_manifest_labels(tmp_path):
    for mode in ["dry-run","real","plan"]:
        atomic_json(tmp_path/f"runs/{mode}/manifest.json",{"schema":"r2p_run/1","mode":mode,"task":"t","rounds":{}})
    result=data(tmp_path).runs()
    assert {r["mode"]:r["label"] for r in result}=={"dry-run":"SIM DRY RUN","real":"REAL","plan":"PLAN"}


@contextmanager
def http_server(adapter):
    server=make_server(adapter,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def get(url):
    with urllib.request.urlopen(url,timeout=3) as response:
        return response.read(),response.headers


def test_mock_http_serves_four_panels_runs_and_honest_data(tmp_path):
    adapter=DashboardData(tmp_path,config(),mock=True)
    with http_server(adapter) as base:
        html,headers=get(base+"/")
        for label in (b"REAL",b"REAL2SIM",b"GAME",b"POLICY",b"Tic-tac-toe"):
            assert label in html
        assert b"Run history" in get(base+"/runs")[0]
        state=json.loads(get(base+"/api/state")[0])
        assert state["mock"] and state["twin"]["synthetic"] and state["real"]["frame"] is None
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base+"/static/%2e%2e/config.json")
        assert error.value.code==404
        with pytest.raises(urllib.error.HTTPError):
            get(base+"/qr.png")


def test_qr_is_locally_decodable():
    import cv2
    import numpy as np
    url="https://www.roblox.com/games/12345/Real2Play"
    image=cv2.imdecode(np.frombuffer(qr_png(url),np.uint8),cv2.IMREAD_GRAYSCALE)
    decoded,_,_=cv2.QRCodeDetector().detectAndDecode(image)
    assert decoded==url


def test_qr_route_uses_configured_game_url(tmp_path):
    adapter=DashboardData(tmp_path,config(game_url="https://www.roblox.com/games/12345"),mock=True)
    with http_server(adapter) as base:
        image,headers=get(base+"/qr.png")
        assert image.startswith(b"\x89PNG") and headers["Content-Type"]=="image/png"
