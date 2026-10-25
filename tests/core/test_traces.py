import copy
import json
import numpy as np
import pytest

from r2p.tasks import load_task
from r2p.traces import join_upload_parts, read_trace, validate_trace, write_trace


def data():
    meta = dict(task=load_task("cube_in_bowl"),source="expert",seed=4,success=True,
                commands=[[0,.1,"pick(cube)"]],rig="so101_bench")
    ticks = [dict(t=i/30,q=[0.1]*6,q_target=[0.2]*6,poses={"cube":{"xyz":[.2,0,.019],"yaw_deg":0}},
                  grip_closed=False,command="pick(cube)",phase="approach") for i in range(3)]
    return meta,ticks


def test_round_trip_preserves_labels_extra_fields_and_inputs(tmp_path):
    meta,ticks = data()
    ticks[0]["q_exec"] = [-.2]*6
    saved = copy.deepcopy((meta,ticks))
    path = write_trace(tmp_path/"nested"/"trace.jsonl",meta,(tick for tick in ticks))
    assert (meta,ticks) == saved
    recovered_meta,recovered_ticks = read_trace(path)
    assert recovered_meta["schema"] == "r2p_trace/1"
    assert recovered_meta["task_name"] == "cube_in_bowl"
    assert recovered_ticks == ticks
    assert not validate_trace(recovered_meta,recovered_ticks)
    assert json.loads(path.read_text().splitlines()[0]) == {"meta":recovered_meta}


def test_numpy_values_and_default_metadata(tmp_path):
    meta,ticks = data()
    meta["task"] = "cube_in_bowl"
    ticks[0]["q"] = np.arange(6,dtype=np.float32)
    ticks[0]["q_target"] = np.ones(6)
    ticks[0]["grip_closed"] = np.bool_(True)
    ticks[0].pop("command")
    ticks[0].pop("phase")
    path = write_trace(tmp_path/"trace.jsonl",meta,ticks)
    _,actual = read_trace(path)
    assert actual[0]["q"] == list(range(6))
    assert actual[0]["command"] is None and actual[0]["phase"] is None


@pytest.mark.parametrize("change", ["missing_label","wrong_joints","nan","bad_time","gap","bad_pose","bad_grip"])
def test_invalid_ticks_are_rejected_without_overwriting(tmp_path,change):
    meta,ticks = data()
    path = write_trace(tmp_path/"trace.jsonl",meta,ticks)
    original = path.read_bytes()
    if change == "missing_label": ticks[0].pop("q_target")
    elif change == "wrong_joints": ticks[0]["q_target"] = [0]*5
    elif change == "nan": ticks[0]["q_target"][0] = float("nan")
    elif change == "bad_time": ticks[1]["t"] = 0
    elif change == "gap": ticks[1]["t"] = .1
    elif change == "bad_pose": ticks[0]["poses"]["cube"]["xyz"] = [0,0]
    elif change == "bad_grip": ticks[0]["grip_closed"] = 1
    with pytest.raises(ValueError):
        write_trace(path,meta,ticks)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("field,value", [("schema","old"),("source","scripted"),("task_name","wrong"),
    ("success",1),("seed",True),("scene",[]),("commands",[[.1,0,"home"]]),
    ("commands",[[0,.2,"home"],[.1,.3,"release"]]), ("rig",5)])
def test_invalid_metadata(tmp_path,field,value):
    meta,ticks = data()
    meta[field] = value
    with pytest.raises(ValueError):
        write_trace(tmp_path/"trace.jsonl",meta,ticks)


@pytest.mark.parametrize("text", ["",'{}\n','{"meta":{}}\n\n','{"meta":{}}\n{"meta":{}}\n',
    '{"meta":{}}\n{"meta_final":{}}\n','{"meta":{}}\nnot-json\n'])
def test_read_rejects_malformed_envelopes(tmp_path,text):
    path=tmp_path/"bad.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_trace(path)


def upload():
    meta,ticks=data()
    meta["success"]=False
    text="\n".join(json.dumps(record) for record in [{"meta":meta},*ticks,{"meta_final":{"success":True,"fail_reason":None,"commands":[[0,.1,"pick(cube)"]]}}])+"\n"
    return text,ticks


def test_upload_join_sorts_parts_merges_trailer_and_saves(tmp_path):
    text,ticks=upload()
    cut=len(text)//3
    parts=[{"part":2,"data":text[2*cut:]},{"part":0,"data":text[:cut]},{"part":1,"data":text[cut:2*cut]}]
    path=tmp_path/"joined.jsonl"
    meta,actual=join_upload_parts(parts,path)
    assert meta["success"] and actual == ticks
    assert read_trace(path) == (meta,actual)
    assert "meta_final" not in path.read_text()
    assert join_upload_parts({0:text[:cut],1:text[cut:]}) == (meta,actual)


def test_upload_chunks_can_split_utf8_codepoints():
    text,_=upload()
    text=text.replace('"approach"','"café"')
    encoded=text.encode("utf-8")
    cut=encoded.index(b"\xc3")+1
    _,ticks=join_upload_parts([encoded[:cut],encoded[cut:]])
    assert ticks[0]["phase"] == "café"


@pytest.mark.parametrize("parts", [[],{1:""},[{"part":0,"data":""},{"part":0,"data":""}],
                                   [{"part":0,"data":""},{"part":2,"data":""}]])
def test_upload_missing_duplicate_parts(parts):
    with pytest.raises(ValueError):
        join_upload_parts(parts)


def test_upload_cannot_change_identity_or_insert_metadata_mid_trace():
    text,_=upload()
    text=text.replace('"meta_final": {"success": true','"meta_final": {"task_name": "different", "success": true')
    with pytest.raises(ValueError,match="cannot change"):
        join_upload_parts([text])
    text,_=upload()
    lines=text.splitlines()
    lines.insert(2,'{"meta_final":{"success":true}}')
    with pytest.raises(ValueError,match="Unexpected metadata"):
        join_upload_parts(["\n".join(lines)])


def test_join_roblox_http_payloads(tmp_path):
    meta,ticks = data()
    meta.update(source="human",success=False)
    parts = [dict(trace_id="human_123",part=1,final=True,ticks=ticks[1:],meta_final={"success":True}),
             dict(trace_id="human_123",part=0,final=False,meta=meta,ticks=ticks[:1])]
    saved = copy.deepcopy(parts)
    actual_meta,actual_ticks = join_upload_parts(parts,tmp_path/"http.jsonl")
    assert actual_meta["success"] and actual_ticks == ticks
    assert parts == saved
    assert read_trace(tmp_path/"http.jsonl") == (actual_meta,actual_ticks)
    assert join_upload_parts({p["part"]:p for p in parts}) == (actual_meta,actual_ticks)


@pytest.mark.parametrize("change", ["different_id","no_final","early_final","mid_meta","mid_meta_final"])
def test_join_rejects_incomplete_or_inconsistent_http_payloads(change):
    meta,ticks=data()
    parts=[dict(trace_id="one",part=0,final=False,meta=meta,ticks=ticks[:1]),
           dict(trace_id="one",part=1,final=True,ticks=ticks[1:])]
    if change == "different_id": parts[1]["trace_id"]="two"
    elif change == "no_final": parts[1]["final"]=False
    elif change == "early_final": parts[0]["final"]=True
    elif change == "mid_meta": parts[1]["meta"]=meta
    elif change == "mid_meta_final": parts[0]["meta_final"]={"success":True}
    with pytest.raises(ValueError):
        join_upload_parts(parts)
